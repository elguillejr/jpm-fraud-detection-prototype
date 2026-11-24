"""
JPMorgan Chase AI Fraud Detection - mock real-time scoring service (MIST 7400 prototype).

This is the "Deployment and Operations" stage of the value chain in miniature: the model
trained in JPM_Fraud_Prototype.ipynb runs behind an HTTP API, the way an authorization
system would call a fraud score before approving a card payment.

Each call returns:
  - a fraud score (0 to 1)
  - a decision tier: APPROVE, CHALLENGE (step-up check such as a one-time code) or DECLINE
  - the top reason codes (SHAP) for any flagged transaction
  - the model version and how long scoring took
Every decision is written to an audit log (outputs/audit_log.jsonl), so a disputed decline
can be traced to the model version, score and reasons (report Section 6.2).

Run from the project folder:
    uvicorn demo.scoring_service:app --port 8000
"""
import json, time
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "outputs" / "fraud_xgb_model.json"
META_PATH = ROOT / "outputs" / "model_meta.json"
AUDIT_PATH = ROOT / "outputs" / "audit_log.jsonl"

if not MODEL_PATH.exists() or not META_PATH.exists():
    raise SystemExit("Model files not found. Run JPM_Fraud_Prototype.ipynb (Run All) first: "
                     "it saves outputs/fraud_xgb_model.json and outputs/model_meta.json.")

META = json.loads(META_PATH.read_text())
FEATURES = META["features"]
# Three-tier routing cutoffs from the notebook (Section 7): the thresholds the cost analysis
# picks at a $5 and a $50 false-alarm cost. Fall back to older metadata if needed.
CHALLENGE_T = META.get("tier_stepup_threshold", META["threshold"])
DECLINE_T = META.get("tier_decline_threshold", 0.90)

model = xgb.XGBClassifier()
model.load_model(MODEL_PATH)
booster = model.get_booster()
explainer = shap.TreeExplainer(model)

app = FastAPI(title="JPM Fraud Scoring Service (MIST 7400 prototype)", version=META["model_version"])


class Transaction(BaseModel):
    transaction_id: str
    amount: float = Field(ge=0, description="Transaction amount in USD")
    v: List[float] = Field(min_length=28, max_length=28,
                           description="The 28 anonymized PCA features V1..V28")
    explain: Optional[bool] = Field(None, description="Force reason codes even if approved")


def to_frame(txns: List[Transaction]) -> pd.DataFrame:
    rows = [t.v + [float(np.log1p(t.amount))] for t in txns]
    return pd.DataFrame(rows, columns=FEATURES)


def decide(score: float) -> str:
    if score >= DECLINE_T:
        return "DECLINE"
    if score >= CHALLENGE_T:
        return "CHALLENGE"
    return "APPROVE"


def reasons(x_row: pd.DataFrame, k: int = 3):
    contrib = explainer.shap_values(x_row)[0]
    top = np.argsort(-contrib)[:k]          # features pushing the score UP the most
    return [{"feature": FEATURES[i], "contribution": round(float(contrib[i]), 3)} for i in top]


def audit(entry: dict):
    with AUDIT_PATH.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def score_one(t: Transaction) -> dict:
    x = to_frame([t])
    t0 = time.perf_counter()
    s = float(booster.inplace_predict(x)[0])          # model only: this is the latency that matters
    model_ms = (time.perf_counter() - t0) * 1000
    decision = decide(s)

    rc, explain_ms = [], 0.0
    if decision != "APPROVE" or t.explain:
        t1 = time.perf_counter()
        rc = reasons(x)
        explain_ms = (time.perf_counter() - t1) * 1000

    result = {"transaction_id": t.transaction_id, "amount": t.amount, "fraud_score": round(s, 4),
              "decision": decision, "reason_codes": rc, "model_version": META["model_version"],
              "model_ms": round(model_ms, 3), "explain_ms": round(explain_ms, 3)}
    audit({**result, "ts": time.time()})
    return result


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse("/docs")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": META["model_version"]}


@app.get("/model-card")
def model_card():
    return {**META, "challenge_threshold": CHALLENGE_T, "decline_threshold": DECLINE_T,
            "decision_tiers": {"APPROVE": f"score < {CHALLENGE_T}",
                               "CHALLENGE": f"{CHALLENGE_T} <= score < {DECLINE_T}",
                               "DECLINE": f"score >= {DECLINE_T}"}}


@app.post("/score")
def score(t: Transaction):
    return score_one(t)


@app.post("/score/batch")
def score_batch(txns: List[Transaction]):
    if len(txns) > 5000:
        raise HTTPException(413, "Send at most 5,000 transactions per batch")
    return [score_one(t) for t in txns]


@app.get("/audit/{transaction_id}")
def audit_lookup(transaction_id: str):
    """Trace a disputed decision back to its score, reasons and model version."""
    if not AUDIT_PATH.exists():
        raise HTTPException(404, "No decisions logged yet")
    hits = [json.loads(l) for l in AUDIT_PATH.open() if f'"transaction_id": "{transaction_id}"' in l]
    if not hits:
        raise HTTPException(404, f"No decision found for {transaction_id}")
    return hits[-1]
