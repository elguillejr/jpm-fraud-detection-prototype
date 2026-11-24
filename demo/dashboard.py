"""
JPMorgan Chase AI Fraud Detection - live operations dashboard (MIST 7400 prototype).

Replays the held-out test period (56,962 card transactions, in time order) through the
scoring API as if they were arriving live, and shows what a fraud-operations team would
watch: decisions, fraud caught and missed, false alarms, dollars saved, latency, and a
drift monitor that raises a retraining trigger.

Two ways to run it:
  - With the scoring API (local demo):  bash demo/run_demo.sh
  - Self-contained (hosted demo, e.g. Streamlit Community Cloud):  streamlit run demo/dashboard.py
    If no API answers at the configured address, the dashboard loads the same scoring code
    in-process, so the results are identical.
"""
import sys
import time
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import requests
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "creditcard.csv"
DATA_TEST = ROOT / "demo" / "test_stream.csv.gz"   # test period only, small enough for GitHub
V = [f"V{i}" for i in range(1, 29)]
WINDOW = 5000                 # transactions per monitoring window
RECALL_TRIGGER = 0.60         # ASSUMPTION: retrain if rolling recall falls below 60%
LAST_N_FRAUDS = 15            # rolling recall is judged on the most recent 15 frauds

st.set_page_config(page_title="JPM Fraud Ops (prototype)", layout="wide")


@st.cache_data
def load_stream(test_start: int) -> pd.DataFrame:
    if DATA_TEST.exists():
        return pd.read_csv(DATA_TEST)
    df = pd.read_csv(DATA)
    df["Class"] = df["Class"].astype(str).str.strip("'").astype(int)
    return df.iloc[test_start:].reset_index(names="row")


@st.cache_resource
def local_service():
    """Load the scoring service in-process (used when no API is running, e.g. when hosted)."""
    sys.path.insert(0, str(ROOT))
    from demo import scoring_service
    return scoring_service


def api(path, method="get", timeout=60, **kw):
    url = st.session_state.api_url.rstrip("/") + path
    r = getattr(requests, method)(url, timeout=timeout, **kw)
    r.raise_for_status()
    return r.json()


def get_card():
    if st.session_state.mode == "api":
        try:
            return api("/model-card", timeout=3)
        except Exception:
            st.session_state.mode = "embedded"
    return local_service().model_card()


def score_batch(payload):
    if st.session_state.mode == "api":
        return api("/score/batch", method="post", json=payload)
    svc = local_service()
    return [svc.score_one(svc.Transaction(**p)) for p in payload]


def audit_record(txn_id):
    if st.session_state.mode == "api":
        return api(f"/audit/{txn_id}")
    return next(r for r in reversed(st.session_state.log) if r["transaction_id"] == txn_id)


def reset():
    st.session_state.pos = 0
    st.session_state.log = []
    st.session_state.running = False


# ---------- setup ----------
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://localhost:8000"
    st.session_state.mode = "api"
    reset()

with st.sidebar:
    st.header("Replay controls")
    try:
        card = get_card()
    except Exception as e:
        st.error(f"Could not load the scoring model. Run JPM_Fraud_Prototype.ipynb first.\n\n{e}")
        st.stop()
    if st.session_state.mode == "api":
        st.caption(f"Scoring via API at {st.session_state.api_url}")
    else:
        st.caption("Scoring in-process (self-contained mode)")
    batch = st.select_slider("Transactions per tick", [100, 250, 500, 1000, 2000], value=500)
    drift = st.toggle("Simulate a new fraud pattern", value=False,
                      help="From now on, fraud transactions are altered to look more like normal "
                           "spending, the way fraudsters adapt to a model. Watch rolling recall fall.")
    c1, c2 = st.columns(2)
    if c1.button("Start" if not st.session_state.running else "Pause", use_container_width=True, type="primary"):
        st.session_state.running = not st.session_state.running
    if c2.button("Reset", use_container_width=True):
        reset()
    st.divider()
    st.caption(f"**Model** {card['model_version']}  \n"
               f"**Challenge** at score ≥ {card['challenge_threshold']}  \n"
               f"**Decline** at score ≥ {card['decline_threshold']}  \n"
               f"**Friction cost** ${card.get('stepup_cost_usd', 0):.0f} per step-up, "
               f"${card.get('decline_cost_usd', 0):.0f} per wrong decline")
    st.caption("Labels are revealed as soon as a transaction is scored so the model can be graded. "
               "In production, fraud labels arrive days or weeks later.")

stream = load_stream(card["test_start_row"])
STOP_RATE = card.get("stepup_stop_rate", 1.0)
STEPUP_COST = card.get("stepup_cost_usd", card["fp_cost_usd"])
DECLINE_COST = card.get("decline_cost_usd", card["fp_cost_usd"])

# ---------- process one tick ----------
if st.session_state.running and st.session_state.pos < len(stream):
    chunk = stream.iloc[st.session_state.pos: st.session_state.pos + batch].copy()
    if drift:
        is_f = chunk["Class"] == 1
        chunk.loc[is_f, ["V14", "V12", "V10", "V4"]] *= 0.35   # pull fraud toward normal behaviour
    payload = [{"transaction_id": str(r.row), "amount": float(r.Amount), "v": [float(getattr(r, c)) for c in V]}
               for r in chunk.itertuples()]
    results = score_batch(payload)
    for r, (_, row) in zip(results, chunk.iterrows()):
        r["actual"] = int(row["Class"])
        r["drift"] = drift
    st.session_state.log.extend(results)
    st.session_state.pos += len(chunk)
    if st.session_state.pos >= len(stream):
        st.session_state.running = False
        st.rerun()

# ---------- metrics ----------
st.title("Fraud operations: live scoring")
st.caption("MIST 7400 prototype · JPMorgan Chase AI fraud detection · replay of the held-out test period "
           "(ULB card data, Dal Pozzolo et al., 2015)")

log = pd.DataFrame(st.session_state.log)
st.progress(st.session_state.pos / len(stream),
            text=f"{st.session_state.pos:,} of {len(stream):,} transactions scored")

if log.empty:
    st.info("Press **Start** in the sidebar to begin streaming transactions through the model.")
    st.stop()

log["flagged"] = log["decision"] != "APPROVE"
tp = int((log.flagged & (log.actual == 1)).sum())
fn = int((~log.flagged & (log.actual == 1)).sum())
fp = int((log.flagged & (log.actual == 0)).sum())
fraud_total = tp + fn
# Same cost model as the notebook's three-tier table (Section 7)
fraud, legit = log.actual == 1, log.actual == 0
ch, dc = log.decision == "CHALLENGE", log.decision == "DECLINE"
intercepted = (dc & fraud).sum() + STOP_RATE * (ch & fraud).sum()
lost_usd = (log.loc[~log.flagged & fraud, "amount"].sum()
            + (1 - STOP_RATE) * log.loc[ch & fraud, "amount"].sum())
friction_usd = (ch & legit).sum() * STEPUP_COST + (dc & legit).sum() * DECLINE_COST
no_model_cost = log.loc[fraud, "amount"].sum()
model_cost = lost_usd + friction_usd
saved_pct = (no_model_cost - model_cost) / no_model_cost if no_model_cost else 0

tiles = [
    ("Transactions", f"{len(log):,}", "scored so far"),
    ("Flagged", f"{int(log.flagged.sum()):,}",
     f"{(log.decision == 'DECLINE').sum()} declined, {(log.decision == 'CHALLENGE').sum()} challenged"),
    ("Fraud intercepted", f"{intercepted:.1f} of {fraud_total}",
     f"{tp} flagged; step-up stops {STOP_RATE:.0%}" if fraud_total else "no fraud yet"),
    ("Good customers affected", f"{fp}",
     f"{int((ch & legit).sum())} challenged, {int((dc & legit).sum())} declined"),
    ("Net loss avoided", f"${no_model_cost - model_cost:,.0f}", f"{saved_pct:.0%} of fraud cost, net of friction"),
    ("Scoring latency p99", f"{np.percentile(log.model_ms, 99):.2f} ms",
     f"median {log.model_ms.median():.2f} ms, budget 100 ms"),
]
for col, (label, value, note) in zip(st.columns(6), tiles):
    col.metric(label, value)
    col.caption(note)

# ---------- drift monitor ----------
log["window"] = log.index // WINDOW
win = log.groupby("window").apply(lambda g: pd.Series({
    "Alert rate per 10k": g.flagged.mean() * 10000,
    "Frauds": int(g.actual.sum()),
    "Recall": ((g.flagged & (g.actual == 1)).sum() / g.actual.sum()) if g.actual.sum() else np.nan,
    "Drift on": bool(g.drift.any())}), include_groups=False)
win.index = [f"W{i + 1:02d}" for i in win.index]

left, right = st.columns([3, 2])
with left:
    st.subheader("Drift monitor")
    recent = log[log.actual == 1].tail(LAST_N_FRAUDS)
    if len(recent) >= LAST_N_FRAUDS:
        r_recall = recent.flagged.mean()
        if r_recall < RECALL_TRIGGER:
            st.error(f"**Retraining trigger.** The model caught {r_recall:.0%} of the last {LAST_N_FRAUDS} frauds, "
                     f"below the {RECALL_TRIGGER:.0%} threshold. Escalate to model risk for validation and retraining.")
        else:
            st.success(f"Model healthy. It caught {r_recall:.0%} of the last {LAST_N_FRAUDS} frauds "
                       f"(trigger below {RECALL_TRIGGER:.0%}).")
    else:
        st.info(f"Waiting for {LAST_N_FRAUDS} frauds before judging recall ({len(recent)} so far).")
    st.line_chart(win[["Recall"]], height=220)
    st.caption(f"Recall per window of {WINDOW:,} transactions. Small fraud counts make single windows noisy.")

with right:
    st.subheader("Flagged decisions")
    fl = log[log.flagged]
    mix = (pd.crosstab(fl.decision, np.where(fl.actual == 1, "Fraud", "False alarm"))
           .reindex(index=["CHALLENGE", "DECLINE"], columns=["Fraud", "False alarm"]).fillna(0))
    st.bar_chart(mix, height=220, color=["#E45756", "#9D9D9D"])
    st.caption(f"{(log.decision == 'APPROVE').sum():,} transactions approved without friction.")
    st.dataframe(win.round(2), use_container_width=True)

# ---------- flagged transactions ----------
st.subheader("Flagged transactions (most recent first)")
flagged = log[log.flagged].iloc[::-1].copy()
if flagged.empty:
    st.write("Nothing flagged yet.")
else:
    flagged["Outcome"] = np.where(flagged.actual == 1, "Fraud caught", "False alarm")
    flagged["Top reasons"] = flagged.reason_codes.apply(
        lambda rc: ", ".join(f"{r['feature']} (+{r['contribution']:.2f})" for r in rc))
    st.dataframe(flagged[["transaction_id", "amount", "fraud_score", "decision", "Outcome", "Top reasons"]]
                 .rename(columns={"transaction_id": "Txn", "amount": "Amount $", "fraud_score": "Score",
                                  "decision": "Decision"}).head(50),
                 use_container_width=True, hide_index=True)

    st.subheader("Why was this flagged? (audit trail)")
    pick = st.selectbox("Transaction", flagged.transaction_id.head(50).tolist())
    rec = audit_record(pick)
    a, b = st.columns([1, 2])
    a.metric("Fraud score", f"{rec['fraud_score']:.3f}")
    a.write(f"**Decision:** {rec['decision']}  \n**Amount:** ${rec['amount']:,.2f}  \n"
            f"**Model:** {rec['model_version']}")
    rc = pd.DataFrame(rec["reason_codes"])
    b.altair_chart(alt.Chart(rc).mark_bar(color="#E45756").encode(
        x=alt.X("contribution:Q", title="Pushes score toward fraud",
                scale=alt.Scale(domain=[0, max(4.0, float(rc.contribution.max()) * 1.1)])),
        y=alt.Y("feature:N", sort="-x", title=None)).properties(height=160), use_container_width=True)
    b.caption("SHAP contribution to the fraud score (log-odds). The features are anonymized in this "
              "dataset, so reasons cannot be translated into plain language (report Section 5.4).")

if st.session_state.running:
    time.sleep(0.05)
    st.rerun()
