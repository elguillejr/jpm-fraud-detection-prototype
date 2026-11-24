# Live Fraud-Scoring Demo (MIST 7400 Prototype)

The notebook trains and evaluates the model. This demo shows the same model **deployed**: a
scoring API that an authorization system would call, and an operations dashboard that watches
it. It maps to the Deployment and Operations stage of the value chain (report Section 2.5) and
to the governance controls in Section 6. The demo includes an audit trail, reason codes and a
drift-triggered retraining alert.

## Parts

| File | What it is |
|---|---|
| `scoring_service.py` | FastAPI service. `POST /score` returns a fraud score, a decision (APPROVE / CHALLENGE / DECLINE), the top 3 SHAP reason codes for flagged transactions, the model version and scoring time. Every decision is logged to `outputs/audit_log.jsonl`. `GET /audit/{id}` traces a disputed decision. `GET /model-card` shows thresholds and model details. |
| `dashboard.py` | Streamlit dashboard. Replays the 56,962 held-out test transactions in time order through the API and shows fraud caught, false alarms, loss avoided, latency, a drift monitor and an audit-trail viewer. |
| `run_demo.sh` | Starts both. |

## Run it on a Mac

From the project folder, in Terminal:

```bash
pip3 install -r requirements.txt -r demo/requirements-demo.txt
jupyter nbconvert --to notebook --execute JPM_Fraud_Prototype.ipynb --inplace   # or open it and Run All
bash demo/run_demo.sh
```

The dashboard opens in the browser at http://localhost:8501. The API docs, which are also
worth showing in the video, are at http://localhost:8000/docs. Press Ctrl+C in Terminal to
stop both.

## Design choices (state these as assumptions)

- **Decision tiers** come from the notebook's three-tier routing (Section 7), read from `outputs/model_meta.json`. Step-up starts at the threshold the cost analysis picks for a $5 false alarm, and hard decline starts at the one it picks for a $50 false alarm. Middle scores get a step-up check (for example, a one-time code) instead of a hard decline, which reduces friction for good customers.
- **Cost model** is the same as the notebook's: $5 friction per step-up of a good customer, $50 per wrong decline, and step-up stops 90% of the fraud sent to it. All three are *assumptions*.
- **Retraining trigger** = the model catches fewer than 60% of the last 15 frauds. *Assumption*, chosen for the demo.
- **Labels** are revealed as soon as a transaction is scored so the model can be graded live. In production, fraud labels arrive days or weeks later (Dal Pozzolo et al., 2018).
- **"Simulate a new fraud pattern"** shrinks the four strongest fraud signals (V14, V12, V10, V4) in fraud transactions to 35% of their size. This imitates fraudsters adapting to the model. It is a simulation, not real drift.

## Check that it matches the notebook

With drift off, a full replay ends with the same results as the "Three-tier total" row of
`outputs/table6_three_tier_routing.csv`: frauds intercepted, good customers affected and net
cost. It is the same model, the same cutoffs and the same cost model. On the official run
(step-up at 0.19, decline at 0.86): 59.7 of 75 frauds intercepted, 75 good customers affected
(66 challenged, 9 declined), net cost $3,202 vs $7,729 with no model ($4,527 avoided, 59%).

## Hosted version

When no API answers at the configured address, the dashboard loads `scoring_service.py`
in-process and reads `demo/test_stream.csv.gz` (the test period only). This is how it runs
on Streamlit Community Cloud. The results are identical.

## Suggested 2-minute demo script (slide 11)

1. **(15 s) Architecture.** Show http://localhost:8000/docs: "The model runs as a service the payment system calls before approving a card."
2. **(30 s) Live replay.** Start at 2,000 transactions per tick. Point at recall, false alarms and latency: "about 2 ms per decision against a 100 ms budget."
3. **(30 s) One decision.** Open a declined fraud in the audit-trail viewer: score, decision, reason codes, model version. "Every decision can be traced if a customer disputes it."
4. **(30 s) Drift.** Partway through, switch on *Simulate a new fraud pattern*. Recall falls and the retraining trigger fires. "This is why monitoring and retraining are budgeted in the financial model."
5. **(15 s) Limitation.** The features are anonymized, so reason codes cannot be shown in plain English.

## AI use

Claude (Anthropic) drafted the code for the scoring service and dashboard. The Prototype Lead
ran, tested and reviewed it.
