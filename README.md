# AI Fraud Detection Prototype: JPMorgan Chase Case Study

A scaled-down, end-to-end transaction-fraud detection system: model training and evaluation in a Jupyter notebook, a real-time scoring API, and a live operations dashboard. Built for **MIST 7400: Advanced AI Value Chain & Enterprise Implementation** (Terry College of Business, University of Georgia) as the applied component of a case study on JPMorgan Chase's AI fraud detection program.

[![Open the notebook in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/elguillejr/jpm-fraud-detection-prototype/blob/main/JPM_Fraud_Prototype.ipynb)

**Live demo:** [jpm-fraud-demo-mist7400.streamlit.app](https://jpm-fraud-demo-mist7400.streamlit.app). Click **Start** to replay the test period as live traffic. The app sleeps when idle, so the first load can take a minute.

> This is an academic project. It uses a public dataset, not JPMorgan Chase data, and is not affiliated with or endorsed by JPMorgan Chase & Co.

## What it does

| Layer | File | What it shows |
|---|---|---|
| Model development and evaluation | `JPM_Fraud_Prototype.ipynb` | Time-ordered split, walk-forward validation, three models compared, cost-based thresholds, three-tier routing, SHAP explanations, a fairness proxy, a drift check, a latency benchmark and a model card |
| Real-time scoring service | `demo/scoring_service.py` | FastAPI endpoint that returns a fraud score, a decision (approve / step-up / decline), reason codes, the model version and the scoring time, with an audit log of every decision |
| Operations dashboard | `demo/dashboard.py` | Streamlit app that replays the test period as live traffic: fraud intercepted, customers affected, loss avoided, latency, a drift monitor with a retraining trigger, and an audit-trail viewer |

## Key results (held-out test period: 56,962 transactions, 75 frauds)

| Measure | Result |
|---|---|
| XGBoost PR-AUC | **0.793** (logistic regression 0.752; Isolation Forest 0.039) |
| Cost-tuned threshold ($15 per false alarm) | 0.54: 57 of 75 frauds caught, 17 false alarms |
| Test-period fraud cost | $7,729 with no model, $2,893 with the model (**62.6% lower**) |
| Three-tier routing (step-up ≥ 0.19, decline ≥ 0.86) | 59.7 frauds intercepted; most flagged good customers get a step-up check instead of a decline |
| Scoring latency, single transaction | Median 0.77 ms, p99 1.10 ms against a 100 ms authorization budget (Apple silicon) |

Results were produced with Python 3.14, XGBoost 3.4.1 and SHAP 0.54.1. Other hardware or library versions can shift them slightly.

## Run it locally

```bash
git clone https://github.com/elguillejr/jpm-fraud-detection-prototype.git
cd jpm-fraud-detection-prototype
pip install -r requirements.txt jupyter

# 1. Train and evaluate (downloads the dataset from OpenML on first run, about 150 MB)
jupyter nbconvert --to notebook --execute JPM_Fraud_Prototype.ipynb --inplace

# 2. Start the scoring API and the dashboard
bash demo/run_demo.sh
```

Then open the dashboard at http://localhost:8501 and the API docs at http://127.0.0.1:8000/docs.

The trained model (`outputs/fraud_xgb_model.json`) and a test-period extract (`demo/test_stream.csv.gz`) are included, so the dashboard can run without the full dataset: `streamlit run demo/dashboard.py`.

## Host the live demo (Streamlit Community Cloud, free)

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. **Create app → Deploy a public app from GitHub**. Repository: `elguillejr/jpm-fraud-detection-prototype`, branch `main`, main file path `demo/dashboard.py`.
3. Deploy. With no API running, the dashboard scores in-process with the same model and code, so the results are identical.

## Repository layout

```
JPM_Fraud_Prototype.ipynb      notebook (model, evaluation, routing, latency, model card)
demo/                          scoring API, dashboard, run script, test-period data
outputs/                       trained model, metadata, 300 DPI figures, result tables
requirements.txt
```

## Data

ULB/Worldline credit card fraud dataset: 284,807 European card transactions from September 2013, with 492 frauds (0.17%). Features V1–V28 are anonymized principal components.
Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. *2015 IEEE Symposium Series on Computational Intelligence*, 159–166. Available on [OpenML (ID 1597)](https://www.openml.org/d/1597) and [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## Key assumptions

$15 cost per false alarm; $5 friction per step-up and $50 per wrong decline; a step-up challenge stops 90% of the fraud sent to it; retraining triggers when the model catches fewer than 60% of the last 15 frauds; 100 ms scoring budget. Each assumption is varied or discussed in the notebook.

## Credits

Prototype by Guillermo Ledezma (Prototype Lead), for a MIST 7400 group project.
AI use: code was drafted with help from Claude (Anthropic) and reviewed, run and tested by the author. All results come from executing the code.
