#!/bin/bash
# Starts the scoring API and the dashboard. Run from the project folder:  bash demo/run_demo.sh
cd "$(dirname "$0")/.." || exit 1
if [ ! -f outputs/model_meta.json ]; then
  echo "Run JPM_Fraud_Prototype.ipynb (Run All) first. It saves the model the demo uses."; exit 1
fi
rm -f outputs/audit_log.jsonl
python3 -m uvicorn demo.scoring_service:app --port 8000 &
API_PID=$!
trap "kill $API_PID" EXIT
sleep 3
echo ""
echo "Dashboard: http://localhost:8501   API docs: http://127.0.0.1:8000/docs   (Ctrl+C to stop)"
python3 -m streamlit run demo/dashboard.py --server.headless true --browser.gatherUsageStats false
