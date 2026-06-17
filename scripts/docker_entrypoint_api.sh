#!/bin/sh
set -e

MLFLOW_HOST="${MLFLOW_HOST:-mlflow}"
MLFLOW_PORT="${MLFLOW_PORT:-5000}"
MLFLOW_URI="${MLFLOW_TRACKING_URI:-http://${MLFLOW_HOST}:${MLFLOW_PORT}}"

echo "Waiting for MLflow at ${MLFLOW_URI} ..."
python - <<'PY'
import os, sys, time, urllib.request
uri = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
for attempt in range(60):
    try:
        with urllib.request.urlopen(uri, timeout=3):
            print("MLflow is reachable")
            sys.exit(0)
    except Exception:
        time.sleep(2)
print("MLflow not reachable after 120s", file=sys.stderr)
sys.exit(1)
PY

export MLFLOW_TRACKING_URI="${MLFLOW_URI}"

echo "Ensuring a Production model is available..."
python scripts/seed_model.py

echo "Starting FastAPI on port 8080..."
exec uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --workers 1
