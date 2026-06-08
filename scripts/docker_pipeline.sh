#!/usr/bin/env bash
# Full Docker pipeline: MLflow + data prep + ingest + multiple training runs
set -euo pipefail

RUNS="${1:-3}"
MODEL_TYPE="${2:-baseline}"
DATA_SAMPLE_SIZE="${3:-5000}"

cd "$(dirname "$0")/.."

echo "=== Building Docker images ==="
docker compose build

echo ""
echo "=== Starting MLflow server ==="
docker compose up mlflow -d

echo ""
echo "=== Downloading sample data ==="
docker compose run --rm app scripts/prepare_data.py --sample-size "$DATA_SAMPLE_SIZE"

echo ""
echo "=== Running data ingest pipeline ==="
docker compose run --rm app -m src.data.ingest \
  --input-path data/raw/imdb_train.csv \
  --sample-size "$DATA_SAMPLE_SIZE"

echo ""
echo "=== Running $RUNS training experiments ($MODEL_TYPE) ==="
docker compose run --rm app run_experiments.py --runs "$RUNS" --model-type "$MODEL_TYPE"

echo ""
echo "=== Done ==="
echo "MLflow UI: http://localhost:5000"
echo "Models saved to: ./models/"
