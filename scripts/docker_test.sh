#!/usr/bin/env bash
# Run the full pytest suite inside Docker (CI-equivalent).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

IMAGE="${TEST_IMAGE:-sentiment-test:latest}"

echo "=== Building test image ==="
docker build -f Dockerfile.test -t "$IMAGE" .

mkdir -p htmlcov

echo "=== Lint (Ruff) ==="
docker run --rm "$IMAGE" \
  ruff check src tests scripts train.py update_model.py

echo "=== Pytest with coverage ==="
docker run --rm \
  -v "$ROOT/htmlcov:/app/htmlcov" \
  "$IMAGE" \
  pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

echo "=== Data pipeline smoke ==="
docker run --rm "$IMAGE" python -m src.data.ingest --help

echo ""
echo "All Docker tests passed."
