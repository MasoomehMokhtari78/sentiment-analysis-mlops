# Week 3 — Model Serving & Optimization

This guide explains how to integrate and run the Week 3 components added to the [sentiment-analysis-mlops](https://github.com/MasoomehMokhtari78/sentiment-analysis-mlops) project.

See also: [ARCHITECTURE.md](./ARCHITECTURE.md) (system flowcharts) · [TECHNOLOGIES.md](./TECHNOLOGIES.md) (stack overview)

## New Components

```text
src/serving/
├── app.py                    # FastAPI application (Swagger at /docs)
├── config.py                 # Environment-driven serving settings
├── schemas.py                # Pydantic request/response models
├── model_loader.py           # MLflow Model Registry loader + fallback
├── batching.py               # Async micro-batching for concurrent requests
├── benchmarking.py           # Latency & memory measurement utilities
├── optimization/
│   └── quantization.py       # INT8 quantization (ONNX + optional PyTorch)
└── predictors/
    ├── baseline.py           # Sequential sklearn inference
    ├── batched.py            # Vectorized batch inference
    └── quantized.py          # INT8 ONNX Runtime inference

scripts/
├── promote_model.py          # Promote registry version to Production
└── run_benchmark.py          # Baseline vs optimized benchmark report

Dockerfile.api                # Optimized inference container
requirements-api.txt          # Minimal API runtime dependencies
reports/week3/                # Generated benchmark tables (after run)
```

## Prerequisites

Complete Week 1–2 first:

1. `dvc repro` — processed train/test data
2. `python train.py --model-type baseline` — train + register `SentimentBaselineModel`
3. Promote the model for serving:

```bash
python scripts/promote_model.py --stage Production
```

## 1. Install API Dependencies

```bash
pip install -r requirements-api.txt
```

## 2. Run the FastAPI Service (Local)

```bash
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080 --reload
```

Open interactive API docs:

- Swagger UI: http://localhost:8080/docs
- ReDoc: http://localhost:8080/redoc

### Example Requests

**Health check**

```bash
curl http://localhost:8080/health
```

**Single prediction (async micro-batching — default)**

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was absolutely wonderful!\"}"
```

**Explicit vectorized batch**

```bash
curl -X POST "http://localhost:8080/predict/batch?variant=batched" \
  -H "Content-Type: application/json" \
  -d "{\"texts\": [\"great film\", \"terrible acting\", \"it was okay\"]}"
```

**INT8 quantized inference**

```bash
curl -X POST "http://localhost:8080/predict?variant=quantized" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Not bad, not great.\"}"
```

## 3. Docker Deployment (recommended)

The API container is self-contained: on startup it waits for MLflow, trains/seeds a
baseline model if none exists, promotes it to Production, then serves FastAPI.

**One-command build + test (PowerShell):**

```powershell
.\scripts\docker_test_api.ps1
```

**Manual steps:**

```bash
docker compose build mlflow api
docker compose up -d mlflow api
```

Wait ~1–2 minutes for the API to seed the model and start, then verify:

```bash
curl http://localhost:8080/health
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"This movie was absolutely wonderful!\"}"
```

Swagger UI: http://localhost:8080/docs

**If Docker build fails with `input/output error`:** restart Docker Desktop, free disk
space, then run `docker builder prune -f` and retry.

Or build the API image only:

```bash
docker build -f Dockerfile.api -t sentiment-api:latest .
docker run --rm -p 8080:8080 \
  -e MLFLOW_TRACKING_URI=sqlite:///mlflow.db \
  -v "%cd%/models:/app/models" \
  sentiment-api:latest
```

## 4. Optimization Techniques (Chapter 22)

| Technique | Module | Description |
|-----------|--------|-------------|
| **Vectorization & Batching** | `predictors/batched.py`, `batching.py` | Routes multiple texts through one `predict` / `predict_proba` call; async queue coalesces concurrent `/predict` requests |
| **Quantization (INT8)** | `optimization/quantization.py` | Exports sklearn pipeline to ONNX, applies dynamic UINT8 quantization via ONNX Runtime |

For DistilBERT (PyTorch), use `quantize_pytorch_model()` in `quantization.py` when serving transformer weights.

## 5. Benchmark & Week 3 Report Table

Generate the comparative Baseline vs Optimized metrics:

```bash
python scripts/run_benchmark.py --samples 200 --runs 5 --output-dir reports/week3
```

Outputs:

- `reports/week3/week3_benchmark.json` — machine-readable metrics
- `reports/week3/week3_benchmark.md` — markdown table for your report

The table compares:

- **baseline** — sequential sklearn inference
- **batched** — vectorized batch processing
- **quantized** — INT8 ONNX model

Metrics include mean/P50/P95 latency, throughput, peak memory, and model size.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking backend |
| `MLFLOW_MODEL_NAME` | `SentimentBaselineModel` | Registry model name |
| `MLFLOW_MODEL_STAGE` | `Production` | Stage to load |
| `FALLBACK_MODEL_PATH` | `models/baseline_model.pkl` | Local fallback if registry unavailable |
| `BATCH_MAX_SIZE` | `32` | Max async micro-batch size |
| `BATCH_WAIT_MS` | `25` | Max wait before flushing partial batch |
| `QUANTIZED_MODEL_DIR` | `models/quantized` | ONNX artifact directory |

## Integration Checklist

- [ ] Week 2 model registered in MLflow (`SentimentBaselineModel`)
- [ ] Model promoted to `Production` stage
- [ ] `pip install -r requirements-api.txt`
- [ ] API starts and `/health` returns `healthy`
- [ ] `/docs` shows OpenAPI schema
- [ ] Benchmark report generated in `reports/week3/`
- [ ] Docker image builds and serves on port 8080
