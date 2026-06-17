# Architecture & System Flow

This document describes how the **Sentiment Analysis MLOps** project works end-to-end: data versioning, training, model registry, serving, and deployment.

For technology background, see [TECHNOLOGIES.md](./TECHNOLOGIES.md).  
For Week 3 setup instructions, see [WEEK3_INTEGRATION.md](./WEEK3_INTEGRATION.md).

---

## 1. End-to-End Overview

The project follows three phases:

| Phase | Purpose | Key entry points |
|-------|---------|------------------|
| **Data** | Version, clean, and split IMDB reviews | `dvc repro`, `src/data/` |
| **Training** | Train models and log to MLflow | `train.py`, `scripts/promote_model.py` |
| **Serving** | Expose optimized inference via FastAPI | `src/serving/`, `Dockerfile.api` |

```mermaid
flowchart TB
    subgraph DATA["Phase 1 — Data (DVC)"]
        RAW["Raw CSVs<br/>data/raw/imdb_train.csv<br/>data/raw/imdb_test.csv"]
        DVC["dvc repro"]
        PROC["Ingest & Process<br/>src/data/ingest.py<br/>src/data/processor.py"]
        OUT["Processed splits<br/>data/processed/train.csv<br/>data/processed/test.csv"]
        RAW --> DVC --> PROC --> OUT
    end

    subgraph TRAIN["Phase 2 — Training (MLflow)"]
        TR["train.py<br/>--model-type baseline | distilbert"]
        BM["BaselineModel<br/>TF-IDF + Logistic Regression"]
        DB["DistilBERTModel<br/>optional transformer"]
        REG["MLflow Model Registry<br/>SentimentBaselineModel"]
        PICKLE["Local artifacts<br/>models/baseline_model.pkl"]
        OUT --> TR
        TR --> BM
        TR --> DB
        BM --> REG
        BM --> PICKLE
    end

    subgraph DEPLOY["Phase 3 — Serving (Week 3)"]
        PROM["promote_model.py<br/>→ Production stage"]
        API["FastAPI app<br/>src/serving/app.py :8080"]
        REG --> PROM --> API
        PICKLE -.->|fallback| API
    end

    subgraph OPS["Operations"]
        BENCH["run_benchmark.py<br/>reports/week3/"]
        DOCKER["docker compose<br/>mlflow + api"]
        API --> BENCH
        DOCKER --> API
    end
```

---

## 2. Data Pipeline

DVC defines a single reproducible stage in `dvc.yaml`. When source code or raw data changes, `dvc repro` re-runs ingestion and regenerates processed CSVs.

```mermaid
flowchart LR
    subgraph INPUT
        T["imdb_train.csv"]
        E["imdb_test.csv"]
    end

    subgraph INGEST["src/data/ingest.py (Prefect flow)"]
        L["IMDBReviewLoader<br/>load CSV"]
        C["BasicTextCleaner<br/>lowercase, strip HTML/URLs<br/>PII masking"]
        S["Stratified train/test split<br/>80/20"]
        SAVE["save_processed_data<br/>idempotent write"]
    end

    subgraph OUTPUT
        TRAIN["train.csv<br/>text, cleaned_text, sentiment"]
        TEST["test.csv"]
    end

    T --> L
    E --> L
    L --> C --> S --> SAVE
    SAVE --> TRAIN
    SAVE --> TEST
```

### Per-review processing steps

1. **Load** — Read CSV; normalize column names (`review` → `text`, `label` → `sentiment`).
2. **Clean** — PII anonymization, URL/HTML removal, whitespace normalization, optional lowercasing.
3. **Filter** — Drop rows with empty `cleaned_text` after cleaning.
4. **Split** — Stratified 80/20 train/test split (class balance preserved).
5. **Save** — Write to `data/processed/` (skipped if files already exist — idempotent).

Raw datasets are tracked with **DVC** (`.dvc` files under `data/raw/`). This keeps large files out of Git while preserving lineage between code and data.

---

## 3. Training & Experiment Tracking

```mermaid
flowchart TB
    START(["python train.py<br/>--model-type baseline"])
    LOAD["Load processed train.csv + test.csv"]
    CHOOSE{Model type?}

    subgraph BASELINE["Baseline (default)"]
        B1["TfidfVectorizer<br/>ngrams 1–2, max 10k features"]
        B2["LogisticRegression<br/>balanced classes"]
        B3["sklearn Pipeline"]
        B1 --> B2 --> B3
    end

    subgraph BERT["DistilBERT (optional)"]
        D1["Tokenizer + DistilBERT"]
        D2["Fine-tune on GPU/CPU"]
        D1 --> D2
    end

    EVAL["Evaluate: accuracy, precision,<br/>recall, F1"]
    MLF["MLflow run<br/>params + metrics + artifacts"]
    SAVE["Save locally<br/>models/baseline_model.pkl"]
    REG["Register SentimentBaselineModel<br/>mlflow.sklearn.log_model"]

    START --> LOAD --> CHOOSE
    CHOOSE -->|baseline| BASELINE
    CHOOSE -->|distilbert| BERT
    BASELINE --> EVAL
    BERT --> EVAL
    EVAL --> MLF
    EVAL --> SAVE
    BASELINE --> REG

    subgraph MLFLOW["MLflow backend"]
        SQLITE["SQLite mlflow.db<br/>or http://mlflow:5000"]
        EXP["Experiment: sentiment-analysis"]
        VER["Model versions<br/>None → Staging → Production"]
        SQLITE --> EXP --> VER
    end

    MLF --> MLFLOW
    REG --> VER
```

### Promotion to Production

After training registers a model version, promote it for serving:

```bash
python scripts/promote_model.py --stage Production
```

The API loads `models:/SentimentBaselineModel/Production` by default.

---

## 4. Docker Deployment Topology

`docker-compose.yml` orchestrates three services:

| Service | Image | Port | Role |
|---------|-------|------|------|
| `mlflow` | `Dockerfile` | 5000 | MLflow tracking server + artifact store |
| `app` | `Dockerfile` | — | Training / experiment container |
| `api` | `Dockerfile.api` | 8080 | FastAPI inference service |

```mermaid
flowchart TB
    subgraph COMPOSE["docker compose"]
        MLF["mlflow service<br/>:5000<br/>SQLite + artifacts volume"]
        APP["app service<br/>training / experiments<br/>depends on mlflow"]
        API["api service<br/>Dockerfile.api<br/>:8080"]
    end

    MLF -->|healthy| API
    MLF --> APP

    subgraph API_STARTUP["API container startup<br/>docker_entrypoint_api.py"]
        W1["Wait for MLflow HTTP"]
        W2["seed_model.py<br/>train mini model if needed<br/>register + promote Production"]
        W3["uvicorn src.serving.app:app"]
        W1 --> W2 --> W3
    end

    API --> API_STARTUP

    VOL1[("mlflow_data volume")]
    VOL2["./models mounted"]
    VOL3["./reports mounted"]

    MLF --- VOL1
    API --- VOL2
    API --- VOL3
```

### Docker bootstrap (`seed_model.py`)

When the API container starts:

1. Check if a **Production** model exists in MLflow → if yes, skip.
2. Otherwise train a minimal baseline on synthetic sample texts.
3. Save to `models/baseline_model.pkl`.
4. Register in MLflow and promote to **Production**.

This makes the API container self-contained for demos without requiring a full `dvc repro` + `train.py` run first.

---

## 5. API Startup & Model Loading

```mermaid
flowchart TB
    LIFESPAN["FastAPI lifespan<br/>src/serving/app.py"]

    LOAD["model_loader.load_sklearn_from_registry()"]
    TRY{MLflow registry<br/>models:/SentimentBaselineModel/Production}
    OK["Load sklearn Pipeline"]
    FALL["Fallback: joblib<br/>models/baseline_model.pkl"]

    BASE["SklearnPredictor<br/>sequential inference"]
    BATCH["BatchedSklearnPredictor<br/>vectorized predict_batch"]
    AB["AsyncBatchProcessor<br/>micro-batch queue"]
    QUANT["export_and_quantize_sklearn()<br/>ONNX → INT8"]
    QRUN["QuantizedOnnxPredictor<br/>ONNX Runtime"]

    LIFESPAN --> LOAD --> TRY
    TRY -->|success| OK
    TRY -->|MlflowException| FALL
    OK --> BASE
    FALL --> BASE
    OK --> BATCH
    FALL --> BATCH
    BATCH --> AB
    OK --> QUANT --> QRUN

    READY(["Serving ready<br/>/health /docs /predict"])
    BASE --> READY
    AB --> READY
    QRUN --> READY
```

At startup the app builds **three inference paths** from one loaded sklearn pipeline:

- **baseline** — one text at a time
- **batched** — vectorized batch calls
- **quantized** — INT8 ONNX model (exported on first start if missing)

---

## 6. Inference Request Flows

### Single prediction — `POST /predict`

Default variant is `async_batched` (concurrent micro-batching).

```mermaid
flowchart TB
    CLIENT["Client POST /predict<br/>{ text: \"...\" }"]
    Q{variant query param}

    CLIENT --> Q

    Q -->|async_batched default| ABQ["AsyncBatchProcessor queue"]
    ABQ --> WAIT["Wait up to BATCH_WAIT_MS (25ms)<br/>or BATCH_MAX_SIZE (32)"]
    WAIT --> VEC["BatchedSklearnPredictor.predict_batch()"]
    VEC --> RESP["PredictResponse<br/>sentiment, confidence, probabilities"]

    Q -->|baseline| SEQ["SklearnPredictor.predict_one()"]
    Q -->|batched| BAT["BatchedSklearnPredictor.predict_one()"]
    Q -->|quantized| ONNX["QuantizedOnnxPredictor<br/>INT8 ONNX Runtime"]

    SEQ --> RESP
    BAT --> RESP
    ONNX --> RESP
```

### Batch prediction — `POST /predict/batch`

```mermaid
flowchart LR
    C["Client sends texts[]"] --> V{variant}
    V -->|baseline| P1["Sequential loop"]
    V -->|batched| P2["Single vectorized call"]
    V -->|quantized| P3["ONNX batch inference"]
    P1 --> R["BatchPredictResponse"]
    P2 --> R
    P3 --> R
```

### API endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health, loaded model, enabled optimizations |
| `/model/info` | GET | Registry URI, stage, class labels |
| `/predict` | POST | Single-text prediction (async batching by default) |
| `/predict/batch` | POST | Explicit multi-text batch |
| `/docs` | GET | Swagger UI (OpenAPI) |
| `/redoc` | GET | ReDoc API reference |

**Sentiment classes:** `positive`, `negative`, `neutral`

---

## 7. Optimization Layer (Week 3)

| Technique | Module | When used |
|-----------|--------|-----------|
| **Async micro-batching** | `src/serving/batching.py` | Default `/predict` — coalesces concurrent requests |
| **Vectorized batching** | `src/serving/predictors/batched.py` | `/predict/batch` or `variant=batched` |
| **INT8 quantization** | `src/serving/optimization/quantization.py` | sklearn → ONNX → UINT8; `variant=quantized` |

```mermaid
flowchart LR
    SK["sklearn Pipeline<br/>TF-IDF + LR"]
    SK --> ONNX["model.onnx"]
    ONNX --> INT8["model.int8.onnx"]
    INT8 --> ORT["ONNX Runtime<br/>faster CPU inference<br/>smaller model size"]
```

Benchmark all variants:

```bash
python scripts/run_benchmark.py --samples 200 --runs 5 --output-dir reports/week3
```

Outputs:

- `reports/week3/week3_benchmark.json` — machine-readable metrics
- `reports/week3/week3_benchmark.md` — comparison table (latency, throughput, memory, model size)

---

## 8. CI/CD Pipeline

```mermaid
flowchart LR
    PUSH["push to main/develop<br/>or PR"] --> TEST["pytest + coverage"]
    TEST -->|push to main only| DOCKER["docker build sentiment-api"]
    DOCKER --> SMOKE["import smoke test<br/>in container"]
```

Defined in `.github/workflows/ci.yml`:

1. **test** — install deps, run `pytest` with coverage on every push/PR.
2. **docker** — build and smoke-test the Docker image on pushes to `main` (after tests pass).

---

## 9. Typical Workflows

### Local development (full pipeline)

```bash
dvc repro
python train.py --model-type baseline
python scripts/promote_model.py --stage Production
uvicorn src.serving.app:app --host 0.0.0.0 --port 8080
python scripts/run_benchmark.py
```

### Docker (recommended for serving)

```powershell
.\scripts\docker_test_api.ps1
```

Or manually:

```bash
docker compose build mlflow api
docker compose up -d mlflow api
curl http://localhost:8080/health
```

### Request lifecycle (happy path)

```
Client → POST /predict { "text": "This movie was wonderful!" }
      → AsyncBatchProcessor queues text
      → flush as batch → BatchedSklearnPredictor
      → TF-IDF transform → LogisticRegression predict_proba
      → JSON: { sentiment, confidence, probabilities, model_variant }
```

---

## 10. Project Structure

```text
sentiment-analysis-mlops/
├── data/
│   ├── raw/              # DVC-tracked source CSVs
│   └── processed/        # DVC pipeline output (train/test)
├── src/
│   ├── data/
│   │   ├── ingest.py     # Prefect-orchestrated ingestion flow
│   │   └── processor.py  # Loaders, cleaners, splits
│   ├── models/
│   │   ├── baseline.py   # TF-IDF + Logistic Regression
│   │   └── distilbert.py # Optional transformer model
│   └── serving/
│       ├── app.py        # FastAPI application
│       ├── model_loader.py
│       ├── batching.py
│       ├── benchmarking.py
│       ├── optimization/ # ONNX INT8 quantization
│       └── predictors/   # baseline, batched, quantized
├── scripts/
│   ├── seed_model.py     # Docker bootstrap
│   ├── promote_model.py  # Registry stage promotion
│   ├── run_benchmark.py  # Week 3 benchmarks
│   └── docker_entrypoint_api.py
├── models/               # Local .pkl + quantized ONNX artifacts
├── reports/week3/        # Benchmark reports
├── dvc.yaml              # Data pipeline definition
├── train.py              # Training + MLflow logging
├── docker-compose.yml    # mlflow + app + api services
├── Dockerfile            # MLflow / training image
└── Dockerfile.api        # Production inference image
```

---

## 11. Environment Variables (Serving)

| Variable | Default | Purpose |
|----------|---------|---------|
| `MLFLOW_TRACKING_URI` | `sqlite:///mlflow.db` | MLflow tracking backend |
| `MLFLOW_MODEL_NAME` | `SentimentBaselineModel` | Registry model name |
| `MLFLOW_MODEL_STAGE` | `Production` | Stage to load |
| `FALLBACK_MODEL_PATH` | `models/baseline_model.pkl` | Local fallback if registry unavailable |
| `BATCH_MAX_SIZE` | `32` | Max async micro-batch size |
| `BATCH_WAIT_MS` | `25` | Max wait before flushing partial batch |
| `QUANTIZED_MODEL_DIR` | `models/quantized` | ONNX artifact directory |
