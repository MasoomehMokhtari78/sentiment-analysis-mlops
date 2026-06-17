# Sentiment Analysis MLOps Pipeline

This repository contains a robust, reproducible sentiment analysis pipeline. The project is designed with a **Data-Centric** focus, ensuring that data quality, versioning, and experiment tracking are prioritized alongside model architecture.

## 📖 Documentation

| Document | Description |
|----------|-------------|
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | End-to-end system flowcharts and deployment topology |
| [docs/TECHNOLOGIES.md](docs/TECHNOLOGIES.md) | Technologies used and where they appear in the project |
| [docs/WEEK3_INTEGRATION.md](docs/WEEK3_INTEGRATION.md) | Week 3 serving, optimization, and Docker setup |
| [docs/WEEK4_INTEGRATION.md](docs/WEEK4_INTEGRATION.md) | Week 4 CI/CD and Docker-based testing (implemented so far) |
| [docs/WEEK4_TODO.md](docs/WEEK4_TODO.md) | Phase 4 full plan — monitoring & drift (pending) |

---

## 🚀 Key Features

- **Automated Data Pipeline:** A custom `DataProcessor` handles outlier removal and synthetic data augmentation (applied only to training sets to prevent leakage).
- **Reproducibility with DVC:** The entire ingestion and processing workflow is version-controlled with **DVC (Data Version Control)**, ensuring that data and code are always synchronized.
- **Persistent Experiment Tracking:** Uses **MLflow** with a **SQLite** backend to ensure ACID-compliant, reliable logging of hyperparameters, metrics, and model artifacts.
- **Model Registry:** All training runs are registered in the MLflow Model Registry for easy version management and deployment.

---

## 🏗️ Pipeline Workflow

The project follows a linear, automated flow:

1.  **Raw Data** → 2. **Ingest/Process** (`dvc.yaml` orchestrates `ingest.py`) → 3. **Training** (`train.py`) → 4. **Tracking** (MLflow Registry).

---

## 🛠️ Getting Started

### 1. Prerequisites

Ensure you have the required dependencies installed (usually `pip install -r requirements.txt`).

### 2. DVC Setup

Initialize DVC to track data versions:

```bash
dvc init
```

---

## 🏃 Execution Guide

To reproduce the pipeline or execute new runs, follow this sequence:

### Step 1: Run the Data Pipeline

Automate the data ingestion and processing logic:

```bash
dvc repro
```

_This command checks your dependencies (code and raw data). If changes are detected, it re-runs the processing step and generates the latest `processed/` data._

### Step 2: Train the Model

Execute the training script:

```bash
python train.py --model-type baseline
```

_(The training script is pre-configured to log experiments to `sqlite:///mlflow.db`)_

### Step 3: Monitor Experiments

Launch the MLflow UI to inspect your results:

```bash
mlflow ui
```

Navigate to `http://localhost:5000` in your browser to view metrics, parameters, and the registered models.

---

## 📂 Project Structure

```text
sentiment-analysis-api/
├── data/
│   ├── raw/          # Original, versioned source data
│   └── processed/    # Generated via dvc.yaml
├── src/
│   ├── data/
│   │   ├── ingest.py     # Orchestrates data flow
│   │   └── processor.py  # Cleaning/Augmentation logic
│   └── models/
│       ├── train.py      # Training loop & MLflow logging
│       ├── baseline.py
│       └── distilbert.py
├── dvc.yaml              # Pipeline definition
├── mlflow.db             # Persistent SQLite tracking backend
└── README.md
```

---

## 🛠️ Technical Decisions

- **Why SQLite for MLflow?** We migrated from the default file-based storage to SQLite to ensure **ACID compliance** and to avoid file-locking errors during concurrent runs. This establishes a stable foundation for moving from local development to production-grade deployment.
- **Why DVC?** DVC acts as the "source of truth." By defining stages in `dvc.yaml`, we ensure that manual data handling is eliminated, and the model training always uses the exact version of the data it was intended for.

---

_This project was developed with a focus on MLOps best practices, prioritizing reproducibility, lineage, and automation._
