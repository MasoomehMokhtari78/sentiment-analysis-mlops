"""FastAPI service for sentiment prediction, feedback logging, and monitoring."""
from __future__ import annotations

import csv
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.monitoring.drift_detector import detect_text_drift


MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/baseline_model.pkl"))
REFERENCE_DATA_PATH = Path(os.getenv("REFERENCE_DATA_PATH", "data/processed/train.csv"))
LOG_DIR = Path(os.getenv("MONITORING_LOG_DIR", "logs"))
INFERENCE_LOG_PATH = LOG_DIR / "inference_data.csv"
FEEDBACK_LOG_PATH = LOG_DIR / "feedback.csv"
HEALTH_LOG_PATH = LOG_DIR / "system_health.jsonl"
LABEL_MAP = {
    "0": "negative",
    "1": "positive",
    "negative": "negative",
    "positive": "positive",
}

app = FastAPI(title="Sentiment Analysis API", version="1.0.0")

model_lock = Lock()
log_lock = Lock()

model: Optional[Any] = None

runtime_metrics = {
    "request_count": 0,
    "prediction_count": 0,
    "feedback_count": 0,
    "total_latency_ms": 0.0,
    "last_request_at": None,
}


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1)


class PredictionResponse(BaseModel):
    sentiment: str
    probabilities: Dict[str, float]
    latency_ms: float


class FeedbackRequest(BaseModel):
    text: str = Field(..., min_length=1)
    predicted_label: Optional[str] = None
    correct_label: Optional[str] = None
    is_accurate: Optional[bool] = None
    notes: Optional[str] = None


# =========================
# STARTUP
# =========================

@app.on_event("startup")
def startup_event():
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    _ensure_csv(INFERENCE_LOG_PATH, ["timestamp", "text", "prediction", "latency_ms"])
    _ensure_csv(
        FEEDBACK_LOG_PATH,
        ["timestamp", "text", "predicted_label", "correct_label", "is_accurate", "notes"],
    )

    _load_model()


# =========================
# ENDPOINTS
# =========================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest):
    start = time.perf_counter()

    active_model = _get_model()

    try:
        # =========================
        # SAFE PREDICTION BLOCK
        # =========================

        raw_prediction = active_model.predict([payload.text])[0]
        sentiment = _label_to_name(raw_prediction)

        # probabilities
        probs = active_model.predict_proba([payload.text])[0]
        class_labels = _model_classes(active_model)

        probabilities = {
            _label_to_name(class_labels[i]): float(probs[i])
            for i in range(min(len(class_labels), len(probs)))
        }

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {repr(exc)}") from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 3)

    _record_prediction(payload.text, sentiment, latency_ms)

    return {
        "sentiment": sentiment,
        "probabilities": probabilities,
        "latency_ms": latency_ms,
    }


@app.post("/feedback")
def feedback(payload: FeedbackRequest):
    with log_lock:
        _append_csv(
            FEEDBACK_LOG_PATH,
            {
                "timestamp": _utc_now(),
                "text": payload.text,
                "predicted_label": payload.predicted_label or "",
                "correct_label": payload.correct_label or "",
                "is_accurate": "" if payload.is_accurate is None else payload.is_accurate,
                "notes": payload.notes or "",
            },
        )

        runtime_metrics["feedback_count"] += 1

    return {"status": "logged", "feedback_count": runtime_metrics["feedback_count"]}


@app.get("/metrics")
def metrics():
    snapshot = _metrics_snapshot()

    with log_lock:
        with HEALTH_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(snapshot) + "\n")

    return snapshot


@app.get("/drift-check")
def drift_check():
    try:
        report = detect_text_drift(
            reference_path=REFERENCE_DATA_PATH,
            production_path=INFERENCE_LOG_PATH,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Drift check failed: {exc}") from exc

    report["checked_at"] = _utc_now()
    return report


# =========================
# INTERNALS
# =========================

def _load_model():
    global model
    with model_lock:
        if model is None:
            artifact = joblib.load(MODEL_PATH)
            model = _unwrap_model_artifact(artifact)


def _get_model() -> Any:
    if model is None:
        _load_model()
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return model


def _unwrap_model_artifact(artifact: Any) -> Any:
    if hasattr(artifact, "predict"):
        return artifact

    if isinstance(artifact, dict):
        for key in ("model", "pipeline"):
            candidate = artifact.get(key)
            if hasattr(candidate, "predict"):
                return candidate

        for candidate in artifact.values():
            if hasattr(candidate, "predict"):
                return candidate

    raise ValueError(f"Could not find a predict-capable model in {MODEL_PATH}")


def _model_classes(active_model: Any):
    if hasattr(active_model, "classes_"):
        return list(active_model.classes_)

    if hasattr(active_model, "named_steps"):
        for step in reversed(active_model.named_steps.values()):
            if hasattr(step, "classes_"):
                return list(step.classes_)

    return []


def _label_to_name(label: Any) -> str:
    if isinstance(label, np.generic):
        label = label.item()
    return LABEL_MAP.get(str(label), str(label))


def _record_prediction(text: str, prediction: str, latency_ms: float):
    with log_lock:
        _append_csv(
            INFERENCE_LOG_PATH,
            {
                "timestamp": _utc_now(),
                "text": text,
                "prediction": prediction,
                "latency_ms": latency_ms,
            },
        )

        runtime_metrics["request_count"] += 1
        runtime_metrics["prediction_count"] += 1
        runtime_metrics["total_latency_ms"] += latency_ms
        runtime_metrics["last_request_at"] = _utc_now()


def _metrics_snapshot() -> Dict[str, object]:
    count = runtime_metrics["prediction_count"]

    avg_latency = (
        runtime_metrics["total_latency_ms"] / count if count else 0.0
    )

    return {
        "timestamp": _utc_now(),
        "request_count": runtime_metrics["request_count"],
        "prediction_count": count,
        "feedback_count": runtime_metrics["feedback_count"],
        "average_latency_ms": round(avg_latency, 3),
        "last_request_at": runtime_metrics["last_request_at"],
    }


def _ensure_csv(path: Path, fieldnames):
    if path.exists():
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv(path: Path, row: Dict[str, object]):
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
