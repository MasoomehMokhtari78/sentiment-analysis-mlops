"""Load models from the MLflow Model Registry with local fallback."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException
from sklearn.pipeline import Pipeline

from src.serving.config import ServingSettings, serving_settings

logger = logging.getLogger(__name__)


class ModelLoadError(RuntimeError):
    """Raised when no model can be loaded from registry or disk."""


def build_model_uri(settings: ServingSettings | None = None) -> str:
    settings = settings or serving_settings
    return f"models:/{settings.registered_model_name}/{settings.model_stage}"


def load_sklearn_from_registry(
    settings: ServingSettings | None = None,
) -> tuple[Pipeline, str]:
    """
    Load the registered sklearn model from MLflow Model Registry.

    Returns:
        (sklearn_pipeline, resolved_model_uri)
    """
    settings = settings or serving_settings
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    model_uri = build_model_uri(settings)

    try:
        model = mlflow.sklearn.load_model(model_uri)
        logger.info("Loaded model from MLflow registry: %s", model_uri)
        return model, model_uri
    except MlflowException as exc:
        logger.warning("Registry load failed for %s: %s", model_uri, exc)
        return _load_fallback_pipeline(settings), f"file://{settings.fallback_model_path}"


def _load_fallback_pipeline(settings: ServingSettings) -> Pipeline:
    path = Path(settings.fallback_model_path)
    if not path.exists():
        raise ModelLoadError(
            f"Could not load model from registry and fallback not found: {path}"
        )

    payload: dict[str, Any] = joblib.load(path)
    model = payload.get("model") if isinstance(payload, dict) else payload
    if model is None:
        raise ModelLoadError(f"Invalid fallback model payload at {path}")

    logger.info("Loaded fallback model from %s", path)
    return model


def ensure_production_model(
    settings: ServingSettings | None = None,
    promote_latest: bool = False,
) -> str:
    """
    Verify a Production-stage model exists; optionally promote latest version.

    Useful for local Week 3 setup when training only registered the model
    without transitioning it to Production.
    """
    settings = settings or serving_settings
    client = MlflowClient(tracking_uri=settings.mlflow_tracking_uri)

    try:
        versions = client.get_latest_versions(
            settings.registered_model_name,
            stages=[settings.model_stage],
        )
        if versions:
            return build_model_uri(settings)
    except MlflowException:
        pass

    if not promote_latest:
        raise ModelLoadError(
            f"No {settings.model_stage} version for '{settings.registered_model_name}'. "
            "Run training or promote a version: "
            "python scripts/promote_model.py --stage Production"
        )

    latest = client.get_latest_versions(settings.registered_model_name, stages=["None"])
    if not latest:
        raise ModelLoadError(f"No registered versions for '{settings.registered_model_name}'")

    version = latest[0].version
    client.transition_model_version_stage(
        name=settings.registered_model_name,
        version=version,
        stage=settings.model_stage,
        archive_existing_versions=True,
    )
    logger.info(
        "Promoted %s v%s → %s",
        settings.registered_model_name,
        version,
        settings.model_stage,
    )
    return build_model_uri(settings)
