"""Bootstrap a minimal baseline model for Docker serving (no DVC data required)."""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.models.baseline import BaselineModel
from src.serving.config import serving_settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SAMPLE_TEXTS = [
    "this movie was absolutely wonderful and moving",
    "loved every minute great acting and story",
    "fantastic film highly recommend it",
    "terrible waste of time hated every scene",
    "boring predictable and badly acted",
    "worst movie I have ever seen",
    "it was okay nothing special",
    "average film not bad not great",
    "mediocre story but decent performances",
] * 15

SAMPLE_LABELS = (
    ["positive"] * 45
    + ["negative"] * 45
    + ["neutral"] * 45
)


def _try_load_production(client: MlflowClient, model_name: str) -> str | None:
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return f"models:/{model_name}/Production"
    except Exception:
        pass
    return None


def _register_in_mlflow(model: BaselineModel, model_name: str, client: MlflowClient) -> str | None:
    """Log model to MLflow registry; returns registry URI or None on failure."""
    try:
        with mlflow.start_run(run_name="docker_seed_baseline"):
            mlflow.log_params({"model_type": "baseline", "source": "docker_seed"})
            mlflow.sklearn.log_model(
                sk_model=model.model,
                artifact_path="model",
                registered_model_name=model_name,
            )

        latest = client.get_latest_versions(model_name, stages=["None"])
        if latest:
            client.transition_model_version_stage(
                name=model_name,
                version=latest[0].version,
                stage="Production",
                archive_existing_versions=True,
            )
            logger.info("Promoted %s v%s to Production", model_name, latest[0].version)
            return f"models:/{model_name}/Production"
    except Exception as exc:
        logger.warning("MLflow registration skipped (API will use local fallback): %s", exc)
    return None


def seed() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", serving_settings.mlflow_tracking_uri)
    model_name = os.getenv("MLFLOW_MODEL_NAME", serving_settings.registered_model_name)
    output_path = Path(serving_settings.fallback_model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)

    registry_uri = _try_load_production(client, model_name)
    if registry_uri:
        logger.info("Production model already exists at %s — skipping seed", registry_uri)
        return registry_uri

    if output_path.exists():
        register = os.getenv("SEED_REGISTER_MLFLOW", "").lower() in ("1", "true", "yes")
        if register:
            logger.info("Local fallback exists at %s — attempting MLflow registration", output_path)
            model = BaselineModel()
            model.load(str(output_path))
            registry_uri = _register_in_mlflow(model, model_name, client)
            if registry_uri:
                return registry_uri
        logger.info("Using local fallback: %s", output_path)
        return f"file://{output_path}"

    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis"))

    model = BaselineModel(random_state=42)
    model.train(SAMPLE_TEXTS, SAMPLE_LABELS)
    model.save(str(output_path))
    logger.info("Saved fallback model to %s", output_path)

    registry_uri = _register_in_mlflow(model, model_name, client)
    if registry_uri:
        return registry_uri

    logger.info("Serving will load from local fallback: %s", output_path)
    return f"file://{output_path}"


if __name__ == "__main__":
    uri = seed()
    print(f"Model ready at {uri}")
