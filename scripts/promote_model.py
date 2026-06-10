"""Promote the latest MLflow registry version to Production for serving."""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.config import serving_settings
from src.serving.model_loader import ensure_production_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote MLflow model to a registry stage")
    parser.add_argument("--stage", default="Production", help="Target stage (default: Production)")
    parser.add_argument(
        "--model-name",
        default=serving_settings.registered_model_name,
        help="Registered model name",
    )
    args = parser.parse_args()

    serving_settings.registered_model_name = args.model_name
    serving_settings.model_stage = args.stage

    uri = ensure_production_model(serving_settings, promote_latest=True)
    logger.info("Model ready at %s", uri)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
