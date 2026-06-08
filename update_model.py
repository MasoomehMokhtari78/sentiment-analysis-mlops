"""
update_model.py - Week 2: Stateful/Stateless model update strategy
"""
import argparse
import logging
import mlflow
import os
from mlflow import MlflowClient

from config import settings
from src.data import YelpReviewLoader, BasicTextCleaner, DataProcessor
from src.models import BaselineModel, DistilBERTModel

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
mlflow.set_experiment("Sentiment_Analysis_Project")

client = MlflowClient()


def get_production_model_uri(model_name: str) -> str | None:
    """Get URI of current Production model from registry."""
    try:
        versions = client.get_latest_versions(model_name, stages=["Production"])
        if versions:
            return f"models:/{model_name}/Production"
    except Exception:
        pass
    return None


def transition_to_staging(model_name: str, run_id: str):
    """Register new run and move to Staging."""
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
    client.transition_model_version_stage(
        name=model_name,
        version=result.version,
        stage="Staging"
    )
    logger.info(f"{model_name} v{result.version} → Staging")
    return result.version


def load_new_data(data_path=None, sample_size=None):
    loader = YelpReviewLoader()
    cleaner = BasicTextCleaner()
    processor = DataProcessor(loader, cleaner)
    return processor.process(
        file_path=data_path or settings.data_path,
        sample_size=sample_size or settings.sample_size,
        test_size=settings.test_size,
        random_state=settings.random_state
    )


# ── Stateless: retrain from scratch ──────────────────────────────────────────
def stateless_update(data_path=None, sample_size=None):
    """Baseline (Random Forest) - stateless retrain."""
    logger.info("Stateless update: retraining baseline from scratch")

    train_df, test_df = load_new_data(data_path, sample_size)

    with mlflow.start_run(run_name="stateless_update_baseline") as run:
        mlflow.log_params({
            "update_strategy": "stateless",
            "model_type": "baseline",
            "sample_size": sample_size or settings.sample_size,
        })

        model = BaselineModel(random_state=settings.random_state)
        model.train(train_df['cleaned_text'], train_df['sentiment'])
        metrics = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])

        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model.model, "model")

        transition_to_staging("baseline_sentiment", run.info.run_id)

    return metrics


# ── Stateful: fine-tune existing model ───────────────────────────────────────
def stateful_update(data_path=None, sample_size=None):
    """DistilBERT - stateful fine-tune on new data."""
    logger.info("Stateful update: fine-tuning DistilBERT on new data")

    train_df, test_df = load_new_data(data_path, sample_size)

    # Load existing Production model weights if available
    prod_uri = get_production_model_uri("distilbert_sentiment")
    if prod_uri:
        logger.info(f"Loading production model from {prod_uri}")
        loaded = mlflow.pyfunc.load_model(prod_uri)
        model = loaded._model_impl  # unwrap to get DistilBERTModel instance
    else:
        logger.info("No production model found, fine-tuning from pretrained weights")
        model = DistilBERTModel(
            max_length=settings.max_length,
            batch_size=settings.batch_size,
            random_state=settings.random_state
        )

    with mlflow.start_run(run_name="stateful_update_distilbert") as run:
        mlflow.log_params({
            "update_strategy": "stateful",
            "model_type": "distilbert",
            "sample_size": sample_size or settings.sample_size,"fine_tuned_from": prod_uri or "pretrained",
        })

        model.train(train_df['cleaned_text'], train_df['sentiment'])
        metrics = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])

        mlflow.log_metrics(metrics)
        mlflow.pyfunc.log_model("model", python_model=model)

        transition_to_staging("distilbert_sentiment", run.info.run_id)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Update models - stateful or stateless")
    parser.add_argument("--model-type", choices=["baseline", "distilbert", "both"], default="both")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    args = parser.parse_args()

    if args.model_type in ("baseline", "both"):
        metrics = stateless_update(args.data_path, args.sample_size)
        logger.info(f"Baseline updated | accuracy: {metrics['accuracy']:.4f}")

    if args.model_type in ("distilbert", "both"):
        metrics = stateful_update(args.data_path, args.sample_size)
        logger.info(f"DistilBERT updated | accuracy: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
