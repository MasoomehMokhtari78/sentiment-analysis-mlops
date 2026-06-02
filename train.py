import argparse
import logging
from pathlib import Path
import json
import pandas as pd
from config import settings

import mlflow
import mlflow.sklearn

from src.data import IMDBReviewLoader, BasicTextCleaner, DataProcessor
from src.models import BaselineModel, DistilBERTModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_model(model_type="baseline", data_path=None, sample_size=None, output_dir="models"):
    """
    Train a sentiment analysis model with MLflow tracking using pre-processed data.
    """
    logger.info(f"Starting training for {model_type} model")

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("sentiment-analysis")

    # [MLflow] Start an active experiment run
    with mlflow.start_run(run_name=f"{model_type}_training"):
        
        # Determine the directory where processed data is stored
        processed_dir = Path(data_path or settings.processed_data_dir)
        
        # [MLflow] Log basic execution parameters
        mlflow.log_params({
            "model_type": model_type,
            "processed_data_dir": str(processed_dir),
            "test_size": settings.test_size,
            "random_state": settings.random_state
        })

        # Load already processed data
        logger.info(f"Loading processed data from {processed_dir}")
        try:
            train_df = pd.read_csv(processed_dir / "train.csv")
            test_df = pd.read_csv(processed_dir / "test.csv")
        except FileNotFoundError:
            logger.error(f"Processed files not found in {processed_dir}. Did you run ingest.py?")
            raise

        # Optional: Apply sample size if provided via CLI
        if sample_size:
            train_df = train_df.sample(min(sample_size, len(train_df)), random_state=settings.random_state)
            logger.info(f"Sampled training data to {len(train_df)} rows")
        
        # [MLflow] Log data-centric metrics
        mlflow.log_metric("train_samples", len(train_df))
        mlflow.log_metric("test_samples", len(test_df))

        # Model initialization
        if model_type == "baseline":
            model = BaselineModel(random_state=settings.random_state)
            mlflow.log_params({
                "vectorizer_ngram_range": "(1, 2)",
                "vectorizer_max_features": 5000
            })
        elif model_type == "distilbert":
            model = DistilBERTModel(
                max_length=settings.max_length,
                batch_size=settings.batch_size,
                random_state=settings.random_state
            )
            mlflow.log_params({
                "max_length": settings.max_length,
                "batch_size": settings.batch_size,
                "epochs": 3,
                "learning_rate": 2e-5
            })
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        # Train the model
        logger.info(f"Training {model_type}...")
        model.train(train_df['cleaned_text'], train_df['sentiment'])

        # Evaluate the model
        logger.info(f"Evaluating {model_type}...")
        metrics = model.evaluate(test_df['cleaned_text'], test_df['sentiment'])

        # [MLflow] Log all evaluation metrics
        mlflow.log_metrics({
            "accuracy": metrics['accuracy'],
            "precision": metrics['precision'],
            "recall": metrics['recall'],
            "f1_score": metrics['f1_score']
        })

        # Local save logic
        output_path = Path(output_dir) / f"{model_type}_model"
        if model_type == "baseline":
            output_path = output_path.with_suffix('.pkl')
        model.save(str(output_path))

        # [MLflow] Log model and metrics as artifacts
        metrics_file = Path(output_dir) / f"{model_type}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        mlflow.log_artifact(str(metrics_file))
        
        if model_type == "baseline":
            mlflow.sklearn.log_model(
            sk_model=model.model, 
            artifact_path="model",
            registered_model_name="SentimentBaselineModel" 
            )
            logger.info("Baseline model logged and registered.")
        else:
            # Log the whole directory for DistilBERT
            mlflow.log_artifacts(str(output_path), artifact_path="model_weights")

        logger.info(f"Training and tracking complete for {model_type}")
        return model, metrics


def main():
    parser = argparse.ArgumentParser(description="Train sentiment analysis models")
    parser.add_argument("--model-type", type=str, choices=["baseline", "distilbert", "both"], default="baseline")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default="models")

    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.model_type == "both":
        train_model("baseline", args.data_path, args.sample_size, args.output_dir)
        train_model("distilbert", args.data_path, args.sample_size, args.output_dir)
    else:
        train_model(args.model_type, args.data_path, args.sample_size, args.output_dir)


if __name__ == "__main__":
    main()
