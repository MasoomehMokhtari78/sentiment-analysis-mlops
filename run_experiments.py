"""
Run multiple training experiments and log each run to MLflow.

Usage:
    python run_experiments.py --runs 3 --model-type baseline
    python run_experiments.py --runs 5 --sample-sizes 500,1000,2000,3000,5000
"""
import argparse
import logging
import os
import time

import mlflow

from train import train_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_sample_sizes(runs: int, sizes_arg: str | None) -> list[int]:
    if sizes_arg:
        return [int(s.strip()) for s in sizes_arg.split(",")]
    return [1000 * (i + 1) for i in range(runs)]


def main():
    parser = argparse.ArgumentParser(description="Run multiple MLflow training experiments")
    parser.add_argument("--runs", type=int, default=3, help="Number of training runs")
    parser.add_argument("--model-type", choices=["baseline", "distilbert"], default="baseline")
    parser.add_argument(
        "--sample-sizes",
        type=str,
        default=None,
        help="Comma-separated sample sizes per run (overrides --runs count)",
    )
    parser.add_argument("--output-dir", type=str, default="models")
    parser.add_argument("--wait-mlflow", type=int, default=0, help="Seconds to wait for MLflow server")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis")

    if args.wait_mlflow > 0:
        logger.info("Waiting %ds for MLflow server at %s...", args.wait_mlflow, tracking_uri)
        time.sleep(args.wait_mlflow)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    sample_sizes = parse_sample_sizes(args.runs, args.sample_sizes)
    logger.info("Planning %d runs with sample sizes: %s", len(sample_sizes), sample_sizes)

    results = []
    for i, sample_size in enumerate(sample_sizes, start=1):
        logger.info("=== Run %d/%d | model=%s | sample_size=%d ===", i, len(sample_sizes), args.model_type, sample_size)
        try:
            _, metrics = train_model(
                model_type=args.model_type,
                sample_size=sample_size,
                output_dir=args.output_dir,
            )
            results.append({"run": i, "sample_size": sample_size, "metrics": metrics, "status": "ok"})
            logger.info(
                "Run %d done — accuracy=%.4f f1=%.4f",
                i,
                metrics["accuracy"],
                metrics["f1_score"],
            )
        except Exception as exc:
            logger.error("Run %d failed: %s", i, exc)
            results.append({"run": i, "sample_size": sample_size, "status": "failed", "error": str(exc)})

    succeeded = sum(1 for r in results if r["status"] == "ok")
    logger.info("Finished %d/%d runs successfully. View results at %s", succeeded, len(results), tracking_uri)
    return results


if __name__ == "__main__":
    main()
