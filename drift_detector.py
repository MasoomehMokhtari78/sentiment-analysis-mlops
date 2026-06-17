"""CLI wrapper for the PSI drift detector."""
import argparse
import json

from src.monitoring.drift_detector import detect_text_drift


def main():
    parser = argparse.ArgumentParser(description="Run PSI drift detection.")
    parser.add_argument("--reference-path", default="data/processed/train.csv")
    parser.add_argument("--production-path", default="logs/inference_data.csv")
    parser.add_argument("--buckets", type=int, default=10)
    args = parser.parse_args()

    report = detect_text_drift(
        reference_path=args.reference_path,
        production_path=args.production_path,
        buckets=args.buckets,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
