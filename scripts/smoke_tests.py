"""Integration smoke tests for the sentiment analysis Docker pipeline."""
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

PASS = 0
FAIL = 0


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  PASS  {name}")
    else:
        FAIL += 1
        msg = f"  FAIL  {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def test_mlflow_reachable():
    print("\n[1] MLflow server")
    uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    try:
        with urllib.request.urlopen(uri, timeout=10) as resp:
            check("MLflow HTTP reachable", resp.status == 200, f"status={resp.status}")
    except Exception as exc:
        check("MLflow HTTP reachable", False, str(exc))


def test_processed_data():
    print("\n[2] Processed data")
    processed = Path("data/processed")
    train_path = processed / "train.csv"
    test_path = processed / "test.csv"

    check("train.csv exists", train_path.exists())
    check("test.csv exists", test_path.exists())

    if not (train_path.exists() and test_path.exists()):
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    required = {"cleaned_text", "sentiment"}

    check("train has required columns", required.issubset(train_df.columns), str(train_df.columns.tolist()))
    check("test has required columns", required.issubset(test_df.columns), str(test_df.columns.tolist()))
    check("train set non-empty", len(train_df) > 100, f"rows={len(train_df)}")
    check("test set non-empty", len(test_df) > 20, f"rows={len(test_df)}")
    check("no empty cleaned texts", train_df["cleaned_text"].str.len().min() > 0)
    check("train has both sentiment classes", train_df["sentiment"].nunique() >= 2, str(train_df["sentiment"].value_counts().to_dict()))


def test_text_cleaner():
    print("\n[3] Text cleaner")
    from src.data.processor import BasicTextCleaner

    cleaner = BasicTextCleaner(lowercase=True, remove_special=True, anonymize_pii=True)
    raw = "Contact me at test@example.com this movie was AMAZING"
    cleaned = cleaner.clean(raw)

    check("PII masked", "anonymized_email" in cleaned)
    check("lowercased", cleaned == cleaned.lower())
    check("non-empty output", len(cleaned) > 0)


def test_baseline_model_load_and_predict():
    print("\n[4] Baseline model")
    from src.models.baseline import BaselineModel

    model_path = Path("models/baseline_model.pkl")
    check("model file exists", model_path.exists())

    if not model_path.exists():
        return

    model = BaselineModel()
    model.load(str(model_path))
    check("model loads", model.is_trained)

    samples = [
        "this film was wonderful and moving",
        "boring waste of time hated every minute",
    ]
    preds = model.predict(samples)
    check("batch predict returns list", isinstance(preds, list) and len(preds) == 2, str(preds))

    proba = model.predict_proba(samples[0])
    check("predict_proba returns dict", isinstance(proba, dict) and len(proba) > 0)


def test_baseline_model_evaluate():
    print("\n[5] Model evaluation")
    from src.models.baseline import BaselineModel

    model_path = Path("models/baseline_model.pkl")
    test_path = Path("data/processed/test.csv")
    if not (model_path.exists() and test_path.exists()):
        print("  SKIP  evaluation (missing model or test data)")
        return

    model = BaselineModel()
    model.load(str(model_path))
    test_df = pd.read_csv(test_path).head(500)

    metrics = model.evaluate(test_df["cleaned_text"], test_df["sentiment"])
    acc = metrics.get("accuracy", 0)
    f1 = metrics.get("f1_score", 0)

    check("accuracy reported", "accuracy" in metrics, str(metrics))
    check("accuracy > 50%", acc > 0.5, f"accuracy={acc:.4f}")
    check("f1_score > 50%", f1 > 0.5, f"f1={f1:.4f}")
    print(f"        metrics: accuracy={acc:.4f}  f1={f1:.4f}  precision={metrics.get('precision', 0):.4f}")


def test_mlflow_experiment_runs():
    print("\n[6] MLflow experiment runs")
    import mlflow
    from mlflow.tracking import MlflowClient

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "sentiment-analysis")

    try:
        mlflow.set_tracking_uri(tracking_uri)
        client = MlflowClient(tracking_uri=tracking_uri)
        exp = client.get_experiment_by_name(experiment_name)
        check("experiment exists", exp is not None, experiment_name)

        if exp:
            runs = client.search_runs(experiment_ids=[exp.experiment_id], max_results=20)
            check("at least 1 logged run", len(runs) >= 1, f"found={len(runs)}")
            completed = [r for r in runs if r.data.metrics.get("accuracy") is not None]
            check("at least 1 run with accuracy metric", len(completed) >= 1, f"completed={len(completed)}")
            if completed:
                latest = completed[0]
                acc = latest.data.metrics["accuracy"]
                print(f"        best recent run: {latest.info.run_name}  accuracy={acc:.4f}")
    except Exception as exc:
        check("MLflow API accessible", False, str(exc))


def main():
    print("=" * 50)
    print("Sentiment Analysis — Smoke Tests")
    print("=" * 50)

    test_mlflow_reachable()
    test_processed_data()
    test_text_cleaner()
    test_baseline_model_load_and_predict()
    test_baseline_model_evaluate()
    test_mlflow_experiment_runs()

    print("\n" + "=" * 50)
    print(f"Results: {PASS} passed, {FAIL} failed")
    print("=" * 50)

    return 0 if FAIL == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
