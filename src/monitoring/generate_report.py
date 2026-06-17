import json
import os
from src.monitoring.drift import load_imdb_drift
from src.monitoring.evaluation import full_evaluation
from src.monitoring.metrics import calculate_window_accuracy


def generate_report():

    print("Generating FINAL Week 4 Monitoring Report...")

    # ------------------------
    # 1. DRIFT (REAL IMDB)
    # ------------------------
    drift = load_imdb_drift(
        "data/raw/imdb_train.csv",
        "data/raw/imdb_test.csv"
    )

    # ------------------------
    # 2. DUAL EVALUATION
    # ------------------------
    evaluation_result = full_evaluation(
        model=None,  
        test_path="data/raw/imdb_test.csv"
    )

    offline_metrics = evaluation_result["offline"]
    online_metrics = evaluation_result["online"]
    health_score = evaluation_result["system_health_score"]
    system_status = evaluation_result["status"]

    # ------------------------
    # 3. FEEDBACK STATS
    # ------------------------
    feedback_path = "data/feedback.jsonl"
    feedback_count = 0

    if os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_count = sum(1 for _ in f)

    # ------------------------
    # 4. FINAL REPORT STRUCTURE
    # ------------------------
    report = {
        "drift": drift,

        "offline_metrics": offline_metrics,
        "online_metrics": online_metrics,

        "feedback_samples": feedback_count,

        "system_health_score": round(health_score, 3),
        "system_status": system_status,

        "recommendation": (
            "Retrain model immediately"
            if system_status == "CRITICAL"
            else "System stable"
        )
    }

    # ------------------------
    # 5. SAVE REPORT
    # ------------------------
    os.makedirs("reports/week4", exist_ok=True)

    output_path = "reports/week4/monitoring_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # ------------------------
    # 6. PRINT RESULT
    # ------------------------
    print("\nFINAL REPORT GENERATED:\n")
    print(json.dumps(report, indent=4))

    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    generate_report()