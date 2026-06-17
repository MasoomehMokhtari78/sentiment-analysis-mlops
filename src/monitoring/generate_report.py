import json
import os

from src.monitoring.retraining import retrain_model
from src.monitoring.drift import load_imdb_drift
from src.monitoring.evaluation import full_evaluation
from src.monitoring.metrics import calculate_window_accuracy

# 🔥 NEW: ALERT SYSTEM
from src.monitoring.alerts import drift_alert, metrics_alert, retrain_alert


# =========================================================
# MAIN REPORT GENERATOR
# =========================================================
def generate_report():

    print("Generating FINAL Week 4 Monitoring Report...")

    # ------------------------
    # 1. FULL DRIFT ANALYSIS
    # ------------------------
    drift = load_imdb_drift(
        "data/raw/imdb_train.csv",
        "data/raw/imdb_test.csv"
    )

    # ------------------------
    # 2. DRIFT ALERT (NEW)
    # ------------------------
    drift_alert(drift)

    # ------------------------
    # 3. AUTO RETRAINING TRIGGER
    # ------------------------
    should_retrain = (
        drift.get("covariate_drift", {}).get("drift_detected", False)
        or drift.get("label_drift", {}).get("drift_detected", False)
    )

    if should_retrain:
        print("🚨 Drift detected → Triggering retraining...")
        retrain_result = retrain_model()
        print("🔄 Model updated after retraining")
    else:
        print("✅ No significant drift → skipping retraining")
        retrain_result = {"status": "skipped"}

    # ------------------------
    # 4. RETRAIN ALERT (NEW)
    # ------------------------
    retrain_alert(retrain_result)

    # ------------------------
    # 5. DUAL EVALUATION
    # ------------------------
    evaluation_result = full_evaluation(
        model=None,
        test_path="data/raw/imdb_test.csv"
    )

    offline_metrics = evaluation_result.get("offline", {})
    online_metrics = evaluation_result.get("online", {})
    health_score = evaluation_result.get("system_health_score", 0.0)
    system_status = evaluation_result.get("status", "UNKNOWN")

    # ------------------------
    # 6. FEEDBACK STATS
    # ------------------------
    feedback_path = "data/feedback.jsonl"
    feedback_count = 0

    if os.path.exists(feedback_path):
        with open(feedback_path, "r", encoding="utf-8") as f:
            feedback_count = sum(1 for _ in f)

    # ------------------------
    # 7. METRICS
    # ------------------------
    metrics_result = calculate_window_accuracy(window_size=50)

    # 🔥 NEW: METRICS ALERT
    metrics_alert(metrics_result)

    # ------------------------
    # 8. DRIFT-BASED HEALTH ADJUSTMENT
    # ------------------------
    drift_penalty = 0.0

    covariate = drift.get("covariate_drift", {})
    label = drift.get("label_drift", {})
    concept = drift.get("concept_drift", {})
    domain = drift.get("domain_shift", {})

    if covariate.get("drift_detected"):
        drift_penalty += 0.2

    if label.get("drift_detected"):
        drift_penalty += 0.2

    if concept.get("drift_detected"):
        drift_penalty += 0.3

    if domain.get("drift_detected"):
        drift_penalty += 0.3

    final_health_score = max(0.0, round(health_score - drift_penalty, 3))

    # ------------------------
    # 9. STATUS LOGIC
    # ------------------------
    if final_health_score < 0.4:
        system_status = "CRITICAL"
    elif final_health_score < 0.7:
        system_status = "WARNING"
    else:
        system_status = "HEALTHY"

    # ------------------------
    # 10. FINAL REPORT
    # ------------------------
    report = {
        "drift": drift,

        "retraining": retrain_result,

        "metrics": metrics_result,

        "offline_metrics": offline_metrics,
        "online_metrics": online_metrics,

        "feedback_samples": feedback_count,

        "system_health_score": final_health_score,
        "system_status": system_status,

        "recommendation": (
            "Retrain model executed or required"
            if system_status != "HEALTHY"
            else "System stable"
        )
    }

    # ------------------------
    # 11. SAVE REPORT
    # ------------------------
    os.makedirs("reports/week4", exist_ok=True)

    output_path = "reports/week4/monitoring_report.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=4)

    # ------------------------
    # 12. PRINT
    # ------------------------
    print("\nFINAL REPORT GENERATED:\n")
    print(json.dumps(report, indent=4))

    print(f"\nSaved to: {output_path}")


# =========================================================
# ENTRY POINT
# =========================================================
if __name__ == "__main__":
    generate_report()