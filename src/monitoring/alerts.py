import json
import os
from datetime import datetime


ALERT_PATH = "reports/week4/alerts.jsonl"


# =========================================================
# CORE LOGGER
# =========================================================
def _write_alert(alert):
    os.makedirs(os.path.dirname(ALERT_PATH), exist_ok=True)

    with open(ALERT_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(alert) + "\n")


def log_alert(alert_type, message, severity="INFO", meta=None):

    alert = {
        "timestamp": str(datetime.now()),
        "type": alert_type,
        "severity": severity,
        "message": message,
        "meta": meta or {}
    }

    _write_alert(alert)

    print(f"[{severity}] {alert_type}: {message}")


# =========================================================
# DRIFT ALERT
# =========================================================
def drift_alert(drift):

    cov = drift.get("covariate_drift", {})
    lab = drift.get("label_drift", {})
    con = drift.get("concept_drift", {})
    dom = drift.get("domain_shift", {})

    if cov.get("drift_detected"):
        log_alert("DRIFT", "Covariate drift detected", "WARNING", cov)

    if lab.get("drift_detected"):
        log_alert("DRIFT", "Label drift detected", "WARNING", lab)

    if con.get("drift_detected"):
        log_alert("DRIFT", "Concept drift detected", "CRITICAL", con)

    if dom.get("drift_detected"):
        log_alert("DRIFT", "Domain shift detected", "CRITICAL", dom)


# =========================================================
# METRICS ALERT
# =========================================================
def metrics_alert(metrics):

    acc = metrics.get("accuracy")

    if acc is None:
        return

    if acc < 0.6:
        log_alert(
            "METRICS",
            "Severe accuracy drop detected",
            "CRITICAL",
            metrics
        )
    elif acc < 0.75:
        log_alert(
            "METRICS",
            "Moderate accuracy drop detected",
            "WARNING",
            metrics
        )


# =========================================================
# RETRAIN ALERT
# =========================================================
def retrain_alert(status):

    if status.get("status") == "retrained":
        log_alert(
            "RETRAIN",
            "Model retrained successfully",
            "INFO",
            status
        )
    else:
        log_alert(
            "RETRAIN",
            "Retraining skipped",
            "INFO",
            status
        )