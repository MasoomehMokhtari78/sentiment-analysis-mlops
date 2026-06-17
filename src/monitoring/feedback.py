import json
import os
from datetime import datetime

FEEDBACK_FILE = "data/feedback.jsonl"


def log_prediction(text, prediction, true_label=None, confidence=None):
    """
    Store prediction with optional delayed label support (REAL MLOPS STYLE)
    """

    os.makedirs("data", exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "text": text,
        "prediction": prediction,
        "true_label": true_label,
        "confidence": confidence
    }

    with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_feedback():
    if not os.path.exists(FEEDBACK_FILE):
        return []

    data = []
    with open(FEEDBACK_FILE, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data