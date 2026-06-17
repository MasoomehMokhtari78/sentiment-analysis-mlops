from src.monitoring.feedback import load_feedback


# =========================================================
# WINDOW-BASED ACCURACY (MAIN METRIC)
# =========================================================
def calculate_window_accuracy(window_size=50):

    data = load_feedback()

    labeled = [d for d in data if d.get("true_label") is not None]

    if len(labeled) == 0:
        return {
            "accuracy": None,
            "message": "no labeled data available"
        }

    window = labeled[-window_size:]

    correct = sum(
        1 for item in window
        if item["prediction"] == item["true_label"]
    )

    total = len(window)

    return {
        "accuracy": round(correct / total, 3) if total > 0 else None,
        "total_samples": total,
        "correct": correct,
        "window_size": window_size
    }


# =========================================================
# BACKWARD COMPATIBILITY 
# =========================================================
def calculate_accuracy():
    """
    Alias for older tests
    """
    return calculate_window_accuracy()