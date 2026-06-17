from src.monitoring.feedback import load_feedback


def calculate_window_accuracy(window_size=50):
    """
    REAL MLOPS: accuracy over sliding window
    """

    data = load_feedback()

    # فقط labeled data
    labeled = [d for d in data if d.get("true_label") is not None]

    if len(labeled) == 0:
        return {
            "accuracy": None,
            "message": "no labeled data available"
        }

    # آخرین window
    window = labeled[-window_size:]

    correct = 0
    total = len(window)

    for item in window:
        if item["prediction"] == item["true_label"]:
            correct += 1

    return {
        "accuracy": round(correct / total, 3),
        "total_samples": total,
        "correct": correct,
        "window_size": window_size
    }