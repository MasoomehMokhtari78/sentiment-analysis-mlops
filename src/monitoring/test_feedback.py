from src.monitoring.feedback import log_prediction
import os


def test_log_prediction_creates_file():

    path = "data/feedback.jsonl"

    # run function
    log_prediction(
        text="this movie was amazing",
        prediction="positive",
        true_label="positive",
        confidence=0.92
    )

    # check file exists
    assert os.path.exists(path)


def test_log_prediction_content():

    path = "data/feedback.jsonl"

    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    last_line = lines[-1]

    assert "this movie was amazing" in last_line