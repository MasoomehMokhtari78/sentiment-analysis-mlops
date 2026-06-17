from src.models.baseline import BaselineModel


TRAIN_TEXTS = [
    "amazing great wonderful movie",
    "excellent acting and plot",
    "loved this film a lot",
    "awful boring bad movie",
    "terrible and disappointing",
    "worst film ever",
]
TRAIN_LABELS = ["positive", "positive", "positive", "negative", "negative", "negative"]


def test_baseline_model_train_predict_and_evaluate() -> None:
    model = BaselineModel(max_features=200, max_iter=200, random_state=42)
    model.train(TRAIN_TEXTS, TRAIN_LABELS)

    preds = model.predict(["great story", "bad acting"])
    assert isinstance(preds, list)
    assert len(preds) == 2
    assert set(preds).issubset({"positive", "negative"})

    probs = model.predict_proba("great story")
    assert isinstance(probs, dict)
    assert "positive" in probs or "negative" in probs
    assert abs(sum(probs.values()) - 1.0) < 0.01

    metrics = model.evaluate(TRAIN_TEXTS, TRAIN_LABELS)
    for key in ("accuracy", "precision", "recall", "f1_score"):
        assert key in metrics
