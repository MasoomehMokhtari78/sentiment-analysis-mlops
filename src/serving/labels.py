"""Map sklearn / IMDB numeric labels to API sentiment strings."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np
from sklearn.pipeline import Pipeline

STANDARD_CLASSES = ("negative", "neutral", "positive")

_IMDB_LABEL_MAP = {
    0: "negative",
    1: "positive",
    "0": "negative",
    "1": "positive",
}


def normalize_sentiment_label(label: Any) -> str:
    if isinstance(label, (np.integer, int)):
        mapped = _IMDB_LABEL_MAP.get(int(label))
        if mapped:
            return mapped
    key = str(label).lower()
    return _IMDB_LABEL_MAP.get(key, key)


def pipeline_class_labels(model: Pipeline) -> list[Any]:
    classifier = model.named_steps.get("classifier")
    if classifier is None and getattr(model, "steps", None):
        classifier = model.steps[-1][1]
    classes_ = getattr(classifier, "classes_", None)
    if classes_ is None:
        return []
    return list(classes_)


def expand_probabilities(
    model_classes: Sequence[Any],
    proba_row: Sequence[float],
) -> dict[str, float]:
    scores = {name: 0.0 for name in STANDARD_CLASSES}
    for raw_cls, score in zip(model_classes, proba_row, strict=True):
        name = normalize_sentiment_label(raw_cls)
        if name in scores:
            scores[name] = float(score)
    return scores
