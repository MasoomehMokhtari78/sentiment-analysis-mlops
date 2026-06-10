"""Baseline sklearn inference (one request at a time)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sklearn.pipeline import Pipeline

from src.serving.config import serving_settings
from src.serving.labels import (
    expand_probabilities,
    normalize_sentiment_label,
    pipeline_class_labels,
)
from src.serving.predictors.base import BasePredictor, PredictionResult


class SklearnPredictor(BasePredictor):
    """Sequential sklearn Pipeline inference — baseline for benchmarking."""

    variant = "baseline"

    def __init__(self, model: Pipeline, classes: Sequence[str] | None = None):
        self.model = model
        self.model_classes = (
            list(classes) if classes is not None else pipeline_class_labels(model)
        )
        if not self.model_classes:
            self.model_classes = list(serving_settings.classes)

    def predict_one(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        texts_list = list(texts)
        labels = self.model.predict(texts_list)
        probas = self.model.predict_proba(texts_list)

        results: list[PredictionResult] = []
        for label, proba_row in zip(labels, probas, strict=True):
            proba_map = expand_probabilities(self.model_classes, proba_row)
            sentiment = normalize_sentiment_label(label)
            confidence = float(proba_map.get(sentiment, float(np.max(proba_row))))
            results.append(
                PredictionResult(
                    sentiment=sentiment,
                    confidence=confidence,
                    probabilities=proba_map,
                )
            )
        return results
