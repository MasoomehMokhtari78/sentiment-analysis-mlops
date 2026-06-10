"""Batched sklearn inference with explicit vectorization."""
from __future__ import annotations

from typing import Sequence

from sklearn.pipeline import Pipeline

from src.serving.predictors.baseline import SklearnPredictor


class BatchedSklearnPredictor(SklearnPredictor):
    """
    Vectorized batch predictor.

    Unlike the baseline path (which may call predict per request), this backend
  always routes inputs through a single vectorized `predict` / `predict_proba`
    call — the core optimization from Chapter 22 (batching & vectorization).
    """

    variant = "batched"

    def __init__(self, model: Pipeline, classes: Sequence[str] | None = None):
        super().__init__(model=model, classes=classes)

    def predict_batch(self, texts: Sequence[str]) -> list:
        if not texts:
            return []
        return super().predict_batch(texts)
