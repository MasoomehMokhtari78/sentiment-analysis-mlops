"""Inference backends: baseline, batched, and quantized."""

from src.serving.predictors.base import BasePredictor, PredictionResult
from src.serving.predictors.baseline import SklearnPredictor
from src.serving.predictors.batched import BatchedSklearnPredictor
from src.serving.predictors.quantized import QuantizedOnnxPredictor

__all__ = [
    "BasePredictor",
    "PredictionResult",
    "SklearnPredictor",
    "BatchedSklearnPredictor",
    "QuantizedOnnxPredictor",
]
