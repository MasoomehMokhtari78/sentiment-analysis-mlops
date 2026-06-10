"""Abstract predictor interface for serving."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class PredictionResult:
    sentiment: str
    confidence: float
    probabilities: dict[str, float]


class BasePredictor(ABC):
    """Common contract for all inference backends."""

    variant: str = "baseline"

    @abstractmethod
    def predict_one(self, text: str) -> PredictionResult:
        """Predict sentiment for a single text."""

    @abstractmethod
    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        """Vectorized batch prediction."""

    def warmup(self, sample_text: str = "great movie") -> None:
        """Prime caches and JIT paths before serving traffic."""
        self.predict_one(sample_text)
