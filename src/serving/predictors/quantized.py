"""INT8 ONNX Runtime inference backend."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

from src.serving.config import serving_settings
from src.serving.labels import (
    STANDARD_CLASSES,
    expand_probabilities,
    normalize_sentiment_label,
)
from src.serving.predictors.base import BasePredictor, PredictionResult


class QuantizedOnnxPredictor(BasePredictor):
    """Serve the dynamically quantized ONNX export of the sklearn pipeline."""

    variant = "quantized"

    def __init__(
        self,
        quantized_model_path: str | Path,
        classes: Sequence[str] | None = None,
    ):
        import onnxruntime as ort

        self.model_classes = (
            list(classes) if classes is not None else list(serving_settings.classes)
        )
        self._session = ort.InferenceSession(
            str(quantized_model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._label_output = self._resolve_output("label")
        self._proba_output = self._resolve_output("probabilities")

    def _resolve_output(self, keyword: str) -> str | None:
        for node in self._session.get_outputs():
            if keyword in node.name.lower():
                return node.name
        outputs = self._session.get_outputs()
        if keyword == "label" and outputs:
            return outputs[0].name
        if keyword == "probabilities" and len(outputs) > 1:
            return outputs[1].name
        return None

    def predict_one(self, text: str) -> PredictionResult:
        return self.predict_batch([text])[0]

    def predict_batch(self, texts: Sequence[str]) -> list[PredictionResult]:
        texts_list = list(texts)
        inputs = np.array([[t] for t in texts_list], dtype=object)

        feed = {self._input_name: inputs}
        if self._label_output and self._proba_output:
            labels, probas = self._session.run(
                [self._label_output, self._proba_output],
                feed,
            )
        elif self._label_output:
            labels = self._session.run([self._label_output], feed)[0]
            probas = None
        else:
            raise RuntimeError("ONNX model has no usable classification outputs")

        results: list[PredictionResult] = []
        for idx, label in enumerate(labels):
            sentiment = normalize_sentiment_label(label)
            if probas is not None:
                proba_row = probas[idx]
                proba_map = expand_probabilities(self.model_classes, proba_row)
                confidence = float(proba_map.get(sentiment, float(np.max(proba_row))))
            else:
                proba_map = {cls: 0.0 for cls in STANDARD_CLASSES}
                if sentiment in proba_map:
                    proba_map[sentiment] = 1.0
                confidence = 1.0

            results.append(
                PredictionResult(
                    sentiment=sentiment,
                    confidence=confidence,
                    probabilities=proba_map,
                )
            )
        return results
