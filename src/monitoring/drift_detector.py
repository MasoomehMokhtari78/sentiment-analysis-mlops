"""
Dependency-light PSI-based data drift detection.

The detector compares simple numeric text features from production inference
logs against the processed training baseline. It is intentionally independent
from model training and MLflow registration code.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


NO_CHANGE_THRESHOLD = 0.1
MODERATE_SHIFT_THRESHOLD = 0.25


@dataclass(frozen=True)
class PSIResult:
    feature: str
    psi: float
    status: str


def classify_psi(psi_value: float) -> str:
    """Map a PSI value to a monitoring status."""
    if psi_value < NO_CHANGE_THRESHOLD:
        return "No change"
    if psi_value <= MODERATE_SHIFT_THRESHOLD:
        return "Moderate shift"
    return "Significant shift"


def population_stability_index(
    expected: Iterable[float],
    actual: Iterable[float],
    buckets: int = 10,
    epsilon: float = 1e-6,
) -> float:
    """
    Calculate Population Stability Index from scratch using numpy.

    expected: baseline/reference values, usually training data.
    actual: current production values, usually inference logs.
    buckets: number of quantile buckets built from the expected distribution.
    epsilon: small smoothing value to avoid division by zero.
    """
    expected_values = _clean_numeric(expected)
    actual_values = _clean_numeric(actual)

    if expected_values.size == 0 or actual_values.size == 0:
        raise ValueError("PSI requires non-empty expected and actual arrays.")

    quantiles = np.linspace(0, 1, buckets + 1)
    breakpoints = np.quantile(expected_values, quantiles)
    breakpoints = np.unique(breakpoints)

    if breakpoints.size <= 2:
        min_value = float(np.min(expected_values))
        max_value = float(np.max(expected_values))
        if min_value == max_value:
            max_value = min_value + 1.0
        breakpoints = np.linspace(min_value, max_value, buckets + 1)

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts, _ = np.histogram(expected_values, bins=breakpoints)
    actual_counts, _ = np.histogram(actual_values, bins=breakpoints)

    expected_percents = expected_counts / max(len(expected_values), 1)
    actual_percents = actual_counts / max(len(actual_values), 1)

    expected_percents = np.where(expected_percents == 0, epsilon, expected_percents)
    actual_percents = np.where(actual_percents == 0, epsilon, actual_percents)

    psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
    return float(np.sum(psi_values))


def build_text_features(df: pd.DataFrame, text_column: str = "cleaned_text") -> pd.DataFrame:
    """Create stable numeric features for text drift checks."""
    if text_column not in df.columns:
        if "text" in df.columns:
            text_column = "text"
        else:
            raise ValueError(f"Missing text column. Expected '{text_column}' or 'text'.")

    text = df[text_column].fillna("").astype(str)
    word_counts = text.str.split().str.len()

    return pd.DataFrame(
        {
            "char_length": text.str.len(),
            "word_count": word_counts,
            "avg_word_length": text.apply(_average_word_length),
        }
    )


def detect_text_drift(
    reference_path: str | Path = "data/processed/train.csv",
    production_path: str | Path = "logs/inference_data.csv",
    features: List[str] | None = None,
    buckets: int = 10,
) -> Dict[str, object]:
    """
    Compare production inference text features against the training baseline.

    Returns a JSON-serializable report suitable for FastAPI responses or logs.
    """
    reference_path = Path(reference_path)
    production_path = Path(production_path)

    if not reference_path.exists():
        raise FileNotFoundError(f"Reference data not found: {reference_path}")
    if not production_path.exists():
        return {
            "status": "insufficient_data",
            "message": f"No production inference log found at {production_path}.",
            "features": [],
        }

    reference_df = pd.read_csv(reference_path)
    production_df = pd.read_csv(production_path)

    if production_df.empty:
        return {
            "status": "insufficient_data",
            "message": "Production inference log is empty.",
            "features": [],
        }

    reference_features = build_text_features(reference_df)
    production_features = build_text_features(production_df, text_column="text")
    selected_features = features or list(reference_features.columns)

    results = []
    for feature in selected_features:
        psi = population_stability_index(
            reference_features[feature],
            production_features[feature],
            buckets=buckets,
        )
        results.append(PSIResult(feature=feature, psi=round(psi, 6), status=classify_psi(psi)))

    overall_status = _overall_status(results)
    return {
        "status": overall_status,
        "reference_rows": int(len(reference_df)),
        "production_rows": int(len(production_df)),
        "thresholds": {
            "no_change": f"PSI < {NO_CHANGE_THRESHOLD}",
            "moderate_shift": f"{NO_CHANGE_THRESHOLD} <= PSI <= {MODERATE_SHIFT_THRESHOLD}",
            "significant_shift": f"PSI > {MODERATE_SHIFT_THRESHOLD}",
        },
        "features": [result.__dict__ for result in results],
    }


def _clean_numeric(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _average_word_length(text: str) -> float:
    words = text.split()
    if not words:
        return 0.0
    return float(sum(len(word) for word in words) / len(words))


def _overall_status(results: List[PSIResult]) -> str:
    statuses = {result.status for result in results}
    if "Significant shift" in statuses:
        return "Significant shift"
    if "Moderate shift" in statuses:
        return "Moderate shift"
    return "No change"
