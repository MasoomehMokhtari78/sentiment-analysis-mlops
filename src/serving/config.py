"""Serving-layer configuration (environment-driven)."""
import os
from dataclasses import dataclass, field


@dataclass
class ServingSettings:
    mlflow_tracking_uri: str = field(
        default_factory=lambda: os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    registered_model_name: str = field(
        default_factory=lambda: os.getenv("MLFLOW_MODEL_NAME", "SentimentBaselineModel")
    )
    model_stage: str = field(
        default_factory=lambda: os.getenv("MLFLOW_MODEL_STAGE", "Production")
    )
    fallback_model_path: str = field(
        default_factory=lambda: os.getenv("FALLBACK_MODEL_PATH", "models/baseline_model.pkl")
    )
    classes: tuple[str, ...] = ("negative", "neutral", "positive")

    # Batching
    batch_max_size: int = field(
        default_factory=lambda: int(os.getenv("BATCH_MAX_SIZE", "32"))
    )
    batch_wait_ms: float = field(
        default_factory=lambda: float(os.getenv("BATCH_WAIT_MS", "25"))
    )

    # Quantization artifacts
    quantized_model_dir: str = field(
        default_factory=lambda: os.getenv("QUANTIZED_MODEL_DIR", "models/quantized")
    )

    # Benchmark defaults
    benchmark_samples: int = field(
        default_factory=lambda: int(os.getenv("BENCHMARK_SAMPLES", "200"))
    )
    benchmark_warmup: int = field(
        default_factory=lambda: int(os.getenv("BENCHMARK_WARMUP", "10"))
    )
    benchmark_runs: int = field(
        default_factory=lambda: int(os.getenv("BENCHMARK_RUNS", "5"))
    )


serving_settings = ServingSettings()
