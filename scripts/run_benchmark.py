"""Run Week 3 latency/memory benchmark: baseline vs optimized paths."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.serving.benchmarking import benchmark_predictor, save_benchmark_report
from src.serving.config import serving_settings
from src.serving.model_loader import load_sklearn_from_registry
from src.serving.optimization.quantization import (
    export_and_quantize_sklearn,
    model_size_bytes,
    summarize_quantization_gain,
)
from src.serving.predictors import (
    BatchedSklearnPredictor,
    QuantizedOnnxPredictor,
    SklearnPredictor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_sample_texts(sample_size: int) -> list[str]:
    test_path = Path("data/processed/test.csv")
    if test_path.exists():
        df = pd.read_csv(test_path)
        return df["cleaned_text"].head(sample_size).astype(str).tolist()

    return [
        "this movie was fantastic and emotional",
        "boring waste of time",
        "average film nothing special",
    ] * max(1, sample_size // 3)


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark inference optimizations")
    parser.add_argument("--samples", type=int, default=serving_settings.benchmark_samples)
    parser.add_argument("--warmup", type=int, default=serving_settings.benchmark_warmup)
    parser.add_argument("--runs", type=int, default=serving_settings.benchmark_runs)
    parser.add_argument("--output-dir", default="reports/week3")
    args = parser.parse_args()

    texts = _load_sample_texts(args.samples)
    logger.info("Benchmarking with %d texts per run (%d runs)", len(texts), args.runs)

    pipeline, _ = load_sklearn_from_registry()
    fallback_path = Path(serving_settings.fallback_model_path)

    baseline_predictor = SklearnPredictor(pipeline)
    batched_predictor = BatchedSklearnPredictor(pipeline)

    baseline_size_mb = round(model_size_bytes(fallback_path) / (1024 * 1024), 3)

    results = [
        benchmark_predictor(
            baseline_predictor,
            texts,
            warmup=args.warmup,
            runs=args.runs,
            model_size_mb=baseline_size_mb,
        ),
        benchmark_predictor(
            batched_predictor,
            texts,
            warmup=args.warmup,
            runs=args.runs,
            model_size_mb=baseline_size_mb,
        ),
    ]

    try:
        quantized_path = export_and_quantize_sklearn(
            pipeline, serving_settings.quantized_model_dir
        )
        quantized_predictor = QuantizedOnnxPredictor(quantized_path)
        size_info = summarize_quantization_gain(fallback_path, quantized_path)
        logger.info("Quantization size summary: %s", size_info)
        results.append(
            benchmark_predictor(
                quantized_predictor,
                texts,
                warmup=args.warmup,
                runs=args.runs,
                model_size_mb=size_info["optimized_size_mb"],
            )
        )
    except Exception as exc:
        logger.warning(
            "INT8 quantization unavailable, skipping quantized benchmark: %s", exc
        )

    json_path, md_path = save_benchmark_report(results, args.output_dir)
    print(f"\nBenchmark complete.\n  JSON: {json_path}\n  Markdown: {md_path}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
