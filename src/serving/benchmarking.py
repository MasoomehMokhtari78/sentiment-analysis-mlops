"""Measure inference latency and memory before/after optimizations."""
from __future__ import annotations

import gc
import json
import logging
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import psutil

from src.serving.predictors.base import BasePredictor

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkMetrics:
    variant: str
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    throughput_rps: float
    peak_memory_mb: float
    model_size_mb: float | None = None


def _percentile(values: Sequence[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = int(round((pct / 100.0) * (len(ordered) - 1)))
    return ordered[index]


def _memory_mb() -> float:
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)


def benchmark_predictor(
    predictor: BasePredictor,
    texts: Sequence[str],
    *,
    warmup: int = 10,
    runs: int = 5,
    model_size_mb: float | None = None,
) -> BenchmarkMetrics:
    """Benchmark a predictor over repeated batch inference loops."""
    warmup_texts = list(texts[: max(1, min(warmup, len(texts)))])
    for _ in range(warmup):
        predictor.predict_batch(warmup_texts)

    gc.collect()
    mem_before = _memory_mb()
    latencies_ms: list[float] = []

    for _ in range(runs):
        start = time.perf_counter()
        predictor.predict_batch(texts)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        latencies_ms.append(elapsed_ms)

    peak_memory_mb = max(_memory_mb() - mem_before, 0.0)
    total_samples = len(texts) * runs
    total_seconds = sum(latencies_ms) / 1000.0

    return BenchmarkMetrics(
        variant=predictor.variant,
        mean_latency_ms=round(statistics.mean(latencies_ms), 3),
        p50_latency_ms=round(_percentile(latencies_ms, 50), 3),
        p95_latency_ms=round(_percentile(latencies_ms, 95), 3),
        throughput_rps=round(total_samples / total_seconds, 2) if total_seconds else 0.0,
        peak_memory_mb=round(peak_memory_mb, 3),
        model_size_mb=model_size_mb,
    )


def build_comparison_table(results: Sequence[BenchmarkMetrics]) -> list[dict]:
    if not results:
        return []

    baseline = next((r for r in results if r.variant == "baseline"), results[0])
    table: list[dict] = []

    for row in results:
        latency_gain = (
            round(100.0 * (1 - row.mean_latency_ms / baseline.mean_latency_ms), 2)
            if baseline.mean_latency_ms
            else 0.0
        )
        memory_gain = (
            round(100.0 * (1 - row.peak_memory_mb / baseline.peak_memory_mb), 2)
            if baseline.peak_memory_mb
            else 0.0
        )
        table.append(
            {
                "variant": row.variant,
                "mean_latency_ms": row.mean_latency_ms,
                "p50_latency_ms": row.p50_latency_ms,
                "p95_latency_ms": row.p95_latency_ms,
                "throughput_rps": row.throughput_rps,
                "peak_memory_mb": row.peak_memory_mb,
                "model_size_mb": row.model_size_mb,
                "latency_improvement_pct_vs_baseline": latency_gain,
                "memory_improvement_pct_vs_baseline": memory_gain,
            }
        )
    return table


def render_markdown_table(table: Sequence[dict]) -> str:
    headers = [
        "Variant",
        "Mean Latency (ms)",
        "P50 (ms)",
        "P95 (ms)",
        "Throughput (req/s)",
        "Peak Memory (MB)",
        "Model Size (MB)",
        "Latency Δ vs Baseline",
        "Memory Δ vs Baseline",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in table:
        lines.append(
            "| {variant} | {mean_latency_ms} | {p50_latency_ms} | {p95_latency_ms} | "
            "{throughput_rps} | {peak_memory_mb} | {model_size_mb} | "
            "{latency_improvement_pct_vs_baseline}% | {memory_improvement_pct_vs_baseline}% |".format(
                variant=row["variant"],
                mean_latency_ms=row["mean_latency_ms"],
                p50_latency_ms=row["p50_latency_ms"],
                p95_latency_ms=row["p95_latency_ms"],
                throughput_rps=row["throughput_rps"],
                peak_memory_mb=row["peak_memory_mb"],
                model_size_mb=row.get("model_size_mb", "N/A"),
                latency_improvement_pct_vs_baseline=row["latency_improvement_pct_vs_baseline"],
                memory_improvement_pct_vs_baseline=row["memory_improvement_pct_vs_baseline"],
            )
        )
    return "\n".join(lines)


def save_benchmark_report(
    results: Sequence[BenchmarkMetrics],
    output_dir: str | Path,
) -> tuple[Path, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    table = build_comparison_table(results)
    json_path = output_dir / "week3_benchmark.json"
    md_path = output_dir / "week3_benchmark.md"

    payload = {
        "metrics": [asdict(r) for r in results],
        "comparison_table": table,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md_content = "# Week 3 — Inference Optimization Benchmark\n\n"
    md_content += render_markdown_table(table)
    md_content += "\n"
    md_path.write_text(md_content, encoding="utf-8")

    logger.info("Benchmark report saved to %s and %s", json_path, md_path)
    return json_path, md_path
