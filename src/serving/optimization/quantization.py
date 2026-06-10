"""8-bit quantization for sklearn (ONNX) and PyTorch (DistilBERT) models."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


def get_quantized_model_paths(output_dir: str | Path) -> tuple[Path, Path]:
    """Return (onnx_model_path, quantized_onnx_path)."""
    root = Path(output_dir)
    return root / "model.onnx", root / "model.int8.onnx"


def export_and_quantize_sklearn(
    model: Pipeline,
    output_dir: str | Path,
    *,
    n_features: int = 5000,
    force: bool = False,
) -> Path:
    """
    Export sklearn Pipeline to ONNX and apply dynamic INT8 quantization.

    For TF-IDF + LogisticRegression (our baseline), ONNX Runtime dynamic
    quantization reduces model size and often improves CPU inference latency.
    """
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType
    from onnxruntime.quantization import QuantType, quantize_dynamic

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path, quantized_path = get_quantized_model_paths(output_dir)

    if quantized_path.exists() and not force:
        logger.info("Reusing existing quantized model at %s", quantized_path)
        return quantized_path

    # Fitted pipeline: no converter options needed (max_features is already learned).
    initial_type = [("input", StringTensorType([None, 1]))]

    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        target_opset=15,
    )
    onnx_path.write_bytes(onnx_model.SerializeToString())
    logger.info("Exported ONNX model to %s", onnx_path)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quantized_path),
        weight_type=QuantType.QUInt8,
    )
    logger.info("Quantized ONNX model saved to %s", quantized_path)
    return quantized_path


def quantize_pytorch_model(
    model_dir: str | Path,
    output_dir: str | Path,
    *,
    force: bool = False,
) -> Path:
    """
    Apply PyTorch dynamic INT8 quantization to a saved DistilBERT checkpoint.

    Use this path when serving the transformer model from Week 2 artifacts.
    """
    import torch
    from transformers import DistilBertForSequenceClassification

    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    quantized_path = output_dir / "distilbert_int8.pt"

    if quantized_path.exists() and not force:
        logger.info("Reusing existing quantized PyTorch model at %s", quantized_path)
        return quantized_path

    model = DistilBertForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8,
    )
    torch.save(quantized.state_dict(), quantized_path)
    logger.info("Saved quantized PyTorch weights to %s", quantized_path)
    return quantized_path


def model_size_bytes(path: str | Path) -> int:
    """Return on-disk model artifact size in bytes."""
    path = Path(path)
    if path.is_file():
        return path.stat().st_size
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def summarize_quantization_gain(
    baseline_path: str | Path,
    optimized_path: str | Path,
) -> dict[str, Any]:
    baseline_size = model_size_bytes(baseline_path)
    optimized_size = model_size_bytes(optimized_path)
    reduction_pct = (
        100.0 * (1 - optimized_size / baseline_size) if baseline_size else 0.0
    )
    return {
        "baseline_size_mb": round(baseline_size / (1024 * 1024), 3),
        "optimized_size_mb": round(optimized_size / (1024 * 1024), 3),
        "size_reduction_pct": round(reduction_pct, 2),
    }
