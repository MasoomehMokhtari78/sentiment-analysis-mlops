"""FastAPI application — Week 3 model serving."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from src.serving.batching import AsyncBatchProcessor
from src.serving.config import serving_settings
from src.serving.labels import normalize_sentiment_label
from src.serving.model_loader import load_sklearn_from_registry
from src.serving.optimization.quantization import export_and_quantize_sklearn
from src.serving.predictors import (
    BatchedSklearnPredictor,
    QuantizedOnnxPredictor,
    SklearnPredictor,
)
from src.serving.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictRequest,
    PredictResponse,
    ProbabilityScores,
    SentimentLabel,
)

logger = logging.getLogger(__name__)


class ServingState:
    """Application-scoped inference resources."""

    def __init__(self) -> None:
        self.model_uri: str | None = None
        self.baseline: SklearnPredictor | None = None
        self.batched: BatchedSklearnPredictor | None = None
        self.quantized: QuantizedOnnxPredictor | None = None
        self.batch_processor: AsyncBatchProcessor | None = None


state = ServingState()


def _to_response(result, variant: str) -> PredictResponse:
    probs = result.probabilities
    return PredictResponse(
        sentiment=SentimentLabel(result.sentiment),
        confidence=result.confidence,
        probabilities=ProbabilityScores(
            negative=probs.get("negative", 0.0),
            neutral=probs.get("neutral", 0.0),
            positive=probs.get("positive", 0.0),
        ),
        model_variant=variant,
    )


def _get_predictor(variant: Literal["baseline", "batched", "quantized"]):
    if variant == "baseline" and state.baseline:
        return state.baseline
    if variant == "batched" and state.batched:
        return state.batched
    if variant == "quantized" and state.quantized:
        return state.quantized
    raise HTTPException(status_code=503, detail=f"Predictor '{variant}' is not available")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading model from MLflow registry...")
    pipeline, model_uri = load_sklearn_from_registry()
    state.model_uri = model_uri

    state.baseline = SklearnPredictor(pipeline)
    state.batched = BatchedSklearnPredictor(pipeline)
    state.batch_processor = AsyncBatchProcessor(
        predictor=state.batched,
        max_batch_size=serving_settings.batch_max_size,
        max_wait_ms=serving_settings.batch_wait_ms,
    )

    try:
        quantized_path = export_and_quantize_sklearn(
            pipeline,
            serving_settings.quantized_model_dir,
        )
        state.quantized = QuantizedOnnxPredictor(quantized_path)
        state.quantized.warmup()
    except Exception as exc:
        logger.warning("INT8 quantization unavailable, quantized route disabled: %s", exc)

    state.baseline.warmup()
    state.batched.warmup()
    ready = ["baseline", "batched"] + (["quantized"] if state.quantized else [])
    logger.info("Serving stack ready (%s)", ", ".join(ready))

    yield

    if state.batch_processor:
        await state.batch_processor.shutdown()


app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "Week 3 MLOps serving layer. Loads the registered sklearn model from "
        "MLflow and exposes baseline, batched, and INT8-quantized inference paths."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.get("/health", response_model=HealthResponse, tags=["Operations"])
async def health() -> HealthResponse:
    loaded = state.baseline is not None
    optimizations = []
    if state.batched:
        optimizations.append("vectorized_batching")
    if state.quantized:
        optimizations.append("int8_quantization")
    if state.batch_processor:
        optimizations.append("async_micro_batching")

    return HealthResponse(
        status="healthy" if loaded else "degraded",
        model_loaded=loaded,
        model_name=serving_settings.registered_model_name if loaded else None,
        model_stage=serving_settings.model_stage if loaded else None,
        optimizations=optimizations,
    )


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Operations"])
async def model_info() -> ModelInfoResponse:
    if not state.model_uri:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        registered_model_name=serving_settings.registered_model_name,
        model_stage=serving_settings.model_stage,
        model_uri=state.model_uri,
        classes=[
            normalize_sentiment_label(c)
            for c in (state.baseline.model_classes if state.baseline else serving_settings.classes)
        ],
        optimizations_enabled=["vectorized_batching", "int8_quantization", "async_micro_batching"],
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
async def predict(
    request: PredictRequest,
    variant: Literal["baseline", "batched", "quantized", "async_batched"] = Query(
        default="async_batched",
        description=(
            "baseline: sequential sklearn; batched: explicit vectorized batch; "
            "quantized: INT8 ONNX; async_batched: concurrent micro-batching"
        ),
    ),
) -> PredictResponse:
    if variant == "async_batched":
        if not state.batch_processor:
            raise HTTPException(status_code=503, detail="Batch processor not ready")
        result = await state.batch_processor.predict(request.text)
        return _to_response(result, "batched")

    predictor = _get_predictor(variant)
    result = predictor.predict_one(request.text)
    return _to_response(result, variant)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
async def predict_batch(
    request: BatchPredictRequest,
    variant: Literal["baseline", "batched", "quantized"] = Query(default="batched"),
) -> BatchPredictResponse:
    predictor = _get_predictor(variant)
    results = predictor.predict_batch(request.texts)
    return BatchPredictResponse(
        predictions=[_to_response(r, variant) for r in results],
        batch_size=len(results),
        model_variant=variant,
    )
