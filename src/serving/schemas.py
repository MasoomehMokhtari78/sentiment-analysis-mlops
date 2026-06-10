"""Pydantic request/response schemas for the sentiment API."""
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class SentimentLabel(str, Enum):
    negative = "negative"
    neutral = "neutral"
    positive = "positive"


class PredictRequest(BaseModel):
    """Single-text prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        description="Review or sentence to classify",
        examples=["This movie was absolutely wonderful!"],
    )

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("text must not be empty or whitespace only")
        return stripped


class BatchPredictRequest(BaseModel):
    """Explicit batch prediction request."""

    texts: list[str] = Field(
        ...,
        min_length=1,
        max_length=128,
        description="List of texts to classify in one vectorized call",
    )

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, texts: list[str]) -> list[str]:
        cleaned = []
        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                raise ValueError(f"texts[{i}] must not be empty or whitespace only")
            if len(stripped) > 10_000:
                raise ValueError(f"texts[{i}] exceeds maximum length of 10000")
            cleaned.append(stripped)
        return cleaned


class ProbabilityScores(BaseModel):
    negative: float = Field(..., ge=0.0, le=1.0)
    neutral: float = Field(..., ge=0.0, le=1.0)
    positive: float = Field(..., ge=0.0, le=1.0)


class PredictResponse(BaseModel):
    sentiment: SentimentLabel
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: ProbabilityScores
    model_variant: Literal["baseline", "batched", "quantized"] = "baseline"


class BatchPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    batch_size: int
    model_variant: Literal["baseline", "batched", "quantized"] = "batched"


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded"]
    model_loaded: bool
    model_name: str | None = None
    model_stage: str | None = None
    optimizations: list[str] = Field(default_factory=list)


class ModelInfoResponse(BaseModel):
    registered_model_name: str
    model_stage: str
    model_uri: str
    classes: list[str]
    optimizations_enabled: list[str]
