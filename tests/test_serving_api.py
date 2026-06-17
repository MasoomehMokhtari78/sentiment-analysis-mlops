from fastapi.testclient import TestClient

from src.serving.predictors.base import PredictionResult


class _FakePredictor:
    model_classes = ["negative", "neutral", "positive"]

    def warmup(self) -> None:
        return None

    def predict_one(self, text: str) -> PredictionResult:
        return PredictionResult(
            sentiment="positive",
            confidence=0.91,
            probabilities={"negative": 0.03, "neutral": 0.06, "positive": 0.91},
        )

    def predict_batch(self, texts):
        return [self.predict_one(text) for text in texts]


class _FakeBatchProcessor:
    async def predict(self, text: str) -> PredictionResult:
        return PredictionResult(
            sentiment="negative",
            confidence=0.88,
            probabilities={"negative": 0.88, "neutral": 0.07, "positive": 0.05},
        )

    async def shutdown(self) -> None:
        return None


def _build_client():
    from src.serving import app as serving_app

    serving_app.state.model_uri = "file://tests/fake.pkl"
    serving_app.state.baseline = _FakePredictor()
    serving_app.state.batched = _FakePredictor()
    serving_app.state.quantized = _FakePredictor()
    serving_app.state.batch_processor = _FakeBatchProcessor()
    return TestClient(serving_app.app)


def test_health_endpoint() -> None:
    client = _build_client()
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["model_loaded"] is True
    assert payload["status"] in {"healthy", "degraded"}


def test_predict_endpoint_default_async_batched() -> None:
    client = _build_client()
    response = client.post("/predict", json={"text": "great movie"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["sentiment"] == "negative"
    assert payload["model_variant"] == "batched"


def test_predict_endpoint_validation_error() -> None:
    client = _build_client()
    response = client.post("/predict", json={"text": "   "})
    assert response.status_code == 422
