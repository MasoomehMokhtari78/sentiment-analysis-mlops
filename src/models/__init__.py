"""Models module initialization."""
from src.models.base import ModelInterface, BaseModel, ModelEvaluator
from src.models.baseline import BaselineModel

__all__ = [
    'ModelInterface',
    'BaseModel',
    'ModelEvaluator',
    'BaselineModel',
    'DistilBERTModel',
]


def __getattr__(name: str):
    if name == 'DistilBERTModel':
        from src.models.distilbert import DistilBERTModel
        return DistilBERTModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
