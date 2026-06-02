"""Data module initialization."""
from src.data.processor import (
    DataLoaderInterface,
    IMDBReviewLoader,
    TextCleanerInterface,
    BasicTextCleaner,
    DataProcessor
)

__all__ = [
    'DataLoaderInterface',
    'IMDBReviewLoader',
    'TextCleanerInterface',
    'BasicTextCleaner',
    'DataProcessor'
]
