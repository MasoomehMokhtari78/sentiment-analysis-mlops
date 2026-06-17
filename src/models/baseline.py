"""
Baseline sentiment model using TF-IDF and Logistic Regression.
Implements ModelInterface for interchangeability.
"""
from typing import List, Dict, Any, Union
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import logging

from src.models.base import BaseModel, ModelEvaluator

logger = logging.getLogger(__name__)


class BaselineModel(BaseModel):
    """
    Baseline model using TF-IDF vectorization and Logistic Regression.
    Fast training and inference, good baseline performance.
    """
    
    def __init__(
        self,
        max_features: int = 10000,
        ngram_range: tuple = (1, 2),
        max_iter: int = 1000,
        random_state: int = 42
    ):
        """
        Initialize baseline model.
        
        Args:
            max_features: Maximum number of TF-IDF features
            ngram_range: N-gram range for TF-IDF
            max_iter: Maximum iterations for logistic regression
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Create pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=max_features,
                ngram_range=ngram_range,
                stop_words='english',
                strip_accents='unicode',
                lowercase=True
            )),
            ('classifier', LogisticRegression(
                max_iter=max_iter,
                random_state=random_state,
                class_weight='balanced',
                solver='lbfgs'
            ))
        ])
    
    def train(self, X_train: Union[List[str], pd.Series], y_train: Union[List[str], pd.Series]) -> None:
        """
        Train the baseline model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        logger.info("Training baseline model (TF-IDF + Logistic Regression)")
        
        X_train = list(X_train)
        y_train = list(y_train)
        
        self.model.fit(X_train, y_train)
        classifier = self.model.named_steps["classifier"]
        self.classes_ = list(classifier.classes_)
        self.is_trained = True
        
        logger.info("Baseline model training complete")
    
    def predict(self, texts: Union[str, List[str]]) -> Union[str, List[str]]:
        """
        Predict sentiment for text(s).
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            Predicted sentiment(s)
        """
        self._validate_trained()
        
        single_input = isinstance(texts, str)
        texts = self._ensure_list(texts)
        
        predictions = self.model.predict(texts)
        
        return predictions[0] if single_input else predictions.tolist()
    
    def predict_proba(self, texts: Union[str, List[str]]) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Get prediction probabilities.
        
        Args:
            texts: Input text or list of texts
            
        Returns:
            Dictionary or list of dictionaries with class probabilities
        """
        self._validate_trained()
        
        single_input = isinstance(texts, str)
        texts = self._ensure_list(texts)
        
        proba = self.model.predict_proba(texts)
        
        return self._format_probabilities(proba, single_input)
    
    def evaluate(self, X_test: Union[List[str], pd.Series], y_test: Union[List[str], pd.Series]) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            X_test: Test texts
            y_test: Test labels
            
        Returns:
            Dictionary with evaluation metrics
        """
        self._validate_trained()
        
        X_test = list(X_test)
        y_test = list(y_test)
        
        y_pred = self.model.predict(X_test)
        
        metrics = ModelEvaluator.evaluate(y_test, y_pred)
        
        logger.info(f"Baseline Model - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
