"""
Transformer-based sentiment model using DistilBERT.
Implements ModelInterface for interchangeability.
"""
from typing import List, Dict, Any, Union, Optional
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm
import logging
import joblib
from pathlib import Path

from src.models.base import BaseModel, ModelEvaluator

logger = logging.getLogger(__name__)


class SentimentDataset(Dataset):
    """Custom dataset for sentiment analysis with DistilBERT."""
    
    def __init__(self, texts: List[str], labels: Optional[List[int]] = None, tokenizer=None, max_length: int = 128):
        """
        Initialize dataset.
        
        Args:
            texts: List of input texts
            labels: List of labels (optional for prediction)
            tokenizer: DistilBERT tokenizer
            max_length: Maximum sequence length
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return item


class DistilBERTModel(BaseModel):
    """
    Advanced model using DistilBERT for sentiment classification.
    Provides state-of-the-art performance with reasonable compute requirements.
    """
    
    def __init__(
        self,
        max_length: int = 128,
        batch_size: int = 32,
        epochs: int = 3,
        learning_rate: float = 2e-5,
        random_state: int = 42
    ):
        """
        Initialize DistilBERT model.
        
        Args:
            max_length: Maximum token sequence length
            batch_size: Training batch size
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            random_state: Random seed for reproducibility
        """
        super().__init__()
        
        self.max_length = max_length
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.random_state = random_state
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=2  
        ).to(self.device)
        
        # Label mapping
        self.label_map = {0: 0, 1: 1, '0': 0, '1': 1}
        self.reverse_label_map = {0: 0, 1: 1}
    
    def train(self, X_train: Union[List[str], pd.Series], y_train: Union[List[str], pd.Series]) -> None:
        """
        Train the DistilBERT model.
        
        Args:
            X_train: Training texts
            y_train: Training labels
        """
        logger.info("Training DistilBERT model")
        
        X_train = list(X_train)
        y_train_encoded = [self.label_map[label] for label in y_train]
        
        # Create dataset and dataloader
        train_dataset = SentimentDataset(
            X_train,
            y_train_encoded,
            self.tokenizer,
            self.max_length
        )
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        # Setup optimizer and scheduler
        optimizer = AdamW(self.model.parameters(), lr=self.learning_rate)
        total_steps = len(train_loader) * self.epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Training loop
        self.model.train()
        
        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs}")
            total_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch + 1}")
            
            for batch in progress_bar:
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                progress_bar.set_postfix({'loss': loss.item()})
            
            avg_loss = total_loss / len(train_loader)
            logger.info(f"Average loss for epoch {epoch + 1}: {avg_loss:.4f}")
        
        self.is_trained = True
        logger.info("DistilBERT model training complete")
    
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
        
        self.model.eval()
        
        dataset = SentimentDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        predictions = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
                predictions.extend(preds)
        
        result = [self.reverse_label_map[pred] for pred in predictions]
        
        return result[0] if single_input else result
    
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
        
        self.model.eval()
        
        dataset = SentimentDataset(texts, None, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size)
        
        all_probs = []
        
        with torch.no_grad():
            for batch in loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1).cpu().numpy()
                all_probs.extend(probs)
        
        return self._format_probabilities(np.array(all_probs), single_input)
    
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
        
        y_pred = self.predict(X_test)
        
        metrics = ModelEvaluator.evaluate(y_test, y_pred)
        
        logger.info(f"DistilBERT Model - Accuracy: {metrics['accuracy']:.4f}, F1-Score: {metrics['f1_score']:.4f}")
        
        return metrics
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Directory path to save model
        """
        self._validate_trained()
        
        save_dir = Path(path)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save transformer model and tokenizer
        self.model.save_pretrained(save_dir / "model")
        self.tokenizer.save_pretrained(save_dir / "tokenizer")
        
        # Save configuration
        config = {
            'max_length': self.max_length,
            'batch_size': self.batch_size,
            'label_map': self.label_map,
            'reverse_label_map': self.reverse_label_map,
            'classes': self.classes_,
            'is_trained': self.is_trained
        }
        joblib.dump(config, save_dir / "config.pkl")
        
        logger.info(f"DistilBERT model saved to {path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Directory path to load model from
        """
        load_dir = Path(path)
        
        if not load_dir.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")
        
        # Load transformer model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(
            load_dir / "model"
        ).to(self.device)
        self.tokenizer = DistilBertTokenizer.from_pretrained(load_dir / "tokenizer")
        
        # Load configuration
        config = joblib.load(load_dir / "config.pkl")
        self.max_length = config['max_length']
        self.batch_size = config['batch_size']
        self.label_map = config['label_map']
        self.reverse_label_map = config['reverse_label_map']
        self.classes_ = config['classes']
        self.is_trained = config['is_trained']
        
        logger.info(f"DistilBERT model loaded from {path}")
