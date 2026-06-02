"""
Data processing module following SOLID principles.
Single Responsibility: Each class handles one aspect of data processing.
Open/Closed: New loaders or cleaners can be added without modifying existing code.
Dependency Inversion: High-level processor depends on abstractions, not concretions.
"""
from abc import ABC, abstractmethod
from typing import Optional
import json
import re
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class DataLoaderInterface(ABC):
    """Abstract base class for data loaders (Dependency Inversion Principle)."""
    
    @abstractmethod
    def load_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load data from source."""
        pass


class IMDBReviewLoader(DataLoaderInterface):
    """Concrete implementation for loading IMDB CSV review data."""
    
    def load_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Loading IMDB data from {file_path}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
            
        df = pd.read_csv(file_path)
        
        # Map 'text' column if it's named 'review'
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
            
        # Map 'label' to 'sentiment' so DataProcessor can find it
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'sentiment'})
            
        if sample_size:
            df = df.head(sample_size)
            
        return df


class TextCleanerInterface(ABC):
    """Abstract base class for text cleaning (Open/Closed Principle)."""
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text data."""
        pass


class BasicTextCleaner(TextCleanerInterface):
    """
    Basic text cleaning implementation handling formatting and anonymization.
    """
    
    def __init__(self, lowercase: bool = True, remove_special: bool = True, anonymize_pii: bool = True):
        self.lowercase = lowercase
        self.remove_special = remove_special
        self.anonymize_pii = anonymize_pii
    
    def _mask_pii(self, text: str) -> str:
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[ANONYMIZED_EMAIL]', text)
        text = re.sub(r'@\w+', '[ANONYMIZED_USER]', text)
        return text

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        if self.anonymize_pii:
            text = self._mask_pii(text)
        
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'<.*?>', '', text)
        text = ' '.join(text.split())
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special:
            text = re.sub(r'[^a-zA-Z0-9\s\[\]_]', '', text)
        
        return text.strip()


class DataProcessor:
    """
    Orchestration engine coordinating loading, cleaning, 
    outlier removal, and augmentation.
    """
    
    def __init__(self, loader: DataLoaderInterface, cleaner: TextCleanerInterface):
        self.loader = loader
        self.cleaner = cleaner
    
    def remove_outliers(self, df: pd.DataFrame, min_len: int = 5, max_len: int = 1000) -> pd.DataFrame:
        """Removes rows where text length is outside thresholds."""
        df = df.copy()
        df['length'] = df['cleaned_text'].str.len()
        initial_count = len(df)
        df = df[(df['length'] >= min_len) & (df['length'] <= max_len)].drop(columns=['length'])
        logger.info(f"Outlier removal: Removed {initial_count - len(df)} samples.")
        return df

    def augment_data(self, df: pd.DataFrame, factor: int = 2) -> pd.DataFrame:
        """Simple augmentation by replicating samples."""
        # Simple synonym replacement for demo purposes
        aug_df = df.copy()
        aug_df['cleaned_text'] = aug_df['cleaned_text'].str.replace('good', 'excellent', regex=False)
        return pd.concat([df, aug_df], ignore_index=True)

    def process(
        self,
        file_path: str,
        is_training: bool = False,
        sample_size: Optional[int] = None,
        remove_outliers: bool = False,
        augment: bool = False
    ) -> pd.DataFrame:
        
        # 1. Load
        df = self.loader.load_data(file_path, sample_size)
        
        # 2. Clean
        df['cleaned_text'] = df['text'].apply(self.cleaner.clean)
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        # 3. Handle Outliers
        if remove_outliers:
            df = self.remove_outliers(df)
        
        # 4. Augment (Only if it's training data)
        if is_training and augment:
            df = self.augment_data(df, factor=2)
            logger.info(f"Augmented training data. New size: {len(df)}")
            
        return df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/processed") -> bool:
        """Saves data to output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        logger.info(f"Successfully exported data to: {output_dir}")
        return True
