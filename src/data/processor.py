"""
Data processing module following SOLID principles.
Single Responsibility: Each class handles one aspect of data processing.
Open/Closed: New loaders or cleaners can be added without modifying existing code.
Dependency Inversion: High-level processor depends on abstractions, not concretions.
"""
from abc import ABC, abstractmethod
from typing import Tuple, Optional
import json
import re
import pandas as pd
from sklearn.model_selection import train_test_split
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
        
        # Map 'text' column if it's named 'review' (yours is already 'text', so this is safe)
        if 'review' in df.columns:
            df = df.rename(columns={'review': 'text'})
            
        # Map 'label' to 'sentiment' so DataProcessor can find it
        if 'label' in df.columns:
            df = df.rename(columns={'label': 'sentiment'})
            
        if sample_size:
            df = df.head(sample_size)
            
        return df

class YelpReviewLoader(DataLoaderInterface):
    """
    Concrete implementation for loading Yelp review data.
    Handles large JSON files using chunked reading.
    """
    
    def __init__(self, chunk_size: int = 10000):
        self.chunk_size = chunk_size
    
    def load_data(self, file_path: str, sample_size: Optional[int] = None) -> pd.DataFrame:
        logger.info(f"Loading data from {file_path}")
        if not Path(file_path).exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        records = []
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if sample_size and i >= sample_size:
                        break
                    
                    try:
                        record = json.loads(line)
                        if 'text' in record and 'stars' in record:
                            stars = record['stars']
                            
                            if stars <= 2:
                                sentiment = 'negative'
                            elif stars == 3:
                                sentiment = 'neutral'
                            else:
                                sentiment = 'positive'
                            
                            records.append({
                                'text': record['text'],
                                'sentiment': sentiment,
                                'stars': stars
                            })
                    except json.JSONDecodeError:
                        logger.warning(f"Skipping invalid JSON at line {i}")
                        continue
                    
                    if (i + 1) % self.chunk_size == 0:
                        logger.info(f"Loaded {i + 1} records")
            
            logger.info(f"Total records loaded: {len(records)}")
            df = pd.DataFrame(records)
            df = self._balance_classes(df)
            return df
        
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def _balance_classes(self, df: pd.DataFrame, max_per_class: int = 30000) -> pd.DataFrame:
        balanced_dfs = []
        for sentiment in df['sentiment'].unique():
            sentiment_df = df[df['sentiment'] == sentiment]
            if len(sentiment_df) > max_per_class:
                sentiment_df = sentiment_df.sample(n=max_per_class, random_state=42)
            balanced_dfs.append(sentiment_df)
        
        result = pd.concat(balanced_dfs, ignore_index=True)
        result = result.sample(frac=1, random_state=42).reset_index(drop=True)
        logger.info(f"Class distribution after balancing:\n{result['sentiment'].value_counts()}")
        return result


class TextCleanerInterface(ABC):
    """Abstract base class for text cleaning (Open/Closed Principle)."""
    
    @abstractmethod
    def clean(self, text: str) -> str:
        """Clean text data."""
        pass


class BasicTextCleaner(TextCleanerInterface):
    """
    Basic text cleaning implementation handling formatting and anonymization.
    Single Responsibility: Only handles text token scrubbing and PII containment.
    """
    
    def __init__(self, lowercase: bool = True, remove_special: bool = True, anonymize_pii: bool = True):
        self.lowercase = lowercase
        self.remove_special = remove_special
        self.anonymize_pii = anonymize_pii
    
    def _mask_pii(self, text: str) -> str:
        """Mask potential structural PII footprints like email expressions or user handles."""
        # Clean email targets
        text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '[ANONYMIZED_EMAIL]', text)
        # Clean @username metadata tags
        text = re.sub(r'@\w+', '[ANONYMIZED_USER]', text)
        return text

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        
        # 1. Apply Pseudonymization rules first
        if self.anonymize_pii:
            text = self._mask_pii(text)
        
        # Remove URLs and Hyperlinks
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove HTML layout tags
        text = re.sub(r'<.*?>', '', text)
        # Compress whitespace chunks
        text = ' '.join(text.split())
        
        if self.lowercase:
            text = text.lower()
        
        if self.remove_special:
            # Alpha-numeric text scrubbing, retaining string sequence mask tokens `[]`
            text = re.sub(r'[^a-zA-Z0-9\s\[\]_]', '', text)
        
        return text.strip()


class DataProcessor:
    """
    High-level orchestration engine coordinating loading and text adjustments.
    Follows Single Responsibility and Dependency Injection principles.
    """
    
    def __init__(self, loader: DataLoaderInterface, cleaner: TextCleanerInterface):
        self.loader = loader
        self.cleaner = cleaner
    
    def process(
        self,
        file_path: str,
        sample_size: Optional[int] = None,
        test_size: float = 0.2,
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        
        # Execute Dependency Injected Loader
        df = self.loader.load_data(file_path, sample_size)
        
        # Execute Dependency Injected Cleaner with mapping
        logger.info("Running text optimization layer and PII scrubbing loops")
        df['cleaned_text'] = df['text'].apply(self.cleaner.clean)
        
        # Clean down empty processing items
        df = df[df['cleaned_text'].str.len() > 0].reset_index(drop=True)
        
        # Stratified Data Split Sequence
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df['sentiment']
        )
        
        logger.info(f"Splits complete -> Train rows: {len(train_df)} | Test rows: {len(test_df)}")
        return train_df, test_df
    
    def save_processed_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: str = "data/processed") -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_df.to_csv(output_path / "train.csv", index=False)
        test_df.to_csv(output_path / "test.csv", index=False)
        logger.info(f"Successfully exported data components directly to: {output_dir}")