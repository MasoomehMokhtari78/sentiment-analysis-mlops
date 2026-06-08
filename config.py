# config.py
from pathlib import Path

class Settings:

    # Path where processed data is stored/retrieved
    processed_data_dir: str = "data/processed"
    
    sample_size: int = 1000
    test_size: float = 0.2
    random_state: int = 42
    
    # Model hyperparameters
    max_length: int = 128
    batch_size: int = 16

settings = Settings()
