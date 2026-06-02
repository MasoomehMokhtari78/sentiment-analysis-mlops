import argparse
import logging
from prefect import task, flow
from typing import Optional
from src.data.processor import DataProcessor, IMDBReviewLoader, BasicTextCleaner

logger = logging.getLogger(__name__)

# Configure streaming logger format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@task(name="1. Ingest Raw Data")
def load_data_task(loader, input_path, sample_size):
    return loader.load_data(input_path, sample_size)

@task(name="2. Run Text Transformations & PII Masking")
def process_data_task(processor, raw_df):
    logger.info("Running text optimization layer and PII scrubbing loops")
    # Apply cleaning logic directly using the cleaner inside processor
    raw_df['cleaned_text'] = raw_df['text'].apply(processor.cleaner.clean)
    
    # Strip empty processing elements
    raw_df = raw_df[raw_df['cleaned_text'].str.len() > 0].reset_index(drop=True)
    
    # Use sklearn train_test_split logic from DataProcessor
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        raw_df,
        test_size=0.2,
        random_state=42,
        stratify=raw_df['sentiment']
    )
    logger.info(f"Splits complete -> Train rows: {len(train_df)} | Test rows: {len(test_df)}")
    return train_df, test_df

@task(name="3. Save Partitioned Splits")
def save_data_task(processor, train_df, test_df, output_dir):
    # The processor now returns whether it actually performed a write
    was_saved = processor.save_processed_data(train_df=train_df, test_df=test_df, output_dir=output_dir)
    if not was_saved:
        return "SKIPPED"
    return "SUCCESS"


@flow(name="IMDB-Sentiment-Analysis-Pipeline")
def compliance_pipeline(input_path: str, output_dir: str, sample_size: Optional[int] = None):
    print("\n=== Initializing Prefect Orchestrated ML Flow ===")
    
    # Initialize Core SOLID Objects
    loader = IMDBReviewLoader()
    cleaner = BasicTextCleaner(lowercase=True, remove_special=True, anonymize_pii=True)
    processor = DataProcessor(loader=loader, cleaner=cleaner)
    
    # Execute Orchestrated DAG Tasks
    raw_data = load_data_task(loader, input_path, sample_size)
    train_df, test_df = process_data_task(processor, raw_data)
    save_data_task(processor, train_df, test_df, output_dir)
    
    print("=== Prefect Pipeline Run Completed Successfully ===\n")


def main():
    parser = argparse.ArgumentParser(description="Ingest and pipeline-process structural review assets.")
    parser.add_argument("--input-path", type=str, required=True, help="Path to raw input CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save datasets")
    parser.add_argument("--sample-size", type=int, default=None, help="Optional row limit downsampling")

    args = parser.parse_args()

    # Trigger the Prefect Flow execution
    compliance_pipeline(
        input_path=args.input_path, 
        output_dir=args.output_dir, 
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()