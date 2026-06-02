import argparse
import logging
from prefect import task, flow
from typing import Optional
from src.data.processor import DataProcessor, IMDBReviewLoader, BasicTextCleaner

logger = logging.getLogger(__name__)

# Configure streaming logger format
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@task(name="Process Data File")
def process_file_task(processor, file_path, is_training, remove_outliers, augment):
    """
    Orchestrates the processing of a single file. 
    The processor handles loading, cleaning, and (if training) augmentation.
    """
    logger.info(f"Processing {'training' if is_training else 'test'} data from {file_path}")
    return processor.process(
        file_path=file_path,
        is_training=is_training,
        remove_outliers=remove_outliers,
        augment=augment
    )

@task(name="Save Partitioned Splits")
def save_data_task(processor, train_df, test_df, output_dir):
    was_saved = processor.save_processed_data(train_df=train_df, test_df=test_df, output_dir=output_dir)
    return "SUCCESS" if was_saved else "SKIPPED"

@flow(name="IMDB-Sentiment-Analysis-Pipeline")
def compliance_pipeline(
    train_path: str, 
    test_path: str, 
    output_dir: str, 
    remove_outliers: bool, 
    augment: bool
):
    print("\n=== Initializing Data-Centric ML Flow ===")
    
    # Initialize Core SOLID Objects
    loader = IMDBReviewLoader()
    cleaner = BasicTextCleaner(lowercase=True, remove_special=True, anonymize_pii=True)
    processor = DataProcessor(loader=loader, cleaner=cleaner)
    
    # 1. Process Train Data (Allowing Augmentation)
    train_df = process_file_task(
        processor, train_path, is_training=True, 
        remove_outliers=remove_outliers, augment=augment
    )
    
    # 2. Process Test Data (No Augmentation allowed - Data-Centric Safety)
    test_df = process_file_task(
        processor, test_path, is_training=False, 
        remove_outliers=remove_outliers, augment=False
    )
    
    # 3. Save
    save_data_task(processor, train_df, test_df, output_dir)
    
    print("=== Prefect Pipeline Run Completed Successfully ===\n")

def main():
    parser = argparse.ArgumentParser(description="Ingest and pipeline-process structural review assets.")
    parser.add_argument("--train-path", type=str, required=True, help="Path to raw train CSV")
    parser.add_argument("--test-path", type=str, required=True, help="Path to raw test CSV")
    parser.add_argument("--output-dir", type=str, default="data/processed", help="Directory to save datasets")
    
    # Data-Centric Flags
    parser.add_argument("--remove-outliers", action="store_true", help="Remove text outliers")
    parser.add_argument("--augment", action="store_true", help="Perform synthetic augmentation on training data")

    args = parser.parse_args()

    compliance_pipeline(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        remove_outliers=args.remove_outliers,
        augment=args.augment
    )

if __name__ == "__main__":
    main()
