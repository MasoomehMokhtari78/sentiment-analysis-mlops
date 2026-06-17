import argparse
import logging
from prefect import task, flow
from typing import Optional
from src.data.processor import DataProcessor, IMDBReviewLoader, BasicTextCleaner

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# ----------------------------
# PREFECT TASKS
# ----------------------------

@task(name="1. Ingest Raw Data")
def load_data_task(loader, input_path, sample_size):
    return loader.load_data(input_path, sample_size)


@task(name="2. Run Text Transformations & PII Masking")
def process_data_task(processor, raw_df):
    logger.info("Running text optimization layer and PII scrubbing loops")

    raw_df['cleaned_text'] = raw_df['text'].apply(processor.cleaner.clean)

    raw_df = raw_df[raw_df['cleaned_text'].str.len() > 0].reset_index(drop=True)

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
    was_saved = processor.save_processed_data(
        train_df=train_df,
        test_df=test_df,
        output_dir=output_dir
    )

    return "SUCCESS" if was_saved else "SKIPPED"


# ----------------------------
# PREFECT FLOW
# ----------------------------

@flow(name="IMDB-Sentiment-Analysis-Pipeline")
def compliance_pipeline(input_path: str, output_dir: str, sample_size: Optional[int] = None):
    print("\n=== Initializing Prefect Orchestrated ML Flow ===")

    loader = IMDBReviewLoader()
    cleaner = BasicTextCleaner(lowercase=True, remove_special=True, anonymize_pii=True)
    processor = DataProcessor(loader=loader, cleaner=cleaner)

    raw_data = load_data_task(loader, input_path, sample_size)
    train_df, test_df = process_data_task(processor, raw_data)
    save_data_task(processor, train_df, test_df, output_dir)

    print("=== Prefect Pipeline Run Completed Successfully ===\n")


# ----------------------------
# MAIN (SAFE + DVC COMPATIBLE)
# ----------------------------

def main():
    parser = argparse.ArgumentParser(description="Ingest and pipeline-process structural review assets.")

    # NEW MODE (manual / user run)
    parser.add_argument("--input-path", type=str, required=False)

    # OLD MODE (DVC pipeline compatibility)
    parser.add_argument("--train-path", type=str, required=False)
    parser.add_argument("--test-path", type=str, required=False)

    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--sample-size", type=int, default=None)

    args = parser.parse_args()

    # ----------------------------
    # COMPATIBILITY RESOLVER
    # ----------------------------

    if args.input_path:
        input_path = args.input_path
        logger.info("Running SINGLE input mode (input-path)")

    elif args.train_path and args.test_path:
        # IMPORTANT:
        # DVC pipeline compatibility
        # We only ingest train; test is generated via split in flow
        input_path = args.train_path
        logger.info("Running DVC mode (train/test provided)")

    else:
        raise ValueError(
            "Invalid arguments. Use either --input-path OR (--train-path + --test-path)"
        )

    # ----------------------------
    # RUN PIPELINE
    # ----------------------------

    compliance_pipeline(
        input_path=input_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size
    )


if __name__ == "__main__":
    main()