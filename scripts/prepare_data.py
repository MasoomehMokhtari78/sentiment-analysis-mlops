"""Download IMDB sample data if raw CSV is not present."""
import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_imdb(output_path: Path, sample_size: int = 5000) -> Path:
    import pandas as pd
    from datasets import concatenate_datasets, load_dataset

    # IMDB train split is ordered: 25k negatives then 25k positives.
    per_class = max(sample_size // 2, 1)
    logger.info(
        "Downloading balanced IMDB sample (%d per class, %d total)...",
        per_class,
        per_class * 2,
    )
    full_train = load_dataset("stanfordnlp/imdb", split="train")
    negatives = full_train.filter(lambda row: row["label"] == 0).select(range(per_class))
    positives = full_train.filter(lambda row: row["label"] == 1).select(range(per_class))
    dataset = concatenate_datasets([negatives, positives]).shuffle(seed=42)

    df = pd.DataFrame({"text": dataset["text"], "label": dataset["label"]})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved %d rows to %s (labels: %s)", len(df), output_path, df["label"].value_counts().to_dict())
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Prepare IMDB training data")
    parser.add_argument("--output", type=str, default="data/raw/imdb_train.csv")
    parser.add_argument("--sample-size", type=int, default=5000)
    args = parser.parse_args()

    output_path = Path(args.output)
    if output_path.exists():
        logger.info("Data already exists at %s — skipping download", output_path)
        return

    download_imdb(output_path, args.sample_size)


if __name__ == "__main__":
    main()
