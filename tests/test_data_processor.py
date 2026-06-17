from pathlib import Path

import pandas as pd

from src.data.processor import BasicTextCleaner, DataProcessor, IMDBReviewLoader


def _sample_dataframe() -> pd.DataFrame:
    rows = []
    for i in range(15):
        rows.append({"review": f"Great movie {i}", "label": "positive"})
        rows.append({"review": f"Bad movie {i}", "label": "negative"})
    return pd.DataFrame(rows)


def test_basic_text_cleaner_masks_pii_and_normalizes() -> None:
    cleaner = BasicTextCleaner(lowercase=True, remove_special=True, anonymize_pii=True)
    cleaned = cleaner.clean("Email me: User@Test.com <b>Hello</b> https://example.com")

    assert "[anonymized_email]" in cleaned
    assert "<b>" not in cleaned
    assert "https" not in cleaned
    assert cleaned == cleaned.lower()


def test_process_creates_cleaned_text_and_stratified_splits(tmp_path: Path) -> None:
    input_csv = tmp_path / "input.csv"
    _sample_dataframe().to_csv(input_csv, index=False)

    processor = DataProcessor(loader=IMDBReviewLoader(), cleaner=BasicTextCleaner())
    train_df, test_df = processor.process(str(input_csv), test_size=0.2, random_state=42)

    assert "cleaned_text" in train_df.columns
    assert "cleaned_text" in test_df.columns
    assert len(train_df) > len(test_df)
    assert set(train_df["sentiment"].unique()) == {"positive", "negative"}
    assert set(test_df["sentiment"].unique()) == {"positive", "negative"}


def test_save_processed_data_is_idempotent(tmp_path: Path) -> None:
    processor = DataProcessor(loader=IMDBReviewLoader(), cleaner=BasicTextCleaner())
    train_df = pd.DataFrame({"cleaned_text": ["a", "b"], "sentiment": ["positive", "negative"]})
    test_df = pd.DataFrame({"cleaned_text": ["c"], "sentiment": ["positive"]})
    out_dir = tmp_path / "processed"

    first_save = processor.save_processed_data(train_df, test_df, str(out_dir))
    second_save = processor.save_processed_data(train_df, test_df, str(out_dir))

    assert first_save is True
    assert second_save is False
    assert (out_dir / "train.csv").exists()
    assert (out_dir / "test.csv").exists()
