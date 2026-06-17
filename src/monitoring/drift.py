import pandas as pd
from scipy.stats import ks_2samp


def text_length_drift(train_texts, new_texts):
    """
    Drift based on text length distribution (real NLP drift)
    """

    train_lengths = [len(str(t)) for t in train_texts]
    new_lengths = [len(str(t)) for t in new_texts]

    stat, p_value = ks_2samp(train_lengths, new_lengths)

    return {
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05)
    }


def load_imdb_drift(train_path, test_path):
    """
    Load real IMDB data and compute drift
    """

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return text_length_drift(
        train_df["text"],
        test_df["text"]
    )