import pandas as pd
import numpy as np
from drift import ks_drift

# ----------------------------
# Load real IMDB data
# ----------------------------
train_df = pd.read_csv("data/raw/imdb_train.csv")
test_df = pd.read_csv("data/raw/imdb_test.csv")

# ----------------------------
train_lengths = train_df["review"].apply(len).values
test_lengths = test_df["review"].apply(len).values

# ----------------------------
# Drift detection
# ----------------------------
print("Running real IMDB drift detection...")

result = ks_drift(train_lengths, test_lengths)

print("FINAL DRIFT RESULT:", result)