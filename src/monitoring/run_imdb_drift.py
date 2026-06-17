from drift import load_imdb_drift
import json
import os


print("Running REAL IMDB Drift Analysis...")

result = load_imdb_drift(
    "data/raw/imdb_train.csv",
    "data/raw/imdb_test.csv"
)

print(result)

os.makedirs("reports/week4", exist_ok=True)

with open("reports/week4/drift_report.json", "w") as f:
    json.dump(result, f, indent=4)

print("Saved IMDB drift report")