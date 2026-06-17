import numpy as np
import json
from drift import ks_drift

print("Running Drift Detection...")

# fake production simulation (فعلاً جایگزین IMDB)
train = np.random.normal(0, 1, 1000)
new = np.random.normal(0.8, 1, 1000)

result = ks_drift(train, new)

print(result)

# save report
with open("reports/drift_report.json", "w") as f:
    json.dump(result, f, indent=4)

print("Report saved to reports/drift_report.json")