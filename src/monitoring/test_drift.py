from drift import ks_drift
import numpy as np

print("START TEST")

train = np.random.normal(0, 1, 1000)
new = np.random.normal(1, 1, 1000)

result = ks_drift(train, new)

print("DRIFT RESULT:", result)