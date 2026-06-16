import numpy as np
from scipy.stats import ks_2samp

def ks_drift(train_data, new_data, threshold=0.05):
    stat, p_value = ks_2samp(train_data, new_data)

    print("KS Statistic:", stat)
    print("P-value:", p_value)

    return p_value < threshold