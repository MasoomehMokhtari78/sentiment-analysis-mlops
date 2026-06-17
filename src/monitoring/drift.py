import pandas as pd
import numpy as np
from scipy.stats import ks_2samp


# =========================================================
# 1. COVARIATE DRIFT
# =========================================================
def text_length_drift(train_texts, new_texts):

    train_lengths = np.array([len(str(t)) for t in train_texts])
    new_lengths = np.array([len(str(t)) for t in new_texts])

    stat, p_value = ks_2samp(train_lengths, new_lengths)

    return {
        "method": "covariate_text_length_ks",
        "ks_statistic": float(stat),
        "p_value": float(p_value),
        "drift_detected": bool(p_value < 0.05)
    }


# =========================================================
# 2. LABEL DRIFT
# =========================================================
def label_drift(train_labels, new_labels):

    train_dist = pd.Series(train_labels).value_counts(normalize=True)
    new_dist = pd.Series(new_labels).value_counts(normalize=True)

    all_labels = set(train_dist.index).union(set(new_dist.index))

    shift = sum(
        abs(train_dist.get(label, 0) - new_dist.get(label, 0))
        for label in all_labels
    )

    return {
        "method": "label_distribution_shift",
        "distribution_shift": float(shift),
        "drift_detected": bool(shift > 0.2)
    }


# =========================================================
# 3. CONCEPT DRIFT (SAFE + FIXED)
# =========================================================
def concept_drift(model, texts, true_labels):

    if model is None:
        return {
            "method": "concept_confidence_mismatch",
            "drift_detected": False,
            "note": "model_not_loaded"
        }

    probs = model.predict_proba(texts)
    preds = np.argmax(probs, axis=1)
    confidences = np.max(probs, axis=1)

    correct = (preds == true_labels).astype(int)

    correct_conf = confidences[correct == 1].mean() if np.any(correct == 1) else 0
    wrong_conf = confidences[correct == 0].mean() if np.any(correct == 0) else 0

    confidence_gap = float(wrong_conf - correct_conf)

    return {
        "method": "concept_confidence_mismatch",
        "avg_conf_correct": float(correct_conf),
        "avg_conf_wrong": float(wrong_conf),
        "confidence_gap": confidence_gap,
        "drift_detected": bool(confidence_gap > 0.15)
    }


# =========================================================
# 4. DOMAIN SHIFT (FIXED)
# =========================================================
def domain_shift(covariate_result, concept_result):

    concept_flag = concept_result.get("drift_detected")

    if concept_flag is None:
        return {
            "method": "domain_shift_detection",
            "drift_detected": False
        }

    return {
        "method": "domain_shift_detection",
        "drift_detected": bool(
            covariate_result["drift_detected"] and concept_flag
        )
    }


# =========================================================
# 5. FULL PIPELINE
# =========================================================
def run_full_drift_detection(train_df, test_df, model=None):

    covariate = text_length_drift(
        train_df["text"],
        test_df["text"]
    )

    label = label_drift(
        train_df["label"],
        test_df["label"]
    )

    concept = concept_drift(
        model,
        test_df["text"],
        test_df["label"]
    )

    domain = domain_shift(covariate, concept)

    return {
        "covariate_drift": covariate,
        "label_drift": label,
        "concept_drift": concept,
        "domain_shift": domain
    }


# =========================================================
# 6. BACKWARD COMPATIBILITY
# =========================================================
def load_imdb_drift(train_path, test_path):

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    return run_full_drift_detection(train_df, test_df)