import joblib
import os


MODEL_PATH = "models/model.pkl"
VECTORIZER_PATH = "models/vectorizer.pkl"


def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def load_vectorizer():
    if not os.path.exists(VECTORIZER_PATH):
        return None
    return joblib.load(VECTORIZER_PATH)