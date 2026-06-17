import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os


def retrain_model(train_path="data/raw/imdb_train.csv"):
    """
    Auto retraining pipeline when drift is detected
    """

    print("🚀 Starting auto retraining...")

    df = pd.read_csv(train_path)

    X = df["text"]
    y = df["label"]

    # Vectorization
    vectorizer = TfidfVectorizer(max_features=5000)
    X_vec = vectorizer.fit_transform(X)

    # Model training
    model = LogisticRegression(max_iter=1000)
    model.fit(X_vec, y)

    # Save paths
    os.makedirs("models", exist_ok=True)

    joblib.dump(model, "models/model.pkl")
    joblib.dump(vectorizer, "models/vectorizer.pkl")

    print("Retraining completed & model saved!")

    return {
        "status": "retrained",
        "samples": len(df)
    }