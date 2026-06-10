import os
import logging
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import mlflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.encodings = tokenizer(texts, truncation=True, padding=True, max_length=max_length)
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def main():
    # Setup MLflow Tracking
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Sentiment_Analysis_Project")

    # Paths to your processed splits
    train_path = "data/processed/train.csv"
    test_path = "data/processed/test.csv"
    
    if not (os.path.exists(train_path) and os.path.exists(test_path)):
        logger.error("Processed data files not found in data/processed/. Please check paths.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Subsampling for a swift CPU-profiled validation run
    train_df = train_df.sample(n=1000, random_state=42) if len(train_df) > 1000 else train_df
    test_df = test_df.sample(n=200, random_state=42) if len(test_df) > 200 else test_df

    logger.info(f"Loaded {len(train_df)} training and {len(test_df)} testing samples.")

    model_name = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    train_dataset = SentimentDataset(train_df["cleaned_text"].astype(str).tolist(), train_df["sentiment"].tolist(), tokenizer)
    test_dataset = SentimentDataset(test_df["cleaned_text"].astype(str).tolist(), test_df["sentiment"].tolist(), tokenizer)

    output_dir = "models/distilbert"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        warmup_steps=100,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        use_cpu=True  # Explicitly forces CPU execution to prevent hardware mismatch crashes
    )

    with mlflow.start_run(run_name="train_distilbert_baseline"):
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
        )

        logger.info("⚡ Starting DistilBERT training loop...")
        trainer.train()
        
        # Save structural artifacts locally
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        logger.info(f"📦 Uncompressed FP32 baseline saved successfully to {output_dir}")

if __name__ == "__main__":
    main()