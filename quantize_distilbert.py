import os
import time
import torch
import pandas as pd
from pathlib import Path
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Enforce single-thread execution for deterministic CPU profiling
torch.set_num_threads(1)

def get_size_mb(path):
    if os.path.isdir(path):
        return sum(f.stat().st_size for f in Path(path).glob('**/*') if f.is_file()) / (1024 * 1024)
    return os.path.getsize(path) / (1024 * 1024)

def main():
    print("⚡ Starting Week 3 DistilBERT Quantization & Benchmark Pipeline...")

    test_path = "data/processed/test.csv"
    if not os.path.exists(test_path):
        print(f"❌ Error: Test data split missing at {test_path}")
        return

    test_df = pd.read_csv(test_path)
    # Evaluate over 500 samples for clean, swift execution metrics on CPU
    texts = test_df["cleaned_text"].astype(str).tolist()[:500]
    labels = test_df["sentiment"].tolist()[:500]

    model_dir = "models/distilbert"
    if not os.path.exists(model_dir):
        print(f"⚠️ Local model directory not found at {model_dir}. Falling back to vanilla checkpoint.")
        model_dir = "distilbert-base-uncased"

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    baseline_model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    baseline_model.eval()

    baseline_size = get_size_mb(model_dir)

    # 1. Benchmark Baseline (Unoptimized, Sequential Inference)
    print("\n--- Benchmarking Baseline Model (FP32, Sequential) ---")
    base_preds = []
    start_time = time.perf_counter()
    
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = baseline_model(**inputs)
            pred = torch.argmax(outputs.logits, dim=1).item()
            base_preds.append(pred)
            
    base_time = time.perf_counter() - start_time
    base_latency = (base_time / len(texts)) * 1000
    base_acc = accuracy_score(labels, base_preds)

    # 2. Apply Dynamic Quantization (Level 2: FP32 -> INT8)
    print("\n--- Applying Dynamic Quantization (Level 2 INT8) ---")
    quantized_model = torch.quantization.quantize_dynamic(
        baseline_model,
        {torch.nn.Linear},  # Targets structural linear transformation blocks
        dtype=torch.qint8
    )

    # CRITICAL: Save the entire model object to bypass state_dict mapping failures
    quant_model_path = "models/distilbert_quantized.pt"
    os.makedirs("models", exist_ok=True)
    torch.save(quantized_model, quant_model_path)
    quant_size = get_size_mb(quant_model_path)

    # 3. Benchmark Quantized Model (Optimized Level 1 + Level 2: INT8 + Batching)
    print("\n--- Benchmarking Quantized Model (INT8 + Batched Inference) ---")
    quant_preds = []
    batch_size = 32
    start_time = time.perf_counter()

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            inputs = tokenizer(batch_texts, return_tensors="pt", truncation=True, padding=True, max_length=128)
            outputs = quantized_model(**inputs)
            preds = torch.argmax(outputs.logits, dim=1).tolist()
            quant_preds.extend(preds)

    quant_time = time.perf_counter() - start_time
    quant_latency = (quant_time / len(texts)) * 1000
    quant_acc = accuracy_score(labels, quant_preds)

    # 4. Generate Final Week 3 Performance Analysis Matrix
    print("\n" + "="*80)
    print("                  WEEK 3 DISTILBERT PERFORMANCE ANALYSIS MATRIX")
    print("="*80)
    summary_data = {
        "Optimization State": ["Sequential Baseline (FP32)", "Dynamic Quantized + Batch (INT8)"],
        "Model Size (MB)": [f"{baseline_size:.2f} MB", f"{quant_size:.2f} MB"],
        "Avg Latency / Sample": [f"{base_latency:.2f} ms", f"{quant_latency:.2f} ms"],
        "Accuracy": [f"{base_acc:.4f}", f"{quant_acc:.4f}"],
        "Size Reduction": ["0% (Baseline)", f"-{((baseline_size - quant_size)/baseline_size)*100:.1f}%"],
        "Speedup Factor": ["1.0x", f"{base_time / quant_time:.2f}x faster"]
    }
    summary_df = pd.DataFrame(summary_data)
    print(summary_df.to_string(index=False))
    print("="*80)

if __name__ == "__main__":
    main()