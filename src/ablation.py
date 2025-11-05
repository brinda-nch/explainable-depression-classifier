# src/ablation.py
import pandas as pd, torch, random, numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
from explain import compute_rss, compute_ala, load_split, sample_dataset

def seed_all(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

VARIANTS = [
    # model, explain_method, seed, topk
    ("DistilBERT", "IG", 42, 10),
    ("DistilBERT", "Attention", 42, 10),
]

def evaluate_on_split(model, tok, split_name, device):
    """Evaluate model on a dataset split."""
    ds = load_split(split_name)
    preds, labels = [], []
    
    for example in ds:
        text = example["text"]
        label = int(example["label"])
        
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        
        with torch.no_grad():
            logits = model(**enc).logits
            prob = torch.softmax(logits, dim=-1)[0, 1].item()
            pred = int(prob >= 0.5)
        
        preds.append(pred)
        labels.append(label)
    
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        labels, preds, average="macro", zero_division=0
    )
    
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1}


def run_ablation():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    
    for model_name, method, seed, topk in VARIANTS:
        seed_all(seed)
        MODEL_ID = "distilbert-base-uncased"
        ckpt = "results/runs/distilbert" if "Distil" in model_name else "results/runs/roberta"
        print(f"[RUN] {model_name} + {method}")

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        model = AutoModelForSequenceClassification.from_pretrained(
            ckpt,
            attn_implementation="eager"
        )
        model.to(device)
        model.eval()

        # 1️⃣ Evaluate on validation and test sets
        print(f"  → Evaluating on validation set...")
        val_metrics = evaluate_on_split(model, tok, "val", device)
        print(f"  → Evaluating on test set...")
        test_metrics = evaluate_on_split(model, tok, "test", device)
        val_f1, test_f1 = val_metrics["f1_macro"], test_metrics["f1_macro"]

        # 2️⃣ Explainability metrics on sample of validation set
        print(f"  → Computing explainability metrics (RSS, ALA)...")
        val_ds = load_split("val")
        sample = sample_dataset(val_ds, n=50, seed=seed)  # Use 50 samples for speed
        
        rss_scores = []
        ala_scores = []
        
        for i, example in enumerate(sample):
            text = example["text"]
            if i % 10 == 0:
                print(f"    Processing {i+1}/50...")
            
            try:
                rss = compute_rss(model, tok, text, k=topk, runs=3, device=device)
                ala = compute_ala(model, tok, text, device=device)
                rss_scores.append(rss)
                ala_scores.append(ala)
            except Exception as e:
                print(f"    Warning: Failed on example {i}: {e}")
                continue
        
        avg_rss = float(np.mean(rss_scores)) if rss_scores else 0.0
        avg_ala = float(np.mean(ala_scores)) if ala_scores else 0.0

        results.append({
            "Variant": f"{model_name}-{method}",
            "Model": model_name,
            "ExplainMethod": method,
            "Seed": seed,
            "TopK": topk,
            "ValF1": val_f1,
            "TestF1": test_f1,
            "RSS_avg": avg_rss,
            "ALA_avg": avg_ala,
        })
        
        print(f"  → Val F1: {val_f1:.3f}, Test F1: {test_f1:.3f}, RSS: {avg_rss:.3f}, ALA: {avg_ala:.3f}")

    df = pd.DataFrame(results)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    df.to_csv("results/tables/ablation_matrix.csv", index=False)
    print("\n" + "="*60)
    print(df.to_string(index=False))
    print("="*60)
    print("[OK] Ablation matrix saved → results/tables/ablation_matrix.csv")

if __name__=="__main__":
    run_ablation()
