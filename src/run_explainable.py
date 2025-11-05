# src/run_explainable.py
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from explain import (
    load_split, sample_dataset, save_qualitative_bundle,
)

CKPT_DIR = "results/runs/distilbert"  # <- change if your training saved elsewhere
OUT_DIR = "results/explain"

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("[INFO] Loading model:", CKPT_DIR)
    MODEL_ID = "distilbert-base-uncased"  # The original model
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT_DIR, 
        attn_implementation="eager"  # Suppress attention implementation warning
    )
    model.to(device)

    print("[INFO] Loading test split…")
    test_ds = load_split("test")

    print("[INFO] Sampling 100 examples for qualitative bundle…")
    sub = sample_dataset(test_ds, n=100, seed=7)

    print("[INFO] Computing explanations (attention, IG), RSS & ALA; saving artifacts…")
    save_qualitative_bundle(model, tok, sub, out_dir=OUT_DIR, k_top=10, device=device)

    print("[DONE] Gate C targets produced in results/explain/")

if __name__ == "__main__":
    main()
