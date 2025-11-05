#!/usr/bin/env python3
"""Evaluate trained models on the test set."""

import pandas as pd
import numpy as np
from pathlib import Path
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns

def load_split(name, data_dir="data/processed/user_level_ds_hf"):
    """Load a split from HuggingFace DatasetDict."""
    data_path = Path(data_dir)
    ds = load_from_disk(str(data_path))
    split_name = name if name in ds else ("validation" if name == "val" else name)
    if split_name not in ds:
        raise ValueError(f"Split '{name}' not found in dataset at {data_path}")
    return pd.DataFrame(ds[split_name].to_dict())

def evaluate_tfidf_lr():
    """Train TF-IDF+LR on train+val and evaluate on test."""
    print("\n" + "="*60)
    print("Evaluating TF-IDF + Logistic Regression on Test Set")
    print("="*60)
    
    # Load data
    train = load_split("train")
    val = load_split("val")
    test = load_split("test")
    
    # Combine train + val for final training
    train_val = pd.concat([train, val], ignore_index=True)
    print(f"Training on {len(train_val)} samples (train+val)")
    print(f"Testing on {len(test)} samples")
    
    # Train TF-IDF
    print("\nTraining TF-IDF vectorizer...")
    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=50000,
        min_df=5
    )
    X_train_val = tfidf.fit_transform(train_val["text"])
    X_test = tfidf.transform(test["text"])
    
    # Train classifier
    print("Training Logistic Regression...")
    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        solver="liblinear"
    )
    clf.fit(X_train_val, train_val["label"])
    
    # Predict
    print("Evaluating on test set...")
    preds = clf.predict(X_test)
    probs = clf.predict_proba(X_test)[:,1]
    
    # Metrics
    acc = accuracy_score(test["label"], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(test["label"], preds, average="macro")
    auroc = roc_auc_score(test["label"], probs)
    auprc = average_precision_score(test["label"], probs)
    
    results = {
        "model": "TFIDF+LR",
        "split": "test",
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_macro": f1,
        "auroc": auroc,
        "auprc": auprc
    }
    
    print(f"\nTest Results:")
    print(f"  Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  AUPRC:     {auprc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(test["label"], preds)
    print(f"\nConfusion Matrix:")
    print(cm)
    
    return results, test["label"], preds, probs

def evaluate_distilbert():
    """Evaluate DistilBERT on test set."""
    print("\n" + "="*60)
    print("Evaluating DistilBERT on Test Set")
    print("="*60)
    
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
    import torch
    
    # Load model
    model_path = Path("results/runs/distilbert")
    if not model_path.exists():
        print("DistilBERT model not found. Skipping evaluation.")
        return None, None, None, None
    
    # Load dataset
    print("Loading test set...")
    ds = load_from_disk("data/processed/user_level_ds_hf")
    ds_test = ds["test"]
    
    # Tokenize
    print("Tokenizing...")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    
    ds_test = ds_test.map(tokenize, batched=True)
    
    # Load model
    print("Loading trained model...")
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    
    # Note: We'll need to retrain or the model needs to be saved during training
    # For now, let's document that the model should be saved
    print("\nNote: To properly evaluate DistilBERT on test, the model should be saved during training.")
    print("The current validation results are being reported as a proxy.")
    
    return None, None, None, None

def main():
    """Run test evaluation for all models."""
    
    # Create output directory
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    # Evaluate TF-IDF + LR
    tfidf_results, test_labels, tfidf_preds, tfidf_probs = evaluate_tfidf_lr()
    all_results.append(tfidf_results)
    
    # Save confusion matrix for TF-IDF
    cm = confusion_matrix(test_labels, tfidf_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Control', 'Depression'],
                yticklabels=['Control', 'Depression'])
    plt.title('TF-IDF+LR - Test Set Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig("results/figures/test_cm_tfidf_lr.png", dpi=160)
    plt.close()
    print("\nSaved confusion matrix to results/figures/test_cm_tfidf_lr.png")
    
    # Classification report
    print("\nDetailed Classification Report:")
    print(classification_report(test_labels, tfidf_preds, 
                                target_names=['Control', 'Depression']))
    
    # Evaluate DistilBERT (if available)
    # distilbert_results, _, _, _ = evaluate_distilbert()
    # if distilbert_results:
    #     all_results.append(distilbert_results)
    
    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv("results/tables/test_results.csv", index=False)
    print(f"\n✅ Test results saved to results/tables/test_results.csv")
    
    print("\n" + "="*60)
    print("Test Evaluation Complete!")
    print("="*60)

if __name__ == "__main__":
    main()

