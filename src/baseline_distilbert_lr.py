# src/distilbert_baseline.py
from pathlib import Path
import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from datasets import Dataset, load_from_disk
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve, roc_curve
)

from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)

MODEL_ID = "distilbert-base-uncased"
CKPT_DIR = "results/runs/distilbert"
FIG_DIR = Path("results/figures")
TABLES_PATH = Path("results/tables/baselines.csv")


def load_split(name, data_dir="data/processed/user_level_ds_hf"):
    """Load a split from HuggingFace DatasetDict or JSONL format."""
    data_path = Path(data_dir)
    # Try HF on-disk DatasetDict
    if (data_path / "dataset_dict.json").exists():
        ds = load_from_disk(str(data_path))
        split_name = name if name in ds else ("validation" if name == "val" else name)
        if split_name not in ds:
            raise ValueError(f"Split '{name}' not found in dataset at {data_path}")
        df = pd.DataFrame(ds[split_name].to_dict())
        return df

    # Fallback to JSONL files written by your pipeline
    path = Path(f"data/processed/{name}.jsonl")
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}")
    return pd.read_json(path, lines=True)


def _safe_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except Exception:
        return float("nan")


def _safe_auprc(y_true, y_prob):
    try:
        return float(average_precision_score(y_true, y_prob))
    except Exception:
        return float("nan")


def make_hf_dataset(df: pd.DataFrame) -> Dataset:
    # Ensure correct dtypes
    df = df.copy()
    df["text"] = df["text"].astype(str)
    df["label"] = df["label"].astype(int)
    return Dataset.from_pandas(df, preserve_index=False)


def tokenize(ds: Dataset, tok, max_length=256) -> Dataset:
    return ds.map(
        lambda b: tok(
            b["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in ds.column_names if c not in ("text", "label")]
    )


def compute_metrics_fn(eval_pred):
    logits, labels = eval_pred
    # softmax to probs
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exps / exps.sum(axis=-1, keepdims=True)
    y_prob = probs[:, 1]
    y_pred = probs.argmax(axis=1)
    y_true = labels.astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    auroc = _safe_auc(y_true, y_prob)
    auprc = _safe_auprc(y_true, y_prob)

    return {
        "eval_accuracy": acc,
        "eval_precision": prec,
        "eval_recall": rec,
        "eval_f1_macro": f1,
        "eval_auroc": auroc,
        "eval_auprc": auprc,
    }


def plot_and_save(y_true, y_prob, prefix: str):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    y_pred = (y_prob >= 0.5).astype(int)

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion Matrix - {prefix}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"cm_{prefix}.png")
    plt.close()

    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("FPR")
        plt.ylabel("TPR")
        plt.title(f"ROC - {prefix}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"roc_{prefix}.png")
        plt.close()
    except Exception:
        pass

    # PR
    try:
        p, r, _ = precision_recall_curve(y_true, y_prob)
        plt.figure()
        plt.plot(r, p)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"PR - {prefix}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"pr_{prefix}.png")
        plt.close()
    except Exception:
        pass


def append_results_row(row: dict):
    TABLES_PATH.parent.mkdir(parents=True, exist_ok=True)
    try:
        old = pd.read_csv(TABLES_PATH)
        pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(TABLES_PATH, index=False)
    except Exception:
        pd.DataFrame([row]).to_csv(TABLES_PATH, index=False)


def main():
    # Load splits
    train_df = load_split("train")
    val_df = load_split("val")

    train_ds = make_hf_dataset(train_df)
    val_ds = make_hf_dataset(val_df)

    # Model & tokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID, num_labels=2)

    # Tokenize
    train_tok = tokenize(train_ds, tok, max_length=256)
    val_tok = tokenize(val_ds, tok, max_length=256)

    # Data collator
    collator = DataCollatorWithPadding(tok)

    # Training args (optimized for GPU)
    import torch
    use_gpu = torch.cuda.is_available()
    
    args = TrainingArguments(
        output_dir=CKPT_DIR,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=32 if use_gpu else 16,  # Larger batch on GPU
        per_device_eval_batch_size=64 if use_gpu else 32,   # Larger batch on GPU
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="no",
        load_best_model_at_end=False,
        logging_steps=50,
        report_to=[],
        fp16=use_gpu,  # Use fp16 on GPU for speed
        bf16=False,
    )
    
    print(f"[INFO] Using device: {'GPU' if use_gpu else 'CPU'}")
    print(f"[INFO] Batch sizes - Train: {args.per_device_train_batch_size}, Eval: {args.per_device_eval_batch_size}")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_tok,
        eval_dataset=val_tok,
        tokenizer=tok,
        data_collator=collator,
        compute_metrics=compute_metrics_fn,
    )

    trainer.train()

    # Evaluate on val
    metrics = trainer.evaluate()
    # Get per-example probs for plots
    pred = trainer.predict(val_tok)
    logits = pred.predictions
    exps = np.exp(logits - logits.max(axis=-1, keepdims=True))
    probs = exps / exps.sum(axis=-1, keepdims=True)
    y_prob = probs[:, 1]
    y_true = pred.label_ids.astype(int)

    plot_and_save(y_true, y_prob, prefix="distilbert_val")

    row = {
        "model": "DistilBERT(val)",
        "accuracy": metrics.get("eval_accuracy"),
        "precision": metrics.get("eval_precision"),
        "recall": metrics.get("eval_recall"),
        "f1_macro": metrics.get("eval_f1_macro"),
        "auroc": metrics.get("eval_auroc"),
        "auprc": metrics.get("eval_auprc"),
    }
    append_results_row(row)

    # Save metrics JSON beside checkpoint for convenience
    Path(CKPT_DIR).mkdir(parents=True, exist_ok=True)
    with open(Path(CKPT_DIR, "metrics_val.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print("[OK] DistilBERT baseline trained and evaluated.")
    print(f"[OK] Results row appended to {TABLES_PATH}")
    print(f"[OK] Figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
