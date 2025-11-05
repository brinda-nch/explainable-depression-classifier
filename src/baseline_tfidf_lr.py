from pathlib import Path
import json
import pandas as pd
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, precision_recall_curve
)
import matplotlib.pyplot as plt

def load_split(name, data_dir="data/processed/user_level_ds_hf"):
    """Load a split from HuggingFace DatasetDict or JSONL format."""
    data_path = Path(data_dir)
    
    # Try HuggingFace DatasetDict format first
    if (data_path / "dataset_dict.json").exists():
        ds = load_from_disk(str(data_path))
        split_name = name if name in ds else ("validation" if name == "val" else name)
        if split_name not in ds:
            raise ValueError(f"Split '{name}' not found in dataset at {data_path}")
        return pd.DataFrame(ds[split_name].to_dict())
    
    # Fallback to JSONL format
    path = data_path / f"{name}.jsonl"
    if not path.exists():
        path = Path(f"data/processed/{name}.jsonl")
    return pd.read_json(path, lines=True)

def train_tfidf_lr():
    train = load_split("train")
    val = load_split("val")

    tfidf = TfidfVectorizer(
        ngram_range=(1,2),
        max_features=50000,
        min_df=5
    )
    X_train = tfidf.fit_transform(train["text"])
    X_val = tfidf.transform(val["text"])

    clf = LogisticRegression(
        class_weight="balanced",
        max_iter=200,
        solver="liblinear"
    )
    clf.fit(X_train, train["label"])
    preds = clf.predict(X_val)
    probs = clf.predict_proba(X_val)[:,1]

    acc = accuracy_score(val["label"], preds)
    prec, rec, f1, _ = precision_recall_fscore_support(val["label"], preds, average="macro")
    auroc = roc_auc_score(val["label"], probs)
    auprc = average_precision_score(val["label"], probs)

    Path("results/tables").mkdir(parents=True, exist_ok=True)
    metrics = {
        "model": "TFIDF+LR",
        "accuracy": acc, "precision": prec, "recall": rec,
        "f1_macro": f1, "auroc": auroc, "auprc": auprc
    }
    pd.DataFrame([metrics]).to_csv("results/tables/baselines.csv", index=False)
    print(metrics)

    # Confusion matrix + PR curve
    Path("results/figures").mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(val["label"], preds)
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix - TFIDF+LR")
    plt.savefig("results/figures/cm_tfidf_lr.png")

    p, r, _ = precision_recall_curve(val["label"], probs)
    plt.figure()
    plt.plot(r, p)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve - TFIDF+LR")
    plt.savefig("results/figures/pr_tfidf_lr.png")

if __name__ == "__main__":
    train_tfidf_lr()
