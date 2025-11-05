# src/train.py
import json
from pathlib import Path
from typing import Dict
from datasets import load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import joblib

from repro import seed_everything, start_run_dir

def load_processed_dataset(path: str):
    ds = load_from_disk(path)
    return ds

def train_tfidf_lr(ds, run_dir: Path, config: Dict):
    # 1) Build features
    X_train, y_train = ds["train"]["text"], ds["train"]["label"]
    X_val,   y_val   = ds["validation"]["text"], ds["validation"]["label"]
    X_test,  y_test  = ds["test"]["text"], ds["test"]["label"]

    vectorizer = TfidfVectorizer(
        max_features=config.get("max_features", 50000),
        ngram_range=config.get("ngram_range", (1,2)),
        min_df=config.get("min_df", 2),
    )
    Xtr = vectorizer.fit_transform(X_train)
    Xv  = vectorizer.transform(X_val)
    Xte = vectorizer.transform(X_test)

    # 2) Train Logistic Regression
    clf = LogisticRegression(
        max_iter=config.get("max_iter", 2000),
        C=config.get("C", 1.0),
        class_weight=config.get("class_weight", "balanced"),
        n_jobs=-1,
    )
    clf.fit(Xtr, y_train)

    # 3) Evaluate
    def eval_and_dump(split_name, X, y):
        pred = clf.predict(X)
        acc = accuracy_score(y, pred)
        f1  = f1_score(y, pred)
        cm  = confusion_matrix(y, pred).tolist()
        report = classification_report(y, pred, output_dict=True)
        (run_dir / f"{split_name}_report.json").write_text(json.dumps(report, indent=2))
        return {"acc": acc, "f1": f1, "confusion_matrix": cm}

    metrics = {
        "val":  eval_and_dump("val", Xv, y_val),
        "test": eval_and_dump("test", Xte, y_test),
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 4) Save artifacts
    joblib.dump(vectorizer, run_dir / "tfidf.joblib")
    joblib.dump(clf, run_dir / "logreg.joblib")
    print(f"[INFO] Saved artifacts to {run_dir}")

def main():
    # Minimal config
    config = {
        "seed": 42,
        "dataset_path": "data/processed/user_level_ds_hf",  # or user_level_ds
        "max_features": 50000,
        "ngram_range": (1,2),
        "min_df": 2,
        "max_iter": 2000,
        "C": 1.0,
        "class_weight": "balanced",
    }

    seed_everything(config["seed"], deterministic=True)
    run_dir = start_run_dir(base="results/runs", config=config)
    print(f"[INFO] Run dir: {run_dir}")

    ds = load_processed_dataset(config["dataset_path"])
    train_tfidf_lr(ds, run_dir, config)

if __name__ == "__main__":
    main()
