# src/data.py
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict, load_dataset


# ---------- helpers ----------
def _map_to_schema(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # user id
    user_candidates = ["author", "user_id", "username", "user", "creator", "uid"]
    user_col = next((c for c in user_candidates if c in df.columns), None)
    if user_col:
        df = df.rename(columns={user_col: "user_id"})
    else:
        # fallback: synthesize stable ids if dataset lacks authors
        df["user_id"] = pd.Series(range(len(df)), dtype=str)

    # text
    text_candidates = ["body", "text", "selftext", "content", "post", "title"]
    text_col = next((c for c in text_candidates if c in df.columns), None)
    if not text_col:
        raise ValueError("No text column found (tried body/text/selftext/content/post/title).")
    df = df.rename(columns={text_col: "text"})

    # label -> int {0,1}
    if "label" not in df.columns:
        for cand in ["target", "y", "depression_label", "is_depressed", "class"]:
            if cand in df.columns:
                df = df.rename(columns={cand: "label"})
                break
    if "label" not in df.columns:
        raise ValueError("No label column found (expected label/target/y/depression_label/is_depressed/class).")

    mapping = {
        True: 1, False: 0,
        "depressed": 1, "control": 0,
        "pos": 1, "neg": 0,
        "1": 1, "0": 0
    }
    df["label"] = df["label"].map(lambda v: mapping.get(v, v))
    df["label"] = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int)

    # subreddit (optional)
    if "subreddit" not in df.columns:
        df["subreddit"] = "unknown"

    # keep only what we need
    return df[["user_id", "text", "label", "subreddit"]]


def aggregate_user_level(df: pd.DataFrame) -> pd.DataFrame:
    """Concat all posts per user into one document; take majority subreddit."""
    agg = (
        df.groupby("user_id")
          .agg({
              "text": lambda s: "\n\n".join(map(str, s)),
              "label": "max",  # if any post is positive => user positive
              "subreddit": lambda s: s.mode().iat[0] if not s.mode().empty else "unknown",
          })
          .reset_index()
    )
    return agg.rename(columns={"subreddit": "dominant_subreddit"})[
        ["user_id", "text", "label", "dominant_subreddit"]
    ]


def make_user_splits(user_df: pd.DataFrame, seed: int = 42,
                     val_size: float = 0.15, test_size: float = 0.15):
    users = user_df["user_id"].unique()
    train_users, temp_users = train_test_split(
        users, test_size=val_size + test_size, random_state=seed, shuffle=True
    )
    rel_val = val_size / (val_size + test_size)
    val_users, test_users = train_test_split(
        temp_users, test_size=1 - rel_val, random_state=seed, shuffle=True
    )
    train_df = user_df[user_df["user_id"].isin(train_users)].reset_index(drop=True)
    val_df   = user_df[user_df["user_id"].isin(val_users)].reset_index(drop=True)
    test_df  = user_df[user_df["user_id"].isin(test_users)].reset_index(drop=True)
    return train_df, val_df, test_df


def to_hf_datasetdict(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame) -> DatasetDict:
    cols = ["text", "label", "user_id", "dominant_subreddit"]
    return DatasetDict({
        "train": Dataset.from_pandas(train_df[cols], preserve_index=False),
        "validation": Dataset.from_pandas(val_df[cols], preserve_index=False),
        "test": Dataset.from_pandas(test_df[cols], preserve_index=False),
    })


def save_datasetdict(ds: DatasetDict, out_dir: str):
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(p))


# ---------- main entry ----------
def load_hf_reddit_depression_cleaned(out_dir: str = "data/processed/user_level_ds_hf", seed: int = 42):
    """
    Loads 'hugginglearners/reddit-depression-cleaned', maps columns, aggregates to user level,
    creates user-level train/val/test splits, and saves a HF DatasetDict to out_dir.
    """
    print("[INFO] Loading HF dataset: hugginglearners/reddit-depression-cleaned ...")
    raw = load_dataset("hugginglearners/reddit-depression-cleaned")
    df = raw["train"].to_pandas()

    df = _map_to_schema(df)

    # save raw for reproducibility
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/raw/posts.csv", index=False)
    print(f"[INFO] Saved raw CSV: data/raw/posts.csv ({len(df)} rows)")

    user_df = aggregate_user_level(df)
    train_df, val_df, test_df = make_user_splits(user_df, seed=seed)

    ds = to_hf_datasetdict(train_df, val_df, test_df)
    save_datasetdict(ds, out_dir)

    meta = {
        "source": "hugginglearners/reddit-depression-cleaned",
        "total_users": int(len(user_df)),
        "splits": {"train": int(len(train_df)), "val": int(len(val_df)), "test": int(len(test_df))},
    }
    Path(out_dir, "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[INFO] Saved processed dataset to {out_dir}")
    return ds, meta


if __name__ == "__main__":
    load_hf_reddit_depression_cleaned()
