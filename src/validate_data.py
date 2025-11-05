# src/validate_data.py
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter, defaultdict
import argparse
import matplotlib.pyplot as plt
from datasets import load_from_disk

PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    # Phone: require separators (dash/space/parens) to avoid matching arbitrary digit sequences
    "phone": re.compile(r"\b(?:\+?\d{1,3}[\s.-])?(?:\(?\d{3}\)[\s.-]|\d{3}[\s.-])\d{3}[\s.-]\d{4}\b"),
    "url": re.compile(r"https?://\S+"),
    "mention": re.compile(r"@\w+"),
}

def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read_split(dirpath: Path, name: str):
    # Try HuggingFace DatasetDict format first
    if (dirpath / "dataset_dict.json").exists():
        ds = load_from_disk(str(dirpath))
        split_name = name if name in ds else ("validation" if name == "val" else name)
        if split_name not in ds:
            raise ValueError(f"Split '{name}' not found in dataset at {dirpath}")
        return [dict(ex) for ex in ds[split_name]]
    
    # Fallback to JSONL format
    p = dirpath / f"{name}.jsonl"
    assert p.exists(), f"Missing {p}"
    return list(load_jsonl(p))

def no_pii(text: str) -> bool:
    # if you anonymized correctly, raw patterns shouldn't appear
    for k, pat in PII_PATTERNS.items():
        if pat.search(text):
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proc_dir", type=str, default="data/processed/user_level_ds_hf", 
                    help="dir containing train/val/test splits (HF DatasetDict or JSONL)")
    ap.add_argument("--fig_dir", type=str, default="results/figures")
    args = ap.parse_args()

    proc = Path(args.proc_dir)
    figd = Path(args.fig_dir); figd.mkdir(parents=True, exist_ok=True)

    train = read_split(proc, "train")
    val   = read_split(proc, "val")
    test  = read_split(proc, "test")

    # 1) PII check + anonymization markers
    def scan_pii(split, name):
        bad = []
        markers = Counter()
        for ex in split:
            txt = ex["text"]
            if not no_pii(txt):
                bad.append(ex)
            # markers
            markers["[USER]"] += txt.count("[USER]")
            markers["[URL]"]  += txt.count("[URL]")
        print(f"[{name}] PII violations: {len(bad)} | markers: {dict(markers)}")
        return bad, markers

    _ = scan_pii(train, "train")
    _ = scan_pii(val, "val")
    _ = scan_pii(test, "test")

    # 2) Short text check (<5 tokens)
    def count_short(split, name):
        short = sum(1 for ex in split if len(ex["text"].split()) < 5)
        print(f"[{name}] <5-token texts: {short}")
        return short
    _ = count_short(train, "train"); _ = count_short(val, "val"); _ = count_short(test, "test")

    # 3) User leakage check
    sets = { "train": set(ex["user_id"] for ex in train),
             "val":   set(ex["user_id"] for ex in val),
             "test":  set(ex["user_id"] for ex in test) }
    inter_train_val  = sets["train"] & sets["val"]
    inter_train_test = sets["train"] & sets["test"]
    inter_val_test   = sets["val"]   & sets["test"]
    print(f"[LEAKAGE] train∩val={len(inter_train_val)}  train∩test={len(inter_train_test)}  val∩test={len(inter_val_test)}")

    # 4) Class balance
    def label_counts(split):
        c = Counter(ex["label"] for ex in split)
        total = sum(c.values())
        pct = {k: round(100*v/total,1) for k,v in c.items()}
        return c, pct
    for name, split in [("train", train), ("val", val), ("test", test)]:
        c, pct = label_counts(split)
        print(f"[{name}] counts={dict(c)}  pct={pct}")

    # 5) EDA: length histograms + label counts
    def lengths(split):
        return [len(ex["text"].split()) for ex in split]
    for name, split in [("train", train), ("val", val), ("test", test)]:
        # lengths
        L = lengths(split)
        plt.figure()
        plt.hist(L, bins=50)
        plt.title(f"Length histogram ({name})")
        plt.xlabel("tokens"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figd / f"len_hist_{name}.png", dpi=160)
        plt.close()
        # labels
        c, _ = label_counts(split)
        plt.figure()
        plt.bar(list(map(str, c.keys())), list(c.values()))
        plt.title(f"Label counts ({name})")
        plt.xlabel("label"); plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(figd / f"label_counts_{name}.png", dpi=160)
        plt.close()

    print(f"[OK] EDA figures → {figd}")

if __name__ == "__main__":
    main()