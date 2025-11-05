# src/explain.py
from pathlib import Path
import os, json, random
import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from datasets import load_from_disk, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from captum.attr import IntegratedGradients
import shap
from scipy.stats import spearmanr

from lexicon import is_lexicon_token


# -----------------------------
# Helpers: load splits (same logic you used elsewhere)
# -----------------------------
def load_split(name, data_dir="data/processed/user_level_ds_hf"):
    p = Path(data_dir)
    if (p / "dataset_dict.json").exists():
        ds = load_from_disk(str(p))
        split = name if name in ds else ("validation" if name=="val" else name)
        return ds[split]
    # fallback to JSONL
    import pandas as pd
    path = Path(f"data/processed/{name}.jsonl")
    df = pd.read_json(path, lines=True)
    return Dataset.from_pandas(df)


# -----------------------------
# Attention extraction & heatmap saving
# -----------------------------
def get_last_layer_attentions(model, tokenizer, text: str, device="cpu") -> Tuple[np.ndarray, List[str]]:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        out = model(**enc, output_attentions=True)
    # last layer attentions: (num_heads, seq_len, seq_len)
    att = out.attentions[-1].squeeze(0).detach().cpu().numpy()
    # average across heads
    att_avg = att.mean(axis=0)  # (seq_len, seq_len)
    tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return att_avg, tokens


def save_attention_heatmaps(att_avg: np.ndarray, tokens: List[str], out_path: Path, title: str):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 6))
    plt.imshow(att_avg, aspect="auto")
    plt.title(title)
    plt.xlabel("Key tokens")
    plt.ylabel("Query tokens")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Integrated Gradients (Captum)
# -----------------------------
class WrappedForIG(torch.nn.Module):
    """Wrap model forward to return class logit for IG."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        # logit for class 1
        return out.logits[:, 1]


def integrated_gradients_scores(model, tokenizer, text: str, steps=32, device="cpu") -> Tuple[np.ndarray, List[str]]:
    model.eval()
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(device) for k, v in enc.items()}
    wrapper = WrappedForIG(model)

    # Baseline: pad token ids (all [PAD]) of same shape
    baseline_ids = torch.full_like(enc["input_ids"], tokenizer.pad_token_id)
    baseline_mask = torch.zeros_like(enc["attention_mask"])

    ig = IntegratedGradients(wrapper)
    attributions = ig.attribute(
        inputs=(enc["input_ids"], enc["attention_mask"]),
        baselines=(baseline_ids, baseline_mask),
        n_steps=steps,
    )
    # token-level scalar score = |attr| on input_ids (sum over embedding dims is handled internally)
    # Captum returns tuple aligned with inputs; we take first tensor (for input_ids)
    token_scores = attributions[0].detach().abs().sum(dim=-1).squeeze(0).cpu().numpy()
    toks = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
    return token_scores, toks


# -----------------------------
# SHAP (small batch)
# -----------------------------
def shap_scores(model, tokenizer, texts: List[str], device="cpu") -> List[Tuple[List[str], np.ndarray]]:
    """
    Returns list of (tokens, token_scores) per text.
    Uses the transformers pipeline-compatible explainer (kernel SHAP under the hood).
    """
    model.eval()
    from transformers import pipeline
    clf = pipeline("text-classification", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)
    explainer = shap.Explainer(clf)

    results = []
    # SHAP expects raw strings; we map words back from tokenizer tokenizer.tokenize
    shap_values = explainer(texts)  # may take a bit; keep list small (like 20–50)
    for i, sv in enumerate(shap_values.values):
        # sv has shape (num_tokens, num_classes); use class 1
        # But SHAP tokenization might differ; we read tokens from sv.data
        tokens = shap_values.data[i]
        # ensure ndarray
        token_scores = np.array(sv)[:, 1] if sv.ndim == 2 else np.array(sv)
        results.append((tokens, token_scores))
    return results


# -----------------------------
# Top-k helpers & RSS
# -----------------------------
def topk_indices(scores: np.ndarray, k: int) -> set:
    k = min(k, len(scores))
    return set(np.argsort(scores)[-k:])

def jaccard(a: set, b: set) -> float:
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def compute_rss(model, tokenizer, text: str, k=10, runs=5, device="cpu") -> float:
    """Run multiple stochastic passes (enable dropout) and compute Jaccard@k agreement."""
    # enable dropout during eval by turning train() but keep no-grad
    model.train()
    all_sets = []
    with torch.no_grad():
        for r in range(runs):
            torch.manual_seed(42 + r)
            scores, toks = integrated_gradients_scores(model, tokenizer, text, steps=16, device=device)
            all_sets.append(topk_indices(scores, k))
    # pairwise Jaccard
    if len(all_sets) < 2: return 1.0
    pairs = []
    for i in range(len(all_sets)):
        for j in range(i+1, len(all_sets)):
            pairs.append(jaccard(all_sets[i], all_sets[j]))
    return float(np.mean(pairs))


# -----------------------------
# ALA (Attention–Lexicon Alignment)
# -----------------------------
def compute_ala(model, tokenizer, text: str, device="cpu") -> float:
    att, toks = get_last_layer_attentions(model, tokenizer, text, device=device)
    # CLS-to-token attention (row index of CLS attending to others)
    # DistilBERT uses [CLS] at position 0
    cls_row = att[0]  # shape (seq_len,)
    # build mask
    mask = np.array([1 if is_lexicon_token(t.lower().strip("##")) else 0 for t in toks], dtype=float)
    # Spearman correlation between attention weights and mask
    # Keep lengths aligned; ignore special <pad> tails (attention often 0)
    try:
        rho, _ = spearmanr(cls_row, mask)
        return float(rho) if not np.isnan(rho) else 0.0
    except Exception:
        return 0.0


# -----------------------------
# Utility: pick random test subset & save artifact bundle
# -----------------------------
def sample_dataset(ds: Dataset, n: int, seed=0) -> Dataset:
    n = min(n, len(ds))
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    return ds.select(idx[:n])


def save_qualitative_bundle(
    model, tokenizer, ds: Dataset, out_dir="results/explain", k_top=10, device="cpu"
):
    outd = Path(out_dir); outd.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, ex in enumerate(ds):
        text = ex["text"]
        gold = int(ex["label"])
        # pred & prob
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            prob1 = torch.softmax(logits, dim=-1)[0,1].item()
            pred = int(prob1 >= 0.5)

        # Attention heatmap
        att_avg, toks_att = get_last_layer_attentions(model, tokenizer, text, device=device)
        heatmap_path = Path(outd, f"attn_{i:03d}.png")
        save_attention_heatmaps(att_avg, toks_att, heatmap_path, title=f"Attention #{i}")

        # IG top-k
        ig_scores, toks_ig = integrated_gradients_scores(model, tokenizer, text, steps=16, device=device)
        ig_top_idx = np.argsort(ig_scores)[-min(k_top, len(ig_scores)):]
        ig_top = [(int(idx), toks_ig[idx], float(ig_scores[idx])) for idx in ig_top_idx]

        # RSS
        rss = compute_rss(model, tokenizer, text, k=k_top, runs=5, device=device)

        # ALA
        ala = compute_ala(model, tokenizer, text, device=device)

        rows.append({
            "id": i,
            "text": text,
            "gold": gold,
            "pred": pred,
            "prob_depressed": prob1,
            "ig_topk": ig_top,
            "rss": rss,
            "ala": ala,
            "attn_heatmap_path": str(heatmap_path),
        })

    with open(Path(outd, "explain_samples.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    # summary tables
    import pandas as pd
    pd.DataFrame(rows)[["id","gold","pred","prob_depressed","rss","ala"]].to_csv(
        Path(outd, "explain_summary.csv"), index=False
    )
    print(f"[OK] Saved qualitative bundle to {outd}")
