# src/fairness.py
from pathlib import Path
import re, json, numpy as np, pandas as pd
from datasets import Dataset, load_from_disk
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import precision_recall_fscore_support
import torch
import matplotlib.pyplot as plt

from lexicon import SADNESS, FIRST_PERSON, NEGATIONS


# ------------------------------------------------------------
#  Helpers
# ------------------------------------------------------------
def load_split(name, data_dir="data/processed/user_level_ds_hf"):
    p = Path(data_dir)
    if (p / "dataset_dict.json").exists():
        ds = load_from_disk(str(p))
        split = name if name in ds else ("validation" if name=="val" else name)
        return ds[split].to_pandas()
    return pd.read_json(f"data/processed/{name}.jsonl", lines=True)


def subgroup_flags(text):
    t = text.lower()
    gendered = any(w in t for w in [" she ", " her ", " hers ", " he ", " him ", " his ", " they ", " them "])
    therapy = any(w in t for w in ["therapy","therapist","counseling","medication","prozac","zoloft","lexapro","sertraline"])
    return gendered, therapy


def eval_subgroup(df, model, tok, device):
    """Return F1/P/R per subgroup."""
    preds, labels, groups = [], [], []
    for _, r in df.iterrows():
        text = r["text"]; label = int(r["label"])
        gendered, therapy = subgroup_flags(text)
        group = (
            "gendered" if gendered else
            "therapy" if therapy else
            "general"
        )
        enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        with torch.no_grad():
            p = torch.softmax(model(**enc).logits, dim=-1)[0,1].item()
        preds.append(int(p>=0.5))
        labels.append(label)
        groups.append(group)

    df_eval = pd.DataFrame({"label":labels,"pred":preds,"group":groups})
    results=[]
    for g,gdf in df_eval.groupby("group"):
        p,r,f1,_=precision_recall_fscore_support(gdf.label,gdf.pred,average="binary",zero_division=0)
        results.append({"group":g,"precision":p,"recall":r,"f1":f1,"n":len(gdf)})
    base = pd.DataFrame(results)
    
    # Calculate delta F1 vs general group if it exists
    general_f1 = base.loc[base.group=="general","f1"]
    if not general_f1.empty:
        base["Δf1_vs_general"] = base.f1 - float(general_f1.iloc[0])
    else:
        base["Δf1_vs_general"] = 0.0
    return base


def mask_lexicon(text):
    pattern = re.compile("|".join(
        [re.escape(w) for w in list(SADNESS|FIRST_PERSON|NEGATIONS)]
    ), flags=re.IGNORECASE)
    return pattern.sub("[MASK]", text)


def probe_spurious(df, model, tok, device):
    """Mask lexicon tokens and compare accuracy drop."""
    def acc(data):
        c=0
        for _,r in data.iterrows():
            enc=tok(r["text"],return_tensors="pt",truncation=True,max_length=256).to(device)
            with torch.no_grad():
                p=torch.softmax(model(**enc).logits,dim=-1)[0,1].item()
            c += int((p>=0.5)==int(r["label"]))
        return c/len(data)
    acc_orig=acc(df)
    df_masked=df.copy(); df_masked["text"]=df_masked["text"].map(mask_lexicon)
    acc_mask=acc(df_masked)
    drop = acc_orig-acc_mask
    Path("results/figures").mkdir(parents=True,exist_ok=True)
    plt.bar(["Original","Masked"],[acc_orig,acc_mask],color=["blue","orange"])
    plt.title(f"Accuracy Drop after Lexicon Masking ({drop:.3f})")
    plt.ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig("results/figures/spurious_mask_drop.png")
    plt.close()
    return {"acc_orig":acc_orig,"acc_mask":acc_mask,"drop":drop}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "distilbert-base-uncased"
    ckpt="results/runs/distilbert"
    model=AutoModelForSequenceClassification.from_pretrained(
        ckpt, 
        attn_implementation="eager"
    ).to(device)
    tok=AutoTokenizer.from_pretrained(MODEL_ID)

    test=load_split("test")
    # 4.1 Subgroup fairness
    fair_df=eval_subgroup(test,model,tok,device)
    Path("results/tables").mkdir(parents=True,exist_ok=True)
    fair_df.to_csv("results/tables/fairness_subgroups.csv",index=False)
    print("[OK] Fairness subgroup metrics saved → results/tables/fairness_subgroups.csv")

    # 4.2 Spurious correlation probe
    sens=probe_spurious(test.sample(n=min(200,len(test)),random_state=42),model,tok,device)
    json.dump(sens,open("results/tables/spurious_sensitivity.json","w"),indent=2)
    print("[OK] Masking sensitivity saved → results/tables/spurious_sensitivity.json")

    # 4.3 (optional) OOD split — use val as proxy
    val=load_split("val")
    val_acc = probe_spurious(val.sample(n=min(200,len(val)),random_state=1),model,tok,device)["acc_orig"]
    json.dump({"OOD_val_acc":val_acc},open("results/tables/ood_metrics.json","w"),indent=2)
    print("[OK] OOD metrics saved → results/tables/ood_metrics.json")


if __name__=="__main__":
    main()
