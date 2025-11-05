from datasets import load_from_disk
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, DataCollatorWithPadding
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, average_precision_score
import numpy as np
import pandas as pd
import json
from pathlib import Path

def compute_metrics(pred):
    labels = pred.label_ids
    probs = pred.predictions
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(labels, preds, average="macro")
    auroc = roc_auc_score(labels, probs[:,1])
    auprc = average_precision_score(labels, probs[:,1])
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1_macro": f1, "auroc": auroc, "auprc": auprc}

def main():
    # Load dataset
    print("Loading dataset...")
    ds = load_from_disk("data/processed/user_level_ds_hf")
    ds_train = ds["train"]
    ds_val = ds["validation"]
    
    # Tokenize
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    def tokenize(batch): 
        return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=256)
    
    print("Tokenizing dataset...")
    ds_train = ds_train.map(tokenize, batched=True)
    ds_val = ds_val.map(tokenize, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)
    args = TrainingArguments(
        output_dir="results/runs/distilbert",
        eval_strategy="epoch",  # Updated from evaluation_strategy
        save_strategy="no",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=False,
        metric_for_best_model="f1_macro"
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )
    print("Training model...")
    trainer.train()
    
    print("Evaluating model...")
    metrics = trainer.evaluate()
    
    # Save results
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    result_dict = {"model": "DistilBERT", **metrics}
    
    # Read existing results and append
    baselines_path = Path("results/tables/baselines.csv")
    if baselines_path.exists():
        existing = pd.read_csv(baselines_path)
        new_row = pd.DataFrame([result_dict])
        updated = pd.concat([existing, new_row], ignore_index=True)
        updated.to_csv(baselines_path, index=False)
    else:
        pd.DataFrame([result_dict]).to_csv(baselines_path, index=False)
    
    print("\nDistilBERT Results:")
    print(result_dict)

if __name__ == "__main__":
    main()
