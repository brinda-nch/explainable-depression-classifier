# RunPod Setup Instructions

## Quick Start Guide for Running DistilBERT Training on RunPod

### Step 1: Create a RunPod Instance

1. Go to [RunPod.io](https://www.runpod.io) and sign up/login
2. Click **"Deploy"** → **"GPU Pods"**
3. Choose a GPU (recommended):
   - **Budget:** RTX 4090 (~$0.50/hr) - 10 min training
   - **Fast:** RTX A5000 (~$0.80/hr) - 8 min training
   - **Fastest:** A100 (~$2/hr) - 5 min training
4. Select template: **PyTorch 2.0** or **RunPod Pytorch**
5. Click **"Deploy"**

### Step 2: Connect to Your Pod

1. Once pod is running, click **"Connect"**
2. Choose **"Start Jupyter Lab"** or **"Connect via SSH"**
3. Open a terminal in Jupyter Lab

### Step 3: Upload Your Project

**Option A: Via Jupyter Lab Upload**
1. Zip your project folder on your local machine
2. In Jupyter Lab, click upload button
3. Upload and unzip: `unzip explainable-depression-classifier.zip`

**Option B: Via GitHub (Recommended)**
```bash
git clone https://github.com/YOUR_USERNAME/explainable-depression-classifier.git
cd explainable-depression-classifier
```

### Step 4: Setup Environment

Run the setup script:
```bash
chmod +x setup_runpod.sh
./setup_runpod.sh
```

Or manually:
```bash
pip install -r requirements.txt
```

### Step 5: Verify Data Upload

Make sure your data is present:
```bash
ls -la data/processed/user_level_ds_hf/
```

You should see:
- `dataset_dict.json`
- `train/` directory
- `validation/` directory
- `test/` directory

### Step 6: Run Training

```bash
python src/baseline_distilbert_lr.py
```

**Expected Output:**
```
[INFO] Using device: GPU
[INFO] Batch sizes - Train: 32, Eval: 64
Training...
100%|████████████| XXX/XXX [05:00<00:00, X.XXit/s]
[OK] DistilBERT baseline trained and evaluated.
```

### Step 7: Download Results

After training completes, download:
- `results/figures/cm_distilbert_val.png`
- `results/figures/pr_distilbert_val.png`
- `results/figures/roc_distilbert_val.png`
- `results/tables/baselines.csv`

**Download via Jupyter Lab:**
1. Navigate to `results/` folder
2. Right-click files → Download

**Download via terminal (if using SSH):**
```bash
# On RunPod terminal
zip -r results.zip results/

# Then download via Jupyter Lab interface
```

### Step 8: Stop Your Pod

⚠️ **Important:** Don't forget to stop your pod after downloading results to avoid charges!

1. Go back to RunPod dashboard
2. Click **"Stop"** or **"Terminate"** on your pod

---

## Troubleshooting

### Out of Memory Error
Reduce batch size in `src/baseline_distilbert_lr.py`:
```python
per_device_train_batch_size=16  # Reduce from 32
per_device_eval_batch_size=32   # Reduce from 64
```

### Dataset Not Found
Make sure you uploaded the entire `data/processed/user_level_ds_hf/` folder with all subdirectories.

### Slow Training
Check GPU is being used:
```python
import torch
print(torch.cuda.is_available())  # Should be True
print(torch.cuda.get_device_name(0))  # Shows your GPU
```

---

## Performance Comparison

| Environment | Time | Cost |
|-------------|------|------|
| Local CPU | ~60-90 min | Free |
| Local MPS (Mac) | ~30-35 min | Free |
| RTX 4090 | ~10 min | $0.08 |
| RTX A5000 | ~8 min | $0.11 |
| A100 | ~5 min | $0.17 |

---

## Files Generated

After successful run, you'll have:
- ✅ `results/figures/cm_distilbert_val.png` - Confusion Matrix
- ✅ `results/figures/pr_distilbert_val.png` - Precision-Recall Curve
- ✅ `results/figures/roc_distilbert_val.png` - ROC Curve
- ✅ `results/tables/baselines.csv` - Updated with DistilBERT metrics
- ✅ `results/runs/distilbert/metrics_val.json` - Detailed metrics

