#!/bin/bash
# RunPod Setup Script for DistilBERT Training

echo "=========================================="
echo "RunPod Setup for Depression Classifier"
echo "=========================================="

# Update pip
echo "Updating pip..."
pip install --upgrade pip

# Install dependencies
echo "Installing dependencies..."
pip install torch transformers datasets scikit-learn pandas numpy tqdm matplotlib seaborn accelerate

# Verify GPU is available
echo ""
echo "Checking GPU availability..."
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Upload your data/processed/user_level_ds_hf/ folder"
echo "2. Run: python src/baseline_distilbert_lr.py"
echo ""
echo "Expected runtime: 5-10 minutes on GPU"
echo "=========================================="


