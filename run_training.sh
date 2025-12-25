#!/bin/bash
# RunPod Training Script for MiniGPT
set -e

echo "============================================"
echo "MiniGPT Training Script"
echo "============================================"

# Install dependencies
pip install -q torch tqdm matplotlib

# Generate data if needed
if [ ! -f "data/add_only/train.pkl" ]; then
    echo "Generating dataset..."
    python3 generate_add_only.py
fi

# Create output directory
mkdir -p runs

# Train with logging
python3 train_minigpt_4070.py \
  --data data/add_only/train.pkl \
  --tokenizer data/add_only/tokenizer.json \
  --out_dir runs/step1_add_only \
  --epochs 30 --lr 1e-4 --patience 10 \
  --d_model 128 --n_layers 4 --n_heads 4 \
  --batch_size 64 --autoreg_eval \
  2>&1 | tee runs/training.log

echo ""
echo "============================================"
echo "Training complete! Results in runs/step1_add_only/"
echo "============================================"
