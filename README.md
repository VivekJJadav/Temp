# MiniGPT Training for Memorization vs Generalization Research

Training a small GPT model on synthetic tasks to study memorization vs generalization.

## Quick Start

### Recommended Training Command

```bash
python train_minigpt_4070.py \
  --data data/processed/train_normal.pkl \
  --tokenizer data/processed/tokenizer.json \
  --out_dir runs/fix_run_01 \
  --epochs 50 \
  --lr 5e-5 \
  --warmup_steps 1000 \
  --patience 10 \
  --d_model 256 \
  --n_layers 6 \
  --n_heads 8 \
  --batch_size 8 \
  --accumulation_steps 4 \
  --label_smoothing 0.05 \
  --dropout 0.1 \
  --task_curriculum \
  --show_samples \
  --autoreg_eval
```

```bash
python train_minigpt_4070.py \
  --data data/add_only/train.pkl \
  --tokenizer data/add_only/tokenizer.json \
  --out_dir runs/step1_add_only \
  --epochs 30 \
  --lr 1e-4 \
  --warmup_steps 200 \
  --patience 10 \
  --d_model 128 \
  --n_layers 4 \
  --n_heads 4 \
  --batch_size 32 \
  --accumulation_steps 1 \
  --label_smoothing 0.0 \
  --dropout 0.1 \
  --show_samples \
  --autoreg_eval
```


### Key Parameters Explained

| Parameter | Value | Description |
|-----------|-------|-------------|
| `--lr` | 5e-5 | Lower learning rate to prevent model collapse |
| `--epochs` | 50 | More training time for proper learning |
| `--patience` | 10 | Early stopping patience (epochs without improvement) |
| `--warmup_steps` | 1000 | Gradual LR warmup for stable training |
| `--dropout` | 0.1 | Regularization to reduce overfitting |
| `--label_smoothing` | 0.05 | Soft targets for better generalization |
| `--task_curriculum` | flag | Start with easy tasks (copy→rev→add→sort→rel) |
| `--show_samples` | flag | Display sample predictions during eval |
| `--autoreg_eval` | flag | Autoregressive generation for realistic eval |

## Project Structure

```
.
├── train_minigpt_4070.py    # Main training script
├── evaluate_ood.py          # Out-of-distribution evaluation
├── metrics.py               # Memorization/generalization metrics
├── plots.py                 # Visualization utilities
├── data/
│   └── processed/
│       ├── train_normal.pkl
│       ├── train_dedup.pkl
│       ├── train_duplicated.pkl
│       ├── val.pkl
│       └── tokenizer.json
├── ood_data_complete/       # OOD test sets
└── runs/                    # Training outputs & checkpoints
```

## Tasks

The model learns 5 synthetic tasks:
- `[copy]` - Copy the input sequence
- `[rev]` - Reverse the input sequence  
- `[add]` - Add two numbers
- `[sort]` - Sort a list of numbers
- `[rel]` - Relational reasoning

## Evaluation

After training, evaluate OOD generalization:

```bash
python evaluate_ood.py \
  --model runs/fix_run_01/best_model.pth \
  --tokenizer data/processed/tokenizer.json \
  --ood_dir ood_data_complete \
  --out_dir runs/fix_run_01/ood_results
```
