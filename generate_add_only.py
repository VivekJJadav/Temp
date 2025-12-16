# generate_add_only.py
# Generate a minimal add-only dataset for debugging
# Step 1: Verify model can learn simple 1-2 digit addition

import pickle
import json
import random
import os

def generate_add_sample(max_digits=2):
    """Generate a single addition sample with 1-2 digit numbers."""
    max_val = 10 ** max_digits - 1
    a = random.randint(0, max_val)
    b = random.randint(0, max_val)
    result = a + b
    return f"[add] {a} + {b} | {result}"

def main():
    random.seed(42)
    
    out_dir = "data/add_only"
    os.makedirs(out_dir, exist_ok=True)
    
    # Target sizes
    n_train = 5000
    n_val = 500
    n_total = n_train + n_val
    
    print(f"Generating {n_train} training + {n_val} validation samples (ZERO overlap)...")
    
    # Generate unique samples to avoid ANY overlap between train and val
    unique_samples = set()
    max_attempts = n_total * 10  # Prevent infinite loop
    attempts = 0
    
    while len(unique_samples) < n_total and attempts < max_attempts:
        sample = generate_add_sample(max_digits=2)
        unique_samples.add(sample)
        attempts += 1
    
    if len(unique_samples) < n_total:
        print(f"WARNING: Could only generate {len(unique_samples)} unique samples")
    
    # Convert to list and shuffle
    all_samples = list(unique_samples)
    random.shuffle(all_samples)
    
    # Split into train and val (ZERO overlap guaranteed)
    train_samples = all_samples[:n_train]
    val_samples = all_samples[n_train:n_train + n_val]
    
    # Verify no overlap
    train_set = set(train_samples)
    val_set = set(val_samples)
    overlap = train_set & val_set
    print(f"Train unique: {len(train_set)}, Val unique: {len(val_set)}, Overlap: {len(overlap)}")
    assert len(overlap) == 0, "BUG: Train and val sets should not overlap!"
    
    print("Sample examples:")
    for s in train_samples[:5]:
        print(f"  {s}")
    
    # Build tokenizer from training data only
    all_chars = set()
    for s in train_samples:
        all_chars.update(s)
    
    # Create vocabulary
    stoi = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(all_chars):
        if ch not in stoi:
            stoi[ch] = len(stoi)
    itos = {v: k for k, v in stoi.items()}
    
    print(f"Vocabulary size: {len(stoi)}")
    print(f"Characters: {sorted(all_chars)}")
    
    # Encode samples
    def encode(samples):
        encoded = []
        for s in samples:
            ids = [stoi.get(ch, stoi["<unk>"]) for ch in s]
            encoded.append(ids)
        return encoded
    
    train_encoded = encode(train_samples)
    val_encoded = encode(val_samples)
    
    # Save as pickle (matching expected format)
    train_data = {
        "encoded": train_encoded,
        "raw": train_samples,
        "labels": [None] * len(train_samples)
    }
    val_data = {
        "encoded": val_encoded,
        "raw": val_samples,
        "labels": [None] * len(val_samples)
    }
    
    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    
    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    
    # Save tokenizer
    tokenizer = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2)
    
    print(f"\nSaved to {out_dir}/")
    print(f"  train.pkl: {len(train_samples)} samples")
    print(f"  val.pkl: {len(val_samples)} samples")
    print(f"  tokenizer.json: {len(stoi)} tokens")
    
    # Print training command
    print("\n" + "="*60)
    print("TRAINING COMMAND (Step 1 - Add Only):")
    print("="*60)
    print("""
python train_minigpt_4070.py \\
  --data data/add_only/train.pkl \\
  --tokenizer data/add_only/tokenizer.json \\
  --out_dir runs/step1_add_only \\
  --epochs 30 \\
  --lr 1e-4 \\
  --warmup_steps 200 \\
  --patience 10 \\
  --d_model 128 \\
  --n_layers 4 \\
  --n_heads 4 \\
  --batch_size 32 \\
  --accumulation_steps 1 \\
  --label_smoothing 0.0 \\
  --dropout 0.1 \\
  --show_samples \\
  --autoreg_eval
""")

if __name__ == "__main__":
    main()
