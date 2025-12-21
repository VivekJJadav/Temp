# generate_add_only.py
# Generate a minimal add-only dataset for debugging
# Step 1: Verify model can learn simple 1-2 digit addition

import pickle
import json
import random
import os

def generate_add_sample(min_val=0, max_val=99):
    """Generate a single addition sample with numbers in [min_val, max_val]."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    result = a + b
    return f"[add] {a} + {b} | {result}"


def generate_add_sample_ood(min_val=100, max_val=999):
    """Generate OOD addition sample with larger numbers (for val set)."""
    a = random.randint(min_val, max_val)
    b = random.randint(min_val, max_val)
    result = a + b
    return f"[add] {a} + {b} | {result}"

def main():
    random.seed(42)
    
    out_dir = "data/add_only"
    os.makedirs(out_dir, exist_ok=True)
    
    # Target sizes
    n_train = 5000
    n_val = 500
    
    print("=" * 60)
    print("GENERATING ADD-ONLY DATASET (OOD VALIDATION)")
    print("=" * 60)
    print(f"Training:   {n_train} samples, numbers in [0, 99] (1-2 digit)")
    print(f"Validation: {n_val} samples, numbers in [100, 999] (3 digit) <- OOD!")
    print("=" * 60)
    
    # Generate TRAINING samples (1-2 digit numbers: 0-99)
    print(f"\nGenerating {n_train} training samples (0-99 range)...")
    train_unique = set()
    max_attempts = n_train * 10
    attempts = 0
    while len(train_unique) < n_train and attempts < max_attempts:
        sample = generate_add_sample(min_val=0, max_val=99)
        train_unique.add(sample)
        attempts += 1
    train_samples = list(train_unique)
    random.shuffle(train_samples)
    
    # Generate VALIDATION samples (3 digit numbers: 100-999) <- OOD!
    # This tests TRUE generalization: can the model add numbers it never saw?
    print(f"Generating {n_val} OOD validation samples (100-999 range)...")
    val_unique = set()
    attempts = 0
    while len(val_unique) < n_val and attempts < max_attempts:
        sample = generate_add_sample_ood(min_val=100, max_val=999)
        val_unique.add(sample)
        attempts += 1
    val_samples = list(val_unique)
    random.shuffle(val_samples)
    
    # Verify no overlap (should be trivially true since ranges don't overlap)
    train_set = set(train_samples)
    val_set = set(val_samples)
    overlap = train_set & val_set
    print(f"\nTrain: {len(train_set)}, Val: {len(val_set)}, Overlap: {len(overlap)}")
    assert len(overlap) == 0, "BUG: Train and val sets should not overlap!"
    
    print("\nSample TRAIN examples (1-2 digit):")
    for s in train_samples[:3]:
        print(f"  {s}")
    
    print("\nSample VAL examples (3 digit - OOD):")
    for s in val_samples[:3]:
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
