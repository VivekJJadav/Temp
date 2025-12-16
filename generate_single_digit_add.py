# generate_single_digit_add.py
# Generate the simplest possible dataset for debugging
# If this doesn't learn, something is fundamentally broken

import pickle
import json
import os
import random

def main():
    random.seed(42)
    
    out_dir = "data/single_digit_add"
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate ALL single-digit additions (10x10 = 100 combinations)
    train_samples = []
    for a in range(10):
        for b in range(10):
            train_samples.append(f"[add] {a} + {b} | {a+b}")
    
    # Shuffle and repeat to get more samples
    random.shuffle(train_samples)
    train_samples = train_samples * 50  # 5000 samples
    random.shuffle(train_samples)
    
    # Validation: same combinations but shuffled differently
    val_samples = []
    for a in range(10):
        for b in range(10):
            val_samples.append(f"[add] {a} + {b} | {a+b}")
    random.shuffle(val_samples)
    val_samples = val_samples * 5  # 500 samples
    
    print(f"Generated {len(train_samples)} training + {len(val_samples)} validation samples")
    print("Sample examples:")
    for s in train_samples[:5]:
        print(f"  {s}")
    
    # Build tokenizer
    all_chars = set()
    for s in train_samples:
        all_chars.update(s)
    
    stoi = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(all_chars):
        if ch not in stoi:
            stoi[ch] = len(stoi)
    itos = {v: k for k, v in stoi.items()}
    
    print(f"Vocabulary size: {len(stoi)}")
    print(f"Characters: {sorted(all_chars)}")
    
    # Encode
    def encode(samples):
        encoded = []
        for s in samples:
            ids = [stoi.get(ch, stoi["<unk>"]) for ch in s]
            encoded.append(ids)
        return encoded
    
    train_encoded = encode(train_samples)
    val_encoded = encode(val_samples)
    
    # Save
    train_data = {"encoded": train_encoded, "raw": train_samples, "labels": [None] * len(train_samples)}
    val_data = {"encoded": val_encoded, "raw": val_samples, "labels": [None] * len(val_samples)}
    
    with open(os.path.join(out_dir, "train.pkl"), "wb") as f:
        pickle.dump(train_data, f)
    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    
    tokenizer = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2)
    
    print(f"\nSaved to {out_dir}/")
    
    # Verify separator is in vocabulary
    print(f"\nSanity check: '|' has token ID {stoi.get('|', 'MISSING!')}")
    
    print("\n" + "="*60)
    print("TRAINING COMMAND (Single-Digit Add - Must Work):")
    print("="*60)
    print("""
python train_minigpt_4070.py \\
  --data data/single_digit_add/train.pkl \\
  --tokenizer data/single_digit_add/tokenizer.json \\
  --out_dir runs/single_digit_test \\
  --epochs 50 \\
  --lr 1e-4 \\
  --warmup_steps 200 \\
  --patience 15 \\
  --d_model 256 \\
  --n_layers 6 \\
  --n_heads 8 \\
  --batch_size 32 \\
  --accumulation_steps 1 \\
  --label_smoothing 0.0 \\
  --dropout 0.1 \\
  --show_samples \\
  --autoreg_eval

Expected: Token accuracy > 0.9, EM > 0.5 within 10-20 epochs
If this fails, the problem is NOT the model.
""")

if __name__ == "__main__":
    main()
