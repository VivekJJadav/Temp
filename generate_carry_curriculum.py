#!/usr/bin/env python3
"""
Generate carry-based curriculum dataset for addition learning.

Stages:
- Stage 0: No carry (e.g., 12 + 34 = 46)
- Stage 1: Single carry (e.g., 18 + 25 = 43)
- Stage 2: Multi-carry (e.g., 99 + 99 = 198)
"""

import random
import os
import pickle
import json


def count_carries(a: int, b: int) -> int:
    """Count number of carries in a + b."""
    carries = 0
    carry = 0
    while a > 0 or b > 0:
        digit_sum = (a % 10) + (b % 10) + carry
        if digit_sum >= 10:
            carries += 1
            carry = 1
        else:
            carry = 0
        a //= 10
        b //= 10
    return carries


def generate_sample(a: int, b: int) -> str:
    """Generate addition sample string."""
    return f"[add] {a} + {b} | {a + b}"


def generate_samples_by_carry(n_samples: int, min_val: int, max_val: int, 
                               target_carries: int = None, max_carries: int = None) -> list:
    """
    Generate samples filtered by carry count.
    
    Args:
        n_samples: Number of samples to generate
        min_val, max_val: Range for operands
        target_carries: Exact number of carries (if specified)
        max_carries: Maximum carries allowed (if specified, and target_carries is None)
    """
    samples = set()
    max_attempts = n_samples * 100
    attempts = 0
    
    while len(samples) < n_samples and attempts < max_attempts:
        a = random.randint(min_val, max_val)
        b = random.randint(min_val, max_val)
        carries = count_carries(a, b)
        
        # Filter by carry count
        if target_carries is not None:
            if carries != target_carries:
                attempts += 1
                continue
        elif max_carries is not None:
            if carries > max_carries:
                attempts += 1
                continue
        
        sample = generate_sample(a, b)
        samples.add(sample)
        attempts += 1
    
    return list(samples)


def main():
    random.seed(42)
    
    out_dir = "data/carry_curriculum"
    os.makedirs(out_dir, exist_ok=True)
    
    print("=" * 60)
    print("GENERATING CARRY-BASED CURRICULUM DATASET")
    print("=" * 60)
    
    # Stage 0: No carry (0-99 range, must have 0 carries)
    print("\n[Stage 0] No carry examples...")
    stage0_samples = generate_samples_by_carry(2000, 0, 99, target_carries=0)
    print(f"  Generated: {len(stage0_samples)} samples")
    print(f"  Examples: {stage0_samples[:3]}")
    
    # Stage 1: Single carry (0-99 range, must have exactly 1 carry)
    print("\n[Stage 1] Single carry examples...")
    stage1_samples = generate_samples_by_carry(2000, 0, 99, target_carries=1)
    print(f"  Generated: {len(stage1_samples)} samples")
    print(f"  Examples: {stage1_samples[:3]}")
    
    # Stage 2: Multi-carry (0-99 range, must have 2+ carries)
    print("\n[Stage 2] Multi-carry examples...")
    stage2_samples = generate_samples_by_carry(2000, 10, 99, target_carries=2)
    print(f"  Generated: {len(stage2_samples)} samples")
    print(f"  Examples: {stage2_samples[:3]}")
    
    # Validation: Mixed from all stages
    print("\n[Validation] Mixed from all stages...")
    val_samples = []
    val_samples.extend(random.sample(stage0_samples, min(100, len(stage0_samples))))
    val_samples.extend(random.sample(stage1_samples, min(200, len(stage1_samples))))
    val_samples.extend(random.sample(stage2_samples, min(200, len(stage2_samples))))
    random.shuffle(val_samples)
    print(f"  Generated: {len(val_samples)} samples")
    
    # Build tokenizer from all training data
    all_train = stage0_samples + stage1_samples + stage2_samples
    all_chars = set()
    for s in all_train:
        all_chars.update(s)
    
    stoi = {"<pad>": 0, "<unk>": 1}
    for ch in sorted(all_chars):
        if ch not in stoi:
            stoi[ch] = len(stoi)
    itos = {v: k for k, v in stoi.items()}
    
    print(f"\nVocabulary size: {len(stoi)}")
    print(f"Characters: {sorted(all_chars)}")
    
    # Encode function
    def encode(samples):
        encoded = []
        for s in samples:
            ids = [stoi.get(ch, stoi["<unk>"]) for ch in s]
            encoded.append(ids)
        return encoded
    
    # Save each stage
    for stage, samples, name in [
        (0, stage0_samples, "stage0_no_carry"),
        (1, stage1_samples, "stage1_single_carry"),
        (2, stage2_samples, "stage2_multi_carry"),
    ]:
        data = {
            "encoded": encode(samples),
            "raw": samples,
            "labels": [None] * len(samples),
            "stage": stage,
            "stage_name": name
        }
        path = os.path.join(out_dir, f"{name}.pkl")
        with open(path, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved: {path} ({len(samples)} samples)")
    
    # Save combined training data (all stages)
    all_train_data = {
        "encoded": encode(all_train),
        "raw": all_train,
        "labels": [None] * len(all_train)
    }
    with open(os.path.join(out_dir, "train_all.pkl"), "wb") as f:
        pickle.dump(all_train_data, f)
    print(f"Saved: {out_dir}/train_all.pkl ({len(all_train)} samples)")
    
    # Save validation
    val_data = {
        "encoded": encode(val_samples),
        "raw": val_samples,
        "labels": [None] * len(val_samples)
    }
    with open(os.path.join(out_dir, "val.pkl"), "wb") as f:
        pickle.dump(val_data, f)
    print(f"Saved: {out_dir}/val.pkl ({len(val_samples)} samples)")
    
    # Save tokenizer
    tokenizer = {"stoi": stoi, "itos": {str(k): v for k, v in itos.items()}}
    with open(os.path.join(out_dir, "tokenizer.json"), "w") as f:
        json.dump(tokenizer, f, indent=2)
    print(f"Saved: {out_dir}/tokenizer.json ({len(stoi)} tokens)")
    
    # Print summary
    print("\n" + "=" * 60)
    print("CURRICULUM SUMMARY")
    print("=" * 60)
    print(f"Stage 0 (no carry):     {len(stage0_samples)} samples")
    print(f"Stage 1 (single carry): {len(stage1_samples)} samples")
    print(f"Stage 2 (multi-carry):  {len(stage2_samples)} samples")
    print(f"Validation (mixed):     {len(val_samples)} samples")
    print(f"Total training:         {len(all_train)} samples")
    
    print("\n" + "=" * 60)
    print("TRAINING COMMAND:")
    print("=" * 60)
    print("""
# Option 1: Start with stage 0 only
python3 train_minigpt_4070.py \\
  --data data/carry_curriculum/stage0_no_carry.pkl \\
  --val_data data/carry_curriculum/val.pkl \\
  --tokenizer data/carry_curriculum/tokenizer.json \\
  --out_dir runs/carry_curriculum \\
  --epochs 30 --lr 1e-4 --patience 15 \\
  --d_model 256 --n_layers 6 --n_heads 8 \\
  --batch_size 64 --autoreg_eval

# Option 2: All stages at once
python3 train_minigpt_4070.py \\
  --data data/carry_curriculum/train_all.pkl \\
  --val_data data/carry_curriculum/val.pkl \\
  --tokenizer data/carry_curriculum/tokenizer.json \\
  --out_dir runs/carry_all \\
  --epochs 50 --lr 1e-4 --patience 15 \\
  --d_model 256 --n_layers 6 --n_heads 8 \\
  --batch_size 64 --autoreg_eval
""")


if __name__ == "__main__":
    main()
