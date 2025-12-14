# evaluate_ood.py
# Evaluate a trained MiniGPT model on Out-of-Distribution (OOD) test suites
# AND Duplicate-Sensitivity analysis
#
# Usage (OOD mode):
#   python3 evaluate_ood.py --checkpoint runs/normal/best_model.pth --tokenizer data/processed/tokenizer.json --ood_dir ood_data_complete
#
# Usage (Duplication Sensitivity mode):
#   python3 evaluate_ood.py --mode duplication --checkpoint runs/normal/best_model.pth --tokenizer data/processed/tokenizer.json --train_data data/processed/train_normal.pkl --eval_data data/processed/val.pkl

import os
import json
import pickle
import argparse
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from tqdm import tqdm

# ---------- Model Architecture (imported from training script for consistency) ----------
# Import model classes from training script to ensure checkpoint compatibility
try:
    from train_minigpt_4070 import (
        MiniGPT,
        TokenEmbedding,
        PositionalEmbedding,
        RotaryPositionalEmbedding,
        MultiHeadSelfAttention,
        FeedForward,
        SwiGLUFeedForward,
        TransformerBlock
    )
    MODEL_IMPORTED = True
except ImportError:
    # Fallback: define basic model for backward compatibility with old checkpoints
    MODEL_IMPORTED = False
    print("WARNING: Could not import model from train_minigpt_4070.py. Using fallback model.")
    print("         New features (RoPE, SwiGLU) will not work. Consider running from same directory.")
    
    class TokenEmbedding(nn.Module):
        def __init__(self, vocab_size: int, d_model: int):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, d_model)
        def forward(self, tokens: torch.LongTensor):
            return self.emb(tokens)

    class PositionalEmbedding(nn.Module):
        def __init__(self, max_len: int, d_model: int):
            super().__init__()
            self.pos_emb = nn.Embedding(max_len, d_model)
        def forward(self, x: torch.Tensor):
            b, seq = x.shape
            positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(b, seq)
            return self.pos_emb(positions)

    class MultiHeadSelfAttention(nn.Module):
        def __init__(self, d_model: int, n_heads: int, causal: bool = True):
            super().__init__()
            assert d_model % n_heads == 0
            self.d_model = d_model
            self.n_heads = n_heads
            self.head_dim = d_model // n_heads
            self.causal = causal
            self.w_q = nn.Linear(d_model, d_model)
            self.w_k = nn.Linear(d_model, d_model)
            self.w_v = nn.Linear(d_model, d_model)
            self.out = nn.Linear(d_model, d_model)
        def forward(self, x: torch.Tensor):
            b, seq, _ = x.shape
            Q, K, V = self.w_q(x), self.w_k(x), self.w_v(x)
            def reshape_head(t):
                return t.view(b, seq, self.n_heads, self.head_dim).transpose(1, 2)
            Qh, Kh, Vh = reshape_head(Q), reshape_head(K), reshape_head(V)
            scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if self.causal:
                idxs = torch.arange(seq, device=x.device)
                mask = idxs.unsqueeze(0) <= idxs.unsqueeze(1)
                mask = mask.unsqueeze(0).unsqueeze(0)
                scores = scores.masked_fill(~mask, -1e9)
            attn = torch.softmax(scores, dim=-1)
            out_heads = torch.matmul(attn, Vh)
            out = out_heads.transpose(1, 2).contiguous().view(b, seq, self.d_model)
            return self.out(out)

    class FeedForward(nn.Module):
        def __init__(self, d_model: int, d_ff: int):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            )
        def forward(self, x: torch.Tensor):
            return self.net(x)

    class TransformerBlock(nn.Module):
        def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.0):
            super().__init__()
            self.ln1 = nn.LayerNorm(d_model)
            self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True)
            self.ln2 = nn.LayerNorm(d_model)
            self.ffn = FeedForward(d_model, d_ff)
            self.dropout = nn.Dropout(dropout)
        def forward(self, x: torch.Tensor):
            y = self.ln1(x)
            y = self.attn(y)
            x = x + self.dropout(y)
            y = self.ln2(x)
            y = self.ffn(y)
            x = x + self.dropout(y)
            return x

    class MiniGPT(nn.Module):
        def __init__(self, vocab_size: int, max_len: int, d_model: int, n_heads: int, 
                     n_layers: int, d_ff: int, dropout: float = 0.0, use_checkpoint: bool = False,
                     use_rope: bool = False, use_swiglu: bool = False):
            super().__init__()
            if use_rope or use_swiglu:
                raise ValueError("Fallback model does not support RoPE or SwiGLU. Import from train_minigpt_4070.py")
            self.tok_emb = TokenEmbedding(vocab_size, d_model)
            self.pos_emb = PositionalEmbedding(max_len, d_model)
            self.drop = nn.Dropout(dropout)
            self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)])
            self.ln_f = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, vocab_size, bias=False)
            self.max_len = max_len
            self.use_checkpoint = use_checkpoint
        def forward(self, tokens: torch.LongTensor):
            b, seq = tokens.shape
            if seq > self.max_len:
                tokens = tokens[:, -self.max_len:]
                seq = self.max_len
            x = self.tok_emb(tokens) + self.pos_emb(tokens)
            x = self.drop(x)
            for blk in self.blocks:
                x = blk(x)
            x = self.ln_f(x)
            logits = self.head(x)
            return logits


# ---------- OOD Dataset ----------
class OODDataset(Dataset):
    """Load OOD data from pickle and encode using tokenizer."""
    
    def __init__(self, pkl_path: str, stoi: dict, max_len: int):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        self.raw = data.get("raw", [])
        self.category = data.get("category", "unknown")
        self.stoi = stoi
        self.max_len = max_len
        self.pad_id = stoi.get("<pad>", 0)
        self.unk_id = stoi.get("<unk>", 0)
        
        # Encode raw examples on-the-fly
        self.encoded = []
        for text in self.raw:
            ids = [stoi.get(ch, self.unk_id) for ch in text]
            self.encoded.append(ids)
    
    def __len__(self):
        return len(self.encoded)
    
    def __getitem__(self, idx):
        seq = list(self.encoded[idx])
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return {
            "ids": torch.tensor(seq, dtype=torch.long),
            "raw": self.raw[idx]
        }


def collate_fn(batch: List[dict], pad_id: int, max_len: int):
    batch_ids = [b["ids"] for b in batch]
    lengths = [len(x) for x in batch_ids]
    max_batch_len = min(max(lengths), max_len)
    padded = torch.full((len(batch), max_batch_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        seq = ids[-max_batch_len:]
        padded[i, max_batch_len - len(seq):] = seq
    raws = [b["raw"] for b in batch]
    return padded, raws


# ---------- Utility ----------
def decode_ids_to_text(ids: List[int], itos: dict) -> str:
    return "".join([itos.get(i, "<unk>") for i in ids])


def extract_task_tag(raw: str) -> str:
    """Extract task tag like [addition_extrap] from raw text."""
    if raw and raw.startswith("["):
        end = raw.find("]")
        if end > 0:
            return raw[1:end]
    return "unknown"


def extract_expected_output(raw: str) -> str:
    """Extract expected output after the '|' separator."""
    if "|" in raw:
        return raw.split("|", 1)[1].strip()
    return ""


# ---------- Duplication Sensitivity Dataset ----------
class DuplicationSensitivityDataset(Dataset):
    """Dataset that tracks example frequency for duplication sensitivity analysis."""
    
    def __init__(self, eval_pkl_path: str, train_pkl_path: str, stoi: dict, max_len: int):
        """
        Args:
            eval_pkl_path: Path to evaluation data (val or test set)
            train_pkl_path: Path to training data (to compute duplication counts)
            stoi: Token to ID mapping
            max_len: Maximum sequence length
        """
        # Load evaluation data
        with open(eval_pkl_path, "rb") as f:
            eval_data = pickle.load(f)
        
        # Load training data to count duplicates
        with open(train_pkl_path, "rb") as f:
            train_data = pickle.load(f)
        
        self.raw = eval_data.get("raw", [])
        self.stoi = stoi
        self.max_len = max_len
        self.pad_id = stoi.get("<pad>", 0)
        self.unk_id = stoi.get("<unk>", 0)
        
        # Count frequency of each example in training data
        train_raw = train_data.get("raw", [])
        self.train_frequency = Counter(train_raw)
        
        # Encode examples
        self.encoded = []
        self.frequencies = []  # Frequency of each eval example in training
        
        for text in self.raw:
            ids = [stoi.get(ch, self.unk_id) for ch in text]
            self.encoded.append(ids)
            # How many times did this exact example appear in training?
            self.frequencies.append(self.train_frequency.get(text, 0))
        
        # Compute frequency buckets: unseen, 1x, 10x+
        self.bucket_counts = {
            "unseen": sum(1 for f in self.frequencies if f == 0),
            "1x": sum(1 for f in self.frequencies if f == 1),
            "10x+": sum(1 for f in self.frequencies if f >= 10)
        }
    
    def __len__(self):
        return len(self.encoded)
    
    def __getitem__(self, idx):
        seq = list(self.encoded[idx])
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return {
            "ids": torch.tensor(seq, dtype=torch.long),
            "raw": self.raw[idx],
            "frequency": self.frequencies[idx]
        }


def collate_fn_with_freq(batch: List[dict], pad_id: int, max_len: int):
    """Collate function that also returns frequency information."""
    batch_ids = [b["ids"] for b in batch]
    lengths = [len(x) for x in batch_ids]
    max_batch_len = min(max(lengths), max_len)
    padded = torch.full((len(batch), max_batch_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        seq = ids[-max_batch_len:]
        padded[i, max_batch_len - len(seq):] = seq
    raws = [b["raw"] for b in batch]
    freqs = [b["frequency"] for b in batch]
    return padded, raws, freqs


def get_frequency_bucket(freq: int) -> str:
    """Map frequency count to bucket name."""
    if freq == 0:
        return "unseen"
    elif freq == 1:
        return "1x"
    elif freq >= 10:
        return "10x+"
    else:
        return None  # Skip intermediate frequencies (2-9)


@torch.no_grad()
def evaluate_duplication_sensitivity(model, dataloader, device, stoi, itos, pad_id, vocab_size, max_len) -> Dict:
    """Evaluate model with duplication sensitivity analysis."""
    model.eval()
    
    # Overall metrics
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    total_examples = 0
    
    # Per-bucket tracking
    bucket_correct = defaultdict(int)
    bucket_total = defaultdict(int)
    bucket_loss = defaultdict(float)
    bucket_tokens = defaultdict(int)
    
    # Sample predictions per bucket
    bucket_samples = defaultdict(list)
    
    for padded, raws, freqs in tqdm(dataloader, desc="Evaluating duplication sensitivity"):
        padded = padded.to(device)
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        
        with autocast('cuda'):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size),
                target_ids.view(-1),
                ignore_index=pad_id,
                reduction='none'
            ).view(input_ids.shape[0], -1)
        
        preds = logits.argmax(dim=-1).cpu().tolist()
        targets = target_ids.cpu().tolist()
        
        for i, (pred_ids, target_ids_list, raw, freq) in enumerate(zip(preds, targets, raws, freqs)):
            bucket = get_frequency_bucket(freq)
            if bucket is None:  # Skip intermediate frequencies
                continue
            bucket_total[bucket] += 1
            total_examples += 1
            
            # Compute per-example loss
            example_loss = loss[i]
            mask = target_ids[i] != pad_id
            valid_loss = example_loss[mask].mean().item() if mask.sum() > 0 else 0.0
            bucket_loss[bucket] += valid_loss
            bucket_tokens[bucket] += mask.sum().item()
            total_loss += valid_loss
            total_tokens += mask.sum().item()
            
            # Compare predicted vs target tokens
            pred_tokens = [t for t in pred_ids if t != pad_id]
            target_tokens = [t for t in target_ids_list if t != pad_id]
            
            is_exact = (pred_tokens == target_tokens)
            if is_exact:
                exact_matches += 1
                bucket_correct[bucket] += 1
            
            # Store samples per bucket
            if len(bucket_samples[bucket]) < 3:
                expected = extract_expected_output(raw)
                bucket_samples[bucket].append({
                    "raw": raw[:80] + "..." if len(raw) > 80 else raw,
                    "frequency": freq,
                    "predicted": decode_ids_to_text(pred_tokens[-30:], itos),
                    "expected": expected[:30] if expected else "N/A",
                    "correct": is_exact
                })
    
    # Compute per-bucket accuracy
    bucket_accuracy = {}
    bucket_avg_loss = {}
    for bucket in ["unseen", "1x", "10x+"]:
        if bucket_total[bucket] > 0:
            bucket_accuracy[bucket] = bucket_correct[bucket] / bucket_total[bucket]
            bucket_avg_loss[bucket] = bucket_loss[bucket] / bucket_total[bucket]
        else:
            bucket_accuracy[bucket] = None
            bucket_avg_loss[bucket] = None
    
    # Compute memorization score: (accuracy on 10x+) - (accuracy on unseen)
    # Higher = more memorization, Lower/Negative = better generalization
    acc_high_freq = bucket_accuracy.get("10x+") or 0
    acc_unseen = bucket_accuracy.get("unseen") or 0
    memorization_score = acc_high_freq - acc_unseen if acc_high_freq and acc_unseen else None
    
    # Generalization gap: difference between seen and unseen
    acc_seen = sum(bucket_correct[b] for b in ["1x", "10x+"]) / max(1, sum(bucket_total[b] for b in ["1x", "10x+"]))
    generalization_gap = acc_seen - acc_unseen if acc_unseen else None
    
    return {
        "overall_loss": total_loss / total_examples if total_examples > 0 else 0,
        "overall_exact_match": exact_matches / total_examples if total_examples > 0 else 0,
        "total_examples": total_examples,
        "bucket_accuracy": bucket_accuracy,
        "bucket_loss": bucket_avg_loss,
        "bucket_counts": dict(bucket_total),
        "bucket_samples": dict(bucket_samples),
        "memorization_score": memorization_score,
        "generalization_gap": generalization_gap
    }


# ---------- Evaluation ----------
@torch.no_grad()
def evaluate_ood_file(model, dataloader, device, stoi, itos, pad_id, vocab_size, max_len) -> Dict:
    """Evaluate model on a single OOD file."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    partial_matches = 0
    total_examples = 0
    
    # Per-task tracking
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    
    # Store sample predictions for debugging
    sample_predictions = []
    
    for padded, raws in dataloader:
        padded = padded.to(device)
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        
        with autocast('cuda'):
            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, vocab_size), 
                target_ids.view(-1), 
                ignore_index=pad_id
            )
        
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        
        preds = logits.argmax(dim=-1).cpu().tolist()
        targets = target_ids.cpu().tolist()
        
        for pred_ids, target_ids_list, raw in zip(preds, targets, raws):
            task_tag = extract_task_tag(raw)
            task_total[task_tag] += 1
            total_examples += 1
            
            # Compare predicted vs target tokens (ignoring padding)
            pred_tokens = [t for t in pred_ids if t != pad_id]
            target_tokens = [t for t in target_ids_list if t != pad_id]
            
            # Exact match
            is_exact = (pred_tokens == target_tokens)
            if is_exact:
                exact_matches += 1
                task_correct[task_tag] += 1
            
            # Partial match: check if output portion matches
            expected = extract_expected_output(raw)
            if expected:
                pred_text = decode_ids_to_text(pred_tokens, itos)
                if expected in pred_text:
                    partial_matches += 1
            
            # Store samples for inspection
            if len(sample_predictions) < 5:
                sample_predictions.append({
                    "raw": raw[:100] + "..." if len(raw) > 100 else raw,
                    "predicted": decode_ids_to_text(pred_tokens[-50:], itos),
                    "expected": expected[:50] if expected else "N/A",
                    "correct": is_exact
                })
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0.0
    exact_rate = exact_matches / total_examples if total_examples > 0 else 0.0
    partial_rate = partial_matches / total_examples if total_examples > 0 else 0.0
    
    # Per-task accuracy
    task_accuracy = {}
    for task in task_total:
        task_accuracy[task] = task_correct[task] / task_total[task]
    
    return {
        "loss": avg_loss,
        "exact_match": exact_rate,
        "partial_match": partial_rate,
        "total_examples": total_examples,
        "task_accuracy": task_accuracy,
        "samples": sample_predictions
    }


def run_ood_evaluation(args, model, device, stoi, itos, pad_id, vocab_size, max_len, saved_args):
    """Run OOD evaluation mode."""
    # Find all OOD files
    ood_files = sorted([f for f in os.listdir(args.ood_dir) if f.endswith(".pkl")])
    if not ood_files:
        print(f"No .pkl files found in {args.ood_dir}")
        return
    
    print(f"\nFound {len(ood_files)} OOD test files")
    print("=" * 70)
    
    # Evaluate each OOD file
    all_results = {}
    category_results = defaultdict(list)
    
    for ood_file in ood_files:
        ood_path = os.path.join(args.ood_dir, ood_file)
        
        # Load dataset
        dataset = OODDataset(ood_path, stoi, max_len)
        dataloader = DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            shuffle=False,
            collate_fn=lambda batch: collate_fn(batch, pad_id, max_len)
        )
        
        # Evaluate
        results = evaluate_ood_file(model, dataloader, device, stoi, itos, pad_id, vocab_size, max_len)
        all_results[ood_file] = results
        category_results[dataset.category].append(results)
        
        # Print results
        print(f"\n📊 {ood_file}")
        print(f"   Category: {dataset.category}")
        print(f"   Examples: {results['total_examples']}")
        print(f"   Loss: {results['loss']:.4f}")
        print(f"   Exact Match: {results['exact_match']*100:.2f}%")
        print(f"   Partial Match: {results['partial_match']*100:.2f}%")
        
        if results['task_accuracy']:
            print(f"   Per-task breakdown:")
            for task, acc in sorted(results['task_accuracy'].items()):
                print(f"     - {task}: {acc*100:.1f}%")
        
        if args.show_samples and results['samples']:
            print(f"   Sample predictions:")
            for s in results['samples'][:3]:
                status = "✓" if s['correct'] else "✗"
                print(f"     {status} Input: {s['raw'][:60]}...")
                print(f"       Expected: {s['expected']}")
                print(f"       Got: {s['predicted']}")
    
    # Aggregate by category
    print("\n" + "=" * 70)
    print("📈 SUMMARY BY CATEGORY")
    print("=" * 70)
    
    summary = {}
    for category, results_list in sorted(category_results.items()):
        total_examples = sum(r['total_examples'] for r in results_list)
        avg_loss = sum(r['loss'] * r['total_examples'] for r in results_list) / total_examples
        avg_exact = sum(r['exact_match'] * r['total_examples'] for r in results_list) / total_examples
        avg_partial = sum(r['partial_match'] * r['total_examples'] for r in results_list) / total_examples
        
        summary[category] = {
            "total_examples": total_examples,
            "avg_loss": avg_loss,
            "exact_match": avg_exact,
            "partial_match": avg_partial
        }
        
        print(f"\n🏷️  {category.upper()}")
        print(f"   Total Examples: {total_examples}")
        print(f"   Avg Loss: {avg_loss:.4f}")
        print(f"   Exact Match: {avg_exact*100:.2f}%")
        print(f"   Partial Match: {avg_partial*100:.2f}%")
    
    # Overall stats
    total_all = sum(s['total_examples'] for s in summary.values())
    overall_exact = sum(s['exact_match'] * s['total_examples'] for s in summary.values()) / total_all
    overall_loss = sum(s['avg_loss'] * s['total_examples'] for s in summary.values()) / total_all
    
    print("\n" + "=" * 70)
    print("🎯 OVERALL OOD PERFORMANCE")
    print("=" * 70)
    print(f"   Total OOD Examples: {total_all}")
    print(f"   Overall Loss: {overall_loss:.4f}")
    print(f"   Overall Exact Match: {overall_exact*100:.2f}%")
    
    # Save results
    output_path = args.output or os.path.join(args.ood_dir, "ood_evaluation_results.json")
    output_data = {
        "checkpoint": args.checkpoint,
        "model_config": saved_args,
        "per_file_results": {k: {**v, "samples": v["samples"]} for k, v in all_results.items()},
        "category_summary": summary,
        "overall": {
            "total_examples": total_all,
            "loss": overall_loss,
            "exact_match": overall_exact
        }
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


def run_duplication_evaluation(args, model, device, stoi, itos, pad_id, vocab_size, max_len, saved_args):
    """Run duplication sensitivity evaluation mode."""
    print("\n" + "=" * 70)
    print("🔍 DUPLICATION SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Load dataset
    print(f"\nLoading evaluation data: {args.eval_data}")
    print(f"Loading training data for frequency analysis: {args.train_data}")
    
    dataset = DuplicationSensitivityDataset(args.eval_data, args.train_data, stoi, max_len)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda batch: collate_fn_with_freq(batch, pad_id, max_len)
    )
    
    print(f"\nDataset loaded: {len(dataset)} evaluation examples")
    print(f"Frequency distribution in eval set:")
    for bucket, count in dataset.bucket_counts.items():
        pct = count / len(dataset) * 100 if len(dataset) > 0 else 0
        print(f"   {bucket}: {count} ({pct:.1f}%)")
    
    # Evaluate
    results = evaluate_duplication_sensitivity(model, dataloader, device, stoi, itos, pad_id, vocab_size, max_len)
    
    # Print results
    print("\n" + "=" * 70)
    print("📊 RESULTS BY FREQUENCY BUCKET")
    print("=" * 70)
    
    print(f"\n{'Bucket':<12} {'Count':<10} {'Accuracy':<12} {'Loss':<10}")
    print("-" * 44)
    
    for bucket in ["unseen", "1x", "10x+"]:
        count = results['bucket_counts'].get(bucket, 0)
        acc = results['bucket_accuracy'].get(bucket)
        loss = results['bucket_loss'].get(bucket)
        
        acc_str = f"{acc*100:.2f}%" if acc is not None else "N/A"
        loss_str = f"{loss:.4f}" if loss is not None else "N/A"
        
        print(f"{bucket:<12} {count:<10} {acc_str:<12} {loss_str:<10}")
    
    print("\n" + "=" * 70)
    print("📈 KEY METRICS")
    print("=" * 70)
    
    print(f"\n   Overall Exact Match: {results['overall_exact_match']*100:.2f}%")
    print(f"   Overall Loss: {results['overall_loss']:.4f}")
    
    if results['memorization_score'] is not None:
        score = results['memorization_score']
        interpretation = "🔴 High memorization" if score > 0.2 else "🟡 Moderate" if score > 0.05 else "🟢 Good generalization"
        print(f"\n   Memorization Score: {score:.4f} ({interpretation})")
        print(f"   (Score = Acc(high-freq) - Acc(unseen). Higher = more memorization)")
    
    if results['generalization_gap'] is not None:
        gap = results['generalization_gap']
        interpretation = "🔴 Large gap" if gap > 0.3 else "🟡 Moderate gap" if gap > 0.1 else "🟢 Small gap"
        print(f"\n   Generalization Gap: {gap:.4f} ({interpretation})")
        print(f"   (Gap = Acc(seen) - Acc(unseen). Lower = better generalization)")
    
    # Show sample predictions per bucket
    if args.show_samples:
        print("\n" + "=" * 70)
        print("🔎 SAMPLE PREDICTIONS BY BUCKET")
        print("=" * 70)
        
        for bucket in ["unseen", "1x", "10x+"]:
            samples = results['bucket_samples'].get(bucket, [])
            if samples:
                print(f"\n📁 {bucket.upper()} bucket:")
                for s in samples:
                    status = "✓" if s['correct'] else "✗"
                    print(f"   {status} [freq={s['frequency']}] {s['raw']}")
                    print(f"      Expected: {s['expected']}")
                    print(f"      Got: {s['predicted']}")
    
    # Save results
    output_path = args.output or "duplication_sensitivity_results.json"
    output_data = {
        "checkpoint": args.checkpoint,
        "train_data": args.train_data,
        "eval_data": args.eval_data,
        "model_config": saved_args,
        "overall_exact_match": results['overall_exact_match'],
        "overall_loss": results['overall_loss'],
        "total_examples": results['total_examples'],
        "bucket_accuracy": results['bucket_accuracy'],
        "bucket_loss": results['bucket_loss'],
        "bucket_counts": results['bucket_counts'],
        "memorization_score": results['memorization_score'],
        "generalization_gap": results['generalization_gap'],
        "bucket_samples": results['bucket_samples']
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\n✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniGPT on OOD test suites or duplication sensitivity")
    
    # Common arguments
    parser.add_argument("--mode", choices=["ood", "duplication"], default="ood",
                        help="Evaluation mode: 'ood' for OOD testing, 'duplication' for duplication sensitivity")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--tokenizer", required=True, help="Path to tokenizer.json")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output", default=None, help="Output JSON file for results")
    parser.add_argument("--show_samples", action="store_true", help="Show sample predictions")
    
    # OOD mode arguments
    parser.add_argument("--ood_dir", default="ood_data_complete", help="Directory with OOD .pkl files (for --mode ood)")
    
    # Duplication sensitivity mode arguments
    parser.add_argument("--train_data", default=None, help="Path to training .pkl file (for --mode duplication)")
    parser.add_argument("--eval_data", default=None, help="Path to evaluation .pkl file (for --mode duplication)")
    
    args = parser.parse_args()
    
    # Validate arguments based on mode
    if args.mode == "duplication":
        if not args.train_data or not args.eval_data:
            parser.error("--mode duplication requires --train_data and --eval_data")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Mode: {args.mode.upper()}")
    
    # Load tokenizer
    with open(args.tokenizer, "r", encoding="utf-8") as f:
        tok = json.load(f)
    stoi = tok["stoi"]
    itos = {int(k): v for k, v in tok["itos"].items()} if isinstance(list(tok["itos"].keys())[0], str) else tok["itos"]
    pad_id = stoi.get("<pad>", 0)
    vocab_size = len(stoi)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Get model args from checkpoint
    saved_args = checkpoint.get("args", {})
    d_model = saved_args.get("d_model", 256)
    n_heads = saved_args.get("n_heads", 8)
    n_layers = saved_args.get("n_layers", 6)
    d_ff = saved_args.get("d_ff", d_model * 4)
    max_len = saved_args.get("max_len", 128)
    
    print(f"Model config: d_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, max_len={max_len}")
    
    # Get new model features from checkpoint (for compatibility with updated training script)
    use_rope = saved_args.get("use_rope", True)   # RoPE is now default in training script
    use_swiglu = saved_args.get("use_swiglu", False)
    if use_swiglu and not MODEL_IMPORTED:
        raise RuntimeError("Checkpoint uses SwiGLU but model import failed. Run from same directory as train_minigpt_4070.py")
    
    # Build and load model
    model = MiniGPT(
        vocab_size=vocab_size,
        max_len=max_len,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=0.0,  # No dropout during eval
        use_checkpoint=False,
        use_rope=use_rope,
        use_swiglu=use_swiglu
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Run appropriate evaluation
    if args.mode == "ood":
        run_ood_evaluation(args, model, device, stoi, itos, pad_id, vocab_size, max_len, saved_args)
    else:
        run_duplication_evaluation(args, model, device, stoi, itos, pad_id, vocab_size, max_len, saved_args)


if __name__ == "__main__":
    main()

