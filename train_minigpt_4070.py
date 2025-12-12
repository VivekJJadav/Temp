# train_minigpt_4070.py
# Training script for MiniGPT on 8GB RTX 4070 (FP16, checkpointing, accumulation)
# Usage example:
#   python train_minigpt_4070.py --data data/processed/train_normal.pkl --tokenizer data/processed/tokenizer.json --out_dir runs/normal --epochs 6

import os
import json
import math
import time
import argparse
import random
import pickle
from collections import defaultdict
from functools import partial
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.checkpoint import checkpoint
from torch.amp import autocast, GradScaler  # Updated: non-deprecated imports
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------- Repro / device ----------
def set_seed(seed:int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

# ---------- Small MiniGPT model (causal LM) ----------
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size:int, d_model:int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
    def forward(self, tokens:torch.LongTensor):
        return self.emb(tokens)

class PositionalEmbedding(nn.Module):
    """Legacy positional embedding (kept for compatibility)."""
    def __init__(self, max_len:int, d_model:int):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, d_model)
    def forward(self, x:torch.Tensor):
        # x: (batch, seq)
        b, seq = x.shape
        positions = torch.arange(seq, device=x.device).unsqueeze(0).expand(b, seq)
        return self.pos_emb(positions)


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) for better length generalization.
    Encodes relative positions through rotation matrices applied to Q and K.
    """
    def __init__(self, d_model: int, max_len: int = 2048, base: int = 10000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        
        # Precompute the inverse frequencies
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)
        
        # Cache for sin/cos values
        self._cos_cached = None
        self._sin_cached = None
        self._seq_len_cached = 0
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update the cached sin/cos values if sequence length changed."""
        if seq_len > self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_model/2)
            emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, d_model)
            self._cos_cached = emb.cos().unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_model)
            self._sin_cached = emb.sin().unsqueeze(0).unsqueeze(0)
    
    def forward(self, q: torch.Tensor, k: torch.Tensor):
        """
        Apply rotary embeddings to Q and K tensors.
        Args:
            q, k: (batch, n_heads, seq_len, head_dim)
        Returns:
            q_rotated, k_rotated: same shapes as input
        """
        seq_len = q.shape[2]
        self._update_cache(seq_len, q.device)
        
        cos = self._cos_cached[:, :, :seq_len, :].to(q.dtype)
        sin = self._sin_cached[:, :, :seq_len, :].to(q.dtype)
        
        q_rotated = self._apply_rotary(q, cos, sin)
        k_rotated = self._apply_rotary(k, cos, sin)
        
        return q_rotated, k_rotated
    
    def _apply_rotary(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        """Apply rotary embedding to a tensor."""
        # Split into pairs and rotate
        x1 = x[..., ::2]   # Even indices
        x2 = x[..., 1::2]  # Odd indices
        
        cos = cos[..., ::2]
        sin = sin[..., ::2]
        
        # Rotate pairs
        rotated = torch.stack([
            x1 * cos - x2 * sin,
            x1 * sin + x2 * cos
        ], dim=-1).flatten(-2)
        
        return rotated


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model:int, n_heads:int, causal:bool=True, use_rope:bool=True, attn_dropout:float=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.use_rope = use_rope

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
        # Attention dropout (applied after softmax)
        self.attn_dropout = nn.Dropout(attn_dropout)
        
        # RoPE for positional encoding
        if use_rope:
            self.rope = RotaryPositionalEmbedding(self.head_dim)

    def forward(self, x: torch.Tensor):
        # x: (b, seq, d_model)
        b, seq, _ = x.shape
        Q = self.w_q(x)
        K = self.w_k(x)
        V = self.w_v(x)

        def reshape_head(t):
            return t.view(b, seq, self.n_heads, self.head_dim).transpose(1,2)  # b, heads, seq, head_dim

        Qh = reshape_head(Q)
        Kh = reshape_head(K)
        Vh = reshape_head(V)
        
        # Apply RoPE to Q and K
        if self.use_rope:
            Qh, Kh = self.rope(Qh, Kh)

        scores = torch.matmul(Qh, Kh.transpose(-2, -1)) / (self.head_dim ** 0.5)  # b, heads, seq, seq

        if self.causal:
            idxs = torch.arange(seq, device=x.device)
            mask = idxs.unsqueeze(0) <= idxs.unsqueeze(1)  # seq x seq lower triangular
            mask = mask.unsqueeze(0).unsqueeze(0)  # 1 x 1 x seq x seq
            scores = scores.masked_fill(~mask, -1e4)

        attn = torch.softmax(scores, dim=-1)
        attn = self.attn_dropout(attn)  # Apply attention dropout
        out_heads = torch.matmul(attn, Vh)  # b, heads, seq, head_dim
        out = out_heads.transpose(1,2).contiguous().view(b, seq, self.d_model)
        return self.out(out)

class FeedForward(nn.Module):
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x:torch.Tensor):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float=0.0, use_rope:bool=True):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x:torch.Tensor):
        y = self.ln1(x)
        y = self.attn(y)
        x = x + self.dropout(y)
        y = self.ln2(x)
        y = self.ffn(y)
        x = x + self.dropout(y)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size:int, max_len:int, d_model:int, n_heads:int, n_layers:int, d_ff:int, dropout:float=0.0, use_checkpoint:bool=False, use_rope:bool=True):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.use_rope = use_rope
        
        # Only use additive positional embedding if NOT using RoPE
        if not use_rope:
            self.pos_emb = PositionalEmbedding(max_len, d_model)
        
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope=use_rope) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len
        self.use_checkpoint = use_checkpoint
        
        # Weight tying: share input embedding with output projection
        # This reduces parameters and improves generalization
        self.head.weight = self.tok_emb.emb.weight

    def forward(self, tokens:torch.LongTensor):
        # tokens: (b, seq) integers
        b, seq = tokens.shape
        if seq > self.max_len:
            tokens = tokens[:, -self.max_len:]
            seq = self.max_len
        
        x = self.tok_emb(tokens)
        if not self.use_rope:
            x = x + self.pos_emb(tokens)  # Only add positional embedding if not using RoPE
        x = self.drop(x)
        
        if self.use_checkpoint:
            # checkpoint in groups to save memory - split blocks into chunks
            # naive: checkpoint each block (slower but memory-friendly)
            for blk in self.blocks:
                x = checkpoint(blk, x)
        else:
            for blk in self.blocks:
                x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)  # (b, seq, vocab)
        return logits

# ---------- Dataset loader ----------
class PickleDataset(Dataset):
    def __init__(self, pkl_path:str, stoi:dict, max_len:int):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        # Expect dict {"encoded": List[List[int]], "labels": [...], "raw": [...] }
        if not isinstance(data, dict):
            raise RuntimeError("Pickle content unexpected. Expected dict.")
        self.encoded = data.get("encoded", [])
        self.raw = data.get("raw", None)
        self.labels = data.get("labels", None)
        if self.raw is None:
            print("WARNING: 'raw' field not present in pickle; only encoded sequences available.")
        self.stoi = stoi
        self.pad_id = stoi.get("<pad>", 0)
        self.max_len = max_len

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        seq = list(self.encoded[idx])
        # Truncate to max_len
        if len(seq) > self.max_len:
            seq = seq[-self.max_len:]
        return {"ids": torch.tensor(seq, dtype=torch.long),
                "raw": self.raw[idx] if self.raw is not None else None,
                "length": len(self.encoded[idx])}  # Original length for curriculum


class CurriculumDataset(Dataset):
    """
    Wrapper dataset that filters examples by sequence length for curriculum learning.
    Progressively increase max_curriculum_len to include longer examples.
    """
    def __init__(self, base_dataset: PickleDataset, max_curriculum_len: int):
        self.base_dataset = base_dataset
        self.max_curriculum_len = max_curriculum_len
        # Build index of examples that fit within curriculum length
        self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild the list of valid indices based on current max_curriculum_len."""
        self.valid_indices = []
        for i in range(len(self.base_dataset)):
            orig_len = len(self.base_dataset.encoded[i])
            if orig_len <= self.max_curriculum_len:
                self.valid_indices.append(i)
        # Always include at least some examples (fallback to shortest)
        if len(self.valid_indices) == 0:
            lengths = [(i, len(self.base_dataset.encoded[i])) for i in range(len(self.base_dataset))]
            lengths.sort(key=lambda x: x[1])
            # Take shortest 10% as minimum
            min_count = max(1, len(lengths) // 10)
            self.valid_indices = [i for i, _ in lengths[:min_count]]
    
    def set_curriculum_length(self, new_max_len: int):
        """Update the curriculum length and rebuild indices."""
        self.max_curriculum_len = new_max_len
        self._rebuild_indices()
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.base_dataset[real_idx]
    
    def get_length_stats(self):
        """Return stats about current curriculum subset."""
        if not self.valid_indices:
            return {"count": 0, "min_len": 0, "max_len": 0, "avg_len": 0}
        lengths = [len(self.base_dataset.encoded[i]) for i in self.valid_indices]
        return {
            "count": len(lengths),
            "min_len": min(lengths),
            "max_len": max(lengths),
            "avg_len": sum(lengths) / len(lengths)
        }


def collate_fn(batch:List[dict], pad_id:int, max_len:int):
    # batch: list of {"ids": tensor([...]), "raw": str}
    batch_ids = [b["ids"] for b in batch]
    lengths = [len(x) for x in batch_ids]
    max_batch_len = min(max(lengths), max_len)
    padded = torch.full((len(batch), max_batch_len), pad_id, dtype=torch.long)
    for i, ids in enumerate(batch_ids):
        seq = ids[-max_batch_len:]
        padded[i, max_batch_len - len(seq):] = seq  # right-align
    raws = [b["raw"] for b in batch]
    return padded, raws

# ---------- Utility: decode ids -> string ----------
def decode_ids_to_text(ids: List[int], itos: dict):
    return "".join([itos.get(i, "<unk>") for i in ids])

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute the Levenshtein (edit) distance between two strings.
    Returns the minimum number of insertions, deletions, or substitutions
    needed to transform s1 into s2.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            # j+1 instead of j since previous_row and current_row are one character longer
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]

def normalized_edit_distance(s1: str, s2: str) -> float:
    """
    Normalized edit distance: 0.0 = identical, 1.0 = completely different.
    Normalized by max length of the two strings.
    """
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return dist / max(len(s1), len(s2))

# ---------- Autoregressive Generation ----------
@torch.no_grad()
def generate_autoregressive(model, prompt_ids: List[int], stoi: dict, itos: dict, 
                            max_new_tokens: int = 50, device: torch.device = None) -> List[int]:
    """
    Generate tokens autoregressively from a prompt.
    Stops when:
      - max_new_tokens is reached
      - A newline or special stopping pattern is detected
    
    Args:
        model: The MiniGPT model
        prompt_ids: List of token IDs for the prompt (input up to and including "|")
        stoi: String to ID mapping
        itos: ID to string mapping
        max_new_tokens: Maximum tokens to generate
        device: Device to run on
    
    Returns:
        List of generated token IDs (not including prompt)
    """
    model.eval()
    
    # Start with prompt
    generated = list(prompt_ids)
    
    for _ in range(max_new_tokens):
        # Prepare input (truncate if too long for model)
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)
        if input_ids.shape[1] > model.max_len:
            input_ids = input_ids[:, -model.max_len:]
        
        # Forward pass
        logits = model(input_ids)  # (1, seq, vocab)
        
        # Get next token (greedy decoding)
        next_token_logits = logits[0, -1, :]  # (vocab,)
        next_token = next_token_logits.argmax().item()
        
        generated.append(next_token)
        
        # Stopping conditions
        next_char = itos.get(next_token, "")
        if next_char == "\n":
            break
        # Stop if we generate another "|" (unlikely but possible)
        if next_char == "|" and len(generated) > len(prompt_ids) + 5:
            break
    
    # Return only the generated part (after prompt)
    return generated[len(prompt_ids):]

def extract_prompt_and_target(raw: str, stoi: dict) -> Tuple[List[int], str]:
    """
    Split raw example into prompt (input) and target (output).
    Returns prompt_ids (up to and including " | ") and target string.
    """
    if " | " not in raw:
        return [], ""
    
    parts = raw.split(" | ", 1)
    prompt = parts[0] + " | "
    target = parts[1] if len(parts) > 1 else ""
    
    prompt_ids = [stoi.get(ch, stoi.get("<unk>", 1)) for ch in prompt]
    return prompt_ids, target

# ---------- Memory Profiling ----------
def log_memory(prefix: str = ""):
    """Log GPU memory usage."""
    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"{prefix}VRAM: {alloc:.2f}GB allocated, {reserved:.2f}GB reserved, {peak:.2f}GB peak")

# ---------- Loss Curve Plotting ----------
def plot_loss_curves(history: dict, out_dir: str):
    """Plot and save train vs validation loss curves with accuracy metrics."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    # Create figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Subplot 1: Loss curves
    ax1.plot(epochs, history["train_loss"], 'b-o', label='Train Loss', linewidth=2, markersize=6)
    if history["val_loss"]:
        ax1.plot(epochs, history["val_loss"], 'r-s', label='Validation Loss', linewidth=2, markersize=6)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Train vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Exact Match Accuracy
    if history["val_exact"]:
        ax2.plot(epochs, history["val_exact"], 'g-^', label='Exact Match', linewidth=2, markersize=6)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Exact Match Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    else:
        ax2.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=14)
        ax2.set_title('Exact Match Accuracy', fontsize=14, fontweight='bold')
    
    # Subplot 3: Token Accuracy
    if history.get("val_token_acc"):
        ax3.plot(epochs, history["val_token_acc"], 'm-d', label='Token Accuracy', linewidth=2, markersize=6)
        ax3.set_xlabel('Epoch', fontsize=12)
        ax3.set_ylabel('Accuracy', fontsize=12)
        ax3.set_title('Token Accuracy', fontsize=14, fontweight='bold')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    else:
        ax3.text(0.5, 0.5, 'No validation data', ha='center', va='center', fontsize=14)
        ax3.set_title('Token Accuracy', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_path = os.path.join(out_dir, "loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Training metrics plot saved to: {plot_path}")

# ---------- Training / Evaluation loops ----------
def train_epoch(model, optimizer, scaler, dataloader, device, epoch, args, itos):
    model.train()
    total_loss = 0.0
    total_tokens = 0
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train E{epoch}")
    optimizer.zero_grad()
    accumulation = args.accumulation_steps
    for step, (padded, raws) in pbar:
        padded = padded.to(device)  # (b, seq)
        # prepare input/target for causal LM: input = all tokens except last, target = all tokens except first
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        with autocast('cuda'):  # Updated: specify device type
            logits = model(input_ids)  # (b, seq-1, vocab)
            vocab = logits.shape[-1]
            loss = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), ignore_index=args.pad_id)
            loss_val = loss.item()
        scaler.scale(loss / accumulation).backward()
        if (step + 1) % accumulation == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        total_loss += loss_val * input_ids.numel()  # token-level weighting
        total_tokens += input_ids.numel()
        pbar.set_postfix({"loss": f"{(total_loss/total_tokens):.4f}"})
    avg_loss = total_loss / total_tokens
    return avg_loss

def extract_task_type(raw: str) -> str:
    """Extract task type from raw text (e.g., 'math:', 'logic:', 'code:')."""
    if raw is None:
        return "unknown"
    # Common patterns: "task_type: input | output" or "[task_type] input"
    raw_lower = raw.lower().strip()
    for prefix in ["math", "logic", "code", "reasoning", "qa", "translate", "summarize"]:
        if raw_lower.startswith(prefix + ":") or raw_lower.startswith("[" + prefix + "]"):
            return prefix
    # Try to extract from format "task: ..."
    if ":" in raw:
        potential_task = raw.split(":")[0].strip().lower()
        if len(potential_task) <= 20:  # Reasonable task name length
            return potential_task
    return "general"

@torch.no_grad()
def evaluate(model, dataloader, device, args, itos):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    exact_matches = 0
    total_examples = 0
    vocab = args.vocab_size
    
    # Token accuracy tracking
    correct_tokens = 0
    counted_tokens = 0
    
    # Per-task accuracy tracking
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    
    for padded, raws in tqdm(dataloader, desc="Eval"):
        padded = padded.to(device)
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        with autocast('cuda'):  # Updated: specify device type
            logits = model(input_ids)
            loss = F.cross_entropy(logits.reshape(-1, vocab), target_ids.reshape(-1), ignore_index=args.pad_id)
        total_loss += loss.item() * input_ids.numel()
        total_tokens += input_ids.numel()
        
        # Greedy decode predictions
        preds = logits.argmax(dim=-1)  # (b, seq-1)
        
        # Token accuracy: count correct non-padding tokens
        mask = target_ids != args.pad_id
        correct_tokens += ((preds == target_ids) & mask).sum().item()
        counted_tokens += mask.sum().item()
        
        preds_list = preds.cpu().tolist()
        targets = target_ids.cpu().tolist()
        
        for i, (pred_ids, target_ids_list, raw) in enumerate(zip(preds_list, targets, raws)):
            if raw is None:
                continue
            
            # Show samples if requested
            if args.show_samples and i == 0 and random.random() < 0.1:  # Print snippet from ~10% of batches
                # Filter out pad tokens before decoding
                inp_filtered = [t for t in input_ids[i].tolist() if t != args.pad_id]
                tgt_filtered = [t for t in target_ids_list if t != args.pad_id]
                pred_filtered = [p for p, t in zip(pred_ids, target_ids_list) if t != args.pad_id]
                print(f"\n[Sample] Input:  {decode_ids_to_text(inp_filtered, itos)}")
                print(f"[Sample] Target: {decode_ids_to_text(tgt_filtered, itos)}")
                print(f"[Sample] Pred:   {decode_ids_to_text(pred_filtered, itos)}")
            
            task_type = extract_task_type(raw)
            task_total[task_type] += 1
            
            # Compare predicted tokens to target tokens (ignoring padding)
            # Filter predictions based on TARGET positions, not prediction values
            pred_tokens = [p for p, t in zip(pred_ids, target_ids_list) if t != args.pad_id]
            target_tokens = [t for t in target_ids_list if t != args.pad_id]
            
            # Exact match: entire sequence must match
            if pred_tokens == target_tokens:
                exact_matches += 1
                task_correct[task_type] += 1
            total_examples += 1
    
    avg_loss = total_loss / total_tokens
    exact_match_rate = (exact_matches / total_examples) if total_examples > 0 else 0.0
    token_accuracy = (correct_tokens / counted_tokens) if counted_tokens > 0 else 0.0
    
    # Compute per-task accuracy
    task_accuracy = {}
    for task in task_total:
        if task_total[task] > 0:
            task_accuracy[task] = task_correct[task] / task_total[task]
    
    return avg_loss, exact_match_rate, token_accuracy, task_accuracy


@torch.no_grad()
def evaluate_autoregressive(model, dataloader, device, args, stoi, itos):
    """
    Evaluate using true autoregressive generation (not teacher-forced).
    This gives a more realistic measure of model performance.
    """
    model.eval()
    
    exact_matches = 0
    total_examples = 0
    
    # Token accuracy tracking
    correct_tokens = 0
    total_tokens = 0
    
    # Per-task accuracy tracking
    task_correct = defaultdict(int)
    task_total = defaultdict(int)
    
    # Edit distance tracking
    total_edit_distance = 0.0
    
    # Store sample predictions
    sample_preds = []
    
    for padded, raws in tqdm(dataloader, desc="AutoReg Eval"):
        for i, raw in enumerate(raws):
            if raw is None:
                continue
            
            # Extract prompt and target
            prompt_ids, target_str = extract_prompt_and_target(raw, stoi)
            if not prompt_ids or not target_str:
                continue
            
            # Generate autoregressively
            generated_ids = generate_autoregressive(
                model, prompt_ids, stoi, itos,
                max_new_tokens=len(target_str) + 10,  # Give some buffer
                device=device
            )
            
            # Decode generated text
            generated_str = decode_ids_to_text(generated_ids, itos).strip()
            
            # Token accuracy: compare character by character
            target_chars = list(target_str)
            generated_chars = list(generated_str)
            min_len = min(len(target_chars), len(generated_chars))
            for j in range(min_len):
                total_tokens += 1
                if target_chars[j] == generated_chars[j]:
                    correct_tokens += 1
            # Count extra/missing tokens as errors
            total_tokens += abs(len(target_chars) - len(generated_chars))
            
            # Compare with target
            task_type = extract_task_type(raw)
            task_total[task_type] += 1
            total_examples += 1
            
            # Compute edit distance (useful when exact match is 0)
            edit_dist = normalized_edit_distance(generated_str, target_str)
            total_edit_distance += edit_dist
            
            is_match = (generated_str == target_str)
            if is_match:
                exact_matches += 1
                task_correct[task_type] += 1
            
            # Store samples for display
            if args.show_samples and len(sample_preds) < 5:
                sample_preds.append({
                    "prompt": decode_ids_to_text(prompt_ids, itos),
                    "target": target_str,
                    "generated": generated_str,
                    "match": is_match,
                    "edit_dist": edit_dist
                })
    
    # Print samples
    if args.show_samples and sample_preds:
        print("\n[AutoReg Samples]")
        for s in sample_preds:
            status = "✓" if s["match"] else "✗"
            print(f"  {status} Prompt: {s['prompt'][:50]}...")
            print(f"    Target:    '{s['target']}'")
            print(f"    Generated: '{s['generated']}' (edit_dist={s['edit_dist']:.2f})")
    
    exact_match_rate = (exact_matches / total_examples) if total_examples > 0 else 0.0
    token_accuracy = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    avg_edit_distance = (total_edit_distance / total_examples) if total_examples > 0 else 1.0
    
    # Compute per-task accuracy
    task_accuracy = {}
    for task in task_total:
        if task_total[task] > 0:
            task_accuracy[task] = task_correct[task] / task_total[task]
    
    return exact_match_rate, token_accuracy, avg_edit_distance, task_accuracy

# ---------- Argument parsing ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="path to .pkl dataset")
    p.add_argument("--tokenizer", required=True, help="path to tokenizer.json (stoi/itos)")
    p.add_argument("--out_dir", required=True, help="output dir for checkpoints/logs")
    p.add_argument("--val_data", default=None, help="path to validation .pkl (optional, auto-detects val.pkl if not specified)")
    p.add_argument("--epochs", type=int, default=6)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--d_ff", type=int, default=None)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--use_checkpoint", action="store_true", help="use gradient checkpointing in transformer blocks")
    p.add_argument("--save_every", type=int, default=1, help="save checkpoint every N epochs")
    p.add_argument("--seed", type=int, default=1337)
    # Early stopping arguments
    p.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs without improvement)")
    p.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")
    # Memory profiling
    p.add_argument("--profile_memory", action="store_true", help="log GPU memory usage each epoch")
    p.add_argument("--show_samples", action="store_true", help="show sample predictions during evaluation")
    p.add_argument("--autoreg_eval", action="store_true", help="use autoregressive generation for evaluation (slower but more realistic)")
    # Curriculum learning
    p.add_argument("--curriculum", action="store_true", help="enable curriculum learning (train on short sequences first)")
    p.add_argument("--curriculum_start", type=int, default=20, help="starting max sequence length for curriculum")
    p.add_argument("--curriculum_end", type=int, default=None, help="ending max sequence length (defaults to --max_len)")
    p.add_argument("--curriculum_epochs", type=int, default=5, help="epochs to transition from start to end length")
    args = p.parse_args()
    if args.d_ff is None:
        args.d_ff = args.d_model * 4
    return args

# ---------- Main ----------
def main():
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    os.makedirs(args.out_dir, exist_ok=True)

    # load tokenizer
    with open(args.tokenizer, "r", encoding="utf-8") as f:
        tok = json.load(f)
    stoi = tok["stoi"]
    itos = {int(k):v for k,v in tok["itos"].items()} if isinstance(list(tok["itos"].keys())[0], str) else tok["itos"]
    # tolerate either str keys or int keys in itos
    if isinstance(list(tok["itos"].keys())[0], str):
        itos = {int(k):v for k,v in tok["itos"].items()}
    else:
        itos = tok["itos"]

    pad_id = stoi.get("<pad>", 0)
    vocab_size = len(stoi)
    args.pad_id = pad_id
    args.vocab_size = vocab_size

    # dataset & loader
    base_dataset = PickleDataset(args.data, stoi, args.max_len)
    print(f"Training dataset loaded: {len(base_dataset)} examples")
    
    # Curriculum learning setup
    if args.curriculum:
        if args.curriculum_end is None:
            args.curriculum_end = args.max_len
        # Start with short sequences
        curriculum_dataset = CurriculumDataset(base_dataset, args.curriculum_start)
        dataset = curriculum_dataset
        stats = curriculum_dataset.get_length_stats()
        print(f"[Curriculum] Starting with max_len={args.curriculum_start}, "
              f"using {stats['count']} examples (avg_len={stats['avg_len']:.1f})")
    else:
        dataset = base_dataset
        curriculum_dataset = None
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len), num_workers=2, pin_memory=True)

    # Validation data: explicit path > auto-detect > None
    val_loader = None
    if args.val_data:
        val_path = args.val_data
    else:
        val_path = os.path.join(os.path.dirname(args.data), "val.pkl")
        if not os.path.exists(val_path):
            val_path = None
    
    if val_path and os.path.exists(val_path):
        val_dataset = PickleDataset(val_path, stoi, args.max_len)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len), num_workers=2, pin_memory=True)
        print(f"Validation loader ready: {len(val_dataset)} examples from {val_path}")
    else:
        print("WARNING: No validation data found. Training without validation (early stopping disabled).")

    # build model
    model = MiniGPT(vocab_size=vocab_size, max_len=args.max_len, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
                    dropout=args.dropout, use_checkpoint=args.use_checkpoint).to(device)
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # optimizer, scaler, scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda')  # Updated: specify device type
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)

    # Early stopping state
    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    # training loop
    history = {"train_loss": [], "val_loss": [], "val_exact": [], "val_token_acc": [], "task_accuracy": [], "lr": []}

    # Initial memory profiling
    if args.profile_memory:
        torch.cuda.reset_peak_memory_stats()
        log_memory("[Initial] ")

    for epoch in range(1, args.epochs+1):
        start = time.time()
        
        # Curriculum learning: progressively increase max sequence length
        if args.curriculum and curriculum_dataset is not None:
            # Linear interpolation from curriculum_start to curriculum_end
            progress = min(1.0, (epoch - 1) / max(1, args.curriculum_epochs - 1))
            new_max_len = int(args.curriculum_start + progress * (args.curriculum_end - args.curriculum_start))
            
            old_count = len(curriculum_dataset)
            curriculum_dataset.set_curriculum_length(new_max_len)
            new_count = len(curriculum_dataset)
            
            # Rebuild dataloader if dataset size changed significantly
            if new_count != old_count:
                dataloader = DataLoader(curriculum_dataset, batch_size=args.batch_size, shuffle=True,
                                        collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len), 
                                        num_workers=2, pin_memory=True)
            
            stats = curriculum_dataset.get_length_stats()
            print(f"[Curriculum] Epoch {epoch}: max_len={new_max_len}, "
                  f"examples={stats['count']}, avg_len={stats['avg_len']:.1f}")
        
        train_loss = train_epoch(model, optimizer, scaler, dataloader, device, epoch, args, itos)
        
        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        history["lr"].append(current_lr)
        
        print(f"[Epoch {epoch}] train loss = {train_loss:.4f}, LR = {current_lr:.2e} (time {time.time()-start:.1f}s)")
        history["train_loss"].append(train_loss)

        # Memory profiling
        if args.profile_memory:
            log_memory(f"[Epoch {epoch}] ")

        # Validation
        if val_loader is not None:
            val_loss, val_exact, token_acc, task_accuracy = evaluate(model, val_loader, device, args, itos)
            history["val_loss"].append(val_loss)
            history["val_exact"].append(val_exact)
            history["val_token_acc"].append(token_acc)
            history["task_accuracy"].append(task_accuracy)
            print(f"[Epoch {epoch}] val_loss={val_loss:.4f} | exact_match={val_exact:.4f} | token_acc={token_acc:.4f}")
            
            # Print per-task accuracy breakdown
            if task_accuracy:
                print(f"[Epoch {epoch}] Per-task accuracy (teacher-forced):")
                for task, acc in sorted(task_accuracy.items()):
                    print(f"    {task}: {acc:.4f}")
            
            # Optional: Autoregressive evaluation (slower but more realistic)
            if args.autoreg_eval:
                autoreg_exact, autoreg_token_acc, autoreg_edit_dist, autoreg_task_acc = evaluate_autoregressive(
                    model, val_loader, device, args, stoi, itos
                )
                history.setdefault("autoreg_exact", []).append(autoreg_exact)
                history.setdefault("autoreg_token_acc", []).append(autoreg_token_acc)
                history.setdefault("autoreg_edit_dist", []).append(autoreg_edit_dist)
                print(f"[Epoch {epoch}] AutoReg exact_match={autoreg_exact:.4f} | token_acc={autoreg_token_acc:.4f} | edit_dist={autoreg_edit_dist:.4f}")
                if autoreg_task_acc:
                    print(f"[Epoch {epoch}] AutoReg per-task:")
                    for task, acc in sorted(autoreg_task_acc.items()):
                        print(f"    {task}: {acc:.4f}")
            
            # Early stopping check
            if val_loss < best_val_loss - args.min_delta:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = model.state_dict().copy()
                print(f"[Epoch {epoch}] New best val loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                print(f"[Epoch {epoch}] No improvement. Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"\nEarly stopping triggered after {epoch} epochs!")
                    print(f"Best validation loss: {best_val_loss:.4f}")
                    # Restore best model
                    if best_model_state is not None:
                        model.load_state_dict(best_model_state)
                        print("Restored best model weights.")
                    break

        # save checkpoint
        if epoch % args.save_every == 0:
            cp = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optim_state": optimizer.state_dict(),
                "scaler_state": scaler.state_dict(),
                "scheduler_state": scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "args": vars(args)
            }
            cp_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pth")
            torch.save(cp, cp_path)
            print(f"Saved checkpoint: {cp_path}")

    # Save best model separately
    if best_model_state is not None:
        best_path = os.path.join(args.out_dir, "best_model.pth")
        torch.save({"model_state": best_model_state, "val_loss": best_val_loss, "args": vars(args)}, best_path)
        print(f"Saved best model: {best_path}")

    # final save results
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete. History written.")
    
    # Plot loss curves
    plot_loss_curves(history, args.out_dir)
    
    # Final memory summary
    if args.profile_memory:
        log_memory("[Final] ")
        print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")

if __name__ == "__main__":
    main()