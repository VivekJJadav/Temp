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

# TensorBoard for real-time visualization
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False
    print("TensorBoard not available. Install with: pip install tensorboard")

# Import custom modules for metrics and plotting
try:
    from metrics import compute_train_exact_match, compute_ngram_overlap, compute_generation_entropy_stats
    from plots import generate_all_training_plots
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False
    print("WARNING: metrics.py or plots.py not found. Advanced metrics disabled.")

# ---------- Repro / device ----------
def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception:
        pass


def get_eval_rng(seed: int):
    """
    Create a separate RNG for evaluation sampling.
    Keeps eval sampling reproducible independent of training randomness.
    """
    return np.random.default_rng(seed + 42)

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
    """Standard GELU FeedForward network."""
    def __init__(self, d_model:int, d_ff:int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x:torch.Tensor):
        return self.net(x)


class SwiGLUFeedForward(nn.Module):
    """
    SwiGLU FeedForward network (improved FFN from LLaMA/PaLM).
    Uses gating mechanism: SiLU(xW1) * (xW3) followed by W2.
    ~13% more params but better quality per FLOP.
    """
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        # Note: d_ff is the hidden dimension, we use 2/3 * d_ff for gated variant
        # to maintain similar param count as standard FFN
        hidden_dim = int(2 * d_ff / 3)
        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, d_model, bias=False)
        self.w3 = nn.Linear(d_model, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: (SiLU(xW1) * xW3) W2
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class TransformerBlock(nn.Module):
    def __init__(self, d_model:int, n_heads:int, d_ff:int, dropout:float=0.0, use_rope:bool=True, use_swiglu:bool=False):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_heads, causal=True, use_rope=use_rope)
        self.ln2 = nn.LayerNorm(d_model)
        # Choose FFN type based on use_swiglu
        if use_swiglu:
            self.ffn = SwiGLUFeedForward(d_model, d_ff)
        else:
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
    def __init__(self, vocab_size:int, max_len:int, d_model:int, n_heads:int, n_layers:int, d_ff:int, dropout:float=0.0, use_checkpoint:bool=False, use_rope:bool=True, use_swiglu:bool=False):
        super().__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.use_rope = use_rope
        
        # Only use additive positional embedding if NOT using RoPE
        if not use_rope:
            self.pos_emb = PositionalEmbedding(max_len, d_model)
        
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads, d_ff, dropout, use_rope=use_rope, use_swiglu=use_swiglu) for _ in range(n_layers)])
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
                x = checkpoint(blk, x, use_reentrant=False)
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


# ---------- Task-Based Curriculum ----------
# Define task difficulty stages (easier tasks first)
TASK_DIFFICULTY_STAGES = [
    # Stage 0: Easiest - simple copying
    {"tasks": ["copy"], "description": "Copy/repeat tasks"},
    # Stage 1: Reversing (needs to understand order)
    {"tasks": ["rev"], "description": "Reverse sequences"},
    # Stage 2: Easy addition (1-2 digit)
    {"tasks": ["add"], "max_input_len": 15, "description": "Easy addition"},
    # Stage 3: Easy sorting (short lists)
    {"tasks": ["sort"], "max_input_len": 25, "description": "Easy sorting"},
    # Stage 4: Hard addition (3+ digit)
    {"tasks": ["add"], "description": "All addition"},
    # Stage 5: Hard sorting (long lists)
    {"tasks": ["sort"], "description": "All sorting"},
    # Stage 6: Relations (hardest - needs reasoning)
    {"tasks": ["rel"], "description": "Relation reasoning"},
]


def extract_task_from_raw(raw: str) -> str:
    """Extract task type from raw example string."""
    if raw is None:
        return "unknown"
    raw = str(raw).strip()
    # Format: [task] input | output
    if raw.startswith("[") and "]" in raw:
        task = raw[1:raw.index("]")]
        return task.lower()
    return "unknown"


def get_example_difficulty(raw: str, encoded_len: int) -> int:
    """
    Score example difficulty (0 = easiest, higher = harder).
    Used for fine-grained ordering within stages.
    """
    task = extract_task_from_raw(raw)
    
    # Base difficulty by task type
    task_base = {
        "copy": 0,
        "rev": 100,
        "add": 200,
        "sort": 300,
        "rel": 400,
    }.get(task, 500)
    
    # Add length-based sub-difficulty
    return task_base + encoded_len


class TaskCurriculumDataset(Dataset):
    """
    Curriculum dataset that introduces tasks progressively by semantic difficulty.
    
    Instead of filtering by sequence length, this filters by task TYPE,
    introducing easier tasks (copy, reverse) before harder ones (addition, relations).
    """
    
    def __init__(self, base_dataset: PickleDataset, max_stage: int = 0):
        self.base_dataset = base_dataset
        self.max_stage = max_stage
        self.stages = TASK_DIFFICULTY_STAGES
        
        # Pre-compute task type for each example
        self.example_tasks = []
        self.example_difficulties = []
        for i in range(len(base_dataset)):
            raw = base_dataset.raw[i] if base_dataset.raw else None
            task = extract_task_from_raw(raw)
            self.example_tasks.append(task)
            encoded_len = len(base_dataset.encoded[i])
            self.example_difficulties.append(get_example_difficulty(raw, encoded_len))
        
        # Count tasks
        task_counts = defaultdict(int)
        for t in self.example_tasks:
            task_counts[t] += 1
        print(f"[TaskCurriculum] Dataset task distribution: {dict(task_counts)}")
        
        self._rebuild_indices()
    
    def _rebuild_indices(self):
        """Rebuild valid indices based on current max_stage."""
        self.valid_indices = []
        
        # Collect allowed tasks from stages 0 to max_stage
        allowed_tasks = set()
        max_input_lens = {}  # task -> max input length (if specified)
        
        for stage_idx in range(min(self.max_stage + 1, len(self.stages))):
            stage = self.stages[stage_idx]
            for task in stage["tasks"]:
                allowed_tasks.add(task)
                # Update max_input_len (later stages may remove the limit)
                if "max_input_len" in stage:
                    if task not in max_input_lens or max_input_lens[task] < stage["max_input_len"]:
                        max_input_lens[task] = stage["max_input_len"]
                else:
                    # No limit - remove any previous limit
                    max_input_lens[task] = float('inf')
        
        # Filter examples
        for i in range(len(self.base_dataset)):
            task = self.example_tasks[i]
            if task in allowed_tasks:
                # Check input length constraint
                encoded_len = len(self.base_dataset.encoded[i])
                max_len = max_input_lens.get(task, float('inf'))
                if encoded_len <= max_len or max_len == float('inf'):
                    self.valid_indices.append(i)
        
        # Sort by difficulty for smoother curriculum within stage
        self.valid_indices.sort(key=lambda i: self.example_difficulties[i])
    
    def set_stage(self, new_stage: int):
        """Update curriculum stage and rebuild indices."""
        self.max_stage = min(new_stage, len(self.stages) - 1)
        self._rebuild_indices()
    
    def get_current_stage_info(self) -> dict:
        """Get info about current curriculum stage."""
        if self.max_stage < len(self.stages):
            stage = self.stages[self.max_stage]
            return {
                "stage": self.max_stage,
                "tasks": stage["tasks"],
                "description": stage.get("description", ""),
                "num_examples": len(self.valid_indices)
            }
        return {"stage": self.max_stage, "tasks": ["all"], "num_examples": len(self.valid_indices)}
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        return self.base_dataset[real_idx]
    
    def get_stats(self) -> dict:
        """Return stats about current curriculum subset."""
        if not self.valid_indices:
            return {"count": 0, "tasks": {}}
        
        task_counts = defaultdict(int)
        for i in self.valid_indices:
            task_counts[self.example_tasks[i]] += 1
        
        return {
            "count": len(self.valid_indices),
            "tasks": dict(task_counts),
            "stage": self.max_stage,
            "stage_info": self.get_current_stage_info()
        }



def collate_fn(batch:List[dict], pad_id:int, max_len:int, sep_token_id:int = None):
    """
    Collate function that also creates a loss mask.
    
    Uses RIGHT PADDING (sequences left-aligned, padding on right).
    This is better for causal attention because:
    - Real tokens only attend to real tokens (padding is in the "future")
    - Causal mask naturally prevents attending to padding
    - No extra padding mask needed in attention
    
    The loss mask is 1.0 for output tokens (after separator "|") and 0.0 for input tokens.
    This ensures the model is only trained to predict the output, not the input.
    
    Args:
        batch: List of {"ids": tensor, "raw": str}
        pad_id: Padding token ID
        max_len: Maximum sequence length
        sep_token_id: Token ID for separator "|" (if None, no masking)
    
    Returns:
        padded: (batch, seq) padded token IDs
        raws: List of raw strings
        loss_mask: (batch, seq-1) mask where 1.0 = compute loss, 0.0 = ignore
    """
    batch_ids = [b["ids"] for b in batch]
    lengths = [len(x) for x in batch_ids]
    max_batch_len = min(max(lengths), max_len)
    padded = torch.full((len(batch), max_batch_len), pad_id, dtype=torch.long)
    
    # Create loss mask: 1.0 for positions AFTER separator, 0.0 for positions BEFORE/AT separator
    # loss_mask[t] corresponds to predicting padded[t+1] (the target at position t)
    loss_mask = torch.zeros((len(batch), max_batch_len - 1), dtype=torch.float32)
    
    for i, ids in enumerate(batch_ids):
        # RIGHT PADDING: sequence starts at position 0, padding fills the rest
        seq = ids[:max_batch_len]  # truncate from the end if needed
        seq_len = len(seq)
        padded[i, :seq_len] = seq  # left-aligned
        
        # Find separator position and create mask
        if sep_token_id is not None:
            # Find separator in the sequence
            sep_positions = (seq == sep_token_id).nonzero(as_tuple=True)[0]
            if len(sep_positions) > 0:
                # sep_idx is the index of the separator
                sep_idx = sep_positions[0].item()
                # output_start is the first OUTPUT token position (after separator)
                output_start = sep_idx + 1
                
                # loss_mask[t] = 1.0 means we compute loss for predicting padded[t+1]
                # We want loss when target position (t+1) >= output_start AND not padding
                for t in range(max_batch_len - 1):
                    target_pos = t + 1  # the token we're predicting
                    if target_pos >= output_start and target_pos < seq_len:
                        loss_mask[i, t] = 1.0
            else:
                # No separator found, compute loss on all non-pad tokens (fallback)
                for t in range(seq_len - 1):
                    loss_mask[i, t] = 1.0
        else:
            # No separator masking, compute loss on all non-pad tokens
            for t in range(seq_len - 1):
                loss_mask[i, t] = 1.0
    
    raws = [b["raw"] for b in batch]
    return padded, raws, loss_mask

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
                            max_new_tokens: int = 50, device: torch.device = None,
                            temperature: float = 1.0, top_p: float = 1.0, top_k: int = 0) -> List[int]:
    """
    Generate tokens autoregressively from a prompt.
    Supports greedy, temperature scaling, top-k, and nucleus (top-p) sampling.
    
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
        temperature: Sampling temperature (1.0 = normal, <1.0 = more confident, >1.0 = more random)
        top_p: Nucleus sampling threshold (1.0 = disabled, <1.0 = only sample from top cumulative prob)
        top_k: Top-k sampling (0 = disabled, >0 = only sample from top k tokens)
    
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
        next_token_logits = logits[0, -1, :].clone()  # (vocab,)
        
        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature
        
        # Apply top-k filtering
        if top_k > 0:
            top_k = min(top_k, next_token_logits.size(-1))
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][-1]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Apply nucleus (top-p) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Shift: keep first token above threshold too
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_token_logits[indices_to_remove] = float('-inf')
        
        # Sample or greedy
        if temperature == 0 or (top_p == 1.0 and top_k == 0 and temperature == 1.0):
            # Greedy decoding
            next_token = next_token_logits.argmax().item()
        else:
            # Sample from distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        
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
def train_epoch(model, optimizer, scaler, scheduler, dataloader, device, epoch, args, itos):
    model.train()
    total_loss = 0.0
    total_output_tokens = 0  # Only count OUTPUT tokens for loss averaging
    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Train E{epoch}")
    optimizer.zero_grad()
    accumulation = args.accumulation_steps
    
    for step, (padded, raws, loss_mask) in pbar:
        padded = padded.to(device)  # (b, seq)
        loss_mask = loss_mask.to(device)  # (b, seq-1) - 1.0 for output tokens, 0.0 for input
        
        # SANITY CHECK: Print output tokens per batch on first step
        if step == 0 and epoch == 1:
            print(f"\n[Sanity Check] Training on OUTPUT tokens only (after separator)")
            print(f"[Sanity Check] Output tokens per example: {loss_mask.sum(dim=1).tolist()}")
            print(f"[Sanity Check] Total output tokens: {loss_mask.sum().item():.0f}, batch size: {padded.shape[0]}")
        
        # Prepare input/target for causal LM: input = all except last, target = all except first
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        
        with autocast('cuda'):
            logits = model(input_ids)  # (b, seq-1, vocab)
            vocab = logits.shape[-1]
            
            # Compute per-token loss (no reduction)
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, vocab), 
                target_ids.reshape(-1), 
                reduction='none',
                label_smoothing=args.label_smoothing
            )  # (b * seq-1,)
            
            # Reshape and apply mask: only compute loss on OUTPUT tokens (after separator)
            per_token_loss = per_token_loss.reshape(loss_mask.shape)  # (b, seq-1)
            masked_loss = per_token_loss * loss_mask
            
            # Average over masked (output) tokens only
            num_output_tokens = loss_mask.sum()
            if num_output_tokens > 0:
                loss = masked_loss.sum() / num_output_tokens
            else:
                # Fallback: if no output tokens (shouldn't happen), use mean
                loss = per_token_loss.mean()
            
            loss_val = loss.item()
        
        scaler.scale(loss / accumulation).backward()
        
        if (step + 1) % accumulation == 0:
            # Gradient clipping before optimizer step
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step scheduler per optimizer step for warmup
            optimizer.zero_grad()
        
        # Track loss weighted by output tokens only
        total_loss += loss_val * num_output_tokens.item()
        total_output_tokens += num_output_tokens.item()
        pbar.set_postfix({"loss": f"{(total_loss/max(1, total_output_tokens)):.4f}"})
    avg_loss = total_loss / max(1, total_output_tokens)
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
    
    for padded, raws, loss_mask in tqdm(dataloader, desc="Eval"):
        padded = padded.to(device)
        loss_mask = loss_mask.to(device)  # (b, seq-1) - 1.0 for output tokens, 0.0 for input
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        
        with autocast('cuda'):
            logits = model(input_ids)
            
            # Compute per-token loss (no reduction) - MATCHING train_epoch
            per_token_loss = F.cross_entropy(
                logits.reshape(-1, vocab), 
                target_ids.reshape(-1), 
                reduction='none'
            )  # (b * seq-1,)
            
            # Reshape and apply mask: only compute loss on OUTPUT tokens
            per_token_loss = per_token_loss.reshape(loss_mask.shape)  # (b, seq-1)
            masked_loss = per_token_loss * loss_mask
            
            # Average over output tokens only
            num_output_tokens = loss_mask.sum()
            if num_output_tokens > 0:
                loss = masked_loss.sum() / num_output_tokens
            else:
                loss = per_token_loss.mean()
        
        # Track loss weighted by output tokens only (matching train_epoch)
        total_loss += loss.item() * num_output_tokens.item()
        total_tokens += num_output_tokens.item()
        
        # Greedy decode predictions
        preds = logits.argmax(dim=-1)  # (b, seq-1)
        
        # Token accuracy: count correct OUTPUT tokens only (using loss_mask)
        output_mask = loss_mask.bool()  # True for output tokens
        correct_tokens += ((preds == target_ids) & output_mask).sum().item()
        counted_tokens += output_mask.sum().item()
        
        preds_list = preds.cpu().tolist()
        targets = target_ids.cpu().tolist()
        loss_mask_list = loss_mask.cpu().tolist()
        
        for i, (pred_ids, target_ids_list, raw) in enumerate(zip(preds_list, targets, raws)):
            if raw is None:
                continue
            
            mask_i = loss_mask_list[i]  # Get mask for this example
            
            # Show samples if requested
            if args.show_samples and i == 0 and random.random() < 0.1:  # Print snippet from ~10% of batches
                # Filter to OUTPUT tokens only (using loss_mask)
                inp_filtered = [t for t in input_ids[i].tolist() if t != args.pad_id]
                tgt_filtered = [t for t, m in zip(target_ids_list, mask_i) if m == 1.0]
                pred_filtered = [p for p, m in zip(pred_ids, mask_i) if m == 1.0]
                print(f"\n[Sample] Input:  {decode_ids_to_text(inp_filtered, itos)}")
                print(f"[Sample] Target: {decode_ids_to_text(tgt_filtered, itos)}")
                print(f"[Sample] Pred:   {decode_ids_to_text(pred_filtered, itos)}")
            
            task_type = extract_task_type(raw)
            task_total[task_type] += 1
            
            # Compare predicted tokens to target tokens (OUTPUT ONLY using loss_mask)
            pred_tokens = [p for p, m in zip(pred_ids, mask_i) if m == 1.0]
            target_tokens = [t for t, m in zip(target_ids_list, mask_i) if m == 1.0]
            
            # Exact match: entire OUTPUT sequence must match
            if pred_tokens == target_tokens:
                exact_matches += 1
                task_correct[task_type] += 1
            total_examples += 1
    
    avg_loss = total_loss / max(1, total_tokens)
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
    
    for padded, raws, loss_mask in tqdm(dataloader, desc="AutoReg Eval"):
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
    p.add_argument("--minimal_checkpoints", action="store_true", help="save only model weights (no optimizer/scaler), reduces size ~60-70%% for experiments")
    p.add_argument("--seed", type=int, default=1337)
    # Early stopping arguments
    p.add_argument("--patience", type=int, default=3, help="early stopping patience (epochs without improvement)")
    p.add_argument("--min_delta", type=float, default=0.001, help="minimum improvement threshold for early stopping")
    # Memory profiling
    p.add_argument("--profile_memory", action="store_true", help="log GPU memory usage each epoch")
    p.add_argument("--show_samples", action="store_true", help="show sample predictions during evaluation")
    p.add_argument("--autoreg_eval", action="store_true", help="use autoregressive generation for evaluation (slower but more realistic)")
    # Curriculum learning (LENGTH-based - legacy)
    p.add_argument("--curriculum", action="store_true", help="enable LENGTH-based curriculum (train on short sequences first)")
    p.add_argument("--curriculum_start", type=int, default=20, help="starting max sequence length for curriculum")
    p.add_argument("--curriculum_end", type=int, default=None, help="ending max sequence length (defaults to --max_len)")
    p.add_argument("--curriculum_epochs", type=int, default=5, help="epochs to transition from start to end length")
    # Task-based curriculum (SEMANTIC - recommended)
    p.add_argument("--task_curriculum", action="store_true", help="enable TASK-based curriculum (easy tasks first: copy→rev→add→sort→rel)")
    p.add_argument("--task_curriculum_epochs", type=int, default=None, help="epochs to reach final stage (default: 2 per stage)")
    # Training improvements
    p.add_argument("--warmup_steps", type=int, default=500, help="number of warmup steps for learning rate")
    p.add_argument("--max_grad_norm", type=float, default=1.0, help="max gradient norm for clipping")
    p.add_argument("--label_smoothing", type=float, default=0.1, help="label smoothing factor (0 = off)")
    # Memorization analysis
    p.add_argument("--compute_memorization", action="store_true", help="compute train EM and memorization metrics each epoch")
    p.add_argument("--memorization_samples", type=int, default=200, help="number of samples for train memorization check")
    # Model architecture
    p.add_argument("--use_swiglu", action="store_true", help="use SwiGLU FFN instead of standard GELU FFN")
    # Sampling parameters (for autoregressive evaluation)
    p.add_argument("--temperature", type=float, default=1.0, help="sampling temperature (1.0=normal, <1=confident, >1=random)")
    p.add_argument("--top_p", type=float, default=1.0, help="nucleus sampling threshold (1.0=disabled)")
    p.add_argument("--top_k", type=int, default=0, help="top-k sampling (0=disabled)")
    # Resume training
    p.add_argument("--resume", type=str, default=None, help="path to checkpoint to resume training from (e.g., runs/step1/ckpt_epoch5.pth)")
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
    sep_token_id = stoi.get("|", None)  # Separator token for loss masking
    if sep_token_id is None:
        print("WARNING: Separator token '|' not found in tokenizer. Loss will be computed on all tokens.")
    else:
        print(f"Using separator token ID={sep_token_id} for output-only loss masking.")
    
    vocab_size = len(stoi)
    args.pad_id = pad_id
    args.sep_token_id = sep_token_id
    args.vocab_size = vocab_size

    # dataset & loader
    base_dataset = PickleDataset(args.data, stoi, args.max_len)
    print(f"Training dataset loaded: {len(base_dataset)} examples")
    
    # Curriculum learning setup
    task_curriculum_dataset = None
    curriculum_dataset = None
    
    if args.task_curriculum:
        # Task-based curriculum (RECOMMENDED)
        task_curriculum_dataset = TaskCurriculumDataset(base_dataset, max_stage=0)
        dataset = task_curriculum_dataset
        stats = task_curriculum_dataset.get_stats()
        print(f"[TaskCurriculum] Starting at stage 0: {stats['stage_info']['description']}")
        print(f"[TaskCurriculum] Using {stats['count']} examples, tasks: {stats['tasks']}")
        
        # Calculate epochs per stage
        num_stages = len(TASK_DIFFICULTY_STAGES)
        if args.task_curriculum_epochs is None:
            # Default: 2 epochs per stage
            args.task_curriculum_epochs = num_stages * 2
        epochs_per_stage = max(1, args.task_curriculum_epochs // num_stages)
        print(f"[TaskCurriculum] {num_stages} stages, advancing every {epochs_per_stage} epochs")
    elif args.curriculum:
        # Length-based curriculum (legacy)
        if args.curriculum_end is None:
            args.curriculum_end = args.max_len
        curriculum_dataset = CurriculumDataset(base_dataset, args.curriculum_start)
        dataset = curriculum_dataset
        stats = curriculum_dataset.get_length_stats()
        print(f"[LengthCurriculum] Starting with max_len={args.curriculum_start}, "
              f"using {stats['count']} examples (avg_len={stats['avg_len']:.1f})")
    else:
        dataset = base_dataset
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len, sep_token_id=sep_token_id), num_workers=2, pin_memory=True)

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
                                collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len, sep_token_id=sep_token_id), num_workers=2, pin_memory=True)
        print(f"Validation loader ready: {len(val_dataset)} examples from {val_path}")
    else:
        print("WARNING: No validation data found. Training without validation (early stopping disabled).")

    # build model
    model = MiniGPT(vocab_size=vocab_size, max_len=args.max_len, d_model=args.d_model,
                    n_heads=args.n_heads, n_layers=args.n_layers, d_ff=args.d_ff,
                    dropout=args.dropout, use_checkpoint=args.use_checkpoint,
                    use_swiglu=args.use_swiglu).to(device)
    ffn_type = "SwiGLU" if args.use_swiglu else "GELU"
    print(f"Model params: {sum(p.numel() for p in model.parameters()):,} (FFN: {ffn_type})")

    # optimizer, scaler, scheduler with warmup
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler('cuda')  # Updated: specify device type
    
    # Calculate total training steps for scheduler
    steps_per_epoch = len(dataloader) // args.accumulation_steps
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 4)  # Cap warmup at 25% of total
    
    # Linear warmup + Cosine decay scheduler
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        else:
            # Cosine decay
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))  # min LR = 10% of max
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    print(f"Scheduler: {warmup_steps} warmup steps, {total_steps} total steps")

    # Early stopping state
    # TF-based tracking (for diagnostics)
    best_val_loss = float("inf")
    best_TF_model_state = None
    
    # AR-based tracking (ground truth for actual performance)
    best_AR_edit_dist = float("inf")  # Lower is better
    best_AR_model_state = None
    
    patience_counter = 0
    start_epoch = 1

    # Resume from checkpoint if specified
    if args.resume and os.path.exists(args.resume):
        print(f"\n[Resume] Loading checkpoint from {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        scaler.load_state_dict(checkpoint["scaler_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))
        best_AR_edit_dist = checkpoint.get("best_AR_edit_dist", float("inf"))
        start_epoch = checkpoint.get("epoch", 0) + 1
        print(f"[Resume] Resuming from epoch {start_epoch}, best_TF_loss={best_val_loss:.4f}, best_AR_edit={best_AR_edit_dist:.4f}")
        # Load history if exists
        history_path = os.path.join(args.out_dir, "training_history.json")
        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            print(f"[Resume] Loaded training history ({len(history.get('train_loss', []))} epochs)")
    elif args.resume:
        print(f"WARNING: Checkpoint {args.resume} not found, starting fresh.")

    # training loop
    history = history if args.resume and os.path.exists(args.resume) else {"train_loss": [], "val_loss": [], "val_exact": [], "val_token_acc": [], "task_accuracy": [], "lr": [], "train_exact": [], "train_val_gap": []}

    # TensorBoard writer for real-time visualization
    tb_writer = None
    if TENSORBOARD_AVAILABLE:
        tb_log_dir = os.path.join(args.out_dir, "tensorboard")
        tb_writer = SummaryWriter(log_dir=tb_log_dir)
        print(f"TensorBoard logging to: {tb_log_dir}")
        print(f"View with: tensorboard --logdir {tb_log_dir}")

    # Initial memory profiling
    if args.profile_memory:
        torch.cuda.reset_peak_memory_stats()
        log_memory("[Initial] ")

    for epoch in range(start_epoch, args.epochs+1):
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
                                        collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len, sep_token_id=sep_token_id), 
                                        num_workers=2, pin_memory=True)
            
            stats = curriculum_dataset.get_length_stats()
            print(f"[LengthCurriculum] Epoch {epoch}: max_len={new_max_len}, "
                  f"examples={stats['count']}, avg_len={stats['avg_len']:.1f}")
            
            # Log curriculum to TensorBoard
            if tb_writer is not None:
                tb_writer.add_scalar("Curriculum/max_length", new_max_len, epoch)
                # Add vertical line marker when curriculum changes
                if new_count != old_count:
                    tb_writer.add_scalar("Curriculum/length_change", 1, epoch)
        
        # Task-based curriculum: advance stage based on epoch
        if task_curriculum_dataset is not None:
            num_stages = len(TASK_DIFFICULTY_STAGES)
            epochs_per_stage = max(1, args.task_curriculum_epochs // num_stages)
            new_stage = min((epoch - 1) // epochs_per_stage, num_stages - 1)
            
            if new_stage != task_curriculum_dataset.max_stage:
                old_count = len(task_curriculum_dataset)
                task_curriculum_dataset.set_stage(new_stage)
                new_count = len(task_curriculum_dataset)
                
                stats = task_curriculum_dataset.get_stats()
                print(f"[TaskCurriculum] Epoch {epoch}: Advanced to stage {new_stage}")
                print(f"[TaskCurriculum] {stats['stage_info']['description']}: {stats['count']} examples")
                print(f"[TaskCurriculum] Tasks now included: {list(stats['tasks'].keys())}")
                
                # Rebuild dataloader with new curriculum
                if new_count != old_count:
                    dataloader = DataLoader(task_curriculum_dataset, batch_size=args.batch_size, shuffle=True,
                                            collate_fn=partial(collate_fn, pad_id=pad_id, max_len=args.max_len, sep_token_id=sep_token_id), 
                                            num_workers=2, pin_memory=True)
                
                # Log curriculum stage to TensorBoard with vertical line marker
                if tb_writer is not None:
                    tb_writer.add_scalar("Curriculum/task_stage", new_stage, epoch)
                    tb_writer.add_scalar("Curriculum/stage_change", 1, epoch)  # Marker for vertical line
                    tb_writer.add_text("Curriculum/stage_description", stats['stage_info']['description'], epoch)
        
        train_loss = train_epoch(model, optimizer, scaler, scheduler, dataloader, device, epoch, args, itos)
        
        # Get current learning rate (scheduler stepped per optimizer step in train_epoch)
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
            
            # Optional: Autoregressive evaluation (slower but more realistic)
            # Run every 10 epochs or on the last epoch to save time
            autoreg_exact, autoreg_token_acc, autoreg_edit_dist, autoreg_task_acc = None, None, None, None
            if args.autoreg_eval and (epoch % 10 == 0 or epoch == args.epochs):
                print(f"  [AR Eval] Running autoregressive evaluation (epoch {epoch})...")
                autoreg_exact, autoreg_token_acc, autoreg_edit_dist, autoreg_task_acc = evaluate_autoregressive(
                    model, val_loader, device, args, stoi, itos
                )
                history.setdefault("autoreg_exact", []).append(autoreg_exact)
                history.setdefault("autoreg_token_acc", []).append(autoreg_token_acc)
                history.setdefault("autoreg_edit_dist", []).append(autoreg_edit_dist)
            
            # Train memorization check (optional, slower)
            train_em, train_val_gap = None, None
            if args.compute_memorization and METRICS_AVAILABLE:
                train_mem = compute_train_exact_match(
                    model, dataloader, device, args.pad_id, 
                    n_samples=args.memorization_samples, itos=itos
                )
                train_em = train_mem["train_exact_match"]
                history["train_exact"].append(train_em)
                train_val_gap = train_em - val_exact
                history["train_val_gap"].append(train_val_gap)
            
            # ============ TENSORBOARD LOGGING ============
            if tb_writer is not None:
                # Core metrics
                tb_writer.add_scalar("Loss/train", train_loss, epoch)
                tb_writer.add_scalar("Loss/val_TF", val_loss, epoch)
                tb_writer.add_scalar("LearningRate", current_lr, epoch)
                
                # Teacher-forced metrics (diagnostic)
                tb_writer.add_scalar("TF/val_exact_match", val_exact, epoch)
                tb_writer.add_scalar("TF/val_token_accuracy", token_acc, epoch)
                
                # Autoregressive metrics (ground truth) - when available
                if autoreg_exact is not None:
                    tb_writer.add_scalar("AR/val_exact_match", autoreg_exact, epoch)
                    tb_writer.add_scalar("AR/val_token_accuracy", autoreg_token_acc, epoch)
                    tb_writer.add_scalar("AR/val_edit_distance", autoreg_edit_dist, epoch)
                
                # Memorization metrics
                if train_em is not None:
                    tb_writer.add_scalar("Memorization/train_exact_match", train_em, epoch)
                    tb_writer.add_scalar("Memorization/train_val_gap", train_val_gap, epoch)
                
                tb_writer.flush()
            
            # ============ STRUCTURED EPOCH SUMMARY ============
            print()
            print(f"{'='*60}")
            print(f"  EPOCH {epoch} SUMMARY")
            print(f"{'='*60}")
            print()
            
            # Core Metrics
            print("  📊 CORE METRICS")
            print(f"  {'─'*40}")
            print(f"  {'Train Loss:':<20} {train_loss:.4f}")
            print(f"  {'Val Loss (TF-NLL):':<20} {val_loss:.4f}  [diagnostic only]")
            print(f"  {'Learning Rate:':<20} {current_lr:.2e}")
            print()
            
            # Teacher-Forced Evaluation (DIAGNOSTIC ONLY)
            print("  📝 TEACHER-FORCED (diagnostic, not performance)")
            print(f"  {'─'*40}")
            print(f"  {'TF-SeqMatch:':<20} {val_exact:.4f}")
            print(f"  {'TF-TokenAcc:':<20} {token_acc:.4f}")
            print()
            
            # Autoregressive Evaluation (if enabled)
            if args.autoreg_eval and autoreg_exact is not None:
                print("  🔄 AUTOREGRESSIVE (GROUND TRUTH)")
                print(f"  {'─'*40}")
                print(f"  {'AR-ExactMatch:':<20} {autoreg_exact:.4f}  {'✓' if autoreg_exact > 0.1 else '✗'}")
                print(f"  {'AR-TokenAcc:':<20} {autoreg_token_acc:.4f}")
                print(f"  {'AR-EditDist:':<20} {autoreg_edit_dist:.4f}  [← EARLY STOP METRIC]")
                print()
            
            # Memorization Check (if enabled)
            if train_em is not None:
                print("  🧠 MEMORIZATION CHECK")
                print(f"  {'─'*40}")
                print(f"  {'Train Exact Match:':<20} {train_em:.4f}")
                print(f"  {'Train-Val Gap:':<20} {train_val_gap:+.4f}  {'⚠️ HIGH' if train_val_gap > 0.3 else '✓'}")
                print()
            
            # Per-Task Breakdown
            if task_accuracy:
                print("  📋 PER-TASK ACCURACY (Teacher-Forced)")
                print(f"  {'─'*40}")
                for task, acc in sorted(task_accuracy.items()):
                    bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
                    print(f"  {task:<10} {bar} {acc:.2%}")
                print()
            
            # Autoregressive Per-Task (if enabled)
            if args.autoreg_eval and autoreg_task_acc:
                print("  📋 PER-TASK ACCURACY (Autoregressive)")
                print(f"  {'─'*40}")
                for task, acc in sorted(autoreg_task_acc.items()):
                    bar = '█' * int(acc * 20) + '░' * (20 - int(acc * 20))
                    print(f"  {task:<10} {bar} {acc:.2%}")
                print()
            
            print(f"{'='*60}")
            print()
            
            # ============ EARLY STOPPING LOGIC ============
            # When autoreg_eval is enabled, use AR edit distance (ground truth)
            # Otherwise fall back to TF loss (diagnostic mode)
            
            if args.autoreg_eval and autoreg_edit_dist is not None:
                # AR-based early stopping (CORRECT - measures actual task learning)
                if autoreg_edit_dist < best_AR_edit_dist - args.min_delta:
                    best_AR_edit_dist = autoreg_edit_dist
                    patience_counter = 0
                    best_AR_model_state = model.state_dict().copy()
                    print(f"[Epoch {epoch}] 🎯 New best AR edit distance: {autoreg_edit_dist:.4f}")
                else:
                    patience_counter += 1
                    print(f"[Epoch {epoch}] No AR improvement. Patience: {patience_counter}/{args.patience}")
            else:
                # TF-based early stopping (fallback when AR eval disabled)
                if val_loss < best_val_loss - args.min_delta:
                    best_val_loss = val_loss
                    patience_counter = 0
                    best_TF_model_state = model.state_dict().copy()
                    print(f"[Epoch {epoch}] New best TF loss: {val_loss:.4f} (⚠️ diagnostic only)")
                else:
                    patience_counter += 1
                    print(f"[Epoch {epoch}] No TF improvement. Patience: {patience_counter}/{args.patience}")
            
            # Also track TF best independently (for debugging)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_TF_model_state = model.state_dict().copy()
                
            if patience_counter >= args.patience:
                if args.autoreg_eval:
                    print(f"\nEarly stopping triggered after {epoch} epochs!")
                    print(f"Best AR edit distance: {best_AR_edit_dist:.4f}")
                    if best_AR_model_state is not None:
                        model.load_state_dict(best_AR_model_state)
                        print("Restored best AR model weights.")
                else:
                    print(f"\nEarly stopping triggered after {epoch} epochs!")
                    print(f"Best TF loss: {best_val_loss:.4f} (⚠️ TF metric, consider enabling --autoreg_eval)")
                    if best_TF_model_state is not None:
                        model.load_state_dict(best_TF_model_state)
                        print("Restored best TF model weights.")
                break

        # save checkpoint
        if epoch % args.save_every == 0:
            if args.minimal_checkpoints:
                # Minimal checkpoint for experiments (saves ~60-70% space)
                cp = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_AR_edit_dist": best_AR_edit_dist,
                    "args": vars(args)
                }
            else:
                # Full checkpoint for resumable training
                cp = {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_AR_edit_dist": best_AR_edit_dist,
                    "args": vars(args)
                }
            cp_path = os.path.join(args.out_dir, f"ckpt_epoch{epoch}.pth")
            torch.save(cp, cp_path)
            print(f"Saved checkpoint: {cp_path}")

    # Save best models separately (DUAL CHECKPOINT SYSTEM)
    # 1. Best TF model (for debugging/diagnostics)
    if best_TF_model_state is not None:
        best_TF_path = os.path.join(args.out_dir, "best_TF.pth")
        torch.save({"model_state": best_TF_model_state, "val_loss": best_val_loss, "metric": "TF_loss", "args": vars(args)}, best_TF_path)
        print(f"Saved best TF model (diagnostic): {best_TF_path}")
    
    # 2. Best AR model (ACTUAL PERFORMANCE - use this one!)
    if best_AR_model_state is not None:
        best_AR_path = os.path.join(args.out_dir, "best_AR.pth")
        torch.save({"model_state": best_AR_model_state, "AR_edit_dist": best_AR_edit_dist, "metric": "AR_edit_distance", "args": vars(args)}, best_AR_path)
        print(f"Saved best AR model (ground truth): {best_AR_path} ⭐")
        # Also save as best_model.pth for backward compatibility
        best_path = os.path.join(args.out_dir, "best_model.pth")
        torch.save({"model_state": best_AR_model_state, "AR_edit_dist": best_AR_edit_dist, "metric": "AR_edit_distance", "args": vars(args)}, best_path)
    elif best_TF_model_state is not None:
        # Fallback: if no AR model, use TF model (with warning)
        best_path = os.path.join(args.out_dir, "best_model.pth")
        torch.save({"model_state": best_TF_model_state, "val_loss": best_val_loss, "metric": "TF_loss", "args": vars(args)}, best_path)
        print(f"⚠️ Saved best TF model as best_model.pth (enable --autoreg_eval for proper AR checkpoint)")

    # final save results
    with open(os.path.join(args.out_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("\nTraining complete. History written.")
    
    # Generate comprehensive plots using new plots module
    if METRICS_AVAILABLE:
        try:
            plot_paths = generate_all_training_plots(history, args.out_dir)
            print(f"Generated {len(plot_paths)} visualization plots.")
        except Exception as e:
            print(f"Warning: Could not generate all plots: {e}")
            # Fallback to basic plot
            plot_loss_curves(history, args.out_dir)
    else:
        # Fallback to basic plot
        plot_loss_curves(history, args.out_dir)
    
    # Final memory summary
    if args.profile_memory:
        log_memory("[Final] ")
        print(f"Peak VRAM usage: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    
    # Close TensorBoard writer
    if tb_writer is not None:
        tb_writer.close()
        print(f"TensorBoard logs saved to: {os.path.join(args.out_dir, 'tensorboard')}")

if __name__ == "__main__":
    main()