# metrics.py
# Additional metrics for memorization vs. generalization research
# Complements evaluate_ood.py which handles OOD and duplication sensitivity

import torch
import torch.nn.functional as F
from typing import List, Set, Dict, Tuple, Optional
from collections import Counter
import random


# ---------- N-gram Overlap ----------
def extract_ngrams(text: str, n: int = 4) -> Set[str]:
    """Extract all n-grams from a text string."""
    if len(text) < n:
        return set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}


def compute_ngram_overlap(generated_texts: List[str], train_corpus: List[str], n: int = 4) -> Dict[str, float]:
    """
    Compute n-gram overlap between generated texts and training corpus.
    
    Args:
        generated_texts: List of generated output strings
        train_corpus: List of training data strings
        n: N-gram size (default 4)
    
    Returns:
        Dict with overlap metrics:
        - avg_overlap: Average fraction of generated n-grams found in training
        - max_overlap: Maximum overlap for any single generation
        - min_overlap: Minimum overlap (most novel generation)
        - fully_novel_frac: Fraction of generations with 0 overlap
    """
    # Build training n-gram set (do once)
    train_ngrams = set()
    for text in train_corpus:
        train_ngrams.update(extract_ngrams(text, n))
    
    if not train_ngrams:
        return {"avg_overlap": 0.0, "max_overlap": 0.0, "min_overlap": 0.0, "fully_novel_frac": 1.0}
    
    overlaps = []
    fully_novel = 0
    
    for gen_text in generated_texts:
        gen_ngrams = extract_ngrams(gen_text, n)
        if not gen_ngrams:
            overlaps.append(0.0)
            fully_novel += 1
            continue
        
        # What fraction of generated n-grams appear in training?
        overlap_count = len(gen_ngrams & train_ngrams)
        overlap_frac = overlap_count / len(gen_ngrams)
        overlaps.append(overlap_frac)
        
        if overlap_count == 0:
            fully_novel += 1
    
    if not overlaps:
        return {"avg_overlap": 0.0, "max_overlap": 0.0, "min_overlap": 0.0, "fully_novel_frac": 1.0}
    
    return {
        "avg_overlap": sum(overlaps) / len(overlaps),
        "max_overlap": max(overlaps),
        "min_overlap": min(overlaps),
        "fully_novel_frac": fully_novel / len(generated_texts)
    }


# ---------- Edit Distance ----------
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute Levenshtein (edit) distance between two strings.
    Uses dynamic programming with space optimization.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    
    if len(s2) == 0:
        return len(s1)
    
    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return previous_row[-1]


def normalized_edit_distance(s1: str, s2: str) -> float:
    """Normalized edit distance: 0.0 = identical, 1.0 = completely different."""
    if len(s1) == 0 and len(s2) == 0:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return dist / max(len(s1), len(s2))


def compute_min_edit_distance_to_train(generated: str, train_samples: List[str], 
                                        max_samples: int = 100) -> float:
    """
    Find minimum edit distance from a generated string to any training sample.
    
    Args:
        generated: Generated output string
        train_samples: List of training samples to compare against
        max_samples: Max samples to check (for efficiency)
    
    Returns:
        Minimum normalized edit distance (0.0 = exact copy, 1.0 = totally novel)
    """
    if not train_samples:
        return 1.0
    
    # Sample if corpus is large
    samples = train_samples
    if len(train_samples) > max_samples:
        samples = random.sample(train_samples, max_samples)
    
    min_dist = 1.0
    for train_text in samples:
        dist = normalized_edit_distance(generated, train_text)
        min_dist = min(min_dist, dist)
        if min_dist == 0.0:  # Exact match found
            break
    
    return min_dist


def compute_edit_distance_stats(generated_texts: List[str], train_samples: List[str],
                                 max_train_samples: int = 500) -> Dict[str, float]:
    """
    Compute edit distance statistics for a batch of generated texts.
    
    Returns:
        Dict with:
        - avg_min_distance: Average minimum distance to training
        - exact_copies: Fraction that are exact copies (dist = 0)
        - highly_similar: Fraction with dist < 0.2
        - novel: Fraction with dist > 0.5
    """
    if not generated_texts or not train_samples:
        return {"avg_min_distance": 1.0, "exact_copies": 0.0, "highly_similar": 0.0, "novel": 1.0}
    
    # Sample training data for efficiency
    if len(train_samples) > max_train_samples:
        train_subset = random.sample(train_samples, max_train_samples)
    else:
        train_subset = train_samples
    
    distances = []
    for gen_text in generated_texts:
        dist = compute_min_edit_distance_to_train(gen_text, train_subset)
        distances.append(dist)
    
    exact_copies = sum(1 for d in distances if d == 0.0)
    highly_similar = sum(1 for d in distances if d < 0.2)
    novel = sum(1 for d in distances if d > 0.5)
    
    return {
        "avg_min_distance": sum(distances) / len(distances),
        "exact_copies": exact_copies / len(distances),
        "highly_similar": highly_similar / len(distances),
        "novel": novel / len(distances)
    }


# ---------- Output Entropy (Creativity/Diversity) ----------
def compute_output_entropy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Compute entropy of the output distribution.
    Higher entropy = more uncertain/diverse, Lower = more confident/memorized.
    
    Args:
        logits: Shape (batch, seq, vocab) or (seq, vocab)
        temperature: Temperature for softmax scaling
    
    Returns:
        Per-position entropy, shape (batch, seq) or (seq,)
    """
    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=-1)
    log_probs = F.log_softmax(scaled_logits, dim=-1)
    entropy = -torch.sum(probs * log_probs, dim=-1)
    return entropy


def compute_generation_entropy_stats(logits: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, float]:
    """
    Compute entropy statistics for a batch of generations.
    
    Args:
        logits: Shape (batch, seq, vocab)
        pad_mask: Boolean mask where True = valid position (shape: batch, seq)
    
    Returns:
        Dict with entropy statistics
    """
    entropy = compute_output_entropy(logits)  # (batch, seq)
    
    if pad_mask is not None:
        # Mask out padding positions
        entropy = entropy * pad_mask.float()
        valid_counts = pad_mask.sum(dim=-1).float()
        mean_entropy = entropy.sum(dim=-1) / valid_counts.clamp(min=1)
    else:
        mean_entropy = entropy.mean(dim=-1)
    
    return {
        "avg_entropy": mean_entropy.mean().item(),
        "min_entropy": mean_entropy.min().item(),
        "max_entropy": mean_entropy.max().item(),
        "std_entropy": mean_entropy.std().item()
    }


# ---------- Train Exact Match (Memorization Check) ----------
@torch.no_grad()
def compute_train_exact_match(model, dataloader, device, pad_id: int, 
                               n_samples: int = 200, itos: dict = None) -> Dict[str, float]:
    """
    Compute exact match rate on training data.
    High train EM + low val EM = memorization.
    
    Args:
        model: The language model
        dataloader: Training data loader
        device: torch device
        pad_id: Padding token ID
        n_samples: Number of samples to check
        itos: ID to string mapping (optional, for collecting outputs)
    
    Returns:
        Dict with train exact match stats
    """
    model.eval()
    
    exact_matches = 0
    total = 0
    correct_tokens = 0
    total_tokens = 0
    
    for padded, raws in dataloader:
        if total >= n_samples:
            break
        
        padded = padded.to(device)
        input_ids = padded[:, :-1]
        target_ids = padded[:, 1:]
        
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)  # (batch, seq)
        
        # Per-example check
        for i in range(min(preds.shape[0], n_samples - total)):
            pred_ids = preds[i].cpu().tolist()
            target_ids_list = target_ids[i].cpu().tolist()
            
            # Filter padding
            pred_tokens = [p for p, t in zip(pred_ids, target_ids_list) if t != pad_id]
            target_tokens = [t for t in target_ids_list if t != pad_id]
            
            # Exact match
            if pred_tokens == target_tokens:
                exact_matches += 1
            
            # Token accuracy
            for p, t in zip(pred_tokens, target_tokens):
                total_tokens += 1
                if p == t:
                    correct_tokens += 1
            
            total += 1
    
    return {
        "train_exact_match": exact_matches / total if total > 0 else 0.0,
        "train_token_accuracy": correct_tokens / total_tokens if total_tokens > 0 else 0.0,
        "n_samples": total
    }


# ---------- Self-BLEU (Output Diversity) ----------
def compute_self_bleu(generated_texts: List[str], n: int = 4, sample_size: int = 100) -> float:
    """
    Compute Self-BLEU: average BLEU score of each generation against all others.
    Lower Self-BLEU = more diverse outputs.
    
    Args:
        generated_texts: List of generated strings
        n: Max n-gram for BLEU
        sample_size: Max samples to use (for efficiency)
    
    Returns:
        Average self-BLEU score (0-1, lower = more diverse)
    """
    if len(generated_texts) < 2:
        return 0.0
    
    # Sample for efficiency
    if len(generated_texts) > sample_size:
        texts = random.sample(generated_texts, sample_size)
    else:
        texts = generated_texts
    
    # Simple n-gram precision as proxy for BLEU
    total_bleu = 0.0
    count = 0
    
    for i, text in enumerate(texts):
        if not text:
            continue
        
        text_ngrams = extract_ngrams(text, n)
        if not text_ngrams:
            continue
        
        # Compare to all other texts
        other_ngrams = set()
        for j, other in enumerate(texts):
            if i != j:
                other_ngrams.update(extract_ngrams(other, n))
        
        if not other_ngrams:
            continue
        
        # N-gram overlap as BLEU proxy
        overlap = len(text_ngrams & other_ngrams) / len(text_ngrams)
        total_bleu += overlap
        count += 1
    
    return total_bleu / count if count > 0 else 0.0


# ---------- Unified Metrics Computation ----------
def compute_memorization_metrics(
    model,
    train_loader,
    val_loader,
    device,
    pad_id: int,
    stoi: dict,
    itos: dict,
    train_corpus: List[str],
    n_samples: int = 200
) -> Dict[str, float]:
    """
    Compute comprehensive memorization metrics.
    
    Returns:
        Dict with all memorization-related metrics
    """
    metrics = {}
    
    # Train exact match (memorization signal)
    train_em = compute_train_exact_match(model, train_loader, device, pad_id, n_samples, itos)
    metrics.update({f"train_{k}": v for k, v in train_em.items() if k != "n_samples"})
    
    # TODO: Val exact match (compute separately in training loop)
    
    return metrics


# ---------- Test Functions ----------
def test_metrics():
    """Quick test of metric functions."""
    print("Testing metrics.py...")
    
    # Test n-gram extraction
    ngrams = extract_ngrams("hello world", 4)
    assert "hell" in ngrams
    assert "ello" in ngrams
    print("  ✓ N-gram extraction works")
    
    # Test n-gram overlap
    train = ["the quick brown fox", "jumps over the lazy dog"]
    generated = ["the quick fox", "something new"]
    overlap = compute_ngram_overlap(generated, train, n=4)
    assert 0 <= overlap["avg_overlap"] <= 1
    print(f"  ✓ N-gram overlap: {overlap}")
    
    # Test edit distance
    dist = normalized_edit_distance("hello", "hallo")
    assert 0 < dist < 1
    print(f"  ✓ Edit distance (hello->hallo): {dist:.2f}")
    
    # Test entropy
    logits = torch.randn(2, 10, 100)
    entropy_stats = compute_generation_entropy_stats(logits)
    assert entropy_stats["avg_entropy"] > 0
    print(f"  ✓ Entropy stats: avg={entropy_stats['avg_entropy']:.2f}")
    
    # Test self-BLEU
    texts = ["hello world", "hello there", "goodbye world", "something else"]
    self_bleu = compute_self_bleu(texts, n=3)
    print(f"  ✓ Self-BLEU: {self_bleu:.2f}")
    
    print("\nAll metrics tests passed!")


if __name__ == "__main__":
    test_metrics()
