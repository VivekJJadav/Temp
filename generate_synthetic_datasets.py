# generate_synthetic_datasets_final.py
# Final integrated dataset generator (ready-to-run)
# Produces: train_normal.pkl, train_dedup.pkl, train_duplicated.pkl, val.pkl, tokenizer.json, dataset_stats.json

import os
import json
import random
import hashlib
import pickle
from collections import Counter, defaultdict
from typing import List, Iterable, Tuple

import torch
import numpy as np

# -----------------------
# Seed / reproducibility
# -----------------------
SEED = 1337
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.backends.mps.is_available():
    try:
        torch.mps.manual_seed_all(SEED)
    except Exception:
        pass

# -----------------------
# Hyperparameters / Config
# -----------------------
OUT_DIR = "data/processed"
os.makedirs(OUT_DIR, exist_ok=True)

NUM_TOTAL_EXAMPLES = 10000  # lower for quick runs; increase later
TRAIN_RATIO = 0.9

MIX = {"sorting": 0.2, "reversing": 0.2, "addition": 0.2, "copying": 0.2, "relations": 0.2}

# Task-specific (training)
SORT_MIN_L, SORT_MAX_L = 4, 12
REV_MIN_L, REV_MAX_L = 3, 12
ADD_MIN_DIGITS, ADD_MAX_DIGITS = 1, 3
COPY_MIN_L, COPY_MAX_L = 2, 6
COPY_MIN_REPEAT, COPY_MAX_REPEAT = 2, 4
REL_MIN_HOPS, REL_MAX_HOPS = 1, 2

SEPARATOR = " | "
TOKEN_SEP = " "

# Duplication control
DUPLICATE_REPEAT_TOP_K = 200
DUPLICATE_MULTIPLIER = 10

# Dedup control
USE_APPROX_DEDUPE = False

# Tokenizer
MAX_VOCAB = 200

# -----------------------
# Utilities
# -----------------------
def sha1hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

# -----------------------
# Task generators
# -----------------------
def gen_sort_example(min_l=SORT_MIN_L, max_l=SORT_MAX_L) -> str:
    L = random.randint(min_l, max_l)
    seq = [str(random.randint(0, 9)) for _ in range(L)]
    inp = TOKEN_SEP.join(seq)
    out = TOKEN_SEP.join(sorted(seq, key=lambda x: int(x)))
    return f"[sort] {inp}{SEPARATOR}{out}"

def gen_reverse_example(min_l=REV_MIN_L, max_l=REV_MAX_L) -> str:
    L = random.randint(min_l, max_l)
    seq = [str(random.randint(0, 9)) for _ in range(L)]
    inp = TOKEN_SEP.join(seq)
    out = TOKEN_SEP.join(list(reversed(seq)))
    return f"[rev] {inp}{SEPARATOR}{out}"

def gen_add_example(min_digits=ADD_MIN_DIGITS, max_digits=ADD_MAX_DIGITS) -> str:
    a_digits = random.randint(min_digits, max_digits)
    b_digits = random.randint(min_digits, max_digits)
    a = random.randint(0, 10**a_digits - 1)
    b = random.randint(0, 10**b_digits - 1)
    inp = f"{a} + {b}"
    out = str(a + b)
    return f"[add] {inp}{SEPARATOR}{out}"

def gen_copy_example(min_l=COPY_MIN_L, max_l=COPY_MAX_L, min_rep=COPY_MIN_REPEAT, max_rep=COPY_MAX_REPEAT) -> str:
    L = random.randint(min_l, max_l)
    alphabet = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    seq = "".join(random.choice(alphabet) for _ in range(L))
    rep = random.randint(min_rep, max_rep)
    inp = seq
    out = seq * rep
    return f"[copy] {inp}{SEPARATOR}{out}"

def gen_relations_example(min_hops=REL_MIN_HOPS, max_hops=REL_MAX_HOPS) -> str:
    entities = [f"E{i}" for i in range(1, 1000)]
    hops = random.randint(min_hops, max_hops)
    chosen = random.sample(entities, hops + 1)
    lines = []
    for i in range(hops):
        lines.append(f"{chosen[i]} is parent of {chosen[i+1]}.")
    question = f"Who is the ancestor {hops} of {chosen[-1]}?"
    answer = chosen[0]
    inp = " ".join(lines) + " " + question
    return f"[rel] {inp}{SEPARATOR}{answer}"

# Optional OOD generators (for val)
def gen_ood_subtraction() -> str:
    a = random.randint(10, 999)
    b = random.randint(1, a)
    inp = f"{a} - {b}"
    out = str(a - b)
    return f"[sub] {inp}{SEPARATOR}{out}"

def gen_ood_palindrome() -> str:
    L = random.randint(3, 8)
    if random.random() < 0.5:
        half = [str(random.randint(0, 9)) for _ in range(L // 2)]
        if L % 2 == 1:
            seq = half + [str(random.randint(0, 9))] + list(reversed(half))
        else:
            seq = half + list(reversed(half))
        answer = "yes"
    else:
        seq = [str(random.randint(0, 9)) for _ in range(L)]
        answer = "yes" if seq == list(reversed(seq)) else "no"
    inp = TOKEN_SEP.join(seq)
    return f"[pal] {inp} is palindrome?{SEPARATOR}{answer}"

# -----------------------
# Pool builder
# -----------------------
def build_pool(total_examples: int, mix: dict) -> List[Tuple[str,str]]:
    pool = []
    total_w = sum(mix.values())
    weights = {k: v / total_w for k, v in mix.items()}
    counts = {k: int(weights[k] * total_examples) for k in weights}
    assigned = sum(counts.values())
    for _ in range(total_examples - assigned):
        r = random.random() * total_w
        acc = 0
        for k in mix:
            acc += mix[k]
            if r <= acc:
                counts[k] += 1
                break
    print("Counts per task:", counts)
    for k, c in counts.items():
        for _ in range(c):
            if k == "sorting":
                pool.append(("sorting", gen_sort_example()))
            elif k == "reversing":
                pool.append(("reversing", gen_reverse_example()))
            elif k == "addition":
                pool.append(("addition", gen_add_example()))
            elif k == "copying":
                pool.append(("copying", gen_copy_example()))
            elif k == "relations":
                pool.append(("relations", gen_relations_example()))
    random.shuffle(pool)
    return pool

# -----------------------
# Tokenizer builder
# -----------------------
def build_char_tokenizer_from_texts(texts: Iterable[str], max_vocab=MAX_VOCAB):
    counter = Counter()
    for t in texts:
        counter.update(list(t))
    most = [ch for ch, _ in counter.most_common(max_vocab)]
    base = [" ", "|", "+", "-", "*", "/", "?", ".", ",", ":", ";", "(", ")", "'", '"',
            "[", "]",  # For task prefixes like [sort], [add], etc.
            "0","1","2","3","4","5","6","7","8","9"]
    for b in base:
        if b not in most:
            most.append(b)
    vocab = ["<pad>", "<unk>"] + most
    stoi = {s:i for i,s in enumerate(vocab)}
    itos = {i:s for s,i in stoi.items()}
    return stoi, itos

def encode_text(t: str, stoi: dict) -> List[int]:
    return [stoi.get(ch, stoi["<unk>"]) for ch in t]

# -----------------------
# Dedupe exact only
# -----------------------
def dedupe_exact(strings: List[str], labels: List[str]) -> Tuple[List[str], List[str]]:
    seen = set()
    outs, out_labels = [], []
    for s, l in zip(strings, labels):
        h = sha1hex(s)
        if h not in seen:
            seen.add(h)
            outs.append(s)
            out_labels.append(l)
    return outs, out_labels

# -----------------------
# Analysis
# -----------------------
def analyze_dataset(texts: List[str], task_labels: List[str], name: str):
    print(f"\n{'='*60}\n{name} Dataset Statistics\n{'='*60}")
    total = len(texts)
    total_chars = sum(len(t) for t in texts)
    unique = len(set(texts))
    dup_rate = (1 - unique/total) * 100 if total > 0 else 0.0
    lengths = [len(t) for t in texts] if total>0 else [0]
    print(f"Total examples: {total:,}")
    print(f"Total chars: {total_chars:,}, Avg chars/example: {total_chars/total:.1f}")
    print(f"Unique examples: {unique:,}, Duplication rate: {dup_rate:.2f}%")
    if unique < total:
        freq = Counter(texts)
        most_common_count = freq.most_common(1)[0][1]
        print(f"Most duplicated example count: {most_common_count}")
    print(f"Length (chars) min={min(lengths)}, max={max(lengths)}, median={sorted(lengths)[len(lengths)//2]}")
    task_counter = Counter(task_labels)
    print("Per-task breakdown:")
    for task, cnt in task_counter.most_common():
        print(f"  {task:12s}: {cnt:6,} ({cnt/total*100:5.1f}%)")
    return {
        "total": total,
        "unique": unique,
        "dup_rate": dup_rate,
        "task_distribution": dict(task_counter)
    }

# -----------------------
# Save helpers (pickle)
# -----------------------
def save_pickle(obj, path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print("Saved:", path)

# -----------------------
# Equalize stratified (safe)
# -----------------------
def equalize_stratified(texts, labels, target_size, allow_upsample=True):
    cur = len(texts)
    if cur == target_size:
        return texts, labels

    by_task = defaultdict(list)
    for t, l in zip(texts, labels):
        by_task[l].append(t)

    total_w = sum(MIX.values())
    task_targets = {k: int((v / total_w) * target_size) for k, v in MIX.items()}

    current_assigned = sum(task_targets.values())
    diff = target_size - current_assigned
    if diff != 0:
        keys = list(task_targets.keys())
        for _ in range(abs(diff)):
            k = random.choice(keys)
            if diff > 0:
                task_targets[k] += 1
            else:
                task_targets[k] -= 1

    final_texts, final_labels = [], []
    for task, target_count in task_targets.items():
        available = by_task.get(task, [])
        if not available:
            continue

        if len(available) >= target_count:
            chosen = random.sample(available, target_count)
        else:
            if allow_upsample and available:
                chosen = random.choices(available, k=target_count)
            else:
                # conservative: take all available (no replacement)
                chosen = available.copy()

        final_texts.extend(chosen)
        final_labels.extend([task]*len(chosen))

    # fix length mismatches deterministically
    if len(final_texts) > target_size:
        idxs = random.sample(range(len(final_texts)), target_size)
        final_texts = [final_texts[i] for i in idxs]
        final_labels = [final_labels[i] for i in idxs]
    elif len(final_texts) < target_size:
        needed = target_size - len(final_texts)
        if allow_upsample and final_texts:
            pick_idxs = random.choices(range(len(final_texts)), k=needed)
            final_texts += [final_texts[i] for i in pick_idxs]
            final_labels += [final_labels[i] for i in pick_idxs]
        else:
            # conservative: repeat first items across tasks to fill (rare when TARGET_SIZE=min unique)
            while len(final_texts) < target_size:
                for tsk, arr in by_task.items():
                    if not arr: 
                        continue
                    final_texts.append(arr[0])
                    final_labels.append(tsk)
                    if len(final_texts) >= target_size:
                        break

    combined = list(zip(final_texts, final_labels))
    random.shuffle(combined)
    if not combined:
        return [], []
    texts_out, labels_out = zip(*combined)
    return list(texts_out), list(labels_out)

# -----------------------
# Main pipeline
# -----------------------
def produce_and_save(total_examples=NUM_TOTAL_EXAMPLES):
    pool = build_pool(total_examples, MIX)
    texts = [t for _, t in pool]
    labels = [task for task, _ in pool]
    N = len(texts)
    print("Total generated:", N)

    # Shuffle & split
    indices = list(range(N))
    random.Random(SEED).shuffle(indices)
    cutoff = int(N * TRAIN_RATIO)
    train_idx = indices[:cutoff]
    val_idx = indices[cutoff:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    print(f"Train examples: {len(train_texts):,}, Val examples: {len(val_texts):,}")

    # Build tokenizer ONLY from training texts (strict)
    stoi, itos = build_char_tokenizer_from_texts(train_texts, max_vocab=MAX_VOCAB)
    print("Tokenizer size:", len(stoi))

    # Normal
    normal_texts = train_texts.copy()
    normal_labels = train_labels.copy()
    combined = list(zip(normal_texts, normal_labels))
    random.shuffle(combined)
    normal_texts, normal_labels = zip(*combined)
    normal_texts, normal_labels = list(normal_texts), list(normal_labels)

    # Dedup (exact)
    dedup_texts, dedup_labels = dedupe_exact(train_texts, train_labels)
    combined = list(zip(dedup_texts, dedup_labels))
    random.shuffle(combined)
    dedup_texts, dedup_labels = zip(*combined)
    dedup_texts, dedup_labels = list(dedup_texts), list(dedup_labels)

    # Duplicated (per-task top-k oversample)
    dup_texts = train_texts.copy()
    dup_labels = train_labels.copy()
    task_to_texts = defaultdict(list)
    for t, l in zip(train_texts, train_labels):
        task_to_texts[l].append(t)

    for task, t_texts in task_to_texts.items():
        freq = Counter(t_texts)
        most_common = [c for c, _ in freq.most_common(DUPLICATE_REPEAT_TOP_K)]
        for c in most_common:
            for _ in range(DUPLICATE_MULTIPLIER):
                dup_texts.append(c)
                dup_labels.append(task)

    combined = list(zip(dup_texts, dup_labels))
    random.shuffle(combined)
    dup_texts, dup_labels = zip(*combined)
    dup_texts, dup_labels = list(dup_texts), list(dup_labels)

    # Compute unique counts and pick target = min unique
    unique_normal = len(set(normal_texts))
    unique_dedup = len(set(dedup_texts))
    unique_dup = len(set(dup_texts))
    TARGET_SIZE = min(unique_normal, unique_dedup, unique_dup)
    print("Using TARGET_SIZE (min unique):", TARGET_SIZE)

    # ----------------- ROBUST DEDUP REGEN + FINALIZER -----------------
    # desired per-task counts
    task_share = {k: MIX[k] / sum(MIX.values()) for k in MIX}
    target_per_task = {k: int(task_share[k] * TARGET_SIZE) for k in MIX}
    
    # Adjust for rounding errors to ensure sum is exactly TARGET_SIZE
    current_assigned = sum(target_per_task.values())
    diff = TARGET_SIZE - current_assigned
    if diff != 0:
        keys = list(target_per_task.keys())
        # deterministically add to first keys (or random, doesn't matter much for 1-2 items)
        # using random to be fair
        for _ in range(abs(diff)):
            k = random.choice(keys)
            if diff > 0: target_per_task[k] += 1
            else: target_per_task[k] -= 1

    # Build initial dedup_by_task from exact dedupe output
    dedup_by_task = defaultdict(list)
    existing_hashes = set()
    for t, lab in zip(dedup_texts, dedup_labels):
        h = sha1hex(t)
        if h not in existing_hashes:
            existing_hashes.add(h)
            dedup_by_task[lab].append(t)

    # helper generator enforcing uniqueness
    def gen_unique_for_task(task_name, existing_hashes, max_tries=5000):
        tries = 0
        while tries < max_tries:
            tries += 1
            if task_name == "sorting":
                cand = gen_sort_example()
            elif task_name == "reversing":
                cand = gen_reverse_example()
            elif task_name == "addition":
                cand = gen_add_example(min_digits=1, max_digits=5)
            elif task_name == "copying":
                cand = gen_copy_example(min_l=2, max_l=8, min_rep=2, max_rep=6)
            elif task_name == "relations":
                cand = gen_relations_example(min_hops=1, max_hops=3)
            else:
                cand = gen_sort_example()
            h = sha1hex(cand)
            if h not in existing_hashes:
                return cand, h
        return None, None

    # Generate until each task meets target_per_task
    for task_name, desired in target_per_task.items():
        cur = len(dedup_by_task.get(task_name, []))
        if cur >= desired:
            continue
        need = desired - cur
        print(f"[Dedup-regen] Generating {need} unique examples for task {task_name} (have {cur}, want {desired})")
        for _ in range(need):
            cand, h = gen_unique_for_task(task_name, existing_hashes)
            if cand is None:
                raise RuntimeError(f"Failed to generate enough unique examples for task {task_name}")
            dedup_by_task[task_name].append(cand)
            existing_hashes.add(h)

    # Build dedup_texts/dedup_labels exactly
    final_dedup_texts = []
    final_dedup_labels = []
    for task_name in MIX.keys():
        items = dedup_by_task.get(task_name, [])
        if len(items) < target_per_task[task_name]:
            raise RuntimeError(f"Not enough unique items for task {task_name} after regen")
        chosen = items[:target_per_task[task_name]]
        final_dedup_texts.extend(chosen)
        final_dedup_labels.extend([task_name] * len(chosen))

    # Ensure uniqueness
    assert len(final_dedup_texts) == len(set(final_dedup_texts)), "Dedup set still contains duplicates!"

    # Shuffle and assign
    combined = list(zip(final_dedup_texts, final_dedup_labels))
    random.shuffle(combined)
    dedup_texts, dedup_labels = zip(*combined)
    dedup_texts, dedup_labels = list(dedup_texts), list(dedup_labels)
    print("[Dedup-regen] Completed; per-task counts:", {k: len([1 for l in dedup_labels if l==k]) for k in MIX.keys()})

    # ----------------- Equalize sizes (stratified) -----------------
    normal_texts, normal_labels = equalize_stratified(normal_texts, normal_labels, TARGET_SIZE, allow_upsample=True)
    dedup_texts, dedup_labels = equalize_stratified(dedup_texts, dedup_labels, TARGET_SIZE, allow_upsample=False)
    dup_texts, dup_labels = equalize_stratified(dup_texts, dup_labels, TARGET_SIZE, allow_upsample=True)

    print("Final sizes:", len(normal_texts), len(dedup_texts), len(dup_texts))

    # Final sanity checks
    assert len(dedup_texts) == len(set(dedup_texts)), "ERROR: dedup contains duplicates after finalization!"

    def check_dist(labels, name):
        c = Counter(labels)
        total = len(labels)
        for k,v in MIX.items():
            pct = c.get(k,0) / total
            expected = v / sum(MIX.values())
            if abs(pct - expected) > 0.015:
                print(f"WARNING: {name} {k} distribution off by >1.5% ({pct:.3f} vs {expected:.3f})")
    check_dist(normal_labels, "Normal")
    check_dist(dedup_labels, "Dedup")
    check_dist(dup_labels, "Dup")

    # Encode & save
    normal_encoded = [encode_text(t, stoi) for t in normal_texts]
    dedup_encoded = [encode_text(t, stoi) for t in dedup_texts]
    dup_encoded = [encode_text(t, stoi) for t in dup_texts]
    val_encoded = [encode_text(t, stoi) for t in val_texts]

    save_pickle({"encoded": normal_encoded, "labels": normal_labels, "raw": normal_texts},
                os.path.join(OUT_DIR, "train_normal.pkl"))
    save_pickle({"encoded": dedup_encoded, "labels": dedup_labels, "raw": dedup_texts},
                os.path.join(OUT_DIR, "train_dedup.pkl"))
    save_pickle({"encoded": dup_encoded, "labels": dup_labels, "raw": dup_texts},
                os.path.join(OUT_DIR, "train_duplicated.pkl"))
    save_pickle({"encoded": val_encoded, "labels": val_labels, "raw": val_texts},
                os.path.join(OUT_DIR, "val.pkl"))

    # Tokenizer
    with open(os.path.join(OUT_DIR, "tokenizer.json"), "w", encoding="utf-8") as f:
        json.dump({"stoi": stoi, "itos": itos}, f, ensure_ascii=False, indent=2)

    # Stats
    stats_normal = analyze_dataset(normal_texts, normal_labels, "NORMAL")
    stats_dedup = analyze_dataset(dedup_texts, dedup_labels, "DEDUPLICATED")
    stats_dup = analyze_dataset(dup_texts, dup_labels, "DUPLICATED")
    stats_val = analyze_dataset(val_texts, val_labels, "VALIDATION")

    overall = {
        "normal": stats_normal,
        "deduplicated": stats_dedup,
        "duplicated": stats_dup,
        "validation": stats_val,
        "config": {"seed": SEED, "total_examples": total_examples, "train_ratio": TRAIN_RATIO,
                   "mix": MIX, "duplicate_multiplier": DUPLICATE_MULTIPLIER}
    }
    with open(os.path.join(OUT_DIR, "dataset_stats.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print("Saved dataset_stats.json")

if __name__ == "__main__":
    produce_and_save()
