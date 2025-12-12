import random
import json
import pickle
import os

# --------------------------------------------------------
# Utility Functions
# --------------------------------------------------------
def rand_seq(length, low=0, high=9):
    return [str(random.randint(low, high)) for _ in range(length)]

def seq_to_str(seq):
    return " ".join(seq)

# --------------------------------------------------------
# CATEGORY 1: EXTRAPOLATION
# --------------------------------------------------------
def generate_ood_extrapolation_addition(n=500):
    data = []
    for _ in range(n):
        a = random.randint(100, 9999)
        b = random.randint(100, 9999)
        result = a + b
        text = f"[addition_extrap] {a} + {b} | {result}"
        data.append(text)
    return data

def generate_ood_extrapolation_copying(n=500):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    data = []
    for _ in range(n):
        length = random.randint(6, 10)
        reps = random.randint(5, 7)
        s = "".join(random.choice(letters) for _ in range(length))
        out = s * reps
        text = f"[copying_extrap] {s} | {out}"
        data.append(text)
    return data

def generate_ood_extrapolation_relations(n=500):
    data = []
    for _ in range(n):
        depth = random.randint(3, 6)
        ids = [f"E{random.randint(2000, 5000)}" for _ in range(depth + 1)]
        chain = [f"{ids[i]} is parent of {ids[i+1]}." for i in range(depth)]
        question = f"Who is the ancestor {depth} of {ids[-1]}?"
        answer = ids[0]
        text = f"[relations_extrap] {' '.join(chain)} {question} | {answer}"
        data.append(text)
    return data

def generate_ood_extrapolation_reversing(n=500):
    data = []
    for _ in range(n):
        length = random.randint(10, 15)
        seq = rand_seq(length, low=0, high=30)
        rev = seq[::-1]
        text = f"[reversing_extrap] {seq_to_str(seq)} | {seq_to_str(rev)}"
        data.append(text)
    return data

def generate_ood_extrapolation_sorting(n=500):
    data = []
    for _ in range(n):
        length = random.randint(10, 15)
        seq = [str(random.randint(0, 99)) for _ in range(length)]
        sorted_seq = sorted(seq, key=lambda x: int(x))
        text = f"[sorting_extrap] {seq_to_str(seq)} | {seq_to_str(sorted_seq)}"
        data.append(text)
    return data

# --------------------------------------------------------
# CATEGORY 2: INTERPOLATION
# --------------------------------------------------------
def generate_ood_interpolation(n=1000):
    data = []

    for _ in range(n // 3):
        length = random.randint(4, 7)
        seq = rand_seq(length, 0, 9)
        sorted_seq = sorted(seq, key=int)
        text = f"[sorting_interp] {seq_to_str(seq)} | {seq_to_str(sorted_seq)}"
        data.append(text)

    for _ in range(n // 3):
        a = random.randint(1, 99)
        b = random.randint(1, 99)
        result = a + b
        text = f"[addition_interp] {a} + {b} | {result}"
        data.append(text)

    for _ in range(n // 3):
        length = random.randint(4, 6)
        seq = rand_seq(length, 0, 9)
        rev = seq[::-1]
        text = f"[reversing_interp] {seq_to_str(seq)} | {seq_to_str(rev)}"
        data.append(text)

    return data

# --------------------------------------------------------
# CATEGORY 3: COMPOSITIONAL
# --------------------------------------------------------
def generate_ood_compositional(n=1000):
    data = []

    # Sort → Reverse
    for _ in range(n // 5):
        length = random.randint(5, 8)
        seq = rand_seq(length)
        sorted_seq = sorted(seq, key=int)
        reversed_seq = sorted_seq[::-1]
        text = f"[sort_then_reverse] {seq_to_str(seq)} | {seq_to_str(reversed_seq)}"
        data.append(text)

    # Reverse → Sort
    for _ in range(n // 5):
        length = random.randint(5, 8)
        seq = rand_seq(length)
        reversed_seq = seq[::-1]
        sorted_seq = sorted(reversed_seq, key=int)
        text = f"[reverse_then_sort] {seq_to_str(seq)} | {seq_to_str(sorted_seq)}"
        data.append(text)

    # Multi-step addition
    for _ in range(n // 5):
        a = random.randint(5, 30)
        b = random.randint(5, 30)
        c = random.randint(5, 30)
        result = a + b + c
        text = f"[multi_add] {a} + {b} + {c} | {result}"
        data.append(text)

    # Copy → Sort
    for _ in range(n // 5):
        length = random.randint(3, 5)
        seq = rand_seq(length)
        copied = seq * 2
        sorted_result = sorted(copied, key=int)
        text = f"[copy_then_sort] {seq_to_str(seq)} | {seq_to_str(sorted_result)}"
        data.append(text)

    # Nested operations
    for _ in range(n // 5):
        length = random.randint(4, 6)
        seq = rand_seq(length)
        sorted_seq = sorted(seq, key=int)
        reversed_seq = sorted_seq[::-1]
        result = reversed_seq[:3]
        text = f"[nested_ops] {seq_to_str(seq)} | {seq_to_str(result)}"
        data.append(text)

    return data

# --------------------------------------------------------
# CATEGORY 4: NOVEL TASKS
# --------------------------------------------------------
def generate_ood_novel_tasks(n=1000):
    data = []

    # Subtraction
    for _ in range(n // 6):
        a = random.randint(50, 200)
        b = random.randint(10, a)
        text = f"[subtraction] {a} - {b} | {a - b}"
        data.append(text)

    # Multiplication
    for _ in range(n // 6):
        a = random.randint(2, 12)
        b = random.randint(2, 12)
        text = f"[multiplication] {a} * {b} | {a * b}"
        data.append(text)

    # Palindrome
    for _ in range(n // 6):
        length = random.randint(4, 8)
        seq = rand_seq(length)
        is_pal = seq == seq[::-1]
        text = f"[palindrome] {seq_to_str(seq)} | {'yes' if is_pal else 'no'}"
        data.append(text)

    # Find max
    for _ in range(n // 6):
        length = random.randint(5, 10)
        seq = rand_seq(length, 0, 50)
        text = f"[find_max] {seq_to_str(seq)} | {max(int(x) for x in seq)}"
        data.append(text)

    # Deduplicate
    for _ in range(n // 6):
        length = random.randint(6, 10)
        seq = rand_seq(length, 0, 5)
        seen, uniq = set(), []
        for x in seq:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        text = f"[deduplicate] {seq_to_str(seq)} | {seq_to_str(uniq)}"
        data.append(text)

    # Count occurrences
    for _ in range(n // 6):
        length = random.randint(6, 10)
        seq = rand_seq(length, 0, 5)
        target = random.choice(seq)
        text = f"[count] count {target} in {seq_to_str(seq)} | {seq.count(target)}"
        data.append(text)

    return data

# --------------------------------------------------------
# CATEGORY 5: EDGE CASES
# --------------------------------------------------------
def generate_ood_edge_cases(n=500):
    data = []

    # Single element
    for _ in range(n // 6):
        v = str(random.randint(0, 9))
        data.append(f"[sorting_edge] {v} | {v}")

    # All same
    for _ in range(n // 6):
        length = random.randint(4, 8)
        v = str(random.randint(0, 9))
        seq = [v] * length
        data.append(f"[sorting_edge] {seq_to_str(seq)} | {seq_to_str(seq)}")

    # Add 0
    for _ in range(n // 6):
        a = random.randint(1, 100)
        data.append(f"[addition_edge] {a} + 0 | {a}")
        data.append(f"[addition_edge] 0 + {a} | {a}")

    # Already sorted
    for _ in range(n // 6):
        seq = sorted(rand_seq(random.randint(5, 8)), key=int)
        data.append(f"[sorting_edge] {seq_to_str(seq)} | {seq_to_str(seq)}")

    # Reverse sorted
    for _ in range(n // 6):
        seq = sorted(rand_seq(random.randint(5, 8)), key=int, reverse=True)
        data.append(f"[sorting_edge] {seq_to_str(seq)} | {seq_to_str(seq[::-1])}")

    # Max values
    for _ in range(n // 6):
        data.append("[addition_edge] 9999 + 9999 | 19998")

    return data

# --------------------------------------------------------
# CATEGORY 6: SYSTEMATIC CURVES
# --------------------------------------------------------
def generate_ood_systematic(n=1000):
    data = []

    # Length tests
    for length in [8, 10, 12, 15, 20, 25]:
        for _ in range(n // 30):
            seq = rand_seq(length)
            sorted_seq = sorted(seq, key=int)
            data.append(f"[sorting_len{length}] {seq_to_str(seq)} | {seq_to_str(sorted_seq)}")

    # Range tests
    for max_val in [20, 50, 100, 500, 1000]:
        for _ in range(n // 30):
            a = random.randint(1, max_val)
            b = random.randint(1, max_val)
            data.append(f"[addition_range{max_val}] {a} + {b} | {a + b}")

    # Depth tests (relations)
    for depth in [2, 3, 4, 5, 7, 10]:
        for _ in range(n // 30):
            ids = [f"E{random.randint(1000,2000)}" for _ in range(depth+1)]
            chain = [f"{ids[i]} is parent of {ids[i+1]}." for i in range(depth)]
            q = f"Who is the ancestor {depth} of {ids[-1]}?"
            data.append(f"[relations_depth{depth}] {' '.join(chain)} {q} | {ids[0]}")

    return data

# --------------------------------------------------------
# CATEGORY 7: ROBUSTNESS
# --------------------------------------------------------
def generate_ood_robustness(n=500):
    data = []

    # Extra spaces
    for _ in range(n // 4):
        seq = rand_seq(random.randint(5, 8))
        sorted_seq = sorted(seq, key=int)
        seq_str = "  ".join(seq)  # messy spacing
        data.append(f"[sorting_robust] {seq_str} | {seq_to_str(sorted_seq)}")

    # No spaces in addition
    for _ in range(n // 4):
        a = random.randint(10, 99)
        b = random.randint(10, 99)
        data.append(f"[addition_robust] {a}+{b} | {a+b}")

    # Mixed case copying
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    for _ in range(n // 4):
        length = random.randint(4, 6)
        s = "".join(random.choice(letters) for _ in range(length))
        data.append(f"[copying_robust] {s} | {s*3}")

    # Reverse prompt order
    for _ in range(n // 4):
        seq = rand_seq(random.randint(5, 7))
        sorted_seq = sorted(seq, key=int)
        data.append(f"[sorting_robust] sort this: {seq_to_str(seq)} | {seq_to_str(sorted_seq)}")

    return data

# --------------------------------------------------------
# Save Helper
# --------------------------------------------------------
def save_ood_dataset(path, dataset, category):
    data_dict = {
        "raw": dataset,
        "category": category,
        "count": len(dataset),
        "encoded": [],
        "labels": []
    }
    with open(path, "wb") as f:
        pickle.dump(data_dict, f)
    print(f"✓ Saved {len(dataset):4d} examples → {path}")

# --------------------------------------------------------
# MAIN: Generate complete OOD suite
# --------------------------------------------------------
def generate_complete_ood_suite(out_dir="ood_data_complete"):
    os.makedirs(out_dir, exist_ok=True)

    print("="*60)
    print("GENERATING COMPLETE OOD TEST SUITE")
    print("="*60)

    # 1 — Extrapolation
    print("\n[1/7] Extrapolation Tests...")
    save_ood_dataset(f"{out_dir}/ood_extrap_add.pkl", generate_ood_extrapolation_addition(), "extrapolation")
    save_ood_dataset(f"{out_dir}/ood_extrap_copy.pkl", generate_ood_extrapolation_copying(), "extrapolation")
    save_ood_dataset(f"{out_dir}/ood_extrap_rel.pkl", generate_ood_extrapolation_relations(), "extrapolation")
    save_ood_dataset(f"{out_dir}/ood_extrap_rev.pkl", generate_ood_extrapolation_reversing(), "extrapolation")
    save_ood_dataset(f"{out_dir}/ood_extrap_sort.pkl", generate_ood_extrapolation_sorting(), "extrapolation")

    # 2 — Interpolation
    print("\n[2/7] Interpolation Tests...")
    save_ood_dataset(f"{out_dir}/ood_interpolation.pkl", generate_ood_interpolation(), "interpolation")

    # 3 — Compositional
    print("\n[3/7] Compositional Tests...")
    save_ood_dataset(f"{out_dir}/ood_compositional.pkl", generate_ood_compositional(), "compositional")

    # 4 — Novel Tasks
    print("\n[4/7] Novel Task Tests...")
    save_ood_dataset(f"{out_dir}/ood_novel_tasks.pkl", generate_ood_novel_tasks(), "novel")

    # 5 — Edge Cases
    print("\n[5/7] Edge Case Tests...")
    save_ood_dataset(f"{out_dir}/ood_edge_cases.pkl", generate_ood_edge_cases(), "edge")

    # 6 — Systematic Curves
    print("\n[6/7] Systematic Tests...")
    save_ood_dataset(f"{out_dir}/ood_systematic.pkl", generate_ood_systematic(), "systematic")

    # 7 — Robustness
    print("\n[7/7] Robustness Tests...")
    save_ood_dataset(f"{out_dir}/ood_robustness.pkl", generate_ood_robustness(), "robustness")

    print("\n✓ COMPLETE OOD SUITE GENERATED")

# Uncomment to run immediately:
generate_complete_ood_suite()
