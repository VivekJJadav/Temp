# plots.py
# Visualization module for memorization vs. generalization research
# Creates publication-ready plots for training metrics and analysis

import os
import json
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11


# ---------- Color Palette ----------
COLORS = {
    'train': '#2196F3',      # Blue
    'val': '#F44336',        # Red
    'gap': '#9C27B0',        # Purple
    'ood': '#4CAF50',        # Green
    'accent': '#FF9800',     # Orange
    'neutral': '#607D8B',    # Gray
}


# ---------- Training Curves ----------
def plot_train_val_loss(history: Dict, out_dir: str, filename: str = "train_val_loss.png"):
    """Plot training vs validation loss over epochs."""
    epochs = range(1, len(history["train_loss"]) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, history["train_loss"], 'o-', color=COLORS['train'], 
            linewidth=2.5, markersize=8, label='Train Loss')
    
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], 's-', color=COLORS['val'], 
                linewidth=2.5, markersize=8, label='Val Loss')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Loss', fontsize=13)
    ax.set_title('Training vs Validation Loss', fontsize=15, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # Add gap annotation if both exist
    if history.get("val_loss"):
        final_gap = history["val_loss"][-1] - history["train_loss"][-1]
        ax.annotate(f'Final Gap: {final_gap:.3f}', 
                   xy=(epochs[-1], history["val_loss"][-1]),
                   xytext=(epochs[-1] - 1, history["val_loss"][-1] + 0.1),
                   fontsize=10, color=COLORS['gap'],
                   arrowprops=dict(arrowstyle='->', color=COLORS['gap']))
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_train_val_gap(history: Dict, out_dir: str, filename: str = "train_val_gap.png"):
    """Plot the train-val gap over epochs (memorization signal)."""
    if not history.get("val_loss") or not history.get("train_loss"):
        return None
    
    epochs = range(1, len(history["train_loss"]) + 1)
    gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Fill between zero line and gap
    ax.fill_between(epochs, 0, gap, alpha=0.3, color=COLORS['gap'])
    ax.plot(epochs, gap, 'o-', color=COLORS['gap'], linewidth=2.5, markersize=8)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Gap (Val - Train)', fontsize=13)
    ax.set_title('Train-Validation Gap (Overfitting Signal)', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Interpretation annotation
    final_gap = gap[-1]
    if final_gap > 0.2:
        interp = "High gap → Overfitting/Memorization"
        color = '#F44336'
    elif final_gap > 0.05:
        interp = "Moderate gap"
        color = '#FF9800'
    else:
        interp = "Low gap → Good generalization"
        color = '#4CAF50'
    
    ax.annotate(interp, xy=(0.98, 0.95), xycoords='axes fraction',
                fontsize=11, ha='right', color=color, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_exact_match_curves(history: Dict, out_dir: str, filename: str = "exact_match.png"):
    """Plot exact match accuracy curves."""
    if not history.get("val_exact"):
        return None
    
    epochs = range(1, len(history["val_exact"]) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Validation exact match
    ax.plot(epochs, history["val_exact"], 's-', color=COLORS['val'], 
            linewidth=2.5, markersize=8, label='Val Exact Match')
    
    # Train exact match if available
    if history.get("train_exact"):
        ax.plot(epochs, history["train_exact"], 'o-', color=COLORS['train'], 
                linewidth=2.5, markersize=8, label='Train Exact Match')
    
    # Token accuracy if available
    if history.get("val_token_acc"):
        ax.plot(epochs, history["val_token_acc"], '^--', color=COLORS['accent'], 
                linewidth=2, markersize=6, alpha=0.7, label='Val Token Acc')
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Exact Match & Token Accuracy', fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_learning_rate(history: Dict, out_dir: str, filename: str = "learning_rate.png"):
    """Plot learning rate schedule."""
    if not history.get("lr"):
        return None
    
    epochs = range(1, len(history["lr"]) + 1)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, history["lr"], '-', color=COLORS['neutral'], linewidth=2.5)
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Learning Rate', fontsize=13)
    ax.set_title('Learning Rate Schedule', fontsize=15, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ---------- Per-Task Visualization ----------
def plot_per_task_heatmap(task_accuracy_history: List[Dict], out_dir: str, 
                          filename: str = "task_accuracy_heatmap.png"):
    """
    Plot per-task accuracy as a heatmap over epochs.
    
    Args:
        task_accuracy_history: List of dicts, one per epoch, with task->accuracy
        out_dir: Output directory
        filename: Output filename
    """
    if not task_accuracy_history or not any(task_accuracy_history):
        return None
    
    # Extract all task names
    all_tasks = set()
    for epoch_acc in task_accuracy_history:
        if epoch_acc:
            all_tasks.update(epoch_acc.keys())
    
    if not all_tasks:
        return None
    
    tasks = sorted(all_tasks)
    n_epochs = len(task_accuracy_history)
    
    # Build matrix
    matrix = np.zeros((len(tasks), n_epochs))
    for epoch_idx, epoch_acc in enumerate(task_accuracy_history):
        if epoch_acc:
            for task_idx, task in enumerate(tasks):
                matrix[task_idx, epoch_idx] = epoch_acc.get(task, 0)
    
    fig, ax = plt.subplots(figsize=(max(10, n_epochs * 0.8), max(6, len(tasks) * 0.5)))
    
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(n_epochs))
    ax.set_xticklabels([f'E{i+1}' for i in range(n_epochs)])
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks)
    
    # Add values
    for i in range(len(tasks)):
        for j in range(n_epochs):
            val = matrix[i, j]
            color = 'white' if val < 0.5 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=9, color=color)
    
    ax.set_xlabel('Epoch', fontsize=13)
    ax.set_ylabel('Task', fontsize=13)
    ax.set_title('Per-Task Accuracy Over Training', fontsize=15, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Accuracy', fontsize=11)
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_final_task_accuracy_bar(task_accuracy: Dict, out_dir: str,
                                  filename: str = "task_accuracy_bar.png"):
    """Plot final per-task accuracy as a bar chart."""
    if not task_accuracy:
        return None
    
    tasks = sorted(task_accuracy.keys())
    accuracies = [task_accuracy[t] for t in tasks]
    
    fig, ax = plt.subplots(figsize=(max(10, len(tasks) * 0.8), 6))
    
    colors = [COLORS['ood'] if acc > 0.7 else COLORS['accent'] if acc > 0.4 else COLORS['val'] 
              for acc in accuracies]
    
    bars = ax.bar(range(len(tasks)), accuracies, color=colors, edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(tasks, rotation=45, ha='right')
    ax.set_ylabel('Accuracy', fontsize=13)
    ax.set_title('Per-Task Accuracy (Final Epoch)', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ---------- OOD Performance ----------
def plot_ood_summary_bar(ood_results: Dict, out_dir: str, filename: str = "ood_summary.png"):
    """
    Plot OOD performance summary as grouped bar chart.
    
    Args:
        ood_results: Dict with category -> {exact_match, loss, ...}
    """
    if not ood_results:
        return None
    
    categories = sorted(ood_results.keys())
    exact_matches = [ood_results[c].get("exact_match", 0) for c in categories]
    
    fig, ax = plt.subplots(figsize=(max(10, len(categories) * 1.2), 6))
    
    x = range(len(categories))
    bars = ax.bar(x, exact_matches, color=COLORS['ood'], edgecolor='black', linewidth=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha='right')
    ax.set_ylabel('Exact Match Accuracy', fontsize=13)
    ax.set_title('OOD Performance by Category', fontsize=15, fontweight='bold')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add horizontal line for average
    avg = sum(exact_matches) / len(exact_matches)
    ax.axhline(y=avg, color=COLORS['accent'], linestyle='--', linewidth=2, 
               label=f'Average: {avg:.2f}')
    ax.legend(loc='upper right')
    
    # Value labels
    for bar, acc in zip(bars, exact_matches):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
               f'{acc:.2f}', ha='center', fontsize=9)
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


def plot_duplication_sensitivity(dup_results: Dict, out_dir: str, 
                                  filename: str = "duplication_sensitivity.png"):
    """Plot duplication sensitivity analysis results."""
    if not dup_results or not dup_results.get("bucket_accuracy"):
        return None
    
    buckets = ["unseen", "1x", "10x+"]
    accuracies = [dup_results["bucket_accuracy"].get(b, 0) or 0 for b in buckets]
    counts = [dup_results.get("bucket_counts", {}).get(b, 0) for b in buckets]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of accuracy by bucket
    colors = [COLORS['ood'], COLORS['accent'], COLORS['train']]
    bars = ax1.bar(buckets, accuracies, color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_ylabel('Exact Match Accuracy', fontsize=13)
    ax1.set_title('Accuracy by Training Frequency', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3, axis='y')
    
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{acc:.2f}', ha='center', fontsize=11)
    
    # Memorization score annotation
    mem_score = dup_results.get("memorization_score")
    if mem_score is not None:
        color = COLORS['val'] if mem_score > 0.2 else COLORS['accent'] if mem_score > 0.05 else COLORS['ood']
        ax1.annotate(f'Memorization Score: {mem_score:.3f}', 
                    xy=(0.5, 0.95), xycoords='axes fraction',
                    fontsize=12, ha='center', color=color, fontweight='bold')
    
    # Pie chart of counts
    ax2.pie(counts, labels=buckets, autopct='%1.1f%%', colors=colors,
            explode=(0.05, 0, 0), shadow=True, startangle=90)
    ax2.set_title('Distribution of Examples', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ---------- Memorization Dashboard ----------
def plot_memorization_dashboard(history: Dict, out_dir: str, 
                                 ood_summary: Optional[Dict] = None,
                                 dup_results: Optional[Dict] = None,
                                 filename: str = "memorization_dashboard.png"):
    """
    Create a comprehensive dashboard for memorization analysis.
    """
    n_plots = 4
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # 1. Train vs Val Loss
    ax = axes[0]
    epochs = range(1, len(history["train_loss"]) + 1)
    ax.plot(epochs, history["train_loss"], 'o-', color=COLORS['train'], 
            linewidth=2, markersize=6, label='Train')
    if history.get("val_loss"):
        ax.plot(epochs, history["val_loss"], 's-', color=COLORS['val'], 
                linewidth=2, markersize=6, label='Val')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Train vs Val Loss', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Exact Match / Token Accuracy
    ax = axes[1]
    if history.get("val_exact"):
        ax.plot(epochs, history["val_exact"], 's-', color=COLORS['val'], 
                linewidth=2, markersize=6, label='Val EM')
    if history.get("train_exact"):
        ax.plot(epochs, history["train_exact"], 'o-', color=COLORS['train'], 
                linewidth=2, markersize=6, label='Train EM')
    if history.get("val_token_acc"):
        ax.plot(epochs, history["val_token_acc"], '^--', color=COLORS['accent'], 
                linewidth=1.5, markersize=5, alpha=0.7, label='Token Acc')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy Metrics', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    # 3. Train-Val Gap (primary memorization signal)
    ax = axes[2]
    if history.get("val_loss"):
        gap = [v - t for t, v in zip(history["train_loss"], history["val_loss"])]
        ax.fill_between(epochs, 0, gap, alpha=0.3, color=COLORS['gap'])
        ax.plot(epochs, gap, 'o-', color=COLORS['gap'], linewidth=2, markersize=6)
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Gap (Val - Train)')
    ax.set_title('Generalization Gap', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # 4. OOD or Duplication Summary
    ax = axes[3]
    if dup_results and dup_results.get("bucket_accuracy"):
        buckets = ["unseen", "1x", "10x+"]
        accuracies = [dup_results["bucket_accuracy"].get(b, 0) or 0 for b in buckets]
        colors = [COLORS['ood'], COLORS['accent'], COLORS['train']]
        ax.bar(buckets, accuracies, color=colors, edgecolor='black', linewidth=0.5)
        ax.set_ylabel('Accuracy')
        ax.set_title('Duplication Sensitivity', fontweight='bold')
        ax.set_ylim(0, 1.05)
        
        mem_score = dup_results.get("memorization_score", 0) or 0
        ax.annotate(f'Mem. Score: {mem_score:.3f}', 
                   xy=(0.5, 0.95), xycoords='axes fraction',
                   fontsize=11, ha='center', fontweight='bold')
    elif ood_summary:
        categories = sorted(ood_summary.keys())[:6]  # Limit to 6
        accuracies = [ood_summary[c].get("exact_match", 0) for c in categories]
        ax.bar(range(len(categories)), accuracies, color=COLORS['ood'])
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels([c[:10] for c in categories], rotation=45, ha='right')
        ax.set_ylabel('Exact Match')
        ax.set_title('OOD Performance', fontweight='bold')
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, 'No OOD/Duplication data', ha='center', va='center', fontsize=12)
        ax.set_title('OOD/Duplication Analysis', fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Memorization Analysis Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    path = os.path.join(out_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    return path


# ---------- Generate All Plots ----------
def generate_all_training_plots(history: Dict, out_dir: str, 
                                 ood_summary: Optional[Dict] = None,
                                 dup_results: Optional[Dict] = None) -> List[str]:
    """
    Generate all training visualization plots.
    
    Returns:
        List of paths to generated plots
    """
    os.makedirs(os.path.join(out_dir, "plots"), exist_ok=True)
    plot_dir = os.path.join(out_dir, "plots")
    
    generated = []
    
    # Core plots
    path = plot_train_val_loss(history, plot_dir)
    if path: generated.append(path)
    
    path = plot_train_val_gap(history, plot_dir)
    if path: generated.append(path)
    
    path = plot_exact_match_curves(history, plot_dir)
    if path: generated.append(path)
    
    path = plot_learning_rate(history, plot_dir)
    if path: generated.append(path)
    
    # Per-task analysis
    if history.get("task_accuracy"):
        path = plot_per_task_heatmap(history["task_accuracy"], plot_dir)
        if path: generated.append(path)
        
        # Final epoch bar chart
        if history["task_accuracy"][-1]:
            path = plot_final_task_accuracy_bar(history["task_accuracy"][-1], plot_dir)
            if path: generated.append(path)
    
    # OOD summary
    if ood_summary:
        path = plot_ood_summary_bar(ood_summary, plot_dir)
        if path: generated.append(path)
    
    # Duplication sensitivity
    if dup_results:
        path = plot_duplication_sensitivity(dup_results, plot_dir)
        if path: generated.append(path)
    
    # Dashboard
    path = plot_memorization_dashboard(history, plot_dir, ood_summary, dup_results)
    if path: generated.append(path)
    
    print(f"Generated {len(generated)} plots in {plot_dir}/")
    return generated


# ---------- Test ----------
def test_plots():
    """Test plot generation with dummy data."""
    import tempfile
    
    print("Testing plots.py...")
    
    # Create dummy history
    history = {
        "train_loss": [2.5, 1.8, 1.2, 0.8, 0.5, 0.3],
        "val_loss": [2.6, 2.0, 1.5, 1.2, 1.0, 0.9],
        "val_exact": [0.1, 0.2, 0.35, 0.5, 0.6, 0.65],
        "val_token_acc": [0.3, 0.45, 0.55, 0.65, 0.72, 0.78],
        "lr": [3e-4, 2.8e-4, 2.5e-4, 2e-4, 1.5e-4, 1e-4],
        "task_accuracy": [
            {"math": 0.2, "logic": 0.15, "code": 0.1},
            {"math": 0.35, "logic": 0.3, "code": 0.25},
            {"math": 0.5, "logic": 0.45, "code": 0.4},
            {"math": 0.6, "logic": 0.55, "code": 0.5},
            {"math": 0.7, "logic": 0.65, "code": 0.6},
            {"math": 0.75, "logic": 0.7, "code": 0.65},
        ]
    }
    
    dup_results = {
        "bucket_accuracy": {"unseen": 0.4, "1x": 0.6, "10x+": 0.85},
        "bucket_counts": {"unseen": 100, "1x": 50, "10x+": 30},
        "memorization_score": 0.45
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = generate_all_training_plots(history, tmpdir, dup_results=dup_results)
        print(f"  ✓ Generated {len(paths)} test plots")
        for p in paths:
            print(f"    - {os.path.basename(p)}")
    
    print("\nAll plot tests passed!")


if __name__ == "__main__":
    test_plots()
