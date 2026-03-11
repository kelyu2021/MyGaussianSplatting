"""
Visualise Training Losses & Evaluation Metrics
===============================================

Reads the CSV files produced during training and generates publication-quality
matplotlib figures.

Usage
-----
    cd MyGaussianSplatting/gopro360
    python visualize_metrics.py                              # auto-detect latest run
    python visualize_metrics.py --model_path output/gopro360_exp/gopro360_10s
    python visualize_metrics.py --model_path output/gopro360_exp/gopro360_10s --save_dir plots

Generated plots
---------------
  1. ``train_loss.png``        – total loss + L1 loss over iterations
  2. ``train_psnr.png``        – training EMA PSNR over iterations
  3. ``eval_metrics.png``      – test-set L1, PSNR, SSIM, LPIPS at eval checkpoints
  4. ``eval_comparison.png``   – test vs train-view metrics side-by-side
  5. ``points_over_time.png``  – number of Gaussian points at eval iterations
  6. ``summary.png``           – combined 2x4 dashboard of all key curves

If ``--save_dir`` is given the figures are saved there; otherwise they are
shown interactively with ``plt.show()``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
except ImportError:
    print("matplotlib is required.  pip install matplotlib")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════
#  CSV loaders
# ═══════════════════════════════════════════════════════════════════════════

def _parse_float(v: str) -> float:
    """Parse a float from a string, handling tensor(...) representations."""
    import re
    m = re.search(r"tensor\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", v)
    if m:
        return float(m.group(1))
    return float(v)


def load_train_csv(path: str) -> dict[str, np.ndarray]:
    """Load ``train_metrics.csv`` → dict of numpy arrays keyed by column."""
    data: dict[str, list] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for k, v in row.items():
                data.setdefault(k, []).append(_parse_float(v))
    return {k: np.array(v) for k, v in data.items()}


def load_eval_csv(path: str) -> dict[str, dict[str, np.ndarray]]:
    """Load ``eval_metrics.csv`` → ``{split: {col: array}}``."""
    raw: dict[str, dict[str, list]] = {}
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            split = row.pop("split")
            sub = raw.setdefault(split, {})
            for k, v in row.items():
                sub.setdefault(k, []).append(float(v))
    return {s: {k: np.array(v) for k, v in d.items()} for s, d in raw.items()}


# ═══════════════════════════════════════════════════════════════════════════
#  Smoothing helper
# ═══════════════════════════════════════════════════════════════════════════

def ema_smooth(values: np.ndarray, alpha: float = 0.98) -> np.ndarray:
    """Exponential moving average for curve smoothing."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


# ═══════════════════════════════════════════════════════════════════════════
#  Plotting functions
# ═══════════════════════════════════════════════════════════════════════════

def _style():
    """Apply a consistent, clean style across plots."""
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
    })


def plot_train_loss(train: dict, save_path: str | None = None):
    """Plot total loss and L1 loss curves."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    iters = train["iteration"]

    if "loss" in train:
        ax.plot(iters, train["loss"], alpha=0.15, color="C0", linewidth=0.5)
        ax.plot(iters, ema_smooth(train["loss"]), color="C0",
                linewidth=1.5, label="Total Loss (EMA)")
    if "l1_loss" in train:
        ax.plot(iters, train["l1_loss"], alpha=0.15, color="C1", linewidth=0.5)
        ax.plot(iters, ema_smooth(train["l1_loss"]), color="C1",
                linewidth=1.5, label="L1 Loss (EMA)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training Losses")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    return fig


def plot_train_psnr(train: dict, save_path: str | None = None):
    """Plot training PSNR curve."""
    _style()
    fig, ax = plt.subplots(figsize=(10, 5))
    iters = train["iteration"]

    if "ema_psnr" in train:
        ax.plot(iters, train["ema_psnr"], color="C2", linewidth=1.5, label="EMA PSNR")
    if "psnr" in train:
        ax.plot(iters, train["psnr"], alpha=0.15, color="C2", linewidth=0.5)
        ax.plot(iters, ema_smooth(train["psnr"]), color="C2",
                linewidth=1.5, label="PSNR (EMA)")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("Training PSNR")
    ax.legend()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    return fig


def plot_eval_metrics(evals: dict, save_path: str | None = None):
    """Plot test-set L1, PSNR, SSIM, LPIPS at eval iterations."""
    _style()
    test = evals.get("test", {})
    if not test:
        print("  No test-split eval data found – skipping eval_metrics plot.")
        return None

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
    iters = test["iteration"]

    for ax, key, ylabel, color in zip(
        axes,
        ["l1_loss", "psnr", "ssim", "lpips"],
        ["L1 Loss", "PSNR (dB)", "SSIM", "LPIPS"],
        ["C0", "C2", "C3", "C5"],
    ):
        if key in test:
            ax.plot(iters, test[key], marker="o", color=color,
                    linewidth=2, markersize=6)
            ax.set_xlabel("Iteration")
            ax.set_ylabel(ylabel)
            ax.set_title(f"Test {ylabel}")
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Evaluation Metrics (Test Set)", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_eval_comparison(evals: dict, save_path: str | None = None):
    """Side-by-side test vs train-view metrics."""
    _style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))

    for ax, key, ylabel in zip(
        axes,
        ["l1_loss", "psnr", "ssim", "lpips"],
        ["L1 Loss", "PSNR (dB)", "SSIM", "LPIPS"],
    ):
        for split, style, color in [("test", "-o", "C0"), ("train", "--s", "C1")]:
            d = evals.get(split, {})
            if key in d:
                ax.plot(d["iteration"], d[key], style, color=color,
                        linewidth=1.5, markersize=5, label=f"{split} view")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Test vs Train-View Comparison", fontsize=14, y=1.02)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


def plot_points_over_time(evals: dict, save_path: str | None = None):
    """Plot the number of Gaussian points at each eval iteration."""
    _style()
    # Use test split (it has n_points logged)
    d = evals.get("test", evals.get("train", {}))
    if "n_points" not in d:
        print("  No n_points data – skipping points plot.")
        return None

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(d["iteration"], d["n_points"] / 1e6, marker="o",
            color="C4", linewidth=2, markersize=6)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Points (millions)")
    ax.set_title("Number of Gaussians Over Training")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved → {save_path}")
    return fig


def plot_summary_dashboard(train: dict, evals: dict, save_path: str | None = None):
    """Combined dashboard with all key curves."""
    _style()
    iters_t = train.get("iteration", np.array([]))
    has_train_data = len(iters_t) > 0 and ("loss" in train or "l1_loss" in train or "psnr" in train)

    if has_train_data:
        # Full 2x4 layout when training data is available
        fig, axes = plt.subplots(2, 4, figsize=(24, 10))

        # (0,0) Total loss
        ax = axes[0, 0]
        if "loss" in train:
            ax.plot(iters_t, train["loss"], alpha=0.12, color="C0", linewidth=0.5)
            ax.plot(iters_t, ema_smooth(train["loss"]), color="C0", linewidth=1.5)
        ax.set_title("Total Loss"); ax.set_xlabel("Iter"); ax.set_ylabel("Loss")

        # (0,1) L1 loss
        ax = axes[0, 1]
        if "l1_loss" in train:
            ax.plot(iters_t, train["l1_loss"], alpha=0.12, color="C1", linewidth=0.5)
            ax.plot(iters_t, ema_smooth(train["l1_loss"]), color="C1", linewidth=1.5)
        ax.set_title("L1 Loss"); ax.set_xlabel("Iter"); ax.set_ylabel("Loss")

        # (0,2) Training PSNR
        ax = axes[0, 2]
        if "ema_psnr" in train:
            ax.plot(iters_t, train["ema_psnr"], color="C2", linewidth=1.5)
        elif "psnr" in train:
            ax.plot(iters_t, ema_smooth(train["psnr"]), color="C2", linewidth=1.5)
        ax.set_title("Training PSNR"); ax.set_xlabel("Iter"); ax.set_ylabel("PSNR (dB)")

        # (0,3) Number of points
        ax = axes[0, 3]
        d = evals.get("test", evals.get("train", {}))
        if "n_points" in d:
            ax.plot(d["iteration"], d["n_points"] / 1e6, "-o",
                    color="C4", linewidth=2, markersize=5)
        ax.set_title("Gaussian Points"); ax.set_xlabel("Iter"); ax.set_ylabel("M points")

        eval_axes = axes[1]
    else:
        # Compact 2x3 layout when only eval data is available
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))

        # (0,0) Eval L1 Loss
        ax = axes[0, 0]
        for split, style, color in [("test", "-o", "C0"), ("train", "--s", "C1")]:
            d = evals.get(split, {})
            if "l1_loss" in d:
                ax.plot(d["iteration"], d["l1_loss"], style, color=color,
                        linewidth=1.5, markersize=5, label=f"{split}")
        ax.set_title("Eval L1 Loss"); ax.set_xlabel("Iter"); ax.set_ylabel("L1 Loss")
        ax.legend()

        # (0,1) Number of points
        ax = axes[0, 1]
        d = evals.get("test", evals.get("train", {}))
        if "n_points" in d:
            ax.plot(d["iteration"], d["n_points"] / 1e6, "-o",
                    color="C4", linewidth=2, markersize=5)
        ax.set_title("Gaussian Points"); ax.set_xlabel("Iter"); ax.set_ylabel("M points")

        # (0,2) empty
        axes[0, 2].axis("off")

        eval_axes = axes[1]

    # Bottom row: Eval PSNR, SSIM, LPIPS
    ax = eval_axes[0]
    for split, style, color in [("test", "-o", "C0"), ("train", "--s", "C1")]:
        d = evals.get(split, {})
        if "psnr" in d:
            ax.plot(d["iteration"], d["psnr"], style, color=color,
                    linewidth=1.5, markersize=5, label=f"{split}")
    ax.set_title("Eval PSNR"); ax.set_xlabel("Iter"); ax.set_ylabel("PSNR (dB)")
    ax.legend()

    ax = eval_axes[1]
    for split, style, color in [("test", "-o", "C0"), ("train", "--s", "C1")]:
        d = evals.get(split, {})
        if "ssim" in d:
            ax.plot(d["iteration"], d["ssim"], style, color=color,
                    linewidth=1.5, markersize=5, label=f"{split}")
    ax.set_title("Eval SSIM"); ax.set_xlabel("Iter"); ax.set_ylabel("SSIM")
    ax.legend()

    ax = eval_axes[2]
    for split, style, color in [("test", "-o", "C0"), ("train", "--s", "C1")]:
        d = evals.get(split, {})
        if "lpips" in d:
            ax.plot(d["iteration"], d["lpips"], style, color=color,
                    linewidth=1.5, markersize=5, label=f"{split}")
    ax.set_title("Eval LPIPS"); ax.set_xlabel("Iter"); ax.set_ylabel("LPIPS")
    ax.legend()

    if has_train_data:
        # Hide unused slot in 2x4
        axes[1, 3].axis("off")

    for row in axes:
        for ax in row:
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    fig.suptitle("Training Dashboard", fontsize=16, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    return fig


# ═══════════════════════════════════════════════════════════════════════════
#  Auto-detection
# ═══════════════════════════════════════════════════════════════════════════

def find_latest_model_path() -> str | None:
    """Walk ``output/`` and find the directory with the newest train_metrics.csv."""
    root = Path("output")
    if not root.exists():
        return None
    candidates = sorted(root.rglob("train_metrics.csv"), key=os.path.getmtime)
    if candidates:
        return str(candidates[-1].parent)
    # Fallback: look for any model directory with log_images
    candidates = sorted(root.rglob("log_images"), key=os.path.getmtime)
    if candidates:
        return str(candidates[-1].parent)
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Visualise training metrics")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to model output directory "
                             "(contains train_metrics.csv / eval_metrics.csv)")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save plot PNGs "
                             "(omit to show interactively)")
    parser.add_argument("--smooth", type=float, default=0.98,
                        help="EMA smoothing factor (default: 0.98)")
    args = parser.parse_args()

    model_path = args.model_path or find_latest_model_path()
    if model_path is None:
        print("ERROR: No model_path provided and auto-detection failed.\n"
              "  Run:  python visualize_metrics.py --model_path <path>")
        sys.exit(1)

    train_csv = os.path.join(model_path, "train_metrics.csv")
    eval_csv  = os.path.join(model_path, "eval_metrics.csv")

    has_train = os.path.isfile(train_csv)
    has_eval  = os.path.isfile(eval_csv)

    if not has_train and not has_eval:
        print(f"ERROR: No CSV files found in {model_path}\n"
              "  Make sure training was run with the updated train.py "
              "(CSV logging enabled).")
        sys.exit(1)

    train_data = load_train_csv(train_csv) if has_train else {}
    eval_data  = load_eval_csv(eval_csv)   if has_eval  else {}

    save_dir = args.save_dir
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print(f"Model path: {model_path}")
    if has_train:
        n = len(train_data.get("iteration", []))
        print(f"  train_metrics.csv: {n} rows")
    if has_eval:
        for s, d in eval_data.items():
            n = len(d.get("iteration", []))
            print(f"  eval_metrics.csv ({s}): {n} rows")

    # Override global smoothing
    global ema_smooth
    _alpha = args.smooth
    _orig = ema_smooth
    def ema_smooth(v, alpha=_alpha):
        return _orig(v, alpha)

    print("\nGenerating plots...")

    if has_train:
        plot_train_loss(train_data,
                        os.path.join(save_dir, "train_loss.png") if save_dir else None)
        plot_train_psnr(train_data,
                        os.path.join(save_dir, "train_psnr.png") if save_dir else None)

    if has_eval:
        plot_eval_metrics(eval_data,
                          os.path.join(save_dir, "eval_metrics.png") if save_dir else None)
        plot_eval_comparison(eval_data,
                             os.path.join(save_dir, "eval_comparison.png") if save_dir else None)
        plot_points_over_time(eval_data,
                              os.path.join(save_dir, "points_over_time.png") if save_dir else None)

    if has_train or has_eval:
        plot_summary_dashboard(train_data, eval_data,
                               os.path.join(save_dir, "summary.png") if save_dir else None)

    if not save_dir:
        print("\nShowing plots interactively (close windows to exit)...")
        plt.show()
    else:
        print(f"\nAll plots saved to: {save_dir}")


if __name__ == "__main__":
    main()
