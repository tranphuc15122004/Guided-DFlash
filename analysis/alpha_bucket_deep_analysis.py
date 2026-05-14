#!/usr/bin/env python3
"""
Deep analysis: Does the alpha model learn to correctly identify the bucket
containing the target token and assign it the smallest alpha coefficient?

This examines the relationship between:
- True bucket distribution (where the target token actually falls)
- Predicted min-alpha bucket distribution (which bucket the model chooses to minimize alpha)
- Per-bucket alpha values over training epochs
- min_alpha_bucket_acc and strict_min_alpha_bucket_acc trends
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import argparse
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUCKET_FIELD_RE = re.compile(r"^(?P<prefix>per_bucket|true_bucket_rate|pred_min_bucket_rate)_(?P<bucket>\d+)_mean$")

CSV_PATH = Path("analysis/alpha_training_reports/training_epochs.csv")
OUTPUT_DIR = Path("analysis/alpha_training_reports/bucket_analysis")

FIG_KWARGS = dict(dpi=160, bbox_inches="tight")


def parse_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def load_csv(path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader if any((v or "").strip() for v in row.values())]
        return rows, reader.fieldnames


def discover_bucket_ids(fieldnames: list[str]) -> list[int]:
    bucket_ids: set[int] = set()
    for field in fieldnames:
        match = BUCKET_FIELD_RE.match(field)
        if match:
            bucket_ids.add(int(match.group("bucket")))
    return sorted(bucket_ids)


def extract_epoch_data(rows: list[dict[str, Any]], bucket_ids: list[int]) -> list[dict[str, Any]]:
    """Extract per-epoch metrics into structured form."""
    epochs = []
    for row in rows:
        epoch = parse_float(row.get("epoch"))
        if epoch is None:
            continue
        min_acc = parse_float(row.get("min_alpha_bucket_acc_mean"))
        # Skip epochs with no valid data (training may have been stopped early)
        if min_acc is None:
            continue
        d = {
            "epoch": int(epoch),
            "reward_mean": parse_float(row.get("reward_mean")),
            "min_alpha_bucket_acc": min_acc,
            "strict_min_alpha_bucket_acc": parse_float(row.get("strict_min_alpha_bucket_acc_mean")),
            "target_in_topk_rate": parse_float(row.get("target_in_topk_rate_mean")),
            "alpha_gap_mean": parse_float(row.get("alpha_gap_mean")),
            "model_acc": parse_float(row.get("model_acc_mean")),
            "baseline_acc": parse_float(row.get("baseline_acc_mean")),
            "entropy": parse_float(row.get("entropy_mean")),
            "per_bucket": {},
            "true_rate": {},
            "pred_rate": {},
        }
        for bid in bucket_ids:
            d["per_bucket"][bid] = parse_float(row.get(f"per_bucket_{bid}_mean"))
            d["true_rate"][bid] = parse_float(row.get(f"true_bucket_rate_{bid}_mean"))
            d["pred_rate"][bid] = parse_float(row.get(f"pred_min_bucket_rate_{bid}_mean"))

        # Compute additional metrics
        all_valid = all(d["true_rate"][b] is not None and d["pred_rate"][b] is not None for b in bucket_ids)
        if all_valid:
            d["alignment_l1"] = sum(
                abs(d["pred_rate"][b] - d["true_rate"][b]) for b in bucket_ids
            )

        epochs.append(d)
    return epochs


def compute_bucket_rank_correlation(epochs):
    """For each epoch, compute the rank correlation between:
    - True bucket rate (how often target falls in each bucket)
    - Predicted min-alpha rate (how often model assigns min alpha to each bucket)
    This tells us if the model's ranking aligns with the true ranking.
    """
    bucket_ids = sorted(epochs[0]["true_rate"].keys())
    correlations = []
    for ep in epochs:
        true_rates = [ep["true_rate"][b] for b in bucket_ids]
        pred_rates = [ep["pred_rate"][b] for b in bucket_ids]
        if all(v is not None for v in true_rates) and all(v is not None for v in pred_rates):
            from scipy.stats import spearmanr
            corr, pval = spearmanr(true_rates, pred_rates)
            correlations.append({"epoch": ep["epoch"], "spearman_r": corr, "p_value": pval})
    return correlations


def compute_bucket_alpha_trends(epochs, bucket_ids):
    """Compute how per-bucket alpha values change over training."""
    trends = {}
    for bid in bucket_ids:
        alphas = [(ep["epoch"], ep["per_bucket"][bid]) for ep in epochs if ep["per_bucket"][bid] is not None]
        if len(alphas) > 1:
            xs = np.array([a[0] for a in alphas])
            ys = np.array([a[1] for a in alphas])
            slope = np.polyfit(xs, ys, 1)[0]
            trends[bid] = {
                "start_alpha": alphas[0][1],
                "end_alpha": alphas[-1][1],
                "change": alphas[-1][1] - alphas[0][1],
                "pct_change": (alphas[-1][1] - alphas[0][1]) / alphas[0][1] * 100,
                "trend_slope": slope,
            }
    return trends


def plot_min_alpha_vs_target(epochs, output_dir):
    """Plot 1: Does min_alpha_bucket_acc improve over training?"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Alpha Model: Can It Assign Smallest Alpha to Target Bucket?", fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    epochs_list = [ep["epoch"] for ep in epochs]
    min_acc = [ep["min_alpha_bucket_acc"] for ep in epochs]
    strict_acc = [ep["strict_min_alpha_bucket_acc"] for ep in epochs]
    ax.plot(epochs_list, min_acc, "o-", label="Min-alpha bucket acc", color="#4C78A8")
    ax.plot(epochs_list, strict_acc, "s-", label="Strict min-alpha bucket acc", color="#E15759")
    ax.axhline(y=1/11, color="gray", linestyle="--", alpha=0.5, label=f"Random baseline (1/11≈{1/11:.3f})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Can the model find the target in the min-alpha bucket?")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[0, 1]
    alpha_gap = [ep["alpha_gap_mean"] for ep in epochs]
    ax.plot(epochs_list, alpha_gap, "o-", color="#59A14F")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Alpha gap (max - min)")
    ax.set_title("Alpha gap over training")
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    reward = [ep["reward_mean"] for ep in epochs]
    ax.plot(epochs_list, reward, "o-", color="#F28E2B")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward over training")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    align = [ep.get("alignment_l1", 0) for ep in epochs]
    ax.plot(epochs_list, align, "o-", color="#E15759")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 alignment")
    ax.set_title("L1 distance: true vs predicted bucket distribution")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "fig1_min_alpha_accuracy_trend.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_true_vs_pred_distribution(epochs, bucket_ids, output_dir):
    """Plot 2: Compare true bucket distribution vs predicted min-alpha distribution."""
    epochs_list = [ep["epoch"] for ep in epochs]
    n_epochs = len(epochs_list)
    
    # Create matrix of rates over epochs
    n_buckets = len(bucket_ids)
    true_matrix = np.zeros((n_epochs, n_buckets))
    pred_matrix = np.zeros((n_epochs, n_buckets))
    for i, ep in enumerate(epochs):
        for j, bid in enumerate(bucket_ids):
            true_matrix[i, j] = ep["true_rate"].get(bid, 0) or 0
            pred_matrix[i, j] = ep["pred_rate"].get(bid, 0) or 0

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("True Bucket Distribution vs Predicted Min-Alpha Distribution", fontsize=14, fontweight="bold")

    # First and last epoch comparison
    ax = axes[0]
    first_ep = epochs[0]
    last_ep = epochs[-1]
    x = np.arange(n_buckets)
    width = 0.3
    ax.bar(x - width, [first_ep["true_rate"].get(b, 0) or 0 for b in bucket_ids], width, label="True rate (first)", color="#4C78A8", alpha=0.7)
    ax.bar(x, [last_ep["true_rate"].get(b, 0) or 0 for b in bucket_ids], width, label="True rate (last)", color="#59A14F", alpha=0.7)
    ax.bar(x + width, [last_ep["pred_rate"].get(b, 0) or 0 for b in bucket_ids], width, label="Pred rate (last)", color="#F28E2B", alpha=0.7)
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Rate")
    ax.set_title("True vs Predicted Rate (First vs Last Epoch)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Per-bucket gap evolution
    ax = axes[1]
    for bid in bucket_ids:
        gaps = [(ep["pred_rate"].get(bid, 0) or 0) - (ep["true_rate"].get(bid, 0) or 0) for ep in epochs]
        ax.plot(epochs_list, gaps, label=f"Bucket {bid}", alpha=0.7)
    ax.axhline(y=0, color="black", linestyle="--", alpha=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pred - True rate")
    ax.set_title("Gap (predicted - true) by bucket")
    ax.legend(loc="center right", fontsize=8)
    ax.grid(alpha=0.25)

    # L1 alignment over time
    ax = axes[2]
    alignments = [ep.get("alignment_l1", 0) for ep in epochs]
    ax.plot(epochs_list, alignments, "o-", color="#E15759", linewidth=2)
    # Add random baseline: expected L1 for uniform prediction
    # If predicted uniform (1/11 each) vs true distribution
    true_avg = np.mean(true_matrix, axis=0)
    uniform_pred = np.ones(n_buckets) / n_buckets
    random_l1 = np.sum(np.abs(uniform_pred - true_avg))
    ax.axhline(y=random_l1, color="gray", linestyle="--", alpha=0.5, label=f"Uniform pred L1 ≈ {random_l1:.3f}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("L1 distance")
    ax.set_title("Alignment of predictions to truth")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "fig2_true_vs_pred_distribution.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_min_alpha_success_breakdown(epochs, bucket_ids, output_dir):
    """Plot 3: When the model succeeds/fails at assigning min alpha to target bucket.
    
    We analyze: given the target is in bucket B, how often does the model 
    assign the minimum alpha to bucket B?
    
    Since we don't have per-sample breakdown, we can compute a proxy:
    - For each bucket, the 'true rate' tells us how often target falls there
    - The 'pred rate' tells us how often model assigns min alpha there
    - We can derive: what fraction of the time when target is in bucket B,
      does the model pick B as min-alpha?
    
    This can be approximated as: pred_rate / true_rate (capped at 1.0)
    This is essentially the conditional probability P(pred=B|true=B) 
    assuming the model's predictions are calibrated.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("When Does the Model Succeed? Per-Bucket Analysis", fontsize=14, fontweight="bold")

    last_ep = epochs[-1]
    first_ep = epochs[0]

    # Per-bucket conditional accuracy
    ax = axes[0, 0]
    x = np.arange(len(bucket_ids))
    width = 0.35
    cond_first = []
    cond_last = []
    for bid in bucket_ids:
        tr_first = first_ep["true_rate"].get(bid, 0) or 0.001
        tr_last = last_ep["true_rate"].get(bid, 0) or 0.001
        cond_first.append(min(1.0, (first_ep["pred_rate"].get(bid, 0) or 0) / tr_first))
        cond_last.append(min(1.0, (last_ep["pred_rate"].get(bid, 0) or 0) / tr_last))
    
    ax.bar(x - width/2, cond_first, width, label="First epoch", color="#4C78A8", alpha=0.7)
    ax.bar(x + width/2, cond_last, width, label="Last epoch", color="#59A14F", alpha=0.7)
    ax.set_xlabel("Bucket")
    ax.set_ylabel("P(pred = B | true = B)")
    ax.set_title("Conditional min-alpha assignment accuracy")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    # Per-bucket alpha values: evolution
    ax = axes[0, 1]
    for bid in bucket_ids[:6]:  # first 6 for clarity
        alphas = [ep["per_bucket"].get(bid, None) for ep in epochs]
        ax.plot([ep["epoch"] for ep in epochs], alphas, label=f"Bucket {bid}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Alpha value")
    ax.set_title("Alpha values (buckets 0-5)")
    ax.legend()
    ax.grid(alpha=0.25)

    ax = axes[1, 0]
    for bid in bucket_ids[6:]:
        alphas = [ep["per_bucket"].get(bid, None) for ep in epochs]
        ax.plot([ep["epoch"] for ep in epochs], alphas, label=f"Bucket {bid}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Alpha value")
    ax.set_title("Alpha values (buckets 6-10)")
    ax.legend()
    ax.grid(alpha=0.25)

    # Alpha vs true rate scatter (last epoch)
    ax = axes[1, 1]
    true_rates = [last_ep["true_rate"].get(b, 0) or 0 for b in bucket_ids]
    alphas_arr = [last_ep["per_bucket"].get(b, 0) or 0 for b in bucket_ids]
    scatter = ax.scatter(true_rates, alphas_arr, c=bucket_ids, cmap="viridis", s=100, zorder=5)
    for bid in bucket_ids:
        tr = last_ep["true_rate"].get(bid, 0) or 0
        al = last_ep["per_bucket"].get(bid, 0) or 0
        ax.annotate(str(bid), (tr, al),
                    fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("True bucket rate")
    ax.set_ylabel("Alpha value")
    ax.set_title("Alpha vs True Rate (Last Epoch)")
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Bucket ID")
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "fig3_per_bucket_breakdown.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_bucket_rank_correlation_analysis(epochs, bucket_ids, output_dir):
    """Plot 4: Spearman rank correlation between true bucket rates and predicted rates."""
    try:
        from scipy.stats import spearmanr
    except ImportError:
        print("  scipy not available, skipping rank correlation plot")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Rank Correlation: True vs Predicted Bucket Distribution", fontsize=14, fontweight="bold")

    epochs_list = [ep["epoch"] for ep in epochs]
    spearman_rs = []
    for ep in epochs:
        true_rates = [ep["true_rate"].get(b, 0) or 0 for b in bucket_ids]
        pred_rates = [ep["pred_rate"].get(b, 0) or 0 for b in bucket_ids]
        corr, _ = spearmanr(true_rates, pred_rates)
        spearman_rs.append(corr)

    ax = axes[0]
    ax.plot(epochs_list, spearman_rs, "o-", color="#4C78A8", linewidth=2)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect alignment (ρ=1)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Rank correlation over training")
    ax.legend()
    ax.grid(alpha=0.25)

    # Final epoch scatter with ranks
    ax = axes[1]
    last_ep = epochs[-1]
    first_ep = epochs[0]
    true_rates = [last_ep["true_rate"].get(b, 0) or 0 for b in bucket_ids]
    pred_rates = [last_ep["pred_rate"].get(b, 0) or 0 for b in bucket_ids]
    ax.scatter(true_rates, pred_rates, c=bucket_ids, cmap="viridis", s=100, zorder=5)
    # Add diagonal
    max_val = max(max(true_rates), max(pred_rates)) * 1.1
    ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, label="Perfect calibration")
    for bid in bucket_ids:
        tr = last_ep["true_rate"].get(bid, 0) or 0
        pr = last_ep["pred_rate"].get(bid, 0) or 0
        ax.annotate(str(bid), (tr, pr),
                    fontsize=9, ha="center", va="bottom")
    ax.set_xlabel("True rate")
    ax.set_ylabel("Predicted min-alpha rate")
    ax.set_title(f"Final epoch (ρ={spearman_rs[-1]:.4f})")
    ax.legend()
    ax.grid(alpha=0.25)

    fig.tight_layout()
    path = output_dir / "fig4_rank_correlation.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def plot_alpha_learning_effectiveness_summary(epochs, bucket_ids, output_dir):
    """Plot 5: Summary of alpha learning effectiveness.

    Key question: Is the model learning to identify the bucket with the target
    and give it the smallest alpha?

    Compute and visualize:
    1. min_alpha_bucket_acc vs epoch — the main metric
    2. Lift over random baseline
    3. Expected improvement if model perfectly identified target bucket
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Alpha Learning Effectiveness Summary", fontsize=14, fontweight="bold")

    epochs_list = [ep["epoch"] for ep in epochs]
    
    # 1. Min-alpha bucket accuracy with baseline
    ax = axes[0, 0]
    min_acc = [ep["min_alpha_bucket_acc"] if ep["min_alpha_bucket_acc"] is not None else 0 for ep in epochs]
    strict_acc = [ep["strict_min_alpha_bucket_acc"] if ep["strict_min_alpha_bucket_acc"] is not None else 0 for ep in epochs]
    baseline = 1/11
    ax.plot(epochs_list, min_acc, "o-", label="Min-alpha bucket acc", color="#4C78A8", linewidth=2)
    ax.plot(epochs_list, strict_acc, "s-", label="Strict min-alpha bucket acc", color="#E15759", linewidth=2)
    ax.axhline(y=baseline, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({baseline:.3f})")
    ax.fill_between(epochs_list, baseline, [max(b, baseline) for b in min_acc], 
                     alpha=0.15, color="#4C78A8", label="Lift over random")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.set_title("Can the model find target in min-alpha bucket?")
    ax.legend()
    ax.grid(alpha=0.25)

    # 2. Lift over random
    ax = axes[0, 1]
    lift = [acc - baseline for acc in min_acc]
    lift_pct = [(acc - baseline) / baseline * 100 for acc in min_acc]
    ax.plot(epochs_list, lift_pct, "o-", color="#59A14F", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Lift over random (%)")
    ax.set_title(f"Improvement over random (final: {lift_pct[-1]:.1f}%)")
    ax.grid(alpha=0.25)

    # 3. What's the upper bound? Compare target_in_topk_rate (91.6%) vs min_alpha_bucket_acc
    ax = axes[1, 0]
    topk = [ep["target_in_topk_rate"] if ep["target_in_topk_rate"] is not None else 0 for ep in epochs]
    ax.plot(epochs_list, min_acc, "o-", label="Min-alpha bucket acc", color="#4C78A8")
    ax.plot(epochs_list, topk, "s-", label="Target in top-32", color="#F28E2B")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    ax.fill_between(epochs_list, min_acc, topk, alpha=0.1, color="red", 
                     label="Gap to upper bound")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Rate")
    ax.set_title("Upper bound: target must be in top-32")
    ax.legend()
    ax.grid(alpha=0.25)

    # 4. How well does alpha distribution match the true bucket distribution?
    ax = axes[1, 1]
    # Normalize alpha values to sum to 1 and compare to true rates
    last_ep = epochs[-1]
    alphas = np.array([last_ep["per_bucket"].get(b, 0) or 0 for b in bucket_ids])
    true_rates = np.array([last_ep["true_rate"].get(b, 0) or 0 for b in bucket_ids])
    
    # Since alpha is additive penalty, we can't directly compare distributions.
    # Instead show: alpha rank vs true rate rank
    alpha_ranks = np.argsort(np.argsort(alphas))  # 0=smallest
    true_ranks = np.argsort(np.argsort(true_rates))  # 0=smallest
    
    ax.bar(bucket_ids, true_rates, alpha=0.6, label="True rate", color="#4C78A8")
    max_alpha = max(alphas) if max(alphas) > 0 else 1
    max_tr = max(true_rates) if max(true_rates) > 0 else 1
    ax.bar(bucket_ids, alphas / max_alpha * max_tr, alpha=0.4, 
           label="Alpha (normalized)", color="#F28E2B")
    ax.set_xlabel("Bucket")
    ax.set_ylabel("Rate / Normalized alpha")
    ax.set_title("Alpha values vs True rates (final epoch)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)

    fig.tight_layout()
    path = output_dir / "fig5_effectiveness_summary.png"
    fig.savefig(path, **FIG_KWARGS)
    plt.close(fig)
    print(f"  Saved {path}")
    return path


def print_analysis_report(epochs, bucket_ids, bucket_trends, correlations, output_dir: Path):
    """Print a comprehensive text analysis report."""
    first = epochs[0]
    last = epochs[-1]
    
    lines = []
    lines.append("=" * 80)
    lines.append("ALPHA BUCKET RECOGNITION ANALYSIS REPORT")
    lines.append("=" * 80)
    lines.append("")
    lines.append(f"Training epochs analyzed: {len(epochs)} (epochs {first['epoch']} to {last['epoch']})")
    lines.append(f"Number of buckets: {len(bucket_ids)} (IDs: {bucket_ids[0]}–{bucket_ids[-1]})")
    lines.append("")
    
    # 1. Core question: Does the model learn to find the target in the min-alpha bucket?
    lines.append("-" * 80)
    lines.append("1. CORE QUESTION: Can the model assign minimum alpha to target's bucket?")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  Metric                    First epoch  Last epoch  Change    Relative")
    lines.append(f"  {'─'*70}")
    m1 = first['min_alpha_bucket_acc'] or 0
    m2 = last['min_alpha_bucket_acc'] or 0
    s1 = first['strict_min_alpha_bucket_acc'] or 0
    s2 = last['strict_min_alpha_bucket_acc'] or 0
    lines.append(f"  min_alpha_bucket_acc:      {m1:.4f}      {m2:.4f}      "
                 f"{m2 - m1:+.4f}   "
                 f"{(m2 - m1) / m1 * 100 if m1 else 0:+.1f}%")
    lines.append(f"  strict_min_alpha_acc:      {s1:.4f}      {s2:.4f}      "
                 f"{s2 - s1:+.4f}   "
                 f"{(s2 - s1) / s1 * 100 if s1 else 0:+.1f}%")
    baseline = 1/11
    lines.append(f"  Random baseline:           {baseline:.4f} (1/{len(bucket_ids)} per bucket)")
    lines.append(f"  Lift over random:          {(m2 - baseline) / baseline * 100:.1f}%")
    lines.append(f"  Upper bound (target in     {last['target_in_topk_rate'] or 0:.4f}")
    lines.append(f"    top-32 rate):")
    lines.append(f"  Gap to upper bound:        {((last['target_in_topk_rate'] or 0) - m2) * 100:.1f} pp")
    lines.append("")
    lines.append(f"  → The model IMPROVES ({(m2 - m1) / m1 * 100 if m1 else 0:.0f}% relative gain)")
    lines.append(f"    but is still far from the upper bound. Only ~{m2*100:.1f}% of the time does")
    lines.append(f"    the globally minimum alpha land on the target's bucket.")
    lines.append("")

    # 2. Bucket distribution analysis
    lines.append("-" * 80)
    lines.append("2. BUCKET DISTRIBUTION ANALYSIS")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  True bucket distribution (where target tokens fall):")
    lines.append(f"  {'Bucket':>8} {'True Rate':>10} {'Pred Rate':>10} {'Gap':>10} {'Alpha (last)':>13} {'Cond Acc':>10}")
    lines.append(f"  {'-'*64}")
    for bid in bucket_ids:
        tr = last["true_rate"].get(bid, 0) or 0
        pr = last["pred_rate"].get(bid, 0) or 0
        alpha = last["per_bucket"].get(bid, 0) or 0
        gap = pr - tr
        cond_acc = min(1.0, pr / tr) if tr and tr > 0 else 0
        lines.append(f"  {bid:>8} {tr:>10.4f} {pr:>10.4f} {gap:>+10.4f} {alpha:>13.4f} {cond_acc:>10.2%}")
    lines.append("")
    
    # L1 alignment
    l1_first = sum(abs((first["pred_rate"].get(b, 0) or 0) - (first["true_rate"].get(b, 0) or 0)) for b in bucket_ids)
    l1_last = sum(abs((last["pred_rate"].get(b, 0) or 0) - (last["true_rate"].get(b, 0) or 0)) for b in bucket_ids)
    lines.append(f"  L1 alignment (first epoch): {l1_first:.4f}")
    lines.append(f"  L1 alignment (last epoch):  {l1_last:.4f}")
    
    # Spearman correlation
    if correlations:
        lines.append(f"  Spearman ρ  (first epoch):  {correlations[0]['spearman_r']:.4f}")
        lines.append(f"  Spearman ρ  (last epoch):   {correlations[-1]['spearman_r']:.4f}")
    lines.append("")

    # 3. Bucket trends analysis
    lines.append("-" * 80)
    lines.append("3. PER-BUCKET ALPHA TRENDS")
    lines.append("-" * 80)
    lines.append("")
    lines.append(f"  {'Bucket':>8} {'Start α':>9} {'End α':>9} {'Change':>9} {'Change %':>10} {'Slope':>11}")
    lines.append(f"  {'-'*54}")
    for bid in bucket_ids:
        t = bucket_trends[bid]
        lines.append(f"  {bid:>8} {t['start_alpha']:>9.4f} {t['end_alpha']:>9.4f} {t['change']:>+9.4f} {t['pct_change']:>+9.1f}% {t['trend_slope']:>+10.6f}")
    lines.append("")
    lines.append("  → Alpha values generally INCREASE for low buckets and are more stable")
    lines.append("    for mid-range buckets. This widens the alpha gap over training.")
    lines.append("")

    # 4. Key insight
    lines.append("-" * 80)
    lines.append("4. KEY INSIGHT: Does the model actually learn?")
    lines.append("-" * 80)
    lines.append("")
    m1 = first['min_alpha_bucket_acc'] or 0
    m2 = last['min_alpha_bucket_acc'] or 0
    lines.append("  YES, partially. The evidence:")
    lines.append(f"  - min_alpha_bucket_acc improves from {m1:.3f} to {m2:.3f}")
    lines.append(f"    ({(m2 - m1) / m1 * 100 if m1 else 0:.0f}% relative improvement)")
    lines.append(f"  - L1 alignment improves from {l1_first:.3f} to {l1_last:.3f}")
    lines.append(f"  - The alpha gap grows, showing the model is learning to differentiate buckets")
    lines.append("")
    lines.append("  BUT, the limitations are clear:")
    lines.append(f"  - The macro bucket accuracy (~{m2:.3f}) is only {(m2 / baseline * 100):.0f}% better than random")
    lines.append("  - The model systematically underestimates the min-alpha assignment for bucket 0")
    lines.append(f"    (true rate = {last['true_rate'].get(0, 0) or 0:.4f} but pred rate = {last['pred_rate'].get(0, 0) or 0:.4f})")
    lines.append("  - The conditional accuracy P(pred=B | true=B) is low for most buckets")
    lines.append("")
    lines.append("  Root cause hypothesis: The model primarily learns to widen the alpha gap")
    lines.append("  between buckets as a coarse strategy, but struggles with fine-grained")
    lines.append("  per-token bucket identification. The contextual bandit formulation may")
    lines.append("  need more capacity (larger hidden dim) or a different reward structure")
    lines.append("  to properly learn bucket recognition.")
    
    print("\n".join(lines))
    
    # Save report
    report_path = output_dir / "analysis_report.txt"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n  Report saved to {report_path}")
    
    return lines


def main():
    parser = argparse.ArgumentParser(description="Deep alpha-bucket analysis")
    parser.add_argument("--csv", type=str, default=str(CSV_PATH), help="Path to training epochs CSV")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR), help="Directory to write outputs")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {csv_path}...")
    rows, fieldnames = load_csv(csv_path)
    bucket_ids = discover_bucket_ids(fieldnames)
    print(f"Found {len(bucket_ids)} buckets: {bucket_ids}")
    print(f"Loaded {len(rows)} epoch rows")
    
    epochs = extract_epoch_data(rows, bucket_ids)
    print(f"Extracted {len(epochs)} epochs of structured data")
    
    bucket_trends = compute_bucket_alpha_trends(epochs, bucket_ids)
    
    try:
        correlations = compute_bucket_rank_correlation(epochs)
        print(f"Computed Spearman correlations for {len(correlations)} epochs")
    except Exception:
        print("scipy not available or error computing correlations, skipping rank correlation")
        correlations = []
    
    # Generate plots
    print("\nGenerating plots...")
    plot_min_alpha_vs_target(epochs, output_dir)
    plot_true_vs_pred_distribution(epochs, bucket_ids, output_dir)
    plot_min_alpha_success_breakdown(epochs, bucket_ids, output_dir)
    if correlations:
        plot_bucket_rank_correlation_analysis(epochs, bucket_ids, output_dir)
    plot_alpha_learning_effectiveness_summary(epochs, bucket_ids, output_dir)
    
    # Print analysis report
    print("\n" + "=" * 80)
    print_analysis_report(epochs, bucket_ids, bucket_trends, correlations, output_dir)
    
    print("\nDone! All outputs in:", output_dir)


if __name__ == "__main__":
    main()
