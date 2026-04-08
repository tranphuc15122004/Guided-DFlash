import argparse
import csv
import json
import os
import time
import random
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import *
import distributed as dist

NEGATIVE_HIDDEN_MODE = 'shuffle_tokens' 
TOP64_MASK_MODE = True 

""" 
Schema: CDv2
Goal: To statistic and study draft model entropy
"""


def safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    if report_dir:
        path = Path(report_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("analysis") / "results" / f"draft_entropy_{safe_dataset_name(dataset_name)}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _describe_distribution(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {}
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.1)),
        "median": float(np.quantile(values, 0.5)),
        "p90": float(np.quantile(values, 0.9)),
        "max": float(values.max()),
    }


def _build_histogram(values: np.ndarray, bins: int, value_range: tuple[float, float]) -> dict[str, Any]:
    bins = max(2, int(bins))
    counts, bin_edges = np.histogram(values, bins=bins, range=value_range)
    total = max(1, int(values.size))
    rates = counts.astype(np.float64) / float(total)
    return {
        "bin_edges": [float(x) for x in bin_edges.tolist()],
        "counts": [int(x) for x in counts.tolist()],
        "rates": [float(x) for x in rates.tolist()],
    }


def _safe_corrcoef(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if np.allclose(x.std(), 0.0) or np.allclose(y.std(), 0.0):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _safe_mask_mean(values: np.ndarray, mask: np.ndarray) -> float:
    count = int(mask.sum())
    if count == 0:
        return 0.0
    return float(values[mask].mean())


def _build_calibration(confidence: np.ndarray, correctness: np.ndarray, num_bins: int) -> dict[str, Any]:
    num_bins = max(2, int(num_bins))
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    rows: list[dict[str, Any]] = []
    ece = 0.0
    mce = 0.0
    total = max(1, int(confidence.size))

    for i in range(num_bins):
        lo = float(bin_edges[i])
        hi = float(bin_edges[i + 1])
        if i == num_bins - 1:
            mask = (confidence >= lo) & (confidence <= hi)
        else:
            mask = (confidence >= lo) & (confidence < hi)

        count = int(mask.sum())
        if count == 0:
            rows.append(
                {
                    "bin_left": lo,
                    "bin_right": hi,
                    "count": 0,
                    "rate": 0.0,
                    "avg_confidence": 0.0,
                    "accuracy": 0.0,
                    "gap": 0.0,
                }
            )
            continue

        avg_conf = float(confidence[mask].mean())
        acc = float(correctness[mask].mean())
        gap = abs(acc - avg_conf)
        weight = float(count / total)
        ece += weight * gap
        mce = max(mce, gap)

        rows.append(
            {
                "bin_left": lo,
                "bin_right": hi,
                "count": count,
                "rate": weight,
                "avg_confidence": avg_conf,
                "accuracy": acc,
                "gap": float(gap),
            }
        )

    return {
        "num_bins": int(num_bins),
        "ece": float(ece),
        "mce": float(mce),
        "bins": rows,
    }


def _collect_draft_entropy_records(
    *,
    records: list[dict[str, Any]],
    positive_logits: torch.Tensor,
    sampled_draft_ids: torch.Tensor,
    posterior_ids: torch.Tensor,
    acceptance_lengths: torch.Tensor,
    sample_idx: int,
    turn_idx: int,
    decode_step: int,
    block_start: int,
) -> None:
    if positive_logits.numel() == 0:
        return

    logits = positive_logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    entropy = -(probs * log_probs).sum(dim=-1)
    entropy_bits = entropy / np.log(2.0)
    vocab_size = probs.shape[-1]
    entropy_norm = entropy / max(np.log(vocab_size), 1e-12)
    effective_support = torch.exp(entropy)

    top_k = min(10, vocab_size)
    topk_vals, topk_ids = torch.topk(probs, k=top_k, dim=-1)
    top1_prob = topk_vals[..., 0]
    top1_ids = topk_ids[..., 0]
    top2_prob = topk_vals[..., 1] if top_k > 1 else torch.zeros_like(top1_prob)
    top1_margin = top1_prob - top2_prob
    top5_mass = topk_vals[..., : min(5, top_k)].sum(dim=-1)
    top10_mass = topk_vals.sum(dim=-1)

    target_prob = probs.gather(dim=-1, index=posterior_ids.unsqueeze(-1)).squeeze(-1)
    sampled_prob = probs.gather(dim=-1, index=sampled_draft_ids.unsqueeze(-1)).squeeze(-1)
    target_logprob = log_probs.gather(dim=-1, index=posterior_ids.unsqueeze(-1)).squeeze(-1)
    sampled_logprob = log_probs.gather(dim=-1, index=sampled_draft_ids.unsqueeze(-1)).squeeze(-1)

    batch_size, num_steps = sampled_draft_ids.shape
    for b in range(batch_size):
        accepted_len = int(acceptance_lengths[b].item())
        for i in range(num_steps):
            sampled_id = int(sampled_draft_ids[b, i].item())
            target_id = int(posterior_ids[b, i].item())
            positive_top1_id = int(top1_ids[b, i].item())
            sampled_match = int(sampled_id == target_id)
            accepted_by_target = int(i < accepted_len)

            records.append(
                {
                    "sample_idx": int(sample_idx),
                    "turn_idx": int(turn_idx),
                    "decode_step": int(decode_step),
                    "absolute_position": int(block_start + i + 1),
                    "block_position": int(i + 1),
                    "sampled_draft_id": sampled_id,
                    "target_token_id": target_id,
                    "positive_top1_id": positive_top1_id,
                    "sampled_matches_target": sampled_match,
                    "accepted_by_target": accepted_by_target,
                    "positive_argmax_hit": int(positive_top1_id == target_id),
                    "entropy_nats": float(entropy[b, i].item()),
                    "entropy_bits": float(entropy_bits[b, i].item()),
                    "normalized_entropy": float(entropy_norm[b, i].item()),
                    "effective_support": float(effective_support[b, i].item()),
                    "top1_prob": float(top1_prob[b, i].item()),
                    "top2_prob": float(top2_prob[b, i].item()),
                    "top1_margin": float(top1_margin[b, i].item()),
                    "top5_mass": float(top5_mass[b, i].item()),
                    "top10_mass": float(top10_mass[b, i].item()),
                    "target_prob": float(target_prob[b, i].item()),
                    "target_logprob": float(target_logprob[b, i].item()),
                    "sampled_prob_under_positive": float(sampled_prob[b, i].item()),
                    "sampled_logprob_under_positive": float(sampled_logprob[b, i].item()),
                    "target_minus_sampled_prob": float((target_prob[b, i] - sampled_prob[b, i]).item()),
                }
            )


def _build_entropy_band_summary(
    entropy_norm: np.ndarray,
    top1_prob: np.ndarray,
    positive_hit: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    bands = {
        "low_entropy_[0.0,0.4)": (0.0, 0.4),
        "mid_entropy_[0.4,0.7)": (0.4, 0.7),
        "high_entropy_[0.7,1.0]": (0.7, 1.0000001),
    }
    out: dict[str, Any] = {}
    total = max(1, int(entropy_norm.size))
    for name, (lo, hi) in bands.items():
        mask = (entropy_norm >= lo) & (entropy_norm < hi)
        count = int(mask.sum())
        if count == 0:
            out[name] = {
                "count": 0,
                "rate": 0.0,
                "mean_top1_prob": 0.0,
                "positive_argmax_hit_rate": 0.0,
                "accepted_by_target_rate": 0.0,
            }
            continue
        out[name] = {
            "count": count,
            "rate": float(count / total),
            "mean_top1_prob": float(top1_prob[mask].mean()),
            "positive_argmax_hit_rate": float(positive_hit[mask].mean()),
            "accepted_by_target_rate": float(accepted[mask].mean()),
        }
    return out


def _build_entropy_band_distribution(entropy_norm: np.ndarray) -> dict[str, Any]:
    bands = {
        "low_entropy_[0.0,0.4)": (0.0, 0.4),
        "mid_entropy_[0.4,0.7)": (0.4, 0.7),
        "high_entropy_[0.7,1.0]": (0.7, 1.0000001),
    }
    out: dict[str, Any] = {}
    total = max(1, int(entropy_norm.size))
    for name, (lo, hi) in bands.items():
        mask = (entropy_norm >= lo) & (entropy_norm < hi)
        count = int(mask.sum())
        out[name] = {
            "count": count,
            "rate": float(count / total),
        }
    return out


def build_entropy_summary(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    block_size: int,
    decoding_speedup: float,
    avg_acceptance_length: float,
) -> dict[str, Any]:
    meta = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "block_size": int(block_size),
        "model_name_or_path": args.model_name_or_path,
        "draft_name_or_path": args.draft_name_or_path,
        "temperature": float(args.temperature),
        "cd_alpha": float(args.cd_alpha),
        "cd_beta": float(args.cd_beta),
        "negative_context_dropout": float(args.negative_context_dropout),
        "negative_context_noise_std": float(args.negative_context_noise_std),
        "negative_hidden_mode": args.negative_hidden_mode,
        "top64_mask_mode": bool(TOP64_MASK_MODE),
        "decoding_speedup": float(decoding_speedup),
        "avg_acceptance_length": float(avg_acceptance_length),
    }

    if not records:
        return {
            "meta": meta,
            "num_records": 0,
            "note": "No draft-token records were collected. This happens when block_size <= 1.",
        }

    entropy = np.asarray([r["entropy_nats"] for r in records], dtype=np.float64)
    entropy_bits = np.asarray([r["entropy_bits"] for r in records], dtype=np.float64)
    entropy_norm = np.asarray([r["normalized_entropy"] for r in records], dtype=np.float64)
    top1_prob = np.asarray([r["top1_prob"] for r in records], dtype=np.float64)
    top1_margin = np.asarray([r["top1_margin"] for r in records], dtype=np.float64)
    top5_mass = np.asarray([r["top5_mass"] for r in records], dtype=np.float64)
    top10_mass = np.asarray([r["top10_mass"] for r in records], dtype=np.float64)
    target_prob = np.asarray([r["target_prob"] for r in records], dtype=np.float64)
    sampled_prob = np.asarray([r["sampled_prob_under_positive"] for r in records], dtype=np.float64)
    target_minus_sampled_prob = np.asarray([r["target_minus_sampled_prob"] for r in records], dtype=np.float64)
    effective_support = np.asarray([r["effective_support"] for r in records], dtype=np.float64)
    positive_hit = np.asarray([r["positive_argmax_hit"] for r in records], dtype=np.float64)
    sampled_match = np.asarray([r["sampled_matches_target"] for r in records], dtype=np.float64)
    accepted = np.asarray([r["accepted_by_target"] for r in records], dtype=np.float64)

    calibration = _build_calibration(top1_prob, positive_hit, num_bins=args.calibration_bins)

    accepted_mask = accepted == 1
    rejected_mask = accepted == 0
    num_records = int(len(records))
    accepted_count = int(accepted_mask.sum())
    rejected_count = int(rejected_mask.sum())
    entropy_norm_median = float(np.median(entropy_norm))

    def _subset_stat(mask: np.ndarray, values: np.ndarray) -> dict[str, float]:
        if int(mask.sum()) == 0:
            return {}
        return _describe_distribution(values[mask])

    confidence_thresholds: dict[str, Any] = {}
    for threshold in [0.5, 0.7, 0.9]:
        mask = top1_prob >= threshold
        key = f"top1_prob_ge_{threshold:.1f}"
        count = int(mask.sum())
        if count == 0:
            confidence_thresholds[key] = {
                "count": 0,
                "coverage_rate": 0.0,
                "positive_argmax_hit_rate": 0.0,
                "accepted_by_target_rate": 0.0,
            }
            continue
        confidence_thresholds[key] = {
            "count": count,
            "coverage_rate": float(mask.mean()),
            "positive_argmax_hit_rate": float(positive_hit[mask].mean()),
            "accepted_by_target_rate": float(accepted[mask].mean()),
        }

    rejected_focus = {
        "rejected_count": rejected_count,
        "rejected_rate": float(rejected_count / max(1, num_records)),
        "accepted_count": accepted_count,
        "accepted_rate": float(accepted_count / max(1, num_records)),
        "positive_argmax_hit_rate": _safe_mask_mean(positive_hit, rejected_mask),
        "sampled_matches_target_rate": _safe_mask_mean(sampled_match, rejected_mask),
        "entropy_norm_ge_global_median_rate": _safe_mask_mean((entropy_norm >= entropy_norm_median).astype(np.float64), rejected_mask),
        "entropy_nats": _subset_stat(rejected_mask, entropy),
        "entropy_bits": _subset_stat(rejected_mask, entropy_bits),
        "normalized_entropy": _subset_stat(rejected_mask, entropy_norm),
        "effective_support": _subset_stat(rejected_mask, effective_support),
        "top1_prob": _subset_stat(rejected_mask, top1_prob),
        "top1_margin": _subset_stat(rejected_mask, top1_margin),
        "top5_mass": _subset_stat(rejected_mask, top5_mass),
        "top10_mass": _subset_stat(rejected_mask, top10_mass),
        "target_prob": _subset_stat(rejected_mask, target_prob),
        "sampled_prob_under_positive": _subset_stat(rejected_mask, sampled_prob),
        "target_minus_sampled_prob": _subset_stat(rejected_mask, target_minus_sampled_prob),
        "rejected_minus_accepted": {
            "entropy_nats_mean": _safe_mask_mean(entropy, rejected_mask) - _safe_mask_mean(entropy, accepted_mask),
            "normalized_entropy_mean": _safe_mask_mean(entropy_norm, rejected_mask) - _safe_mask_mean(entropy_norm, accepted_mask),
            "top1_prob_mean": _safe_mask_mean(top1_prob, rejected_mask) - _safe_mask_mean(top1_prob, accepted_mask),
            "top1_margin_mean": _safe_mask_mean(top1_margin, rejected_mask) - _safe_mask_mean(top1_margin, accepted_mask),
            "target_prob_mean": _safe_mask_mean(target_prob, rejected_mask) - _safe_mask_mean(target_prob, accepted_mask),
        },
        "calibration": _build_calibration(
            top1_prob[rejected_mask],
            positive_hit[rejected_mask],
            num_bins=args.calibration_bins,
        ),
        "entropy_bands": _build_entropy_band_distribution(entropy_norm[rejected_mask]),
        "histograms": {
            "top1_prob": _build_histogram(top1_prob[rejected_mask], bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "normalized_entropy": _build_histogram(entropy_norm[rejected_mask], bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "top1_margin": _build_histogram(top1_margin[rejected_mask], bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "target_prob": _build_histogram(target_prob[rejected_mask], bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "sampled_prob_under_positive": _build_histogram(sampled_prob[rejected_mask], bins=args.histogram_bins, value_range=(0.0, 1.0)),
        },
    }

    return {
        "meta": meta,
        "num_records": num_records,
        "num_unique_samples": int(len(set(int(r["sample_idx"]) for r in records))),
        "num_unique_turns": int(len(set((int(r["sample_idx"]), int(r["turn_idx"])) for r in records))),
        "rates": {
            "positive_argmax_hit_rate": float(positive_hit.mean()),
            "sampled_matches_target_rate": float(sampled_match.mean()),
            "accepted_by_target_rate": float(accepted.mean()),
        },
        "entropy": {
            "entropy_nats": _describe_distribution(entropy),
            "normalized_entropy": _describe_distribution(entropy_norm),
            "effective_support": _describe_distribution(effective_support),
            "accepted_subset_entropy": _subset_stat(accepted_mask, entropy),
            "rejected_subset_entropy": _subset_stat(rejected_mask, entropy),
        },
        "confidence": {
            "top1_prob": _describe_distribution(top1_prob),
            "top1_margin": _describe_distribution(top1_margin),
            "top5_mass": _describe_distribution(top5_mass),
            "top10_mass": _describe_distribution(top10_mass),
            "target_prob": _describe_distribution(target_prob),
            "sampled_prob_under_positive": _describe_distribution(sampled_prob),
            "confidence_thresholds": confidence_thresholds,
        },
        "rejected_token_focus": rejected_focus,
        "entropy_bands": _build_entropy_band_summary(entropy_norm, top1_prob, positive_hit, accepted),
        "correlation": {
            "corr_top1_prob_vs_positive_hit": _safe_corrcoef(top1_prob, positive_hit),
            "corr_top1_margin_vs_positive_hit": _safe_corrcoef(top1_margin, positive_hit),
            "corr_normalized_entropy_vs_positive_hit": _safe_corrcoef(entropy_norm, positive_hit),
            "corr_top1_prob_vs_accepted": _safe_corrcoef(top1_prob, accepted),
            "corr_normalized_entropy_vs_accepted": _safe_corrcoef(entropy_norm, accepted),
        },
        "calibration": calibration,
        "histograms": {
            "top1_prob": _build_histogram(top1_prob, bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "normalized_entropy": _build_histogram(entropy_norm, bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "top1_margin": _build_histogram(top1_margin, bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "target_prob": _build_histogram(target_prob, bins=args.histogram_bins, value_range=(0.0, 1.0)),
            "sampled_prob_under_positive": _build_histogram(sampled_prob, bins=args.histogram_bins, value_range=(0.0, 1.0)),
        },
    }


def maybe_create_entropy_plots(
    report_dir: Path,
    records: list[dict[str, Any]],
    summary: dict[str, Any],
) -> list[str]:
    if not records:
        return []

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        logger.warning(f"Skip plotting because matplotlib is unavailable: {exc}")
        return []

    top1_prob = np.asarray([r["top1_prob"] for r in records], dtype=np.float64)
    entropy_norm = np.asarray([r["normalized_entropy"] for r in records], dtype=np.float64)
    positive_hit = np.asarray([r["positive_argmax_hit"] for r in records], dtype=np.float64)
    accepted = np.asarray([r["accepted_by_target"] for r in records], dtype=np.float64)

    plot_files: list[str] = []

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(top1_prob[accepted == 1], bins=60, alpha=0.65, label="Accepted")
    plt.hist(top1_prob[accepted == 0], bins=60, alpha=0.65, label="Rejected")
    plt.xlabel("Top-1 probability (positive draft)")
    plt.ylabel("Count")
    plt.title("Distribution of Draft Top-1 Probability")
    plt.legend()
    p1 = report_dir / "draft_top1_probability_hist.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    plot_files.append(p1.name)

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(entropy_norm[accepted == 1], bins=60, alpha=0.65, label="Accepted")
    plt.hist(entropy_norm[accepted == 0], bins=60, alpha=0.65, label="Rejected")
    plt.xlabel("Normalized entropy")
    plt.ylabel("Count")
    plt.title("Distribution of Draft Normalized Entropy")
    plt.legend()
    p2 = report_dir / "draft_normalized_entropy_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    plot_files.append(p2.name)

    rejected_mask = accepted == 0
    if int(rejected_mask.sum()) > 0:
        plt.figure(figsize=(8.0, 4.8))
        plt.hist(entropy_norm[rejected_mask], bins=60, alpha=0.85, color="#c44e52")
        plt.xlabel("Normalized entropy")
        plt.ylabel("Count")
        plt.title("Rejected Tokens: Normalized Entropy")
        p2b = report_dir / "draft_rejected_normalized_entropy_hist.png"
        plt.tight_layout()
        plt.savefig(p2b, dpi=160)
        plt.close()
        plot_files.append(p2b.name)

        plt.figure(figsize=(8.0, 4.8))
        plt.hist(top1_prob[rejected_mask], bins=60, alpha=0.85, color="#4c72b0")
        plt.xlabel("Top-1 probability")
        plt.ylabel("Count")
        plt.title("Rejected Tokens: Top-1 Probability")
        p2c = report_dir / "draft_rejected_top1_probability_hist.png"
        plt.tight_layout()
        plt.savefig(p2c, dpi=160)
        plt.close()
        plot_files.append(p2c.name)

    cal = summary.get("calibration", {})
    cal_bins = cal.get("bins", [])
    if cal_bins:
        xs = [(row["bin_left"] + row["bin_right"]) * 0.5 for row in cal_bins]
        ys_conf = [row["avg_confidence"] for row in cal_bins]
        ys_acc = [row["accuracy"] for row in cal_bins]

        plt.figure(figsize=(7.2, 5.2))
        plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="black", label="Perfect calibration")
        plt.plot(xs, ys_conf, marker="o", linewidth=1.8, label="Avg confidence")
        plt.plot(xs, ys_acc, marker="s", linewidth=1.8, label="Empirical accuracy")
        plt.xlabel("Confidence bin center")
        plt.ylabel("Value")
        plt.title(f"Reliability Diagram (ECE={cal.get('ece', 0.0):.4f})")
        plt.ylim(0.0, 1.0)
        plt.xlim(0.0, 1.0)
        plt.legend()
        p3 = report_dir / "draft_reliability_diagram.png"
        plt.tight_layout()
        plt.savefig(p3, dpi=160)
        plt.close()
        plot_files.append(p3.name)

    max_points = 20000
    if top1_prob.size > max_points:
        idx = np.linspace(0, top1_prob.size - 1, max_points, dtype=np.int64)
        top1_sc = top1_prob[idx]
        entropy_sc = entropy_norm[idx]
        hit_sc = positive_hit[idx]
    else:
        top1_sc = top1_prob
        entropy_sc = entropy_norm
        hit_sc = positive_hit

    plt.figure(figsize=(7.8, 5.2))
    plt.scatter(
        top1_sc,
        entropy_sc,
        c=hit_sc,
        cmap="coolwarm",
        alpha=0.35,
        s=10,
    )
    plt.xlabel("Top-1 probability")
    plt.ylabel("Normalized entropy")
    plt.title("Confidence vs Entropy (Color = Positive Argmax Hit)")
    p4 = report_dir / "draft_confidence_vs_entropy_scatter.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=160)
    plt.close()
    plot_files.append(p4.name)

    return plot_files


def write_entropy_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# Draft Entropy and Confidence Analysis")
    lines.append("")

    if summary.get("num_records", 0) == 0:
        lines.append("No draft entropy records were collected.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    rates = summary.get("rates", {})
    entropy = summary.get("entropy", {})
    confidence = summary.get("confidence", {})
    rejected_focus = summary.get("rejected_token_focus", {})
    cal = summary.get("calibration", {})
    corr = summary.get("correlation", {})

    lines.append("## Overall")
    lines.append(f"- Number of draft-token records: **{summary['num_records']}**")
    lines.append(f"- Positive argmax hit rate: **{rates.get('positive_argmax_hit_rate', 0.0) * 100:.2f}%**")
    lines.append(f"- Sampled draft token match rate: **{rates.get('sampled_matches_target_rate', 0.0) * 100:.2f}%**")
    lines.append(f"- Accepted-by-target rate: **{rates.get('accepted_by_target_rate', 0.0) * 100:.2f}%**")
    lines.append(f"- ECE ({cal.get('num_bins', 0)} bins): **{cal.get('ece', 0.0):.5f}**")
    lines.append(f"- MCE: **{cal.get('mce', 0.0):.5f}**")
    lines.append("")

    ent_nats = entropy.get("entropy_nats", {})
    ent_norm = entropy.get("normalized_entropy", {})
    lines.append("## Entropy")
    lines.append(
        f"- Entropy (nats) mean / median / p90: **{ent_nats.get('mean', 0.0):.4f} / {ent_nats.get('median', 0.0):.4f} / {ent_nats.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Normalized entropy mean / median / p90: **{ent_norm.get('mean', 0.0):.4f} / {ent_norm.get('median', 0.0):.4f} / {ent_norm.get('p90', 0.0):.4f}**"
    )
    lines.append("")

    rej_ent_nats = rejected_focus.get("entropy_nats", {})
    rej_ent_norm = rejected_focus.get("normalized_entropy", {})
    rej_top1 = rejected_focus.get("top1_prob", {})
    rej_margin = rejected_focus.get("top1_margin", {})
    rej_cal = rejected_focus.get("calibration", {})
    rej_delta = rejected_focus.get("rejected_minus_accepted", {})

    lines.append("## Rejected-Token Focus")
    lines.append(
        f"- Rejected tokens: **{rejected_focus.get('rejected_count', 0)} / {summary['num_records']} ({rejected_focus.get('rejected_rate', 0.0) * 100:.2f}%)**"
    )
    lines.append(
        f"- Rejected entropy (nats) mean / median / p90: **{rej_ent_nats.get('mean', 0.0):.4f} / {rej_ent_nats.get('median', 0.0):.4f} / {rej_ent_nats.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Rejected normalized entropy mean / median / p90: **{rej_ent_norm.get('mean', 0.0):.4f} / {rej_ent_norm.get('median', 0.0):.4f} / {rej_ent_norm.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Rejected top1-prob mean / median / p90: **{rej_top1.get('mean', 0.0):.4f} / {rej_top1.get('median', 0.0):.4f} / {rej_top1.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Rejected top1-margin mean / median / p90: **{rej_margin.get('mean', 0.0):.4f} / {rej_margin.get('median', 0.0):.4f} / {rej_margin.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Rejected positive-argmax-hit rate: **{rejected_focus.get('positive_argmax_hit_rate', 0.0) * 100:.2f}%**"
    )
    lines.append(
        f"- Rejected ECE ({rej_cal.get('num_bins', 0)} bins): **{rej_cal.get('ece', 0.0):.5f}**"
    )
    lines.append(
        f"- Rejected minus accepted mean entropy (nats): **{rej_delta.get('entropy_nats_mean', 0.0):+.4f}**"
    )
    lines.append(
        f"- Rejected minus accepted mean top1-prob: **{rej_delta.get('top1_prob_mean', 0.0):+.4f}**"
    )
    lines.append("")

    lines.append("### Rejected Entropy Bands")
    lines.append("| Band | Count | Rate within rejected |")
    lines.append("|---|---:|---:|")
    for k, v in rejected_focus.get("entropy_bands", {}).items():
        lines.append(f"| {k} | {v['count']} | {v['rate'] * 100:.2f}% |")
    lines.append("")

    top1 = confidence.get("top1_prob", {})
    margin = confidence.get("top1_margin", {})
    lines.append("## Confidence")
    lines.append(
        f"- Top-1 probability mean / median / p90: **{top1.get('mean', 0.0):.4f} / {top1.get('median', 0.0):.4f} / {top1.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Top-1 margin mean / median / p90: **{margin.get('mean', 0.0):.4f} / {margin.get('median', 0.0):.4f} / {margin.get('p90', 0.0):.4f}**"
    )
    lines.append("")

    lines.append("## Correlation")
    lines.append(
        f"- corr(top1_prob, positive_hit): **{corr.get('corr_top1_prob_vs_positive_hit', 0.0):.4f}**"
    )
    lines.append(
        f"- corr(normalized_entropy, positive_hit): **{corr.get('corr_normalized_entropy_vs_positive_hit', 0.0):.4f}**"
    )
    lines.append(
        f"- corr(top1_prob, accepted): **{corr.get('corr_top1_prob_vs_accepted', 0.0):.4f}**"
    )
    lines.append(
        f"- corr(normalized_entropy, accepted): **{corr.get('corr_normalized_entropy_vs_accepted', 0.0):.4f}**"
    )
    lines.append("")

    lines.append("## Calibration by Confidence Bin")
    lines.append("| Bin | Count | Rate | Avg confidence | Accuracy | Gap |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for row in cal.get("bins", []):
        bin_label = f"[{row['bin_left']:.1f}, {row['bin_right']:.1f})"
        lines.append(
            f"| {bin_label} | {row['count']} | {row['rate'] * 100:.2f}% | {row['avg_confidence']:.4f} | {row['accuracy']:.4f} | {row['gap']:.4f} |"
        )
    lines.append("")

    lines.append("## Entropy Bands")
    lines.append("| Band | Count | Rate | Mean top1 prob | Positive hit rate | Accepted rate |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for k, v in summary.get("entropy_bands", {}).items():
        lines.append(
            f"| {k} | {v['count']} | {v['rate'] * 100:.2f}% | {v['mean_top1_prob']:.4f} | {v['positive_argmax_hit_rate'] * 100:.2f}% | {v['accepted_by_target_rate'] * 100:.2f}% |"
        )
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    rejected_records = [r for r in records if r.get("accepted_by_target", 0) == 0]

    lines.append("## Highest-Entropy Rejected Tokens")
    lines.append("| sample | turn | step | pos | block_pos | entropy | top1_prob | top1_margin | hit | accepted | target_id | top1_id | sampled_id |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    high_entropy = sorted(rejected_records, key=lambda x: x["normalized_entropy"], reverse=True)[:max_rows]
    for rec in high_entropy:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {block_position} | "
            "{normalized_entropy:.4f} | {top1_prob:.4f} | {top1_margin:.4f} | {positive_argmax_hit} | "
            "{accepted_by_target} | {target_token_id} | {positive_top1_id} | {sampled_draft_id} |".format(**rec)
        )
    lines.append("")

    lines.append("## Most-Confident Rejected Tokens")
    lines.append("| sample | turn | step | pos | top1_prob | entropy | margin | target_id | top1_id | sampled_id | accepted |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    confident_rejected = sorted(rejected_records, key=lambda x: x["top1_prob"], reverse=True)[:max_rows]
    for rec in confident_rejected:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {top1_prob:.4f} | "
            "{normalized_entropy:.4f} | {top1_margin:.4f} | {target_token_id} | {positive_top1_id} | "
            "{sampled_draft_id} | {accepted_by_target} |".format(**rec)
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")

def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by some CUDA kernels for reproducible matmul behavior.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

# construct negative sample by randomly replacing the batch's first token (target model generated token) 
def build_negative_block_output_random(block_output_ids: torch.Tensor, vocab_size: int , gen : torch.Generator = None) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(0, vocab_size, (batch_size,), device=block_output_ids.device , generator=gen)
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids

def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    mode: str = "mask_zero",
    gen : torch.Generator = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()
    batch_size, context_len = negative_target_hidden.shape[:2]

    if mode == "mask_zero":
        if dropout_ratio > 0.0:
            token_keep_mask = (
                torch.rand(
                    negative_target_hidden.shape[:2],
                    device=negative_target_hidden.device,
                    generator=gen
                ) >= dropout_ratio
            ).unsqueeze(-1)
            negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)
    elif mode == "shuffle_tokens":
        if context_len > 1:
            per_batch_indices = []
            for _ in range(batch_size):
                perm = torch.randperm(context_len, device=negative_target_hidden.device, generator=gen)
                per_batch_indices.append(perm)
            gather_index = torch.stack(per_batch_indices, dim=0).unsqueeze(-1).expand(-1, -1, negative_target_hidden.shape[-1])
            negative_target_hidden = torch.gather(negative_target_hidden, dim=1, index=gather_index)
    else:
        raise ValueError(
            f"Unsupported negative hidden mode: {mode}. "
            "Use 'mask_zero' or 'shuffle_tokens'."
        )

    if noise_std > 0.0:
        noise = torch.randn(
            negative_target_hidden.shape,
            dtype=negative_target_hidden.dtype,
            device=negative_target_hidden.device,
            generator=gen
        )
        negative_target_hidden = negative_target_hidden + noise_std * noise

    return negative_target_hidden

def compute_contrastive_draft_logits(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    block_output_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    draft_position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    block_size: int,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    negative_hidden_mode: str,
    gen : torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    
    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size , gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        mode=negative_hidden_mode,
        gen=gen,
    )
    
    possitive_noise_embedding = target.model.embed_tokens(block_output_ids)
    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)
    

    paired_noise_embedding = torch.cat([possitive_noise_embedding, negative_noise_embedding], dim=0)
    paired_hidden = torch.cat([target_hidden, negative_target_hidden], dim=0)
    
    paired_position_ids = torch.cat([draft_position_ids, draft_position_ids], dim=0)

    paired_draft_logits = target.lm_head(model(
        target_hidden=paired_hidden,
        noise_embedding=paired_noise_embedding,
        position_ids=paired_position_ids,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )[:, -block_size+1:, :])

    first_draft_logits, second_draft_logits = paired_draft_logits.split(batch_size, dim=0)
    return first_draft_logits, second_draft_logits


def build_cd_candidate_mask(
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)

    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def build_topk_probability_mask(
    reference_logits: torch.Tensor,
    top_k: int = 64,
) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    vocab_size = reference_probs.size(-1)
    k = min(max(1, top_k), vocab_size)

    topk_indices = torch.topk(reference_probs, k=k, dim=-1).indices
    topk_mask = torch.zeros_like(reference_probs, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)
    return topk_mask


def apply_cd_candidate_filter(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    filtered_logits = logits.masked_fill(
        ~candidate_mask,
        torch.finfo(logits.dtype).min,
    )
    return filtered_logits

def _compute_kl_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> float:
    """
    Returns a scalar KL(P_first || P_second) averaged over the vocab dimension.
    """
    p_first = torch.softmax(first_logits, dim=-1)
    p_second = torch.softmax(second_logits, dim=-1)
    # Add a tiny epsilon to avoid log(0)
    kl = (p_first * (torch.log(p_first + 1e-10) - torch.log(p_second + 1e-10))).sum(dim=-1).mean()
    return kl.item()

def log_logits_difference(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    step: int,
) -> None:
    # Compute KL divergence between the two logit distributions
    first_probs = torch.softmax(first_logits, dim=-1)
    second_probs = torch.softmax(second_logits, dim=-1)
    
    # KL(P_first || P_second)
    kl_div = torch.sum(first_probs * (torch.log(first_probs + 1e-10) - torch.log(second_probs + 1e-10)), dim=-1).mean()
    
    # Cosine similarity between logit vectors
    first_logits_flat = first_logits.view(-1, first_logits.size(-1))
    second_logits_flat = second_logits.view(-1, second_logits.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(first_logits_flat, second_logits_flat, dim=-1).mean()
    
    # L2 distance between logits
    l2_dist = torch.norm(first_logits - second_logits, p=2, dim=-1).mean()
    
    logger.info(
        f"Step {step}: KL divergence: {kl_div.item():.4f}, "
        f"Cosine similarity: {cosine_sim.item():.4f}, "
        f"L2 distance: {l2_dist.item():.4f}"
    )

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    cd_alpha: float = 1.0,
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    negative_hidden_mode: str = "mask_zero",
    divergence_accumulator: Optional[List[float]] = None,
    draft_entropy_records: Optional[List[dict[str, Any]]] = None,
    sample_idx: int = -1,
    turn_idx: int = -1,
    seed: int = 0,
) -> SimpleNamespace:
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature , gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    decode_step = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            draft_position_ids = position_ids[:, start - target_hidden.shape[1] : start + block_size]
            positive_draft_logits, negative_draft_logits = compute_contrastive_draft_logits(
                model=model,
                target=target,
                block_output_ids=block_output_ids,
                target_hidden=target_hidden,
                draft_position_ids=draft_position_ids,
                past_key_values_draft=past_key_values_draft,
                block_size=block_size,
                negative_context_dropout=negative_context_dropout,
                negative_context_noise_std=negative_context_noise_std,
                negative_hidden_mode=negative_hidden_mode,
                gen=gen
            )
            past_key_values_draft.crop(start)
            
            # ----- Divergence witness (optional) -----
            if divergence_accumulator is not None:
                kl_val = _compute_kl_divergence(positive_draft_logits, negative_draft_logits)
                divergence_accumulator.append(kl_val)

            if not TOP64_MASK_MODE:
                candidate_mask = build_cd_candidate_mask(
                    reference_logits=positive_draft_logits,
                    beta=beta,
                )
            else:
                candidate_mask = build_topk_probability_mask(
                    reference_logits=positive_draft_logits,
                    top_k=64,
                )
                        
            final_draft_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            
            final_draft_logits = apply_cd_candidate_filter(
                logits=final_draft_logits,
                candidate_mask=candidate_mask,
            )
            
            # Applying positive draft logits to the top candidates in the final draft logits to further reduce the chance of selecting a token that is not favored by the positive draft
            n_keep = min(6, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]
            
            block_output_ids[:, 1:] = sample(final_draft_logits , gen=gen)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature , gen=gen)
        acceptance_lengths_per_batch = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
        acceptance_length = acceptance_lengths_per_batch[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        if block_size > 1 and draft_entropy_records is not None:
            _collect_draft_entropy_records(
                records=draft_entropy_records,
                positive_logits=positive_draft_logits,
                sampled_draft_ids=block_output_ids[:, 1:],
                posterior_ids=posterior[:, :-1],
                acceptance_lengths=acceptance_lengths_per_batch,
                sample_idx=sample_idx,
                turn_idx=turn_idx,
                decode_step=decode_step,
                block_start=start,
            )

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        decode_step += 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    global NEGATIVE_HIDDEN_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-example-rows", type=int, default=20)
    parser.add_argument("--calibration-bins", type=int, default=10)
    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", type=float, default=0.05)
    parser.add_argument("--cd-beta", type=float, default=0.0)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--negative-hidden-mode",
        type=str,
        choices=["mask_zero", "shuffle_tokens"],
        default=NEGATIVE_HIDDEN_MODE,
        help="How to construct negative hidden: mask tokens to zero or shuffle token positions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    args = parser.parse_args()
    NEGATIVE_HIDDEN_MODE = args.negative_hidden_mode
    print(f"Using negative hidden mode: [bold magenta]{args.negative_hidden_mode}[/bold magenta]")

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")
        model_dtype = torch.bfloat16
    else:
        logger.warning("CUDA is unavailable. Falling back to CPU inference; this benchmark will run much slower.")
        device = torch.device("cpu")
        model_dtype = torch.float32

    def has_flash_attn():
        if not use_cuda:
            return False
        if args.deterministic:
            logger.info("Deterministic mode enabled. Forcing SDPA attention backend.")
            return False
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=model_dtype,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=model_dtype,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = [] 
    draft_entropy_records: List[dict[str, Any]] = []

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    
    base_seed = args.seed
    for idx in tqdm(indices, disable=not dist.is_main()):
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            decode_block_sizes = [1] if block_size == 1 else [1, block_size]
            for bs in decode_block_sizes:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    cd_alpha=args.cd_alpha,
                    beta=args.cd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    negative_hidden_mode=args.negative_hidden_mode,
                    divergence_accumulator = divergence_accumulator if bs == block_size else None,
                    draft_entropy_records=draft_entropy_records if bs == block_size else None,
                    sample_idx=idx,
                    turn_idx=turn_index,
                    seed = sample_seed + bs,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        divergence_accumulator = dist.gather(divergence_accumulator, dst=0)
        draft_entropy_records = dist.gather(draft_entropy_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        divergence_accumulator = list(chain(*divergence_accumulator))
        draft_entropy_records = list(chain(*draft_entropy_records))

    if not responses:
        print("No responses were generated.")
        return

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    
    # Report the collected divergence metric (if any)
    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        # Optionally also show min/max for a quick sense of spread
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

    report_dir = build_report_dir(args.dataset, args.report_dir)
    entropy_summary = build_entropy_summary(
        draft_entropy_records,
        args=args,
        block_size=block_size,
        decoding_speedup=float(t1 / tb),
        avg_acceptance_length=float(tau),
    )

    records_jsonl = report_dir / "draft_entropy_records.jsonl"
    records_csv = report_dir / "draft_entropy_records.csv"
    summary_json = report_dir / "draft_entropy_summary.json"
    report_md = report_dir / "draft_entropy_report.md"

    write_jsonl(records_jsonl, draft_entropy_records)
    write_csv(records_csv, draft_entropy_records)
    summary_json.write_text(json.dumps(entropy_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_files = [] if args.no_plots else maybe_create_entropy_plots(report_dir, draft_entropy_records, entropy_summary)
    write_entropy_report_md(
        report_path=report_md,
        summary=entropy_summary,
        records=draft_entropy_records,
        plot_files=plot_files,
        max_rows=args.max_example_rows,
    )

    print(f"Draft entropy records: {entropy_summary.get('num_records', 0)}")
    if entropy_summary.get("num_records", 0) > 0:
        top1_mean = entropy_summary["confidence"]["top1_prob"]["mean"]
        ent_norm_mean = entropy_summary["entropy"]["normalized_entropy"]["mean"]
        ece = entropy_summary["calibration"]["ece"]
        hit_rate = entropy_summary["rates"]["positive_argmax_hit_rate"]
        rejected_focus = entropy_summary.get("rejected_token_focus", {})
        rejected_rate = rejected_focus.get("rejected_rate", 0.0)
        rejected_ent_norm_mean = rejected_focus.get("normalized_entropy", {}).get("mean", 0.0)
        rejected_top1_mean = rejected_focus.get("top1_prob", {}).get("mean", 0.0)
        print(f"Positive argmax hit rate: {hit_rate * 100:.2f}%")
        print(f"Mean top1 probability: {top1_mean:.4f}")
        print(f"Mean normalized entropy: {ent_norm_mean:.4f}")
        print(f"Calibration ECE: {ece:.5f}")
        print(f"Rejected token rate: {rejected_rate * 100:.2f}%")
        print(f"Rejected mean normalized entropy: {rejected_ent_norm_mean:.4f}")
        print(f"Rejected mean top1 probability: {rejected_top1_mean:.4f}")

    print(f"Report directory: {report_dir}")
    print(f"  - {records_jsonl.name}")
    print(f"  - {records_csv.name}")
    print(f"  - {summary_json.name}")
    print(f"  - {report_md.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")

if __name__ == "__main__":
    main()
