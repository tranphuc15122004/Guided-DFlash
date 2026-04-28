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

NEGATIVE_HIDDEN_MODE = 'mask_zero' 
TOP64_MASK_MODE = False 

""" 
CDv2
Goal: Study draft model entropy
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


def select_entropy_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    entropy_logit_source: str,
) -> torch.Tensor:
    if entropy_logit_source == "positive":
        return positive_logits
    if entropy_logit_source == "negative":
        return negative_logits
    raise ValueError(
        f"Unsupported entropy_logit_source={entropy_logit_source}. "
        "Use 'positive' or 'negative'."
    )


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


def init_vocab_probability_histogram_accumulator(max_positions: int, bins: int) -> dict[str, Any]:
    max_positions = max(1, int(max_positions))
    bins = max(2, int(bins))
    return {
        "max_positions": max_positions,
        "bins": bins,
        "bin_edges": np.linspace(0.0, 1.0, bins + 1, dtype=np.float64),
        "counts": np.zeros((max_positions, bins), dtype=np.int64),
        "num_distributions": np.zeros(max_positions, dtype=np.int64),
        "vocab_values": np.zeros(max_positions, dtype=np.int64),
    }


def update_vocab_probability_histogram_accumulator(
    accumulator: Optional[dict[str, Any]],
    probs: torch.Tensor,
) -> None:
    if accumulator is None or probs.numel() == 0:
        return

    max_positions = int(accumulator["max_positions"])
    bin_edges = np.asarray(accumulator["bin_edges"], dtype=np.float64)
    counts = np.asarray(accumulator["counts"], dtype=np.int64)
    num_distributions = np.asarray(accumulator["num_distributions"], dtype=np.int64)
    vocab_values = np.asarray(accumulator["vocab_values"], dtype=np.int64)

    batch_size, num_steps, _ = probs.shape
    use_steps = min(max_positions, int(num_steps))
    probs_np = probs[:, :use_steps, :].detach().float().cpu().numpy()

    for pos in range(use_steps):
        vals = probs_np[:, pos, :].reshape(-1)
        pos_counts, _ = np.histogram(vals, bins=bin_edges)
        counts[pos] += pos_counts.astype(np.int64)
        num_distributions[pos] += int(batch_size)
        vocab_values[pos] += int(vals.size)


def finalize_vocab_probability_histogram_accumulator(
    accumulator: Optional[dict[str, Any]],
) -> list[dict[str, Any]]:
    if accumulator is None:
        return []

    max_positions = int(accumulator["max_positions"])
    bin_edges = np.asarray(accumulator["bin_edges"], dtype=np.float64)
    counts = np.asarray(accumulator["counts"], dtype=np.int64)
    num_distributions = np.asarray(accumulator["num_distributions"], dtype=np.int64)
    vocab_values = np.asarray(accumulator["vocab_values"], dtype=np.int64)

    rows: list[dict[str, Any]] = []
    for pos in range(max_positions):
        total_values = int(vocab_values[pos])
        pos_counts = counts[pos]
        if total_values > 0:
            rates = (pos_counts.astype(np.float64) / float(total_values)).tolist()
        else:
            rates = [0.0 for _ in range(pos_counts.size)]

        rows.append(
            {
                "block_position": int(pos + 1),
                "num_distributions": int(num_distributions[pos]),
                "vocab_values": total_values,
                "bin_edges": [float(x) for x in bin_edges.tolist()],
                "counts": [int(x) for x in pos_counts.tolist()],
                "rates": [float(x) for x in rates],
            }
        )

    return rows


def merge_vocab_probability_histogram_accumulators(
    accumulators: list[dict[str, Any]],
) -> Optional[dict[str, Any]]:
    if not accumulators:
        return None

    first = accumulators[0]
    merged = init_vocab_probability_histogram_accumulator(
        max_positions=int(first["max_positions"]),
        bins=int(first["bins"]),
    )

    for acc in accumulators:
        merged["counts"] += np.asarray(acc["counts"], dtype=np.int64)
        merged["num_distributions"] += np.asarray(acc["num_distributions"], dtype=np.int64)
        merged["vocab_values"] += np.asarray(acc["vocab_values"], dtype=np.int64)

    return merged


def write_vocab_probability_histogram_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "block_position",
        "num_distributions",
        "vocab_values",
        "bin_left",
        "bin_right",
        "count",
        "rate",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            edges = row.get("bin_edges", [])
            counts = row.get("counts", [])
            rates = row.get("rates", [])
            for i in range(max(0, len(edges) - 1)):
                writer.writerow(
                    {
                        "block_position": int(row.get("block_position", 0)),
                        "num_distributions": int(row.get("num_distributions", 0)),
                        "vocab_values": int(row.get("vocab_values", 0)),
                        "bin_left": float(edges[i]),
                        "bin_right": float(edges[i + 1]),
                        "count": int(counts[i]) if i < len(counts) else 0,
                        "rate": float(rates[i]) if i < len(rates) else 0.0,
                    }
                )


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

def cuda_time() -> float:
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


def _collect_draft_entropy_records(
    *,
    records: list[dict[str, Any]],
    analysis_logits: torch.Tensor,
    logits_source: str,
    sampled_draft_ids: torch.Tensor,
    posterior_ids: torch.Tensor,
    acceptance_lengths: torch.Tensor,
    sample_idx: int,
    turn_idx: int,
    decode_step: int,
    block_start: int,
    vocab_probability_hist_accumulator: Optional[dict[str, Any]] = None,
) -> None:
    if analysis_logits.numel() == 0:
        return

    logits = analysis_logits.float()
    probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log_softmax(logits, dim=-1)

    update_vocab_probability_histogram_accumulator(vocab_probability_hist_accumulator, probs)

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
            analyzed_top1_id = int(top1_ids[b, i].item())
            sampled_match = int(sampled_id == target_id)
            accepted_by_target = int(i < accepted_len)

            records.append(
                {
                    "sample_idx": int(sample_idx),
                    "turn_idx": int(turn_idx),
                    "decode_step": int(decode_step),
                    "absolute_position": int(block_start + i + 1),
                    "block_position": int(i + 1),
                    "logits_source": logits_source,
                    "sampled_draft_id": sampled_id,
                    "target_token_id": target_id,
                    "analyzed_top1_id": analyzed_top1_id,
                    "sampled_matches_target": sampled_match,
                    "accepted_by_target": accepted_by_target,
                    "analyzed_argmax_hit": int(analyzed_top1_id == target_id),
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
                    "sampled_prob_under_analyzed": float(sampled_prob[b, i].item()),
                    "sampled_logprob_under_analyzed": float(sampled_logprob[b, i].item()),
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


def _build_distribution_shape_summary(
    top1_prob: np.ndarray,
    effective_support: np.ndarray,
    entropy: np.ndarray,
    positive_hit: np.ndarray,
    accepted: np.ndarray,
) -> dict[str, Any]:
    total = max(1, int(top1_prob.size))

    def _summarize(mask: np.ndarray) -> dict[str, Any]:
        count = int(mask.sum())
        if count == 0:
            return {
                "count": 0,
                "rate": 0.0,
                "mean_top1_prob": 0.0,
                "mean_effective_support": 0.0,
                "mean_entropy_nats": 0.0,
                "positive_argmax_hit_rate": 0.0,
                "accepted_by_target_rate": 0.0,
            }
        return {
            "count": count,
            "rate": float(count / total),
            "mean_top1_prob": float(top1_prob[mask].mean()),
            "mean_effective_support": float(effective_support[mask].mean()),
            "mean_entropy_nats": float(entropy[mask].mean()),
            "positive_argmax_hit_rate": float(positive_hit[mask].mean()),
            "accepted_by_target_rate": float(accepted[mask].mean()),
        }

    top1_bands = {
        "very_sharp_[0.9,1.0]": top1_prob >= 0.9,
        "sharp_[0.7,0.9)": (top1_prob >= 0.7) & (top1_prob < 0.9),
        "medium_[0.5,0.7)": (top1_prob >= 0.5) & (top1_prob < 0.7),
        "flat_[0.0,0.5)": top1_prob < 0.5,
    }
    support_bands = {
        "very_sharp_[1,2]": effective_support <= 2.0,
        "moderate_[2,10)": (effective_support > 2.0) & (effective_support < 10.0),
        "diffuse_[10,+inf)": effective_support >= 10.0,
    }

    return {
        "top1_prob_bands": {name: _summarize(mask) for name, mask in top1_bands.items()},
        "effective_support_bands": {name: _summarize(mask) for name, mask in support_bands.items()},
    }


def _build_block_position_profile(
    block_position: np.ndarray,
    entropy: np.ndarray,
    entropy_norm: np.ndarray,
    effective_support: np.ndarray,
    top1_prob: np.ndarray,
    top5_mass: np.ndarray,
    top10_mass: np.ndarray,
    positive_hit: np.ndarray,
    accepted: np.ndarray,
) -> list[dict[str, Any]]:
    if block_position.size == 0:
        return []

    positions = sorted(int(x) for x in np.unique(block_position))
    total = max(1, int(block_position.size))
    rows: list[dict[str, Any]] = []

    for pos in positions:
        mask = block_position == pos
        count = int(mask.sum())
        if count == 0:
            continue

        rows.append(
            {
                "block_position": int(pos),
                "count": count,
                "rate": float(count / total),
                "positive_argmax_hit_rate": float(positive_hit[mask].mean()),
                "accepted_by_target_rate": float(accepted[mask].mean()),
                "entropy_nats": _describe_distribution(entropy[mask]),
                "normalized_entropy": _describe_distribution(entropy_norm[mask]),
                "effective_support": _describe_distribution(effective_support[mask]),
                "top1_prob": _describe_distribution(top1_prob[mask]),
                "top5_mass": _describe_distribution(top5_mass[mask]),
                "top10_mass": _describe_distribution(top10_mass[mask]),
                "sharpness_rates": {
                    "top1_prob_ge_0.9": float((top1_prob[mask] >= 0.9).mean()),
                    "top1_prob_ge_0.7": float((top1_prob[mask] >= 0.7).mean()),
                    "effective_support_le_2": float((effective_support[mask] <= 2.0).mean()),
                    "effective_support_gt_10": float((effective_support[mask] >= 10.0).mean()),
                },
            }
        )

    return rows


def build_entropy_summary(
    records: list[dict[str, Any]],
    *,
    args: argparse.Namespace,
    block_size: int,
    decoding_speedup: float,
    avg_acceptance_length: float,
    vocab_probability_hist_accumulator: Optional[dict[str, Any]] = None,
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
        "entropy_logit_source": args.entropy_logit_source,
        "top64_mask_mode": bool(TOP64_MASK_MODE),
        "decoding_speedup": float(decoding_speedup),
        "avg_acceptance_length": float(avg_acceptance_length),
    }

    vocab_probability_histogram = finalize_vocab_probability_histogram_accumulator(
        vocab_probability_hist_accumulator
    )

    if not records:
        return {
            "meta": meta,
            "num_records": 0,
            "distribution": {
                "vocab_probability_histogram_by_position": vocab_probability_histogram,
            },
            "note": "No draft-token records were collected. This happens when block_size <= 1.",
        }

    top1_prob = np.asarray([r["top1_prob"] for r in records], dtype=np.float64)
    entropy = np.asarray([r["entropy_nats"] for r in records], dtype=np.float64)
    entropy_norm = np.asarray([r["normalized_entropy"] for r in records], dtype=np.float64)
    effective_support = np.asarray([r["effective_support"] for r in records], dtype=np.float64)
    block_position = np.asarray([r["block_position"] for r in records], dtype=np.int64)
    analyzed_hit = np.asarray([r["analyzed_argmax_hit"] for r in records], dtype=np.float64)
    accepted = np.asarray([r["accepted_by_target"] for r in records], dtype=np.float64)

    top1_bands = {
        "very_sharp_[0.9,1.0]": (top1_prob >= 0.9),
        "sharp_[0.7,0.9)": (top1_prob >= 0.7) & (top1_prob < 0.9),
        "medium_[0.5,0.7)": (top1_prob >= 0.5) & (top1_prob < 0.7),
        "flat_[0.0,0.5)": (top1_prob < 0.5),
    }
    entropy_bands = {
        "low_entropy_[0.0,0.4)": (entropy_norm >= 0.0) & (entropy_norm < 0.4),
        "mid_entropy_[0.4,0.7)": (entropy_norm >= 0.4) & (entropy_norm < 0.7),
        "high_entropy_[0.7,1.0]": (entropy_norm >= 0.7) & (entropy_norm <= 1.0000001),
    }

    total = max(1, int(top1_prob.size))
    top1_band_stats = {
        name: {
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
        }
        for name, mask in top1_bands.items()
    }
    entropy_band_stats = {
        name: {
            "count": int(mask.sum()),
            "rate": float(mask.mean()),
        }
        for name, mask in entropy_bands.items()
    }

    position_rows: list[dict[str, Any]] = []
    for pos in sorted(int(x) for x in np.unique(block_position)):
        mask = block_position == pos
        count = int(mask.sum())
        if count == 0:
            continue
        position_rows.append(
            {
                "block_position": int(pos),
                "count": count,
                "rate": float(count / total),
                "top1_prob": _describe_distribution(top1_prob[mask]),
                "normalized_entropy": _describe_distribution(entropy_norm[mask]),
                "effective_support": _describe_distribution(effective_support[mask]),
            }
        )

    return {
        "meta": meta,
        "num_records": int(total),
        "num_unique_samples": int(len(set(int(r["sample_idx"]) for r in records))),
        "num_unique_turns": int(len(set((int(r["sample_idx"]), int(r["turn_idx"])) for r in records))),
        "quick_quality_context": {
            "analyzed_argmax_hit_rate": float(analyzed_hit.mean()),
            "accepted_by_target_rate": float(accepted.mean()),
        },
        "distribution": {
            "top1_prob": _describe_distribution(top1_prob),
            "entropy_nats": _describe_distribution(entropy),
            "normalized_entropy": _describe_distribution(entropy_norm),
            "effective_support": _describe_distribution(effective_support),
            "top1_prob_bands": top1_band_stats,
            "entropy_bands": entropy_band_stats,
            "histograms": {
                "top1_prob": _build_histogram(top1_prob, bins=args.histogram_bins, value_range=(0.0, 1.0)),
                "normalized_entropy": _build_histogram(entropy_norm, bins=args.histogram_bins, value_range=(0.0, 1.0)),
            },
            "block_position_profile": position_rows,
            "vocab_probability_histogram_by_position": vocab_probability_histogram,
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
    block_position = np.asarray([r["block_position"] for r in records], dtype=np.int64)

    plot_files: list[str] = []

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(top1_prob, bins=60, alpha=0.9, color="#4c72b0")
    plt.xlabel("Top-1 probability (analyzed logits)")
    plt.ylabel("Count")
    plt.title("Distribution of Draft Top-1 Probability (Analyzed)")
    p1 = report_dir / "draft_top1_probability_hist.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    plot_files.append(p1.name)

    plt.figure(figsize=(8.0, 4.8))
    plt.hist(entropy_norm, bins=60, alpha=0.9, color="#c44e52")
    plt.xlabel("Normalized entropy")
    plt.ylabel("Count")
    plt.title("Distribution of Draft Normalized Entropy")
    p2 = report_dir / "draft_normalized_entropy_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    plot_files.append(p2.name)

    unique_positions = sorted(int(x) for x in np.unique(block_position))

    def _plot_position_quantiles(values: np.ndarray, y_label: str, title: str, file_name: str) -> None:
        if len(unique_positions) <= 1:
            return

        xs: list[int] = []
        q10: list[float] = []
        q25: list[float] = []
        q50: list[float] = []
        q75: list[float] = []
        q90: list[float] = []

        for pos in unique_positions:
            mask = block_position == pos
            pos_values = values[mask]
            if pos_values.size == 0:
                continue
            xs.append(pos)
            q = np.quantile(pos_values, [0.1, 0.25, 0.5, 0.75, 0.9])
            q10.append(float(q[0]))
            q25.append(float(q[1]))
            q50.append(float(q[2]))
            q75.append(float(q[3]))
            q90.append(float(q[4]))

        if len(xs) <= 1:
            return

        x_arr = np.asarray(xs, dtype=np.float64)
        plt.figure(figsize=(8.2, 5.0))
        plt.fill_between(x_arr, q10, q90, alpha=0.2, label="p10-p90", color="#4c72b0")
        plt.fill_between(x_arr, q25, q75, alpha=0.3, label="p25-p75", color="#4c72b0")
        plt.plot(x_arr, q50, marker="o", linewidth=2.0, label="median", color="#dd8452")
        plt.xlabel("Block position")
        plt.ylabel(y_label)
        plt.title(title)
        plt.xticks(xs)
        plt.legend()
        out = report_dir / file_name
        plt.tight_layout()
        plt.savefig(out, dpi=160)
        plt.close()
        plot_files.append(out.name)

    _plot_position_quantiles(
        top1_prob,
        "Top-1 probability",
        "Top-1 Probability by Block Position",
        "draft_top1_probability_by_block_position.png",
    )
    _plot_position_quantiles(
        entropy_norm,
        "Normalized entropy",
        "Normalized Entropy by Block Position",
        "draft_normalized_entropy_by_block_position.png",
    )

    vocab_hist_rows = (
        summary.get("distribution", {}).get("vocab_probability_histogram_by_position", [])
    )
    valid_vocab_hist_rows = [r for r in vocab_hist_rows if int(r.get("vocab_values", 0)) > 0]
    if valid_vocab_hist_rows:
        heatmap = np.asarray([r.get("rates", []) for r in valid_vocab_hist_rows], dtype=np.float64)
        if heatmap.ndim == 2 and heatmap.shape[0] > 0 and heatmap.shape[1] > 0:
            positions = [int(r.get("block_position", i + 1)) for i, r in enumerate(valid_vocab_hist_rows)]
            bin_edges = np.asarray(valid_vocab_hist_rows[0].get("bin_edges", []), dtype=np.float64)
            bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:]) if bin_edges.size > 1 else np.array([])

            plt.figure(figsize=(9.0, 5.5))
            plt.imshow(heatmap.T, aspect="auto", origin="lower", cmap="magma")
            plt.colorbar(label="Mean histogram rate")
            plt.xlabel("Block position")
            plt.ylabel("Probability bin")
            plt.title("Mean Vocab-Probability Histogram by Block Position")
            plt.xticks(np.arange(len(positions)), positions)
            if bin_centers.size > 0:
                yticks = np.linspace(0, len(bin_centers) - 1, num=min(6, len(bin_centers)), dtype=int)
                ylabels = [f"{bin_centers[i]:.3f}" for i in yticks]
                plt.yticks(yticks, ylabels)
            out = report_dir / "draft_vocab_probability_histogram_by_position.png"
            plt.tight_layout()
            plt.savefig(out, dpi=160)
            plt.close()
            plot_files.append(out.name)

    return plot_files


def write_entropy_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
) -> None:
    lines: list[str] = []
    lines.append("# Draft Output Probability Shape")
    lines.append("")

    if summary.get("num_records", 0) == 0:
        lines.append("No draft entropy records were collected.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    quality = summary.get("quick_quality_context", {})
    dist = summary.get("distribution", {})
    meta = summary.get("meta", {})
    top1 = dist.get("top1_prob", {})
    entropy_norm = dist.get("normalized_entropy", {})
    effective_support = dist.get("effective_support", {})
    top1_bands = dist.get("top1_prob_bands", {})
    entropy_bands = dist.get("entropy_bands", {})
    position_profile = dist.get("block_position_profile", [])
    vocab_prob_hist = dist.get("vocab_probability_histogram_by_position", [])

    lines.append("## Overall")
    lines.append(f"- Number of draft-token records: **{summary['num_records']}**")
    lines.append(f"- Entropy logits source: **{meta.get('entropy_logit_source', 'positive')}**")
    lines.append(f"- Decoding speedup (bs=1 vs bs=block): **{summary.get('meta', {}).get('decoding_speedup', 0.0):.2f}x**")
    lines.append(f"- Average acceptance length: **{summary.get('meta', {}).get('avg_acceptance_length', 0.0):.2f}**")
    lines.append(f"- Analyzed argmax hit rate (context): **{quality.get('analyzed_argmax_hit_rate', 0.0) * 100:.2f}%**")
    lines.append(f"- Accepted-by-target rate (context): **{quality.get('accepted_by_target_rate', 0.0) * 100:.2f}%**")
    lines.append("")

    lines.append("## Core Distribution Stats")
    lines.append(
        f"- Top-1 probability mean / median / p90: **{top1.get('mean', 0.0):.4f} / {top1.get('median', 0.0):.4f} / {top1.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Normalized entropy mean / median / p90: **{entropy_norm.get('mean', 0.0):.4f} / {entropy_norm.get('median', 0.0):.4f} / {entropy_norm.get('p90', 0.0):.4f}**"
    )
    lines.append(
        f"- Effective support mean / median / p90: **{effective_support.get('mean', 0.0):.4f} / {effective_support.get('median', 0.0):.4f} / {effective_support.get('p90', 0.0):.4f}**"
    )
    lines.append("")

    lines.append("## Top-1 Probability Bands")
    lines.append("| Band | Count | Rate |")
    lines.append("|---|---:|---:|")
    for name, row in top1_bands.items():
        lines.append(f"| {name} | {row['count']} | {row['rate'] * 100:.2f}% |")
    lines.append("")

    lines.append("## Normalized-Entropy Bands")
    lines.append("| Band | Count | Rate |")
    lines.append("|---|---:|---:|")
    for name, row in entropy_bands.items():
        lines.append(f"| {name} | {row['count']} | {row['rate'] * 100:.2f}% |")
    lines.append("")

    if position_profile:
        lines.append("## Position-wise Distribution (Within Block)")
        lines.append("| block_pos | count | rate | top1 median | normalized-entropy median | effective-support median |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for row in position_profile:
            top1_stats = row.get("top1_prob", {})
            entropy_stats = row.get("normalized_entropy", {})
            support_stats = row.get("effective_support", {})
            lines.append(
                f"| {row['block_position']} | {row['count']} | {row['rate'] * 100:.2f}% | {top1_stats.get('median', 0.0):.4f} | {entropy_stats.get('median', 0.0):.4f} | {support_stats.get('median', 0.0):.3f} |"
            )
        lines.append("")

    valid_vocab_hist = [r for r in vocab_prob_hist if int(r.get("vocab_values", 0)) > 0]
    if valid_vocab_hist:
        lines.append("## Mean Histogram of Full-Vocabulary Probabilities (Positions 1-15)")
        lines.append("| block_pos | distributions | vocab_values | mass in first bin | mass in last bin |")
        lines.append("|---:|---:|---:|---:|---:|")
        for row in valid_vocab_hist:
            rates = row.get("rates", [])
            first_bin = float(rates[0]) if rates else 0.0
            last_bin = float(rates[-1]) if rates else 0.0
            lines.append(
                f"| {int(row.get('block_position', 0))} | {int(row.get('num_distributions', 0))} | {int(row.get('vocab_values', 0))} | {first_bin * 100:.4f}% | {last_bin * 100:.4f}% |"
            )
        lines.append("")
        lines.append(
            "Detailed per-bin values are saved in `draft_vocab_probability_hist_by_position.csv` and `draft_entropy_summary.json`."
        )
        lines.append("")

    lines.append("## Interpreting Sharp vs Flat")
    lines.append(
        "- More sharp: higher top1 probability, lower normalized entropy, lower effective support."
    )
    lines.append(
        "- More flat: lower top1 probability, higher normalized entropy, higher effective support."
    )
    lines.append(
        f"- Quick ratio (top1>=0.7 vs top1<0.5): **{(top1_bands.get('very_sharp_[0.9,1.0]', {}).get('rate', 0.0) + top1_bands.get('sharp_[0.7,0.9)', {}).get('rate', 0.0)) * 100:.2f}% vs {top1_bands.get('flat_[0.0,0.5)', {}).get('rate', 0.0) * 100:.2f}%**"
    )
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")

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
    entropy_logit_source: str = "positive",
    divergence_accumulator: Optional[List[float]] = None,
    draft_entropy_records: Optional[List[dict[str, Any]]] = None,
    vocab_probability_hist_accumulator: Optional[dict[str, Any]] = None,
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
            entropy_logits = select_entropy_logits(
                positive_logits=positive_draft_logits,
                negative_logits=negative_draft_logits,
                entropy_logit_source=entropy_logit_source,
            )
            _collect_draft_entropy_records(
                records=draft_entropy_records,
                analysis_logits=entropy_logits,
                logits_source=entropy_logit_source,
                sampled_draft_ids=block_output_ids[:, 1:],
                posterior_ids=posterior[:, :-1],
                acceptance_lengths=acceptance_lengths_per_batch,
                sample_idx=sample_idx,
                turn_idx=turn_idx,
                decode_step=decode_step,
                block_start=start,
                vocab_probability_hist_accumulator=vocab_probability_hist_accumulator,
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
    parser.add_argument("--histogram-bins", type=int, default=40)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", type=float, default=0.1)
    parser.add_argument("--cd-beta", type=float, default=0.0)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--entropy-logit-source",
        type=str,
        choices=["positive", "negative"],
        default="positive",
        help="Select which draft logits are analyzed for entropy/statistics.",
    )
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
    vocab_probability_hist_accumulator = init_vocab_probability_histogram_accumulator(
        max_positions=15,
        bins=args.histogram_bins,
    )

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
            for bs in [1, block_size]:
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
                    entropy_logit_source=args.entropy_logit_source,
                    divergence_accumulator=divergence_accumulator if bs == block_size else None,
                    draft_entropy_records=draft_entropy_records if bs == block_size else None,
                    vocab_probability_hist_accumulator=vocab_probability_hist_accumulator if bs == block_size else None,
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
        vocab_probability_hist_accumulator = dist.gather(vocab_probability_hist_accumulator, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        divergence_accumulator = list(chain(*divergence_accumulator))
        draft_entropy_records = list(chain(*draft_entropy_records))
        vocab_probability_hist_accumulator = merge_vocab_probability_histogram_accumulators(
            vocab_probability_hist_accumulator
        )

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
        vocab_probability_hist_accumulator=vocab_probability_hist_accumulator,
    )

    records_jsonl = report_dir / "draft_entropy_records.jsonl"
    records_csv = report_dir / "draft_entropy_records.csv"
    vocab_hist_csv = report_dir / "draft_vocab_probability_hist_by_position.csv"
    summary_json = report_dir / "draft_entropy_summary.json"
    report_md = report_dir / "draft_entropy_report.md"

    write_jsonl(records_jsonl, draft_entropy_records)
    write_csv(records_csv, draft_entropy_records)
    write_vocab_probability_histogram_csv(
        vocab_hist_csv,
        entropy_summary.get("distribution", {}).get("vocab_probability_histogram_by_position", []),
    )
    summary_json.write_text(json.dumps(entropy_summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_files = [] if args.no_plots else maybe_create_entropy_plots(report_dir, draft_entropy_records, entropy_summary)
    write_entropy_report_md(
        report_path=report_md,
        summary=entropy_summary,
        records=draft_entropy_records,
        plot_files=plot_files,
    )

    print(f"Draft entropy records: {entropy_summary.get('num_records', 0)}")
    if entropy_summary.get("num_records", 0) > 0:
        quality = entropy_summary.get("quick_quality_context", {})
        dist_stats = entropy_summary.get("distribution", {})
        source_name = entropy_summary.get("meta", {}).get("entropy_logit_source", "positive")
        top1_mean = dist_stats.get("top1_prob", {}).get("mean", 0.0)
        ent_norm_mean = dist_stats.get("normalized_entropy", {}).get("mean", 0.0)
        support_mean = dist_stats.get("effective_support", {}).get("mean", 0.0)
        hit_rate = quality.get("analyzed_argmax_hit_rate", 0.0)
        accepted_rate = quality.get("accepted_by_target_rate", 0.0)
        top1_bands = dist_stats.get("top1_prob_bands", {})
        very_sharp = top1_bands.get("very_sharp_[0.9,1.0]", {}).get("rate", 0.0)
        sharp = top1_bands.get("sharp_[0.7,0.9)", {}).get("rate", 0.0)
        flat = top1_bands.get("flat_[0.0,0.5)", {}).get("rate", 0.0)
        print(f"Analyzed logits source: {source_name}")
        print(f"Analyzed argmax hit rate (context): {hit_rate * 100:.2f}%")
        print(f"Accepted-by-target rate (context): {accepted_rate * 100:.2f}%")
        print(f"Mean top1 probability: {top1_mean:.4f}")
        print(f"Mean normalized entropy: {ent_norm_mean:.4f}")
        print(f"Mean effective support: {support_mean:.4f}")
        print(
            f"Sharpness check (top1>=0.7 vs top1<0.5): {(very_sharp + sharp) * 100:.2f}% vs {flat * 100:.2f}%"
        )

    print(f"Report directory: {report_dir}")
    print(f"  - {records_jsonl.name}")
    print(f"  - {records_csv.name}")
    print(f"  - {vocab_hist_csv.name}")
    print(f"  - {summary_json.name}")
    print(f"  - {report_md.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")

if __name__ == "__main__":
    main()
