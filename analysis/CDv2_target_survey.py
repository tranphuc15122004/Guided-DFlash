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
FINAL_OVERRIDE_KEEP = 6


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


def _safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _build_rank_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    if report_dir:
        out = Path(report_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("analysis") / "results" / f"cdv2_target_rank_{_safe_dataset_name(dataset_name)}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n")
        return

    fieldnames = sorted({k for row in rows for k in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _describe_distribution(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "p10": 0.0,
            "median": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }

    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _subset_distribution(values: np.ndarray, mask: np.ndarray) -> dict[str, Any]:
    if values.size == 0 or mask.size == 0 or mask.shape[0] != values.shape[0]:
        return _describe_distribution(np.asarray([], dtype=np.float64))
    return _describe_distribution(values[mask])


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    token_logit = logits_1d[token_id]
    return int((logits_1d > token_logit).sum().item()) + 1


def _compute_target_token_stats(logits_1d: torch.Tensor, target_token_id: int) -> dict[str, Any]:
    logits = logits_1d.float()
    rank = _token_rank_from_logits(logits, target_token_id)
    logprob = float(torch.log_softmax(logits, dim=-1)[target_token_id].item())
    return {
        "rank": int(rank),
        "logprob": logprob,
        "top1_hit": int(rank <= 1),
        "top5_hit": int(rank <= 5),
        "top10_hit": int(rank <= 10),
    }


def _build_target_rank_record(
    *,
    sample_idx: int,
    turn_idx: int,
    decode_step: int,
    start: int,
    step_offset: int,
    reject_offset: int,
    acceptance_length: int,
    target_token_id: int,
    sampled_draft_token_id: int,
    positive_step_logits: torch.Tensor,
    negative_step_logits: torch.Tensor,
    cd_raw_step_logits: torch.Tensor,
    cd_masked_step_logits: torch.Tensor,
    cd_final_step_logits: torch.Tensor,
    candidate_mask_step: torch.Tensor,
    n_keep: int,
) -> dict[str, Any]:
    positive_stats = _compute_target_token_stats(positive_step_logits, target_token_id)
    negative_stats = _compute_target_token_stats(negative_step_logits, target_token_id)
    cd_raw_stats = _compute_target_token_stats(cd_raw_step_logits, target_token_id)
    cd_masked_stats = _compute_target_token_stats(cd_masked_step_logits, target_token_id)
    cd_final_stats = _compute_target_token_stats(cd_final_step_logits, target_token_id)

    target_in_candidate_mask = int(candidate_mask_step[target_token_id].item())
    is_nkeep_overridden = int(step_offset < n_keep)

    record = {
        "sample_idx": int(sample_idx),
        "turn_idx": int(turn_idx),
        "decode_step": int(decode_step),
        "block_start_position": int(start),
        "absolute_position": int(start + step_offset + 1),
        "block_position": int(step_offset + 1),
        "acceptance_length": int(acceptance_length),
        "reject_offset_in_block": int(reject_offset),
        "reject_block_position": int(reject_offset + 1),
        "position_offset_from_reject": int(step_offset - reject_offset),
        "cohort_reject_only": int(step_offset == reject_offset),
        "cohort_reject_to_tail": 1,
        "target_token_id": int(target_token_id),
        "sampled_draft_token_id": int(sampled_draft_token_id),
        "target_in_candidate_mask": int(target_in_candidate_mask),
        "target_filtered_by_mask": int(1 - target_in_candidate_mask),
        "is_nkeep_overridden": int(is_nkeep_overridden),

        "positive_target_rank": int(positive_stats["rank"]),
        "positive_target_logprob": float(positive_stats["logprob"]),
        "positive_top1_hit": int(positive_stats["top1_hit"]),
        "positive_top5_hit": int(positive_stats["top5_hit"]),
        "positive_top10_hit": int(positive_stats["top10_hit"]),

        "negative_target_rank": int(negative_stats["rank"]),
        "negative_target_logprob": float(negative_stats["logprob"]),
        "negative_top1_hit": int(negative_stats["top1_hit"]),
        "negative_top5_hit": int(negative_stats["top5_hit"]),
        "negative_top10_hit": int(negative_stats["top10_hit"]),

        "cd_raw_target_rank": int(cd_raw_stats["rank"]),
        "cd_raw_target_logprob": float(cd_raw_stats["logprob"]),
        "cd_raw_top1_hit": int(cd_raw_stats["top1_hit"]),
        "cd_raw_top5_hit": int(cd_raw_stats["top5_hit"]),
        "cd_raw_top10_hit": int(cd_raw_stats["top10_hit"]),

        "cd_masked_target_rank": int(cd_masked_stats["rank"]),
        "cd_masked_target_logprob": float(cd_masked_stats["logprob"]),
        "cd_masked_top1_hit": int(cd_masked_stats["top1_hit"]),
        "cd_masked_top5_hit": int(cd_masked_stats["top5_hit"]),
        "cd_masked_top10_hit": int(cd_masked_stats["top10_hit"]),

        "cd_final_target_rank": int(cd_final_stats["rank"]),
        "cd_final_target_logprob": float(cd_final_stats["logprob"]),
        "cd_final_top1_hit": int(cd_final_stats["top1_hit"]),
        "cd_final_top5_hit": int(cd_final_stats["top5_hit"]),
        "cd_final_top10_hit": int(cd_final_stats["top10_hit"]),
    }

    record.update(
        {
            "delta_rank_negative_vs_positive": int(record["positive_target_rank"] - record["negative_target_rank"]),
            "delta_logprob_negative_vs_positive": float(
                record["negative_target_logprob"] - record["positive_target_logprob"]
            ),

            "delta_rank_cd_raw_vs_positive": int(record["positive_target_rank"] - record["cd_raw_target_rank"]),
            "delta_logprob_cd_raw_vs_positive": float(
                record["cd_raw_target_logprob"] - record["positive_target_logprob"]
            ),

            "delta_rank_cd_masked_vs_positive": int(
                record["positive_target_rank"] - record["cd_masked_target_rank"]
            ),
            "delta_logprob_cd_masked_vs_positive": float(
                record["cd_masked_target_logprob"] - record["positive_target_logprob"]
            ),

            "delta_rank_cd_final_vs_positive": int(record["positive_target_rank"] - record["cd_final_target_rank"]),
            "delta_logprob_cd_final_vs_positive": float(
                record["cd_final_target_logprob"] - record["positive_target_logprob"]
            ),

            "delta_rank_cd_masked_vs_cd_raw": int(record["cd_raw_target_rank"] - record["cd_masked_target_rank"]),
            "delta_logprob_cd_masked_vs_cd_raw": float(
                record["cd_masked_target_logprob"] - record["cd_raw_target_logprob"]
            ),

            "delta_rank_cd_final_vs_cd_masked": int(
                record["cd_masked_target_rank"] - record["cd_final_target_rank"]
            ),
            "delta_logprob_cd_final_vs_cd_masked": float(
                record["cd_final_target_logprob"] - record["cd_masked_target_logprob"]
            ),

            "gain_top1_cd_final_vs_positive": int(
                (record["cd_final_top1_hit"] == 1) and (record["positive_top1_hit"] == 0)
            ),
            "loss_top1_cd_final_vs_positive": int(
                (record["cd_final_top1_hit"] == 0) and (record["positive_top1_hit"] == 1)
            ),
            "gain_top5_cd_final_vs_positive": int(
                (record["cd_final_top5_hit"] == 1) and (record["positive_top5_hit"] == 0)
            ),
            "loss_top5_cd_final_vs_positive": int(
                (record["cd_final_top5_hit"] == 0) and (record["positive_top5_hit"] == 1)
            ),
            "gain_top10_cd_final_vs_positive": int(
                (record["cd_final_top10_hit"] == 1) and (record["positive_top10_hit"] == 0)
            ),
            "loss_top10_cd_final_vs_positive": int(
                (record["cd_final_top10_hit"] == 0) and (record["positive_top10_hit"] == 1)
            ),
        }
    )

    return record


def _build_cohort_position_profile(
    records: list[dict[str, Any]],
    cohort_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not records:
        return [], []

    total = max(1, len(records))
    profile_rows: list[dict[str, Any]] = []
    csv_rows: list[dict[str, Any]] = []

    positions = sorted(int(row["block_position"]) for row in records)
    for pos in sorted(set(positions)):
        subset = [row for row in records if int(row["block_position"]) == pos]
        count = len(subset)
        if count == 0:
            continue

        pos_rank = np.asarray([row["positive_target_rank"] for row in subset], dtype=np.float64)
        neg_rank = np.asarray([row["negative_target_rank"] for row in subset], dtype=np.float64)
        cd_raw_rank = np.asarray([row["cd_raw_target_rank"] for row in subset], dtype=np.float64)
        cd_masked_rank = np.asarray([row["cd_masked_target_rank"] for row in subset], dtype=np.float64)
        cd_final_rank = np.asarray([row["cd_final_target_rank"] for row in subset], dtype=np.float64)

        delta_rank = np.asarray([row["delta_rank_cd_final_vs_positive"] for row in subset], dtype=np.float64)
        delta_lp = np.asarray([row["delta_logprob_cd_final_vs_positive"] for row in subset], dtype=np.float64)
        target_in_mask = np.asarray([row["target_in_candidate_mask"] for row in subset], dtype=np.float64)
        overridden = np.asarray([row["is_nkeep_overridden"] for row in subset], dtype=np.float64)

        profile_row = {
            "block_position": int(pos),
            "count": int(count),
            "rate": float(count / total),
            "positive_target_rank": _describe_distribution(pos_rank),
            "negative_target_rank": _describe_distribution(neg_rank),
            "cd_raw_target_rank": _describe_distribution(cd_raw_rank),
            "cd_masked_target_rank": _describe_distribution(cd_masked_rank),
            "cd_final_target_rank": _describe_distribution(cd_final_rank),
            "delta_rank_cd_final_vs_positive": _describe_distribution(delta_rank),
            "delta_logprob_cd_final_vs_positive": _describe_distribution(delta_lp),
            "target_in_candidate_mask_rate": float(target_in_mask.mean()),
            "is_nkeep_overridden_rate": float(overridden.mean()),
        }
        profile_rows.append(profile_row)

        csv_rows.append(
            {
                "cohort": cohort_name,
                "block_position": int(pos),
                "count": int(count),
                "rate": float(count / total),
                "positive_rank_mean": float(pos_rank.mean()),
                "negative_rank_mean": float(neg_rank.mean()),
                "cd_raw_rank_mean": float(cd_raw_rank.mean()),
                "cd_masked_rank_mean": float(cd_masked_rank.mean()),
                "cd_final_rank_mean": float(cd_final_rank.mean()),
                "delta_rank_cd_final_vs_positive_mean": float(delta_rank.mean()),
                "delta_logprob_cd_final_vs_positive_mean": float(delta_lp.mean()),
                "target_in_candidate_mask_rate": float(target_in_mask.mean()),
                "is_nkeep_overridden_rate": float(overridden.mean()),
            }
        )

    return profile_rows, csv_rows


def _build_cohort_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "num_records": 0,
            "target_in_candidate_mask_rate": 0.0,
            "target_filtered_by_mask_rate": 0.0,
            "is_nkeep_overridden_rate": 0.0,
            "rank_distribution": {},
            "logprob_distribution": {},
            "topk_hit_rates": {},
            "cd_impact_vs_positive": {},
            "mask_effect": {},
            "override_effect": {},
            "hit_change_cd_final_vs_positive": {},
        }

    variant_names = ["positive", "negative", "cd_raw", "cd_masked", "cd_final"]

    def arr(name: str, dtype: np.dtype = np.float64) -> np.ndarray:
        return np.asarray([row[name] for row in records], dtype=dtype)

    target_in_mask = arr("target_in_candidate_mask")
    target_filtered = arr("target_filtered_by_mask")
    overridden = arr("is_nkeep_overridden")
    filtered_mask = target_filtered.astype(np.int64) == 1
    overridden_mask = overridden.astype(np.int64) == 1

    rank_distribution = {
        variant: _describe_distribution(arr(f"{variant}_target_rank"))
        for variant in variant_names
    }
    logprob_distribution = {
        variant: _describe_distribution(arr(f"{variant}_target_logprob"))
        for variant in variant_names
    }

    topk_hit_rates = {
        variant: {
            "top1": float(arr(f"{variant}_top1_hit").mean()),
            "top5": float(arr(f"{variant}_top5_hit").mean()),
            "top10": float(arr(f"{variant}_top10_hit").mean()),
        }
        for variant in variant_names
    }

    compare_variants = ["negative", "cd_raw", "cd_masked", "cd_final"]
    cd_impact_vs_positive = {
        variant: {
            "delta_rank": _describe_distribution(arr(f"delta_rank_{variant}_vs_positive")),
            "delta_logprob": _describe_distribution(arr(f"delta_logprob_{variant}_vs_positive")),
            "improved_rank_rate": float((arr(f"delta_rank_{variant}_vs_positive") > 0).mean()),
            "improved_logprob_rate": float((arr(f"delta_logprob_{variant}_vs_positive") > 0).mean()),
        }
        for variant in compare_variants
    }

    mask_delta_rank = arr("delta_rank_cd_masked_vs_cd_raw")
    mask_delta_lp = arr("delta_logprob_cd_masked_vs_cd_raw")
    override_delta_rank = arr("delta_rank_cd_final_vs_cd_masked")
    override_delta_lp = arr("delta_logprob_cd_final_vs_cd_masked")

    mask_effect = {
        "all": {
            "delta_rank": _describe_distribution(mask_delta_rank),
            "delta_logprob": _describe_distribution(mask_delta_lp),
            "improved_rank_rate": float((mask_delta_rank > 0).mean()),
            "improved_logprob_rate": float((mask_delta_lp > 0).mean()),
        },
        "target_filtered_only": {
            "count": int(filtered_mask.sum()),
            "delta_rank": _subset_distribution(mask_delta_rank, filtered_mask),
            "delta_logprob": _subset_distribution(mask_delta_lp, filtered_mask),
            "improved_rank_rate": float((mask_delta_rank[filtered_mask] > 0).mean()) if filtered_mask.any() else 0.0,
            "improved_logprob_rate": float((mask_delta_lp[filtered_mask] > 0).mean()) if filtered_mask.any() else 0.0,
        },
    }

    override_effect = {
        "all": {
            "delta_rank": _describe_distribution(override_delta_rank),
            "delta_logprob": _describe_distribution(override_delta_lp),
            "improved_rank_rate": float((override_delta_rank > 0).mean()),
            "improved_logprob_rate": float((override_delta_lp > 0).mean()),
        },
        "nkeep_overridden_only": {
            "count": int(overridden_mask.sum()),
            "delta_rank": _subset_distribution(override_delta_rank, overridden_mask),
            "delta_logprob": _subset_distribution(override_delta_lp, overridden_mask),
            "improved_rank_rate": float((override_delta_rank[overridden_mask] > 0).mean()) if overridden_mask.any() else 0.0,
            "improved_logprob_rate": float((override_delta_lp[overridden_mask] > 0).mean()) if overridden_mask.any() else 0.0,
        },
    }

    hit_change_cd_final_vs_positive = {
        "gain_top1_rate": float(arr("gain_top1_cd_final_vs_positive").mean()),
        "loss_top1_rate": float(arr("loss_top1_cd_final_vs_positive").mean()),
        "gain_top5_rate": float(arr("gain_top5_cd_final_vs_positive").mean()),
        "loss_top5_rate": float(arr("loss_top5_cd_final_vs_positive").mean()),
        "gain_top10_rate": float(arr("gain_top10_cd_final_vs_positive").mean()),
        "loss_top10_rate": float(arr("loss_top10_cd_final_vs_positive").mean()),
    }

    return {
        "num_records": int(len(records)),
        "target_in_candidate_mask_rate": float(target_in_mask.mean()),
        "target_filtered_by_mask_rate": float(target_filtered.mean()),
        "is_nkeep_overridden_rate": float(overridden.mean()),
        "rank_distribution": rank_distribution,
        "logprob_distribution": logprob_distribution,
        "topk_hit_rates": topk_hit_rates,
        "cd_impact_vs_positive": cd_impact_vs_positive,
        "mask_effect": mask_effect,
        "override_effect": override_effect,
        "hit_change_cd_final_vs_positive": hit_change_cd_final_vs_positive,
    }


def build_target_rank_summary(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    block_size: int,
    cd_alpha: float,
    cd_beta: float,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    negative_hidden_mode: str,
    decoding_speedup: float,
    avg_acceptance_length: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reject_only_records = [row for row in records if int(row.get("cohort_reject_only", 0)) == 1]
    reject_to_tail_records = [row for row in records if int(row.get("cohort_reject_to_tail", 0)) == 1]
    unique_reject_events = {
        (
            int(row["sample_idx"]),
            int(row["turn_idx"]),
            int(row["decode_step"]),
            int(row["block_start_position"]),
        )
        for row in reject_only_records
    }

    reject_only_summary = _build_cohort_summary(reject_only_records)
    reject_to_tail_summary = _build_cohort_summary(reject_to_tail_records)

    reject_only_profile, reject_only_csv_rows = _build_cohort_position_profile(reject_only_records, "reject_only")
    reject_to_tail_profile, reject_to_tail_csv_rows = _build_cohort_position_profile(reject_to_tail_records, "reject_to_tail")

    reject_only_summary["block_position_profile"] = reject_only_profile
    reject_to_tail_summary["block_position_profile"] = reject_to_tail_profile

    summary = {
        "meta": {
            "dataset": dataset,
            "seed": int(seed),
            "block_size": int(block_size),
            "target_token_definition": "posterior_sampled_token",
            "cd_alpha": float(cd_alpha),
            "cd_beta": float(cd_beta),
            "negative_context_dropout": float(negative_context_dropout),
            "negative_context_noise_std": float(negative_context_noise_std),
            "negative_hidden_mode": negative_hidden_mode,
            "top64_mask_mode": bool(TOP64_MASK_MODE),
            "final_override_keep": int(FINAL_OVERRIDE_KEEP),
            "decoding_speedup": float(decoding_speedup),
            "avg_acceptance_length": float(avg_acceptance_length),
        },
        "num_records_total": int(len(records)),
        "num_reject_events": int(len(unique_reject_events)),
        "cohorts": {
            "reject_only": reject_only_summary,
            "reject_to_tail": reject_to_tail_summary,
        },
        "consistency_checks": {
            "reject_only_count": int(len(reject_only_records)),
            "reject_to_tail_count": int(len(reject_to_tail_records)),
            "reject_only_equals_num_reject_events": bool(len(reject_only_records) == len(unique_reject_events)),
            "reject_to_tail_ge_reject_only": bool(len(reject_to_tail_records) >= len(reject_only_records)),
        },
    }

    return summary, reject_only_csv_rows + reject_to_tail_csv_rows


def _sanitize_md_text(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n")


def write_target_rank_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    tokenizer: AutoTokenizer,
    max_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# CDv2 Target-Token Rank Survey")
    lines.append("")

    if summary.get("num_records_total", 0) == 0:
        lines.append("No reject-based target-rank record was collected.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    meta = summary.get("meta", {})
    lines.append("## Setup")
    lines.append(f"- Dataset: **{meta.get('dataset', '')}**")
    lines.append(f"- Seed: **{meta.get('seed', 0)}**")
    lines.append(f"- Block size: **{meta.get('block_size', 0)}**")
    lines.append(f"- Target token definition: **{meta.get('target_token_definition', 'posterior_sampled_token')}**")
    lines.append(f"- CD alpha / beta: **{meta.get('cd_alpha', 0.0):.4f} / {meta.get('cd_beta', 0.0):.4f}**")
    lines.append(f"- Decoding speedup: **{meta.get('decoding_speedup', 0.0):.2f}x**")
    lines.append(f"- Avg acceptance length: **{meta.get('avg_acceptance_length', 0.0):.2f}**")
    lines.append("")

    lines.append("## Cohort Sizes")
    lines.append(f"- Total records (reject_to_tail): **{summary.get('num_records_total', 0)}**")
    lines.append(f"- Reject events (reject_only): **{summary.get('num_reject_events', 0)}**")
    checks = summary.get("consistency_checks", {})
    lines.append(
        "- Invariant reject_to_tail >= reject_only: "
        f"**{checks.get('reject_to_tail_ge_reject_only', False)}**"
    )
    lines.append("")

    cohorts = summary.get("cohorts", {})
    for cohort_name in ["reject_only", "reject_to_tail"]:
        cohort = cohorts.get(cohort_name, {})
        lines.append(f"## Cohort: {cohort_name}")
        lines.append(f"- Records: **{cohort.get('num_records', 0)}**")
        lines.append(
            f"- Target in candidate mask rate: **{cohort.get('target_in_candidate_mask_rate', 0.0) * 100:.2f}%**"
        )
        lines.append(
            f"- Target filtered by mask rate: **{cohort.get('target_filtered_by_mask_rate', 0.0) * 100:.2f}%**"
        )
        lines.append(
            f"- n_keep overridden rate: **{cohort.get('is_nkeep_overridden_rate', 0.0) * 100:.2f}%**"
        )
        lines.append("")

        topk = cohort.get("topk_hit_rates", {})
        if topk:
            lines.append("### Top-k Hit Rates")
            lines.append("| Variant | Top-1 | Top-5 | Top-10 |")
            lines.append("|---|---:|---:|---:|")
            for variant in ["positive", "negative", "cd_raw", "cd_masked", "cd_final"]:
                row = topk.get(variant, {})
                lines.append(
                    f"| {variant} | {row.get('top1', 0.0) * 100:.2f}% | {row.get('top5', 0.0) * 100:.2f}% | {row.get('top10', 0.0) * 100:.2f}% |"
                )
            lines.append("")

        cd_impact = cohort.get("cd_impact_vs_positive", {})
        if cd_impact:
            lines.append("### Delta vs Positive")
            lines.append("| Variant | Mean delta rank | Median delta rank | Mean delta logprob | Median delta logprob |")
            lines.append("|---|---:|---:|---:|---:|")
            for variant in ["negative", "cd_raw", "cd_masked", "cd_final"]:
                row = cd_impact.get(variant, {})
                dr = row.get("delta_rank", {})
                dl = row.get("delta_logprob", {})
                lines.append(
                    f"| {variant} | {dr.get('mean', 0.0):+.4f} | {dr.get('median', 0.0):+.4f} | {dl.get('mean', 0.0):+.5f} | {dl.get('median', 0.0):+.5f} |"
                )
            lines.append("")

        hit_change = cohort.get("hit_change_cd_final_vs_positive", {})
        if hit_change:
            lines.append("### CD Final Hit Change vs Positive")
            lines.append(
                f"- Top-1 gain/loss: **{hit_change.get('gain_top1_rate', 0.0) * 100:.2f}% / {hit_change.get('loss_top1_rate', 0.0) * 100:.2f}%**"
            )
            lines.append(
                f"- Top-5 gain/loss: **{hit_change.get('gain_top5_rate', 0.0) * 100:.2f}% / {hit_change.get('loss_top5_rate', 0.0) * 100:.2f}%**"
            )
            lines.append(
                f"- Top-10 gain/loss: **{hit_change.get('gain_top10_rate', 0.0) * 100:.2f}% / {hit_change.get('loss_top10_rate', 0.0) * 100:.2f}%**"
            )
            lines.append("")

        mask_effect = cohort.get("mask_effect", {})
        override_effect = cohort.get("override_effect", {})
        if mask_effect or override_effect:
            lines.append("### Mask and Override Effects")
            if mask_effect:
                mask_all = mask_effect.get("all", {})
                lines.append(
                    "- Mask effect (cd_masked - cd_raw) mean delta rank/logprob: "
                    f"**{mask_all.get('delta_rank', {}).get('mean', 0.0):+.4f} / {mask_all.get('delta_logprob', {}).get('mean', 0.0):+.5f}**"
                )
                mask_filtered = mask_effect.get("target_filtered_only", {})
                lines.append(
                    f"- Mask effect when target filtered (count={mask_filtered.get('count', 0)}): "
                    f"mean delta rank/logprob = **{mask_filtered.get('delta_rank', {}).get('mean', 0.0):+.4f} / {mask_filtered.get('delta_logprob', {}).get('mean', 0.0):+.5f}**"
                )
            if override_effect:
                override_all = override_effect.get("all", {})
                lines.append(
                    "- Override effect (cd_final - cd_masked) mean delta rank/logprob: "
                    f"**{override_all.get('delta_rank', {}).get('mean', 0.0):+.4f} / {override_all.get('delta_logprob', {}).get('mean', 0.0):+.5f}**"
                )
                override_subset = override_effect.get("nkeep_overridden_only", {})
                lines.append(
                    f"- Override effect at overridden positions (count={override_subset.get('count', 0)}): "
                    f"mean delta rank/logprob = **{override_subset.get('delta_rank', {}).get('mean', 0.0):+.4f} / {override_subset.get('delta_logprob', {}).get('mean', 0.0):+.5f}**"
                )
            lines.append("")

        profile = cohort.get("block_position_profile", [])
        if profile:
            lines.append("### Position-wise Profile")
            lines.append("| block_pos | count | rate | median pos rank | median neg rank | median cd_raw rank | median cd_masked rank | median cd_final rank | mean delta rank(final-pos) |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
            for row in profile:
                lines.append(
                    "| {pos} | {count} | {rate:.2f}% | {pos_med:.2f} | {neg_med:.2f} | {raw_med:.2f} | {masked_med:.2f} | {final_med:.2f} | {delta_mean:+.4f} |".format(
                        pos=row.get("block_position", 0),
                        count=row.get("count", 0),
                        rate=row.get("rate", 0.0) * 100,
                        pos_med=row.get("positive_target_rank", {}).get("median", 0.0),
                        neg_med=row.get("negative_target_rank", {}).get("median", 0.0),
                        raw_med=row.get("cd_raw_target_rank", {}).get("median", 0.0),
                        masked_med=row.get("cd_masked_target_rank", {}).get("median", 0.0),
                        final_med=row.get("cd_final_target_rank", {}).get("median", 0.0),
                        delta_mean=row.get("delta_rank_cd_final_vs_positive", {}).get("mean", 0.0),
                    )
                )
            lines.append("")

    def token_text(token_id: int) -> str:
        return _sanitize_md_text(tokenizer.decode([int(token_id)], skip_special_tokens=False))

    reject_only_records = [row for row in records if int(row.get("cohort_reject_only", 0)) == 1]
    if reject_only_records:
        lines.append("## Top Reject-Only Cases with Best/Worst CD Final Delta")

        improved = sorted(
            reject_only_records,
            key=lambda row: float(row.get("delta_logprob_cd_final_vs_positive", 0.0)),
            reverse=True,
        )[:max_rows]
        worsened = sorted(
            reject_only_records,
            key=lambda row: float(row.get("delta_logprob_cd_final_vs_positive", 0.0)),
        )[:max_rows]

        lines.append("### Most Improved")
        lines.append("| sample | turn | step | block_pos | target | draft | d_rank(final-pos) | d_logprob(final-pos) | in_mask | overridden |")
        lines.append("|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
        for row in improved:
            lines.append(
                "| {sample_idx} | {turn_idx} | {decode_step} | {block_position} | {target} ({target_id}) | {draft} ({draft_id}) | {d_rank:+d} | {d_lp:+.5f} | {in_mask} | {override} |".format(
                    sample_idx=int(row["sample_idx"]),
                    turn_idx=int(row["turn_idx"]),
                    decode_step=int(row["decode_step"]),
                    block_position=int(row["block_position"]),
                    target=token_text(int(row["target_token_id"])),
                    target_id=int(row["target_token_id"]),
                    draft=token_text(int(row["sampled_draft_token_id"])),
                    draft_id=int(row["sampled_draft_token_id"]),
                    d_rank=int(row["delta_rank_cd_final_vs_positive"]),
                    d_lp=float(row["delta_logprob_cd_final_vs_positive"]),
                    in_mask=int(row["target_in_candidate_mask"]),
                    override=int(row["is_nkeep_overridden"]),
                )
            )
        lines.append("")

        lines.append("### Most Worsened")
        lines.append("| sample | turn | step | block_pos | target | draft | d_rank(final-pos) | d_logprob(final-pos) | in_mask | overridden |")
        lines.append("|---:|---:|---:|---:|---|---|---:|---:|---:|---:|")
        for row in worsened:
            lines.append(
                "| {sample_idx} | {turn_idx} | {decode_step} | {block_position} | {target} ({target_id}) | {draft} ({draft_id}) | {d_rank:+d} | {d_lp:+.5f} | {in_mask} | {override} |".format(
                    sample_idx=int(row["sample_idx"]),
                    turn_idx=int(row["turn_idx"]),
                    decode_step=int(row["decode_step"]),
                    block_position=int(row["block_position"]),
                    target=token_text(int(row["target_token_id"])),
                    target_id=int(row["target_token_id"]),
                    draft=token_text(int(row["sampled_draft_token_id"])),
                    draft_id=int(row["sampled_draft_token_id"]),
                    d_rank=int(row["delta_rank_cd_final_vs_positive"]),
                    d_lp=float(row["delta_logprob_cd_final_vs_positive"]),
                    in_mask=int(row["target_in_candidate_mask"]),
                    override=int(row["is_nkeep_overridden"]),
                )
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_target_rank_console_summary(summary: dict[str, Any]) -> None:
    cohorts = summary.get("cohorts", {})
    for cohort_name in ["reject_only", "reject_to_tail"]:
        cohort = cohorts.get(cohort_name, {})
        num_records = int(cohort.get("num_records", 0))
        if num_records == 0:
            print(f"{cohort_name}: no records.")
            continue

        impact_final = cohort.get("cd_impact_vs_positive", {}).get("cd_final", {})
        dr = impact_final.get("delta_rank", {})
        dl = impact_final.get("delta_logprob", {})

        hit_change = cohort.get("hit_change_cd_final_vs_positive", {})
        mask_effect = cohort.get("mask_effect", {}).get("all", {})
        override_effect = cohort.get("override_effect", {}).get("all", {})

        print(
            f"[{cohort_name}] records={num_records} | "
            f"delta_rank(final-pos) mean/median={dr.get('mean', 0.0):+.4f}/{dr.get('median', 0.0):+.4f} | "
            f"delta_logprob(final-pos) mean/median={dl.get('mean', 0.0):+.5f}/{dl.get('median', 0.0):+.5f}"
        )
        print(
            f"[{cohort_name}] top1 gain/loss={hit_change.get('gain_top1_rate', 0.0) * 100:.2f}%/"
            f"{hit_change.get('loss_top1_rate', 0.0) * 100:.2f}% | "
            f"mask delta rank mean={mask_effect.get('delta_rank', {}).get('mean', 0.0):+.4f} | "
            f"override delta rank mean={override_effect.get('delta_rank', {}).get('mean', 0.0):+.4f}"
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
    rank_records_accumulator: Optional[List[dict[str, Any]]] = None,
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
            
            cd_raw_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            
            cd_masked_logits = apply_cd_candidate_filter(
                logits=cd_raw_logits,
                candidate_mask=candidate_mask,
            )

            cd_final_logits = cd_masked_logits.clone()
            
            # Keep early draft positions close to positive branch to reduce unstable trajectory drift.
            """ n_keep = min(FINAL_OVERRIDE_KEEP, cd_final_logits.size(1))
            cd_final_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :] """
            
            block_output_ids[:, 1:] = sample(cd_final_logits , gen=gen)
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
        acceptance_length = int((block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())

        if rank_records_accumulator is not None and block_size > 1 and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            for step_offset in range(reject_offset, block_size - 1):
                target_token_id = int(posterior[0, step_offset].item())
                sampled_draft_token_id = int(block_output_ids[0, step_offset + 1].item())
                rank_records_accumulator.append(
                    _build_target_rank_record(
                        sample_idx=sample_idx,
                        turn_idx=turn_idx,
                        decode_step=decode_step,
                        start=start,
                        step_offset=step_offset,
                        reject_offset=reject_offset,
                        acceptance_length=acceptance_length,
                        target_token_id=target_token_id,
                        sampled_draft_token_id=sampled_draft_token_id,
                        positive_step_logits=positive_draft_logits[0, step_offset, :],
                        negative_step_logits=negative_draft_logits[0, step_offset, :],
                        cd_raw_step_logits=cd_raw_logits[0, step_offset, :],
                        cd_masked_step_logits=cd_masked_logits[0, step_offset, :],
                        cd_final_step_logits=cd_final_logits[0, step_offset, :],
                        candidate_mask_step=candidate_mask[0, step_offset, :],
                        n_keep=n_keep,
                    )
                )

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", type=float, default=0.1)
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
    parser.add_argument(
        "--rank-report-dir",
        type=str,
        default=None,
        help="Output directory for target-rank artifacts. If omitted, a timestamped folder is created under analysis/results.",
    )
    parser.add_argument(
        "--max-rank-report-rows",
        type=int,
        default=30,
        help="Maximum number of improved/worsened cases shown in markdown report tables.",
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
    target_rank_records: List[dict[str, Any]] = []

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
                    divergence_accumulator = divergence_accumulator,
                    rank_records_accumulator=target_rank_records if bs == block_size else None,
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
        target_rank_records = dist.gather(target_rank_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        target_rank_records = list(chain(*target_rank_records))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    decoding_speedup = t1 / tb
    print(f"Decoding speedup: {decoding_speedup:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    rank_report_dir = _build_rank_report_dir(args.dataset, args.rank_report_dir)

    _write_csv(rank_report_dir / "target_rank_records.csv", target_rank_records)
    _write_jsonl(rank_report_dir / "target_rank_records.jsonl", target_rank_records)

    rank_summary, by_position_rows = build_target_rank_summary(
        target_rank_records,
        dataset=args.dataset,
        seed=args.seed,
        block_size=block_size,
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        negative_context_dropout=args.negative_context_dropout,
        negative_context_noise_std=args.negative_context_noise_std,
        negative_hidden_mode=args.negative_hidden_mode,
        decoding_speedup=decoding_speedup,
        avg_acceptance_length=tau,
    )
    (rank_report_dir / "target_rank_summary.json").write_text(
        json.dumps(rank_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_csv(rank_report_dir / "target_rank_by_position.csv", by_position_rows)
    write_target_rank_report_md(
        rank_report_dir / "target_rank_report.md",
        rank_summary,
        target_rank_records,
        tokenizer,
        max_rows=max(1, args.max_rank_report_rows),
    )

    print(f"Target-rank records (reject_to_tail): {rank_summary.get('num_records_total', 0)}")
    print(f"Target-rank events (reject_only): {rank_summary.get('num_reject_events', 0)}")
    print_target_rank_console_summary(rank_summary)
    print(f"Target-rank artifacts: {rank_report_dir}")
    
    # Report the collected divergence metric (if any)
    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        # Optionally also show min/max for a quick sense of spread
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

if __name__ == "__main__":
    main()
