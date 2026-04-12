import argparse
import csv
import json
import os
import random
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import distributed as dist
import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import *

""" 
Schema: CDv2
Goal: To statistic and study first nagetive token embedding distance and similarity between positive and negative sample
"""


NEGATIVE_HIDDEN_MODE = "mask_zero"

CASE_IMPROVED = "improved_push_to_top1"
CASE_WORSE_DROP = "worsened_drop_from_top1"
CASE_WORSE_NOT_PUSH = "worsened_not_push_to_top1"
CASE_KEEP_TOP1 = "kept_top1"
CASE_LABELS = [
    CASE_IMPROVED,
    CASE_WORSE_DROP,
    CASE_WORSE_NOT_PUSH,
    CASE_KEEP_TOP1,
]

CASE_DISPLAY_NAMES = {
    CASE_IMPROVED: "Improved: target pushed to top-1",
    CASE_WORSE_DROP: "Worsened: target dropped from top-1",
    CASE_WORSE_NOT_PUSH: "Worsened: target still not top-1",
    CASE_KEEP_TOP1: "Stable: target stayed top-1",
}


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def build_negative_block_output_random(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(
        0,
        vocab_size,
        (batch_size,),
        device=block_output_ids.device,
        generator=gen,
    )
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids


def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    mode: str = "mask_zero",
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()
    batch_size, context_len = negative_target_hidden.shape[:2]

    if mode == "mask_zero":
        if dropout_ratio > 0.0:
            token_keep_mask = (
                torch.rand(
                    negative_target_hidden.shape[:2],
                    device=negative_target_hidden.device,
                    generator=gen,
                )
                >= dropout_ratio
            ).unsqueeze(-1)
            negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)
    elif mode == "shuffle_tokens":
        if context_len > 1:
            per_batch_indices = []
            for _ in range(batch_size):
                perm = torch.randperm(
                    context_len,
                    device=negative_target_hidden.device,
                    generator=gen,
                )
                per_batch_indices.append(perm)
            gather_index = (
                torch.stack(per_batch_indices, dim=0)
                .unsqueeze(-1)
                .expand(-1, -1, negative_target_hidden.shape[-1])
            )
            negative_target_hidden = torch.gather(
                negative_target_hidden,
                dim=1,
                index=gather_index,
            )
    else:
        raise ValueError(
            f"Unsupported negative hidden mode: {mode}. Use 'mask_zero' or 'shuffle_tokens'."
        )

    if noise_std > 0.0:
        noise = torch.randn(
            negative_target_hidden.shape,
            dtype=negative_target_hidden.dtype,
            device=negative_target_hidden.device,
            generator=gen,
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
    gen: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    batch_size = block_output_ids.shape[0]

    neg_block_output_ids = build_negative_block_output_random(
        block_output_ids,
        target.config.vocab_size,
        gen=gen,
    )
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        mode=negative_hidden_mode,
        gen=gen,
    )

    positive_noise_embedding = target.model.embed_tokens(block_output_ids)
    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)

    positive_first_embed = positive_noise_embedding[:, 0, :].float()
    negative_first_embed = negative_noise_embedding[:, 0, :].float()
    first_token_l2 = torch.norm(positive_first_embed - negative_first_embed, p=2, dim=-1)
    first_token_cosine_similarity = torch.nn.functional.cosine_similarity(
        positive_first_embed,
        negative_first_embed,
        dim=-1,
    )
    first_token_cosine_distance = 1.0 - first_token_cosine_similarity

    paired_noise_embedding = torch.cat([positive_noise_embedding, negative_noise_embedding], dim=0)
    paired_hidden = torch.cat([target_hidden, negative_target_hidden], dim=0)
    paired_position_ids = torch.cat([draft_position_ids, draft_position_ids], dim=0)

    paired_draft_logits = target.lm_head(
        model(
            target_hidden=paired_hidden,
            noise_embedding=paired_noise_embedding,
            position_ids=paired_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -block_size + 1 :, :]
    )

    first_draft_logits, second_draft_logits = paired_draft_logits.split(batch_size, dim=0)
    step_payload = {
        "positive_first_token_ids": block_output_ids[:, 0].clone(),
        "negative_first_token_ids": neg_block_output_ids[:, 0].clone(),
        "first_token_l2": first_token_l2,
        "first_token_cosine_similarity": first_token_cosine_similarity,
        "first_token_cosine_distance": first_token_cosine_distance,
    }
    return first_draft_logits, second_draft_logits, step_payload


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


def apply_cd_candidate_filter(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


def _compute_kl_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> float:
    p_first = torch.softmax(first_logits, dim=-1)
    p_second = torch.softmax(second_logits, dim=-1)
    kl = (p_first * (torch.log(p_first + 1e-10) - torch.log(p_second + 1e-10))).sum(dim=-1).mean()
    return float(kl.item())


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    token_logit = logits_1d[token_id]
    return int((logits_1d > token_logit).sum().item()) + 1


def _token_str(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def _safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    if report_dir:
        out = Path(report_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("analysis") / "results" / f"cdv2_embed_dist_{_safe_dataset_name(dataset_name)}_{ts}"
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

    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})
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
            "p25": 0.0,
            "median": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "max": 0.0,
        }
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p10": float(np.quantile(values, 0.10)),
        "p25": float(np.quantile(values, 0.25)),
        "median": float(np.quantile(values, 0.50)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "max": float(values.max()),
    }


def _safe_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 0.0
    if float(x.std()) == 0.0 or float(y.std()) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _compute_distance_bin_rows(
    records: list[dict[str, Any]],
    num_bins: int,
) -> list[dict[str, Any]]:
    if not records:
        return []

    l2_values = np.asarray([r["first_token_embedding_l2"] for r in records], dtype=np.float32)
    total = l2_values.size
    if total == 0:
        return []

    num_bins = max(2, int(num_bins))
    num_bins = min(num_bins, total)
    if num_bins <= 0:
        return []

    edges = np.quantile(l2_values, np.linspace(0.0, 1.0, num_bins + 1))
    for i in range(1, edges.shape[0]):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-9

    rows: list[dict[str, Any]] = []
    for i in range(num_bins):
        low = float(edges[i])
        high = float(edges[i + 1])
        if i < num_bins - 1:
            mask = (l2_values >= low) & (l2_values < high)
        else:
            mask = (l2_values >= low) & (l2_values <= high)

        idx = np.where(mask)[0]
        if idx.size == 0:
            continue

        subset = [records[int(j)] for j in idx.tolist()]
        subset_delta_logprob = np.asarray([r["delta_target_logprob"] for r in subset], dtype=np.float32)
        subset_delta_rank = np.asarray([r["delta_target_rank"] for r in subset], dtype=np.float32)
        subset_candidate = np.asarray([r["target_in_candidate_mask"] for r in subset], dtype=np.float32)
        subset_accepted = np.asarray([r["accepted_by_final"] for r in subset], dtype=np.float32)

        rows.append(
            {
                "bin_index": int(i),
                "l2_lower": low,
                "l2_upper": high,
                "bin_center_l2": float((low + high) * 0.5),
                "count": int(len(subset)),
                "rate": float(len(subset) / len(records)),
                "improved_rate": float(np.mean([r["case_type"] == CASE_IMPROVED for r in subset])),
                "drop_from_top1_rate": float(np.mean([r["case_type"] == CASE_WORSE_DROP for r in subset])),
                "not_push_to_top1_rate": float(np.mean([r["case_type"] == CASE_WORSE_NOT_PUSH for r in subset])),
                "keep_top1_rate": float(np.mean([r["case_type"] == CASE_KEEP_TOP1 for r in subset])),
                "mean_delta_target_logprob": float(subset_delta_logprob.mean()),
                "mean_delta_target_rank": float(subset_delta_rank.mean()),
                "target_in_candidate_mask_rate": float(subset_candidate.mean()),
                "accepted_by_final_rate": float(subset_accepted.mean()),
            }
        )

    return rows


def _build_case_embedding_summary(
    records: list[dict[str, Any]],
    args: argparse.Namespace,
    block_size: int,
) -> dict[str, Any]:
    meta = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "block_size": int(block_size),
        "cd_alpha": float(args.cd_alpha),
        "cd_beta": float(args.cd_beta),
        "negative_context_dropout": float(args.negative_context_dropout),
        "negative_context_noise_std": float(args.negative_context_noise_std),
        "negative_hidden_mode": args.negative_hidden_mode,
        "distance_bin_count": int(args.distance_bin_count),
        "analysis_target": "first_token_embedding_distance_in_negative_sample",
        "classification_target": "target_token_top1_behavior",
    }

    if not records:
        return {
            "meta": meta,
            "num_records": 0,
            "case_counts": {label: 0 for label in CASE_LABELS},
            "case_rates": {label: 0.0 for label in CASE_LABELS},
            "case_stats": {label: {} for label in CASE_LABELS},
            "focus_cases": {
                CASE_IMPROVED: {},
                CASE_WORSE_DROP: {},
                CASE_WORSE_NOT_PUSH: {},
            },
            "not_push_subtype_counts": {},
            "not_push_subtype_rates": {},
            "distance_bins": [],
            "global_correlations": {
                "corr_l2_vs_delta_target_logprob": 0.0,
                "corr_l2_vs_delta_target_rank": 0.0,
                "corr_cosine_dist_vs_delta_target_logprob": 0.0,
                "corr_cosine_dist_vs_delta_target_rank": 0.0,
            },
            "global_indicators": {
                "first_token_same_id_rate": 0.0,
                "target_in_candidate_mask_rate": 0.0,
                "accepted_by_final_rate": 0.0,
            },
            "confusion": {
                "positive_wrong_final_wrong": 0,
                "positive_wrong_final_correct": 0,
                "positive_correct_final_wrong": 0,
                "positive_correct_final_correct": 0,
            },
        }

    total = len(records)
    case_counts = {label: int(sum(r["case_type"] == label for r in records)) for label in CASE_LABELS}
    case_rates = {label: float(case_counts[label] / total) for label in CASE_LABELS}

    all_l2 = np.asarray([r["first_token_embedding_l2"] for r in records], dtype=np.float32)
    all_cos_dist = np.asarray([r["first_token_embedding_cosine_distance"] for r in records], dtype=np.float32)
    all_delta_lp = np.asarray([r["delta_target_logprob"] for r in records], dtype=np.float32)
    all_delta_rank = np.asarray([r["delta_target_rank"] for r in records], dtype=np.float32)
    all_same_first = np.asarray([r["first_token_same"] for r in records], dtype=np.float32)
    all_target_in_mask = np.asarray([r["target_in_candidate_mask"] for r in records], dtype=np.float32)
    all_accepted = np.asarray([r["accepted_by_final"] for r in records], dtype=np.float32)

    case_stats: dict[str, dict[str, Any]] = {}
    for label in CASE_LABELS:
        subset = [r for r in records if r["case_type"] == label]
        subset_l2 = np.asarray([r["first_token_embedding_l2"] for r in subset], dtype=np.float32)
        subset_cos_dist = np.asarray([r["first_token_embedding_cosine_distance"] for r in subset], dtype=np.float32)
        subset_delta_lp = np.asarray([r["delta_target_logprob"] for r in subset], dtype=np.float32)
        subset_delta_rank = np.asarray([r["delta_target_rank"] for r in subset], dtype=np.float32)
        subset_target_mask = np.asarray([r["target_in_candidate_mask"] for r in subset], dtype=np.float32)
        subset_accepted = np.asarray([r["accepted_by_final"] for r in subset], dtype=np.float32)

        stat = {
            "count": int(len(subset)),
            "rate": float(len(subset) / total),
            "first_token_embedding_l2": _describe_distribution(subset_l2),
            "first_token_embedding_cosine_distance": _describe_distribution(subset_cos_dist),
            "delta_target_logprob": _describe_distribution(subset_delta_lp),
            "delta_target_rank": _describe_distribution(subset_delta_rank),
            "target_in_candidate_mask_rate": float(subset_target_mask.mean()) if subset else 0.0,
            "accepted_by_final_rate": float(subset_accepted.mean()) if subset else 0.0,
        }

        if label == CASE_WORSE_NOT_PUSH:
            subtype_counts: dict[str, int] = {}
            for row in subset:
                subtype = row.get("not_push_subtype", "unknown")
                subtype_counts[subtype] = subtype_counts.get(subtype, 0) + 1
            stat["subtype_counts"] = subtype_counts
            stat["subtype_rates"] = {
                k: float(v / len(subset)) for k, v in subtype_counts.items()
            } if subset else {}

        case_stats[label] = stat

    not_push_subtype_counts = case_stats.get(CASE_WORSE_NOT_PUSH, {}).get("subtype_counts", {})
    not_push_total = max(case_counts[CASE_WORSE_NOT_PUSH], 1)
    not_push_subtype_rates = {
        key: float(val / not_push_total)
        for key, val in not_push_subtype_counts.items()
    }

    distance_bins = _compute_distance_bin_rows(records, args.distance_bin_count)

    positive_hit = np.asarray([r["positive_hit"] for r in records], dtype=np.int32)
    final_hit = np.asarray([r["final_hit"] for r in records], dtype=np.int32)

    summary = {
        "meta": meta,
        "num_records": int(total),
        "case_counts": case_counts,
        "case_rates": case_rates,
        "case_stats": case_stats,
        "focus_cases": {
            CASE_IMPROVED: case_stats.get(CASE_IMPROVED, {}),
            CASE_WORSE_DROP: case_stats.get(CASE_WORSE_DROP, {}),
            CASE_WORSE_NOT_PUSH: case_stats.get(CASE_WORSE_NOT_PUSH, {}),
        },
        "not_push_subtype_counts": not_push_subtype_counts,
        "not_push_subtype_rates": not_push_subtype_rates,
        "distance_bins": distance_bins,
        "global_correlations": {
            "corr_l2_vs_delta_target_logprob": _safe_corr(all_l2, all_delta_lp),
            "corr_l2_vs_delta_target_rank": _safe_corr(all_l2, all_delta_rank),
            "corr_cosine_dist_vs_delta_target_logprob": _safe_corr(all_cos_dist, all_delta_lp),
            "corr_cosine_dist_vs_delta_target_rank": _safe_corr(all_cos_dist, all_delta_rank),
        },
        "global_indicators": {
            "first_token_same_id_rate": float(all_same_first.mean()),
            "target_in_candidate_mask_rate": float(all_target_in_mask.mean()),
            "accepted_by_final_rate": float(all_accepted.mean()),
        },
        "confusion": {
            "positive_wrong_final_wrong": int(np.sum((positive_hit == 0) & (final_hit == 0))),
            "positive_wrong_final_correct": int(np.sum((positive_hit == 0) & (final_hit == 1))),
            "positive_correct_final_wrong": int(np.sum((positive_hit == 1) & (final_hit == 0))),
            "positive_correct_final_correct": int(np.sum((positive_hit == 1) & (final_hit == 1))),
        },
    }
    return summary


def _sanitize_md_text(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", "\\n")


def _format_case_distance_stats_row(summary: dict[str, Any], case: str) -> dict[str, Any]:
    case_stat = summary.get("focus_cases", {}).get(case, {})
    l2_stat = case_stat.get("first_token_embedding_l2", {})
    cos_stat = case_stat.get("first_token_embedding_cosine_distance", {})
    count = int(case_stat.get("count", 0))

    def _fmt(value: float) -> str:
        if count <= 0:
            return "n/a"
        return f"{value:.4f}"

    return {
        "case_name": CASE_DISPLAY_NAMES.get(case, case),
        "count": count,
        "rate": float(case_stat.get("rate", 0.0) * 100.0),
        "l2_mean": _fmt(float(l2_stat.get("mean", 0.0))),
        "l2_median": _fmt(float(l2_stat.get("median", 0.0))),
        "l2_p90": _fmt(float(l2_stat.get("p90", 0.0))),
        "cos_mean": _fmt(float(cos_stat.get("mean", 0.0))),
        "cos_median": _fmt(float(cos_stat.get("median", 0.0))),
        "cos_p90": _fmt(float(cos_stat.get("p90", 0.0))),
    }


def _write_requested_case_distance_table(lines: list[str], summary: dict[str, Any]) -> None:
    lines.append("## Requested 3-Case Embedding Distance Stats")
    lines.append(
        "| Case | Count | Rate | L2 mean | L2 median | L2 p90 | CosDist mean | CosDist median | CosDist p90 |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in [CASE_IMPROVED, CASE_WORSE_DROP, CASE_WORSE_NOT_PUSH]:
        row = _format_case_distance_stats_row(summary, case)
        lines.append(
            "| {case_name} | {count} | {rate:.2f}% | {l2_mean} | {l2_median} | {l2_p90} | {cos_mean} | {cos_median} | {cos_p90} |".format(
                **row
            )
        )
    lines.append("")


def _write_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# CDv2 First-Token Embedding Distance Analysis")
    lines.append("")

    if summary.get("num_records", 0) == 0:
        lines.append("No analysis record was captured.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Case Definitions")
    lines.append(f"- `{CASE_IMPROVED}`: positive miss, contrastive hits target top-1.")
    lines.append(f"- `{CASE_WORSE_DROP}`: positive hits target top-1, contrastive drops it.")
    lines.append(f"- `{CASE_WORSE_NOT_PUSH}`: both positive and contrastive do not put target at top-1.")
    lines.append(f"- `{CASE_KEEP_TOP1}`: both positive and contrastive keep target at top-1.")
    lines.append("")

    lines.append("## Overall")
    lines.append(f"- Total token-level records: **{summary['num_records']}**")
    lines.append(
        "- First-token same-id rate (negative token equals positive token): "
        f"**{summary['global_indicators']['first_token_same_id_rate'] * 100:.4f}%**"
    )
    lines.append(
        f"- Target in candidate-mask rate: **{summary['global_indicators']['target_in_candidate_mask_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Final accepted-prefix rate at position: **{summary['global_indicators']['accepted_by_final_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Corr(L2, delta_target_logprob): **{summary['global_correlations']['corr_l2_vs_delta_target_logprob']:+.4f}**"
    )
    lines.append(
        f"- Corr(L2, delta_target_rank): **{summary['global_correlations']['corr_l2_vs_delta_target_rank']:+.4f}**"
    )
    lines.append("")

    lines.append("## Focus Case Stats")
    lines.append(
        "| Case | Count | Rate | L2 mean | L2 median | L2 p90 | CosDist mean | Delta logprob mean | Delta rank mean |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for case in [CASE_IMPROVED, CASE_WORSE_DROP, CASE_WORSE_NOT_PUSH]:
        case_stat = summary["focus_cases"].get(case, {})
        l2_stat = case_stat.get("first_token_embedding_l2", {})
        cos_stat = case_stat.get("first_token_embedding_cosine_distance", {})
        dlp_stat = case_stat.get("delta_target_logprob", {})
        drk_stat = case_stat.get("delta_target_rank", {})
        lines.append(
            "| {name} | {count} | {rate:.2f}% | {l2_mean:.4f} | {l2_med:.4f} | {l2_p90:.4f} | {cos_mean:.4f} | {dlp_mean:+.4f} | {drk_mean:+.4f} |".format(
                name=CASE_DISPLAY_NAMES.get(case, case),
                count=case_stat.get("count", 0),
                rate=100.0 * case_stat.get("rate", 0.0),
                l2_mean=l2_stat.get("mean", 0.0),
                l2_med=l2_stat.get("median", 0.0),
                l2_p90=l2_stat.get("p90", 0.0),
                cos_mean=cos_stat.get("mean", 0.0),
                dlp_mean=dlp_stat.get("mean", 0.0),
                drk_mean=drk_stat.get("mean", 0.0),
            )
        )
    lines.append("")

    _write_requested_case_distance_table(lines, summary)

    lines.append("## Not-Push Subtypes")
    subtype_counts = summary.get("not_push_subtype_counts", {})
    subtype_rates = summary.get("not_push_subtype_rates", {})
    if subtype_counts:
        lines.append("| Subtype | Count | Rate within not-push |")
        lines.append("|---|---:|---:|")
        for key in sorted(subtype_counts.keys(), key=lambda x: subtype_counts[x], reverse=True):
            lines.append(
                f"| {key} | {subtype_counts[key]} | {subtype_rates.get(key, 0.0) * 100:.2f}% |"
            )
    else:
        lines.append("No not-push event.")
    lines.append("")

    lines.append("## Distance Bin Analysis")
    distance_bins = summary.get("distance_bins", [])
    if distance_bins:
        lines.append(
            "| Bin | L2 range | Count | Rate | Improve | Drop | Not-push | Keep-top1 | Mean d_logprob | Mean d_rank |"
        )
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for row in distance_bins:
            lines.append(
                "| {bin_index} | [{low:.4f}, {high:.4f}] | {count} | {rate:.2f}% | {improve:.2f}% | {drop:.2f}% | {not_push:.2f}% | {keep:.2f}% | {dlog:+.4f} | {drank:+.4f} |".format(
                    bin_index=row["bin_index"],
                    low=row["l2_lower"],
                    high=row["l2_upper"],
                    count=row["count"],
                    rate=row["rate"] * 100.0,
                    improve=row["improved_rate"] * 100.0,
                    drop=row["drop_from_top1_rate"] * 100.0,
                    not_push=row["not_push_to_top1_rate"] * 100.0,
                    keep=row["keep_top1_rate"] * 100.0,
                    dlog=row["mean_delta_target_logprob"],
                    drank=row["mean_delta_target_rank"],
                )
            )
    else:
        lines.append("Distance-bin analysis unavailable.")
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    def _top_rows(case: str, key: str, reverse: bool) -> list[dict[str, Any]]:
        subset = [r for r in records if r["case_type"] == case]
        subset = sorted(subset, key=lambda x: x[key], reverse=reverse)
        return subset[:max_rows]

    lines.append("## Top Improved Cases")
    lines.append(
        "| sample | turn | step | abs_pos | block_pos | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|")
    for rec in _top_rows(CASE_IMPROVED, "delta_target_logprob", True):
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {block_position} | {first_text} ({first_id}) -> {neg_text} ({neg_id}) | {l2:.4f} | {cd:.4f} | {target_text} ({target_id}) | {pos_text} ({pos_id}) | {final_text} ({final_id}) | {dlp:+.4f} | {dr:+d} | {mask} | {accepted} |".format(
                sample_idx=rec["sample_idx"],
                turn_idx=rec["turn_idx"],
                decode_step=rec["decode_step"],
                absolute_position=rec["absolute_position"],
                block_position=rec["block_position"],
                first_text=_sanitize_md_text(rec["first_token_text"]),
                first_id=rec["first_token_id"],
                neg_text=_sanitize_md_text(rec["negative_first_token_text"]),
                neg_id=rec["negative_first_token_id"],
                l2=rec["first_token_embedding_l2"],
                cd=rec["first_token_embedding_cosine_distance"],
                target_text=_sanitize_md_text(rec["target_token_text"]),
                target_id=rec["target_token_id"],
                pos_text=_sanitize_md_text(rec["positive_pred_text"]),
                pos_id=rec["positive_pred_id"],
                final_text=_sanitize_md_text(rec["final_pred_text"]),
                final_id=rec["final_pred_id"],
                dlp=rec["delta_target_logprob"],
                dr=rec["delta_target_rank"],
                mask=rec["target_in_candidate_mask"],
                accepted=rec["accepted_by_final"],
            )
        )
    lines.append("")

    lines.append("## Top Worsened Cases: Drop From Top-1")
    lines.append(
        "| sample | turn | step | abs_pos | block_pos | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|")
    for rec in _top_rows(CASE_WORSE_DROP, "delta_target_logprob", False):
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {block_position} | {first_text} ({first_id}) -> {neg_text} ({neg_id}) | {l2:.4f} | {cd:.4f} | {target_text} ({target_id}) | {pos_text} ({pos_id}) | {final_text} ({final_id}) | {dlp:+.4f} | {dr:+d} | {mask} | {accepted} |".format(
                sample_idx=rec["sample_idx"],
                turn_idx=rec["turn_idx"],
                decode_step=rec["decode_step"],
                absolute_position=rec["absolute_position"],
                block_position=rec["block_position"],
                first_text=_sanitize_md_text(rec["first_token_text"]),
                first_id=rec["first_token_id"],
                neg_text=_sanitize_md_text(rec["negative_first_token_text"]),
                neg_id=rec["negative_first_token_id"],
                l2=rec["first_token_embedding_l2"],
                cd=rec["first_token_embedding_cosine_distance"],
                target_text=_sanitize_md_text(rec["target_token_text"]),
                target_id=rec["target_token_id"],
                pos_text=_sanitize_md_text(rec["positive_pred_text"]),
                pos_id=rec["positive_pred_id"],
                final_text=_sanitize_md_text(rec["final_pred_text"]),
                final_id=rec["final_pred_id"],
                dlp=rec["delta_target_logprob"],
                dr=rec["delta_target_rank"],
                mask=rec["target_in_candidate_mask"],
                accepted=rec["accepted_by_final"],
            )
        )
    lines.append("")

    lines.append("## Top Worsened Cases: Not Push To Top-1")
    lines.append(
        "| sample | turn | step | abs_pos | block_pos | subtype | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |"
    )
    lines.append("|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---:|")
    not_push_rows = [r for r in records if r["case_type"] == CASE_WORSE_NOT_PUSH]
    not_push_rows = sorted(
        not_push_rows,
        key=lambda x: (x["delta_target_rank"], x["delta_target_logprob"]),
    )[:max_rows]
    for rec in not_push_rows:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {block_position} | {subtype} | {first_text} ({first_id}) -> {neg_text} ({neg_id}) | {l2:.4f} | {cd:.4f} | {target_text} ({target_id}) | {pos_text} ({pos_id}) | {final_text} ({final_id}) | {dlp:+.4f} | {dr:+d} | {mask} | {accepted} |".format(
                sample_idx=rec["sample_idx"],
                turn_idx=rec["turn_idx"],
                decode_step=rec["decode_step"],
                absolute_position=rec["absolute_position"],
                block_position=rec["block_position"],
                subtype=rec.get("not_push_subtype", "n/a"),
                first_text=_sanitize_md_text(rec["first_token_text"]),
                first_id=rec["first_token_id"],
                neg_text=_sanitize_md_text(rec["negative_first_token_text"]),
                neg_id=rec["negative_first_token_id"],
                l2=rec["first_token_embedding_l2"],
                cd=rec["first_token_embedding_cosine_distance"],
                target_text=_sanitize_md_text(rec["target_token_text"]),
                target_id=rec["target_token_id"],
                pos_text=_sanitize_md_text(rec["positive_pred_text"]),
                pos_id=rec["positive_pred_id"],
                final_text=_sanitize_md_text(rec["final_pred_text"]),
                final_id=rec["final_pred_id"],
                dlp=rec["delta_target_logprob"],
                dr=rec["delta_target_rank"],
                mask=rec["target_in_candidate_mask"],
                accepted=rec["accepted_by_final"],
            )
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def _write_insights_md(insights_path: Path, summary: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append(f"# Embed-Distance Insights for `{insights_path.parent.name}`")
    lines.append("")
    lines.append("## Requested Stats")
    _write_requested_case_distance_table(lines, summary)

    focus = summary.get("focus_cases", {})
    improved_count = int(focus.get(CASE_IMPROVED, {}).get("count", 0))
    dropped_count = int(focus.get(CASE_WORSE_DROP, {}).get("count", 0))
    not_push_count = int(focus.get(CASE_WORSE_NOT_PUSH, {}).get("count", 0))

    lines.append("## Quick Read")
    lines.append(f"- Improved cases (target pushed to top-1): {improved_count}")
    lines.append(f"- Worsened cases (target dropped from top-1): {dropped_count}")
    lines.append(f"- Worsened cases (target not pushed to top-1): {not_push_count}")
    lines.append("")

    insights_path.write_text("\n".join(lines), encoding="utf-8")


def _maybe_create_plots(
    report_dir: Path,
    records: list[dict[str, Any]],
    distance_bins: list[dict[str, Any]],
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

    plot_files: list[str] = []
    colors = {
        CASE_IMPROVED: "#2f855a",
        CASE_WORSE_DROP: "#c53030",
        CASE_WORSE_NOT_PUSH: "#dd6b20",
        CASE_KEEP_TOP1: "#2b6cb0",
    }

    plt.figure(figsize=(8.4, 4.8))
    for case in CASE_LABELS:
        vals = [r["first_token_embedding_l2"] for r in records if r["case_type"] == case]
        if vals:
            plt.hist(
                vals,
                bins=70,
                alpha=0.45,
                label=CASE_DISPLAY_NAMES.get(case, case),
                color=colors.get(case),
            )
    plt.xlabel("First-token embedding L2 distance")
    plt.ylabel("Count")
    plt.title("Embedding-Distance Distribution by Case")
    plt.legend()
    p1 = report_dir / "first_token_embed_l2_hist_by_case.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    plot_files.append(p1.name)

    labels = []
    data = []
    for case in CASE_LABELS:
        vals = [r["first_token_embedding_l2"] for r in records if r["case_type"] == case]
        if vals:
            labels.append(CASE_DISPLAY_NAMES.get(case, case))
            data.append(vals)
    if data:
        plt.figure(figsize=(9.0, 5.0))
        plt.boxplot(data, labels=labels, showfliers=False)
        plt.xticks(rotation=15, ha="right")
        plt.ylabel("First-token embedding L2 distance")
        plt.title("Embedding-Distance Boxplot by Case")
        p2 = report_dir / "first_token_embed_l2_boxplot_by_case.png"
        plt.tight_layout()
        plt.savefig(p2, dpi=160)
        plt.close()
        plot_files.append(p2.name)

    max_scatter_points = 15000
    scatter_records = records
    if len(records) > max_scatter_points:
        sample_ids = np.linspace(0, len(records) - 1, max_scatter_points, dtype=np.int64)
        scatter_records = [records[int(i)] for i in sample_ids.tolist()]

    plt.figure(figsize=(8.4, 5.2))
    for case in CASE_LABELS:
        xs = [r["first_token_embedding_l2"] for r in scatter_records if r["case_type"] == case]
        ys = [r["delta_target_logprob"] for r in scatter_records if r["case_type"] == case]
        if xs:
            plt.scatter(
                xs,
                ys,
                s=10,
                alpha=0.38,
                color=colors.get(case),
                label=CASE_DISPLAY_NAMES.get(case, case),
            )
    plt.axhline(0.0, linestyle="--", linewidth=1.0)
    plt.xlabel("First-token embedding L2 distance")
    plt.ylabel("Delta target logprob (contrastive - positive)")
    plt.title("Embedding Distance vs Target LogProb Change")
    plt.legend(markerscale=1.5)
    p3 = report_dir / "first_token_embed_l2_vs_delta_target_logprob.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=160)
    plt.close()
    plot_files.append(p3.name)

    if distance_bins:
        x = [row["bin_center_l2"] for row in distance_bins]
        improve = [row["improved_rate"] for row in distance_bins]
        drop = [row["drop_from_top1_rate"] for row in distance_bins]
        not_push = [row["not_push_to_top1_rate"] for row in distance_bins]
        keep = [row["keep_top1_rate"] for row in distance_bins]

        plt.figure(figsize=(8.4, 4.8))
        plt.plot(x, improve, marker="o", color=colors[CASE_IMPROVED], label=CASE_DISPLAY_NAMES[CASE_IMPROVED])
        plt.plot(x, drop, marker="o", color=colors[CASE_WORSE_DROP], label=CASE_DISPLAY_NAMES[CASE_WORSE_DROP])
        plt.plot(
            x,
            not_push,
            marker="o",
            color=colors[CASE_WORSE_NOT_PUSH],
            label=CASE_DISPLAY_NAMES[CASE_WORSE_NOT_PUSH],
        )
        plt.plot(x, keep, marker="o", color=colors[CASE_KEEP_TOP1], label=CASE_DISPLAY_NAMES[CASE_KEEP_TOP1])
        plt.ylim(0.0, 1.0)
        plt.xlabel("Binned L2 center")
        plt.ylabel("Rate")
        plt.title("Case Rates Over Embedding-Distance Bins")
        plt.legend()
        p4 = report_dir / "case_rates_over_embedding_distance_bins.png"
        plt.tight_layout()
        plt.savefig(p4, dpi=160)
        plt.close()
        plot_files.append(p4.name)

    return plot_files


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
    first_token_case_records: Optional[list[dict[str, Any]]] = None,
    tokenizer: Optional[AutoTokenizer] = None,
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
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature, gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

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
            positive_draft_logits, negative_draft_logits, step_payload = compute_contrastive_draft_logits(
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
                gen=gen,
            )
            past_key_values_draft.crop(start)

            if divergence_accumulator is not None:
                divergence_accumulator.append(_compute_kl_divergence(positive_draft_logits, negative_draft_logits))

            candidate_mask = build_cd_candidate_mask(reference_logits=positive_draft_logits, beta=beta)

            final_draft_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            final_draft_logits = apply_cd_candidate_filter(
                logits=final_draft_logits,
                candidate_mask=candidate_mask,
            )

            # Keep positive scores on top-vocab entries selected by contrastive logits.
            n_keep = min(6, final_draft_logits.size(-1))
            if n_keep > 0:
                top_indices = torch.topk(final_draft_logits, k=n_keep, dim=-1).indices
                positive_top_values = positive_draft_logits.gather(dim=-1, index=top_indices)
                final_draft_logits.scatter_(dim=-1, index=top_indices, src=positive_top_values)

            block_output_ids[:, 1:] = sample(final_draft_logits, gen=gen)
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

        posterior = sample(output.logits, temperature, gen=gen)
        acceptance_length = int(
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )

        if block_size > 1 and first_token_case_records is not None and tokenizer is not None:
            first_token_id = int(step_payload["positive_first_token_ids"][0].item())
            negative_first_token_id = int(step_payload["negative_first_token_ids"][0].item())
            first_token_l2 = float(step_payload["first_token_l2"][0].item())
            first_token_cos_sim = float(step_payload["first_token_cosine_similarity"][0].item())
            first_token_cos_dist = float(step_payload["first_token_cosine_distance"][0].item())
            first_token_same = int(first_token_id == negative_first_token_id)

            seq_len = int(positive_draft_logits.shape[1])
            for offset in range(seq_len):
                target_token_id = int(posterior[0, offset].item())

                positive_step_logits = positive_draft_logits[0, offset, :].float()
                negative_step_logits = negative_draft_logits[0, offset, :].float()
                final_step_logits = final_draft_logits[0, offset, :].float()

                positive_pred_id = int(torch.argmax(positive_step_logits).item())
                negative_pred_id = int(torch.argmax(negative_step_logits).item())
                final_pred_id = int(torch.argmax(final_step_logits).item())

                positive_hit = int(positive_pred_id == target_token_id)
                final_hit = int(final_pred_id == target_token_id)

                positive_logprob = torch.log_softmax(positive_step_logits, dim=-1)
                negative_logprob = torch.log_softmax(negative_step_logits, dim=-1)
                final_logprob = torch.log_softmax(final_step_logits, dim=-1)

                positive_target_logprob = float(positive_logprob[target_token_id].item())
                negative_target_logprob = float(negative_logprob[target_token_id].item())
                final_target_logprob = float(final_logprob[target_token_id].item())

                positive_target_rank = _token_rank_from_logits(positive_step_logits, target_token_id)
                negative_target_rank = _token_rank_from_logits(negative_step_logits, target_token_id)
                final_target_rank = _token_rank_from_logits(final_step_logits, target_token_id)

                delta_target_logprob = float(final_target_logprob - positive_target_logprob)
                delta_target_rank = int(positive_target_rank - final_target_rank)

                if positive_hit == 0 and final_hit == 1:
                    case_type = CASE_IMPROVED
                    case_reason = "contrastive pushes target to top-1"
                    not_push_subtype = "n/a"
                elif positive_hit == 1 and final_hit == 0:
                    case_type = CASE_WORSE_DROP
                    case_reason = "contrastive drops target from top-1"
                    not_push_subtype = "n/a"
                elif positive_hit == 0 and final_hit == 0:
                    case_type = CASE_WORSE_NOT_PUSH
                    case_reason = "target still not top-1 after contrastive"
                    if delta_target_rank > 0:
                        not_push_subtype = "rank_improved_but_not_top1"
                    elif delta_target_rank == 0:
                        not_push_subtype = "rank_unchanged_not_top1"
                    else:
                        not_push_subtype = "rank_worse_not_top1"
                else:
                    case_type = CASE_KEEP_TOP1
                    case_reason = "target remains top-1 in both branches"
                    not_push_subtype = "n/a"

                p_pos = torch.softmax(positive_step_logits, dim=-1)
                p_neg = torch.softmax(negative_step_logits, dim=-1)
                step_kl_pos_vs_neg = float(
                    (p_pos * (torch.log(p_pos + 1e-10) - torch.log(p_neg + 1e-10))).sum().item()
                )

                first_token_case_records.append(
                    {
                        "sample_idx": int(sample_idx),
                        "turn_idx": int(turn_idx),
                        "decode_step": int(decode_step),
                        "block_position": int(offset),
                        "absolute_position": int(start + offset + 1),
                        "first_token_id": int(first_token_id),
                        "first_token_text": _token_str(tokenizer, first_token_id),
                        "negative_first_token_id": int(negative_first_token_id),
                        "negative_first_token_text": _token_str(tokenizer, negative_first_token_id),
                        "first_token_same": int(first_token_same),
                        "first_token_embedding_l2": float(first_token_l2),
                        "first_token_embedding_cosine_similarity": float(first_token_cos_sim),
                        "first_token_embedding_cosine_distance": float(first_token_cos_dist),
                        "target_token_id": int(target_token_id),
                        "target_token_text": _token_str(tokenizer, target_token_id),
                        "positive_pred_id": int(positive_pred_id),
                        "positive_pred_text": _token_str(tokenizer, positive_pred_id),
                        "negative_pred_id": int(negative_pred_id),
                        "negative_pred_text": _token_str(tokenizer, negative_pred_id),
                        "final_pred_id": int(final_pred_id),
                        "final_pred_text": _token_str(tokenizer, final_pred_id),
                        "positive_hit": int(positive_hit),
                        "final_hit": int(final_hit),
                        "case_type": case_type,
                        "case_reason": case_reason,
                        "not_push_subtype": not_push_subtype,
                        "positive_target_logprob": float(positive_target_logprob),
                        "negative_target_logprob": float(negative_target_logprob),
                        "final_target_logprob": float(final_target_logprob),
                        "delta_target_logprob": float(delta_target_logprob),
                        "positive_target_rank": int(positive_target_rank),
                        "negative_target_rank": int(negative_target_rank),
                        "final_target_rank": int(final_target_rank),
                        "delta_target_rank": int(delta_target_rank),
                        "target_in_candidate_mask": int(candidate_mask[0, offset, target_token_id].item()),
                        "candidate_size": int(candidate_mask[0, offset, :].sum().item()),
                        "accepted_by_final": int(offset < acceptance_length),
                        "sampled_by_final_id": int(block_output_ids[0, offset + 1].item()),
                        "sampled_by_final_text": _token_str(tokenizer, int(block_output_ids[0, offset + 1].item())),
                        "step_kl_positive_vs_negative": float(step_kl_pos_vs_neg),
                    }
                )

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        decode_step += 1

        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

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
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--max-example-rows", type=int, default=30)
    parser.add_argument("--distance-bin-count", type=int, default=6)
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

    def has_flash_attn() -> bool:
        if not use_cuda:
            return False
        if args.deterministic:
            logger.info("Deterministic mode enabled. Forcing SDPA attention backend.")
            return False
        try:
            import flash_attn  # noqa: F401

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
    if block_size <= 1:
        raise ValueError("First-token embedding distance analysis requires block_size > 1.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = []
    first_token_case_records: list[dict[str, Any]] = []
    responses = []

    indices = range(dist.rank(), len(dataset), dist.size())
    base_seed = args.seed
    for idx in tqdm(indices, disable=not dist.is_main()):
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
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
                    divergence_accumulator=divergence_accumulator if bs == block_size else None,
                    first_token_case_records=first_token_case_records if bs == block_size else None,
                    tokenizer=tokenizer if bs == block_size else None,
                    sample_idx=idx,
                    turn_idx=turn_index,
                    seed=sample_seed + bs,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        first_token_case_records = dist.gather(first_token_case_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        first_token_case_records = list(chain(*first_token_case_records))

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

    if divergence_accumulator:
        avg_divergence = float(sum(divergence_accumulator) / len(divergence_accumulator))
        print(f"Average KL divergence between positive and negative draft logits: {avg_divergence:.5f}")
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

    report_dir = _build_report_dir(args.dataset, args.report_dir)
    summary = _build_case_embedding_summary(first_token_case_records, args, block_size)

    records_jsonl = report_dir / "first_token_case_records.jsonl"
    records_csv = report_dir / "first_token_case_records.csv"
    summary_json = report_dir / "first_token_embed_summary.json"
    report_md = report_dir / "first_token_embed_report.md"
    insights_md = report_dir / "insights.md"

    _write_jsonl(records_jsonl, first_token_case_records)
    _write_csv(records_csv, first_token_case_records)
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    plot_files = [] if args.no_plots else _maybe_create_plots(report_dir, first_token_case_records, summary.get("distance_bins", []))
    _write_report_md(
        report_path=report_md,
        summary=summary,
        records=first_token_case_records,
        plot_files=plot_files,
        max_rows=args.max_example_rows,
    )
    _write_insights_md(insights_md, summary)

    print(f"First-token analysis records: {summary.get('num_records', 0)}")
    if summary.get("num_records", 0) > 0:
        focus = summary.get("focus_cases", {})
        improved = focus.get(CASE_IMPROVED, {})
        dropped = focus.get(CASE_WORSE_DROP, {})
        not_push = focus.get(CASE_WORSE_NOT_PUSH, {})

        print(
            "Improved (target pushed to top-1): "
            f"{improved.get('count', 0)} ({improved.get('rate', 0.0) * 100:.2f}%)"
        )
        print(
            "Worsened (target dropped from top-1): "
            f"{dropped.get('count', 0)} ({dropped.get('rate', 0.0) * 100:.2f}%)"
        )
        print(
            "Worsened (target not pushed to top-1): "
            f"{not_push.get('count', 0)} ({not_push.get('rate', 0.0) * 100:.2f}%)"
        )

        imp_l2_mean = improved.get("first_token_embedding_l2", {}).get("mean", 0.0)
        drop_l2_mean = dropped.get("first_token_embedding_l2", {}).get("mean", 0.0)
        np_l2_mean = not_push.get("first_token_embedding_l2", {}).get("mean", 0.0)
        print(
            "Mean L2 distance (improved / drop / not-push): "
            f"{imp_l2_mean:.4f} / {drop_l2_mean:.4f} / {np_l2_mean:.4f}"
        )

    print(f"Report directory: {report_dir}")
    print(f"  - {records_jsonl.name}")
    print(f"  - {records_csv.name}")
    print(f"  - {summary_json.name}")
    print(f"  - {report_md.name}")
    print(f"  - {insights_md.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")


if __name__ == "__main__":
    main()