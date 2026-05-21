import argparse
import csv
import importlib.util
import json
import os
import random
import time
from collections import Counter
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

from analysis import token_level_core
from analysis import token_level_reporting
from model import DFlashDraftModel, apply_cd_logits, extract_context_feature, load_and_process_dataset, sample


NEGATIVE_HIDDEN_MODE = "mask_zero"
TOP64_MASK_MODE = False
FINAL_OVERRIDE_KEEP = 7
COMPARE_NAME = "CDv2"


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
    return token_level_core.build_negative_block_output_random(
        block_output_ids=block_output_ids,
        vocab_size=vocab_size,
        gen=gen,
    )


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
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    neg_block_output_ids = build_negative_block_output_random(
        block_output_ids=block_output_ids,
        vocab_size=target.config.vocab_size,
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
    return first_draft_logits, second_draft_logits


def build_cd_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    return token_level_core.build_candidate_mask(reference_logits=reference_logits, beta=beta)


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


def apply_cd_candidate_filter(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return token_level_core.apply_candidate_filter(logits=logits, candidate_mask=candidate_mask)


def _compute_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> float:
    return token_level_core.compute_kl_divergence(first_logits=first_logits, second_logits=second_logits)


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    return token_level_core.token_rank_from_logits(logits_1d=logits_1d, token_id=token_id)


def _token_str(tokenizer: AutoTokenizer, token_id: int) -> str:
    return token_level_core.token_str(tokenizer=tokenizer, token_id=token_id)


def _topk_token_details(
    logits_1d: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> list[dict[str, Any]]:
    return token_level_core.topk_token_details(logits_1d=logits_1d, tokenizer=tokenizer, k=k)


def _estimate_token_hit_probability(
    logits_1d: torch.Tensor,
    token_id: int,
    num_samples: int,
    gen: Optional[torch.Generator] = None,
) -> float:
    return token_level_core.estimate_token_hit_probability(
        logits_1d=logits_1d,
        token_id=token_id,
        num_samples=num_samples,
        gen=gen,
    )


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


def _build_reject_reason(record: dict[str, Any]) -> str:
    return token_level_reporting.build_reject_reason(record)


def _classify_reject_taxonomy(record: dict[str, Any]) -> tuple[str, str]:
    return token_level_reporting.classify_reject_taxonomy(record)


def _safe_dataset_name(name: str) -> str:
    return token_level_reporting.safe_dataset_name(name)


def _build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    return token_level_reporting.build_report_dir(dataset_name, report_dir)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    token_level_reporting.write_jsonl(path, records)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    token_level_reporting.write_csv(path, records)


def _maybe_create_plots(report_dir: Path, records: list[dict[str, Any]]) -> list[str]:
    return token_level_reporting.maybe_create_plots(report_dir, records, compare_name=COMPARE_NAME)


def _build_summary(records: list[dict[str, Any]], args: argparse.Namespace, block_size: int) -> dict[str, Any]:
    return token_level_reporting.build_summary(
        records,
        dataset=args.dataset,
        seed=args.seed,
        block_size=block_size,
        alpha=args.cd_alpha,
        beta=args.cd_beta,
        shadow_num_samples=args.cd_shadow_num_samples,
        negative_context_dropout=args.negative_context_dropout,
        negative_context_noise_std=args.negative_context_noise_std,
        compare_name=COMPARE_NAME,
    )


def _write_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
) -> None:
    token_level_reporting.write_report_md(
        report_path=report_path,
        summary=summary,
        records=records,
        plot_files=plot_files,
        max_rows=max_rows,
        compare_name=COMPARE_NAME,
    )


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


def _build_ar_target_rank_record(
    *,
    sample_idx: int,
    turn_idx: int,
    decode_step: int,
    block_start_position: int,
    block_position: int,
    acceptance_length: int,
    token_source: str,
    target_token_id: int,
    committed_token_id: int,
    positive_step_logits: torch.Tensor,
    tokenizer: AutoTokenizer,
) -> dict[str, Any]:
    target_stats = _compute_target_token_stats(positive_step_logits, target_token_id)
    # Also compute rank of the committed token for comparison
    committed_stats = _compute_target_token_stats(positive_step_logits, committed_token_id)
    return {
        "sample_idx": int(sample_idx),
        "turn_idx": int(turn_idx),
        "decode_step": int(decode_step),
        "block_start_position": int(block_start_position),
        "absolute_position": int(block_start_position + block_position),
        "block_position": int(block_position),
        "acceptance_length": int(acceptance_length),
        "token_source": token_source,
        "target_token_id": int(target_token_id),
        "target_token_text": _token_str(tokenizer, target_token_id),
        "positive_target_rank": int(target_stats["rank"]),
        "positive_target_logprob": float(target_stats["logprob"]),
        "positive_top1_hit": int(target_stats["top1_hit"]),
        "positive_top5_hit": int(target_stats["top5_hit"]),
        "positive_top10_hit": int(target_stats["top10_hit"]),
        "committed_token_id": int(committed_token_id),
        "committed_token_text": _token_str(tokenizer, committed_token_id),
        "committed_rank": int(committed_stats["rank"]),
        "committed_logprob": float(committed_stats["logprob"]),
        "committed_top1_hit": int(committed_stats["top1_hit"]),
        "committed_top5_hit": int(committed_stats["top5_hit"]),
        "committed_top10_hit": int(committed_stats["top10_hit"]),
        "is_after_acceptance_length": int(block_position > acceptance_length),
    }


def _build_ar_target_rank_summary(
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
    top64_mask_mode: bool,
    final_override_keep: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        summary = {
            "meta": {
                "dataset": dataset,
                "seed": int(seed),
                "block_size": int(block_size),
                "target_token_definition": "ar_token_after_acceptance_length",
                "cd_alpha": float(cd_alpha),
                "cd_beta": float(cd_beta),
                "negative_context_dropout": float(negative_context_dropout),
                "negative_context_noise_std": float(negative_context_noise_std),
                "negative_hidden_mode": negative_hidden_mode,
                "top64_mask_mode": bool(top64_mask_mode),
                "final_override_keep": int(final_override_keep),
            },
            "num_records": 0,
            "num_blocks": 0,
            "token_source_counts": {},
            "token_source_rates": {},
            "acceptance_length_distribution": _describe_distribution(np.asarray([], dtype=np.float64)),
            "rank_distribution": _describe_distribution(np.asarray([], dtype=np.float64)),
            "logprob_distribution": _describe_distribution(np.asarray([], dtype=np.float64)),
            "topk_hit_rates": {"top1": 0.0, "top5": 0.0, "top10": 0.0},
            "committed_rank_distribution": _describe_distribution(np.asarray([], dtype=np.float64)),
            "committed_topk_hit_rates": {"top1": 0.0, "top5": 0.0, "top10": 0.0},
            "block_position_profile": [],
        }
        return summary, []

    block_acceptance_lengths: dict[tuple[int, int, int, int], int] = {}
    for row in records:
        block_key = (
            int(row["sample_idx"]),
            int(row["turn_idx"]),
            int(row["decode_step"]),
            int(row["block_start_position"]),
        )
        block_acceptance_lengths.setdefault(block_key, int(row["acceptance_length"]))

    acceptance_lengths = np.asarray(list(block_acceptance_lengths.values()), dtype=np.float64)
    rank_values = np.asarray([row["positive_target_rank"] for row in records], dtype=np.float64)
    logprob_values = np.asarray([row["positive_target_logprob"] for row in records], dtype=np.float64)
    top1_values = np.asarray([row["positive_top1_hit"] for row in records], dtype=np.float64)
    top5_values = np.asarray([row["positive_top5_hit"] for row in records], dtype=np.float64)
    top10_values = np.asarray([row["positive_top10_hit"] for row in records], dtype=np.float64)
    # Committed token stats (for AR suffix, committed != target due to sampling)
    committed_rank_values = np.asarray([row["committed_rank"] for row in records], dtype=np.float64)
    committed_logprob_values = np.asarray([row["committed_logprob"] for row in records], dtype=np.float64)
    committed_top1_values = np.asarray([row["committed_top1_hit"] for row in records], dtype=np.float64)
    committed_top5_values = np.asarray([row["committed_top5_hit"] for row in records], dtype=np.float64)
    committed_top10_values = np.asarray([row["committed_top10_hit"] for row in records], dtype=np.float64)
    token_source_counts = Counter(row.get("token_source", "unknown") for row in records)

    position_rows: list[dict[str, Any]] = []
    for block_position in sorted({int(row["block_position"]) for row in records}):
        subset = [row for row in records if int(row["block_position"]) == block_position]
        source_counts = Counter(row.get("token_source", "unknown") for row in subset)
        ranks = np.asarray([row["positive_target_rank"] for row in subset], dtype=np.float64)
        logprobs = np.asarray([row["positive_target_logprob"] for row in subset], dtype=np.float64)
        top1 = np.asarray([row["positive_top1_hit"] for row in subset], dtype=np.float64)
        top5 = np.asarray([row["positive_top5_hit"] for row in subset], dtype=np.float64)
        top10 = np.asarray([row["positive_top10_hit"] for row in subset], dtype=np.float64)
        committed_ranks = np.asarray([row["committed_rank"] for row in subset], dtype=np.float64)
        committed_top1 = np.asarray([row["committed_top1_hit"] for row in subset], dtype=np.float64)

        count = max(1, len(subset))
        position_rows.append(
            {
                "block_position": int(block_position),
                "count": int(len(subset)),
                "verify_count": int(source_counts.get("verify", 0)),
                "ar_suffix_count": int(source_counts.get("ar_suffix", 0)),
                "verify_rate": float(source_counts.get("verify", 0) / count),
                "ar_suffix_rate": float(source_counts.get("ar_suffix", 0) / count),
                "rank_count": int(ranks.size),
                "rank_mean": float(ranks.mean()) if ranks.size else 0.0,
                "rank_std": float(ranks.std()) if ranks.size else 0.0,
                "rank_min": float(ranks.min()) if ranks.size else 0.0,
                "rank_p10": float(np.quantile(ranks, 0.10)) if ranks.size else 0.0,
                "rank_median": float(np.quantile(ranks, 0.50)) if ranks.size else 0.0,
                "rank_p90": float(np.quantile(ranks, 0.90)) if ranks.size else 0.0,
                "rank_max": float(ranks.max()) if ranks.size else 0.0,
                "logprob_mean": float(logprobs.mean()) if logprobs.size else 0.0,
                "logprob_std": float(logprobs.std()) if logprobs.size else 0.0,
                "logprob_min": float(logprobs.min()) if logprobs.size else 0.0,
                "logprob_p10": float(np.quantile(logprobs, 0.10)) if logprobs.size else 0.0,
                "logprob_median": float(np.quantile(logprobs, 0.50)) if logprobs.size else 0.0,
                "logprob_p90": float(np.quantile(logprobs, 0.90)) if logprobs.size else 0.0,
                "logprob_max": float(logprobs.max()) if logprobs.size else 0.0,
                "top1_rate": float(top1.mean()) if top1.size else 0.0,
                "top5_rate": float(top5.mean()) if top5.size else 0.0,
                "top10_rate": float(top10.mean()) if top10.size else 0.0,
                "committed_rank_mean": float(committed_ranks.mean()) if committed_ranks.size else 0.0,
                "committed_rank_median": float(np.quantile(committed_ranks, 0.50)) if committed_ranks.size else 0.0,
                "committed_top1_rate": float(committed_top1.mean()) if committed_top1.size else 0.0,
            }
        )

    summary = {
        "meta": {
            "dataset": dataset,
            "seed": int(seed),
            "block_size": int(block_size),
            "target_token_definition": "ar_token_after_acceptance_length",
            "cd_alpha": float(cd_alpha),
            "cd_beta": float(cd_beta),
            "negative_context_dropout": float(negative_context_dropout),
            "negative_context_noise_std": float(negative_context_noise_std),
            "negative_hidden_mode": negative_hidden_mode,
            "top64_mask_mode": bool(top64_mask_mode),
            "final_override_keep": int(final_override_keep),
        },
        "num_records": int(len(records)),
        "num_blocks": int(len(block_acceptance_lengths)),
        "token_source_counts": {str(k): int(v) for k, v in token_source_counts.items()},
        "token_source_rates": {
            str(k): float(v / max(1, len(records))) for k, v in token_source_counts.items()
        },
        "acceptance_length_distribution": _describe_distribution(acceptance_lengths),
        "rank_distribution": _describe_distribution(rank_values),
        "logprob_distribution": _describe_distribution(logprob_values),
        "topk_hit_rates": {
            "top1": float(top1_values.mean()) if top1_values.size else 0.0,
            "top5": float(top5_values.mean()) if top5_values.size else 0.0,
            "top10": float(top10_values.mean()) if top10_values.size else 0.0,
        },
        "committed_rank_distribution": _describe_distribution(committed_rank_values),
        "committed_topk_hit_rates": {
            "top1": float(committed_top1_values.mean()) if committed_top1_values.size else 0.0,
            "top5": float(committed_top5_values.mean()) if committed_top5_values.size else 0.0,
            "top10": float(committed_top10_values.mean()) if committed_top10_values.size else 0.0,
        },
        "block_position_profile": position_rows,
    }

    return summary, position_rows


def _write_ar_target_rank_report_md(
    report_path: Path,
    summary: dict[str, Any],
    max_rows: int,
) -> None:
    lines: list[str] = []
    lines.append("# AR Target Token Rank Survey")
    lines.append("")

    meta = summary.get("meta", {})
    lines.append("## Setup")
    lines.append(f"- Dataset: **{meta.get('dataset', '')}**")
    lines.append(f"- Seed: **{meta.get('seed', 0)}**")
    lines.append(f"- Block size: **{meta.get('block_size', 0)}**")
    lines.append(f"- Target token definition: **{meta.get('target_token_definition', '')}**")
    lines.append(f"- CD alpha / beta: **{meta.get('cd_alpha', 0.0):.4f} / {meta.get('cd_beta', 0.0):.4f}**")
    lines.append(
        f"- Negative context dropout / noise: **{meta.get('negative_context_dropout', 0.0):.2f} / {meta.get('negative_context_noise_std', 0.0):.2f}**"
    )
    lines.append(f"- Negative hidden mode: **{meta.get('negative_hidden_mode', '')}**")
    lines.append("")

    lines.append("## Overall")
    lines.append(f"- Records: **{summary.get('num_records', 0)}**")
    lines.append(f"- Blocks: **{summary.get('num_blocks', 0)}**")
    lines.append(f"- Acceptance length mean: **{summary.get('acceptance_length_distribution', {}).get('mean', 0.0):.2f}**")
    lines.append("")
    lines.append("### Rerun Target (argmax) Stats")
    lines.append(f"- Rank mean: **{summary.get('rank_distribution', {}).get('mean', 0.0):.2f}**")
    lines.append(f"- Rank median: **{summary.get('rank_distribution', {}).get('median', 0.0):.2f}**")
    lines.append(
        f"- Top-1 / Top-5 / Top-10: **{summary.get('topk_hit_rates', {}).get('top1', 0.0) * 100:.2f}% / {summary.get('topk_hit_rates', {}).get('top5', 0.0) * 100:.2f}% / {summary.get('topk_hit_rates', {}).get('top10', 0.0) * 100:.2f}%**"
    )
    lines.append("")
    lines.append("### Committed Token (sampled) Stats")
    committed_rank = summary.get('committed_rank_distribution', {})
    committed_topk = summary.get('committed_topk_hit_rates', {})
    lines.append(f"- Rank mean: **{committed_rank.get('mean', 0.0):.2f}**")
    lines.append(f"- Rank median: **{committed_rank.get('median', 0.0):.2f}**")
    lines.append(
        f"- Top-1 / Top-5 / Top-10: **{committed_topk.get('top1', 0.0) * 100:.2f}% / {committed_topk.get('top5', 0.0) * 100:.2f}% / {committed_topk.get('top10', 0.0) * 100:.2f}%**"
    )
    lines.append("")
    source_counts = summary.get("token_source_counts", {})
    lines.append(
        f"- Token source counts: verify **{source_counts.get('verify', 0)}**, ar_suffix **{source_counts.get('ar_suffix', 0)}**"
    )
    lines.append("")

    lines.append("## Per-Position Summary")
    lines.append("| pos | count | verify | ar_suffix | verify_rate | rank_mean | rank_median | logprob_mean | top1 | top5 | top10 | comm_rank | comm_top1 |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    rows = summary.get("block_position_profile", [])
    for row in rows[:max_rows]:
        lines.append(
            "| "
            f"{row.get('block_position', 0)} | "
            f"{row.get('count', 0)} | "
            f"{row.get('verify_count', 0)} | "
            f"{row.get('ar_suffix_count', 0)} | "
            f"{row.get('verify_rate', 0.0) * 100:.2f}% | "
            f"{row.get('rank_mean', 0.0):.2f} | "
            f"{row.get('rank_median', 0.0):.2f} | "
            f"{row.get('logprob_mean', 0.0):.4f} | "
            f"{row.get('top1_rate', 0.0) * 100:.2f}% | "
            f"{row.get('top5_rate', 0.0) * 100:.2f}% | "
            f"{row.get('top10_rate', 0.0) * 100:.2f}% | "
            f"{row.get('committed_rank_mean', 0.0):.2f} | "
            f"{row.get('committed_top1_rate', 0.0) * 100:.2f}% |"
        )

    report_path.write_text("\n".join(lines), encoding="utf-8")


def print_ar_target_rank_console_summary(summary: dict[str, Any]) -> None:
    source_counts = summary.get("token_source_counts", {})
    print(f"AR target-rank records: {summary.get('num_records', 0)}")
    print(f"AR target-rank blocks: {summary.get('num_blocks', 0)}")
    print(
        f"Token source counts: verify {source_counts.get('verify', 0)}, ar_suffix {source_counts.get('ar_suffix', 0)}"
    )
    print(f"Acceptance length mean: {summary.get('acceptance_length_distribution', {}).get('mean', 0.0):.2f}")
    print(f"Positive rank mean: {summary.get('rank_distribution', {}).get('mean', 0.0):.2f}")


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
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
    reject_records: Optional[list[dict[str, Any]]] = None,
    ar_rank_records: Optional[list[dict[str, Any]]] = None,
    block_ar_records: Optional[list[dict[str, Any]]] = None,
    cd_shadow_num_samples: int = 1,
    sample_idx: int = -1,
    turn_idx: int = -1,
    seed: int = 0,
) -> SimpleNamespace:
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=model.device)
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
                gen=gen,
            )
            past_key_values_draft.crop(start)

            if divergence_accumulator is not None:
                divergence_accumulator.append(_compute_kl_divergence(positive_draft_logits, negative_draft_logits))

            if not TOP64_MASK_MODE:
                candidate_mask = build_cd_candidate_mask(reference_logits=positive_draft_logits, beta=beta)
            else:
                candidate_mask = build_topk_probability_mask(reference_logits=positive_draft_logits, top_k=64)

            final_draft_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            final_draft_logits = apply_cd_candidate_filter(logits=final_draft_logits, candidate_mask=candidate_mask)

            # Match CDv2 rollout behavior by overriding early positions with positive logits.
            """ n_keep = min(FINAL_OVERRIDE_KEEP, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :] """

            block_output_ids[:, 1:] = sample(final_draft_logits, gen=gen)
            cd_shadow_sampled_tokens = sample(final_draft_logits, gen=gen)
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
        acceptance_length = int((block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())

        # ── Accept/reject + commit (CD_v2) ──
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        new_start = start + acceptance_length + 1
        acceptance_lengths.append(acceptance_length + 1)
        decode_step += 1

        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        # ── AR rerun prediction (for analysis only) ──
        # rerun_target_ids: target AR argmax at positions acceptance_length+1..block_size-1
        # Khong anh huong output_ids -- chi de so sanh voi verify cua block tiep theo
        rerun_target_ids = block_output_ids.clone()
        if block_size > 1 and acceptance_length < (block_size - 1):
            past_key_values_target.crop(start + acceptance_length + 1)
            prev_token = block_output_ids[:, acceptance_length : acceptance_length + 1]
            for block_position in range(acceptance_length + 1, block_size):
                ar_output = target(
                    prev_token,
                    position_ids=position_ids[:, start + block_position - 1 : start + block_position],
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=False,
                )
                rerun_target_ids[:, block_position] = ar_output.logits[0, -1, :].argmax(dim=-1)
                next_token = sample(ar_output.logits, temperature, gen=gen)
                prev_token = next_token

        # ── Save per-block AR data with absolute positions ──
        if block_ar_records is not None and block_size > 1:
            ar_predictions = {}
            if acceptance_length < (block_size - 1):
                for bp in range(acceptance_length + 1, block_size):
                    abs_pos = int(start + bp)
                    ar_predictions[abs_pos] = int(rerun_target_ids[0, bp].item())
            block_ar_records.append({
                "sample_idx": sample_idx,
                "turn_idx": turn_idx,
                "decode_step": decode_step,
                "block_start": int(start),
                "acceptance_length": acceptance_length,
                "bonus_token": int(posterior[0, acceptance_length].item()),
                "ar_predictions": ar_predictions,
            })

        # ── Analysis records ──
        if ar_rank_records is not None and block_size > 1:
            stop_token_set = set(stop_token_ids or [])
            max_record_position = block_size - 1
            if stop_token_set:
                for block_position in range(1, block_size):
                    token_id = int(block_output_ids[0, block_position].item())
                    if token_id in stop_token_set:
                        max_record_position = block_position
                        break

            for block_position in range(1, max_record_position + 1):
                token_source = "verify" if block_position <= acceptance_length else "ar_suffix"
                committed_token_id = int(block_output_ids[0, block_position].item())
                target_token_id = int(rerun_target_ids[0, block_position].item())
                positive_step_logits = positive_draft_logits[0, block_position - 1, :].float()
                ar_rank_records.append(
                    _build_ar_target_rank_record(
                        sample_idx=sample_idx,
                        turn_idx=turn_idx,
                        decode_step=decode_step,
                        block_start_position=start,
                        block_position=block_position,
                        acceptance_length=acceptance_length,
                        token_source=token_source,
                        target_token_id=target_token_id,
                        committed_token_id=committed_token_id,
                        positive_step_logits=positive_step_logits,
                        tokenizer=tokenizer,
                    )
                )

        if block_size > 1 and reject_records is not None and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            target_token_id = int(posterior[0, reject_offset].item())
            sampled_draft_id = int(block_output_ids[0, reject_offset + 1].item())
            cd_shadow_sampled_id = int(cd_shadow_sampled_tokens[0, reject_offset].item())

            positive_step_logits = positive_draft_logits[0, reject_offset, :].float()
            cd_step_logits = final_draft_logits[0, reject_offset, :].float()
            posterior_step_logits = output.logits[0, reject_offset, :].float()

            reject_records.append(
                token_level_reporting.build_reject_record(
                    tokenizer=tokenizer,
                    sample_idx=sample_idx,
                    turn_idx=turn_idx,
                    decode_step=decode_step,
                    start=start,
                    reject_offset=reject_offset,
                    target_token_id=target_token_id,
                    sampled_draft_id=sampled_draft_id,
                    shadow_sampled_id=cd_shadow_sampled_id,
                    positive_step_logits=positive_step_logits,
                    contrastive_step_logits=cd_step_logits,
                    posterior_step_logits=posterior_step_logits,
                    candidate_mask_step=candidate_mask[0, reject_offset, :],
                    shadow_num_samples=cd_shadow_num_samples,
                    gen=gen,
                )
            )

        past_key_values_target.crop(new_start)
        start = new_start

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
    global NEGATIVE_HIDDEN_MODE, TOP64_MASK_MODE

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", "--vcd-alpha", dest="cd_alpha", type=float, default=0.5)
    parser.add_argument("--cd-beta", "--vcd-beta", dest="cd_beta", type=float, default=0.0)
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
        "--top64-mask-mode",
        action=argparse.BooleanOptionalAction,
        default=TOP64_MASK_MODE,
        help="Use top-64 probability candidate mask instead of beta-threshold mask.",
    )
    parser.add_argument(
        "--cd-shadow-num-samples",
        "--vcd-shadow-num-samples",
        dest="cd_shadow_num_samples",
        type=int,
        default=1,
        help="Number of Monte Carlo samples per reject position to estimate rescue probability.",
    )
    parser.add_argument("--max-example-rows", type=int, default=30)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    args = parser.parse_args()

    NEGATIVE_HIDDEN_MODE = args.negative_hidden_mode
    TOP64_MASK_MODE = args.top64_mask_mode
    print(f"Using negative hidden mode: [bold magenta]{args.negative_hidden_mode}[/bold magenta]")
    print(f"Top64 mask mode: [bold cyan]{TOP64_MASK_MODE}[/bold cyan]")

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn() -> bool:
        if args.deterministic:
            logger.info("Deterministic mode enabled. Forcing SDPA attention backend.")
            return False
        try:
            found = importlib.util.find_spec("flash_attn") is not None
            if found:
                return True
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False
        except Exception:
            logger.warning("Failed to check flash_attn. Falling back to torch.sdpa.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size
    if block_size <= 1:
        raise ValueError("Reject-token analysis requires block_size > 1.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = []
    reject_records: list[dict[str, Any]] = []
    ar_rank_records: list[dict[str, Any]] = []
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
                    tokenizer=tokenizer,
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
                    reject_records=reject_records if bs == block_size else None,
                    ar_rank_records=ar_rank_records if bs == block_size else None,
                    cd_shadow_num_samples=args.cd_shadow_num_samples,
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
        reject_records = dist.gather(reject_records, dst=0)
        ar_rank_records = dist.gather(ar_rank_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        reject_records = list(chain(*reject_records))
        ar_rank_records = list(chain(*ar_rank_records))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

    report_dir = _build_report_dir(args.dataset, args.report_dir)
    summary = _build_summary(reject_records, args, block_size)
    summary_meta = summary.setdefault("meta", {})
    summary_meta["negative_hidden_mode"] = args.negative_hidden_mode
    summary_meta["top64_mask_mode"] = bool(TOP64_MASK_MODE)
    summary_meta["final_override_keep"] = int(FINAL_OVERRIDE_KEEP)

    jsonl_path = report_dir / "reject_records.jsonl"
    csv_path = report_dir / "reject_records.csv"
    summary_path = report_dir / "summary.json"
    report_md_path = report_dir / "report.md"

    _write_jsonl(jsonl_path, reject_records)
    _write_csv(csv_path, reject_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_files = [] if args.no_plots else _maybe_create_plots(report_dir, reject_records)
    _write_report_md(
        report_path=report_md_path,
        summary=summary,
        records=reject_records,
        plot_files=plot_files,
        max_rows=args.max_example_rows,
    )

    print(f"Reject events recorded: {summary.get('reject_events', 0)}")
    if summary.get("reject_events", 0) > 0:
        print(f"Positive hit rate @reject: {summary['positive_hit_rate'] * 100:.2f}%")
        print(f"{COMPARE_NAME} hit rate @reject: {summary['vcd_hit_rate'] * 100:.2f}%")
        print(f"Hit-rate gain ({COMPARE_NAME} - Positive): {summary['hit_rate_gain'] * 100:+.2f}%")
        print(f"{COMPARE_NAME} shadow fix rate @reject: {summary['vcd_shadow_fix_rate'] * 100:.2f}%")
        print(
            f"Mean rescue-prob (Positive / {COMPARE_NAME}-shadow): "
            f"{summary['positive_rescue_prob_est']['mean']:.4f} / {summary['vcd_shadow_rescue_prob_est']['mean']:.4f}"
        )
        print(
            f"Mean rescue-prob gain ({COMPARE_NAME} - Positive): "
            f"{summary['rescue_prob_gain_vcd_minus_positive']['mean']:+.4f}"
        )
    print(f"Report directory: {report_dir}")
    print(f"  - {jsonl_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {report_md_path.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")

    ar_summary, ar_by_position_rows = _build_ar_target_rank_summary(
        ar_rank_records,
        dataset=args.dataset,
        seed=args.seed,
        block_size=block_size,
        cd_alpha=args.cd_alpha,
        cd_beta=args.cd_beta,
        negative_context_dropout=args.negative_context_dropout,
        negative_context_noise_std=args.negative_context_noise_std,
        negative_hidden_mode=args.negative_hidden_mode,
        top64_mask_mode=bool(TOP64_MASK_MODE),
        final_override_keep=int(FINAL_OVERRIDE_KEEP),
    )

    ar_jsonl_path = report_dir / "ar_target_rank_records.jsonl"
    ar_csv_path = report_dir / "ar_target_rank_records.csv"
    ar_summary_path = report_dir / "ar_target_rank_summary.json"
    ar_by_position_path = report_dir / "ar_target_rank_by_position.csv"
    ar_report_md_path = report_dir / "ar_target_rank_report.md"

    _write_jsonl(ar_jsonl_path, ar_rank_records)
    _write_csv(ar_csv_path, ar_rank_records)
    ar_summary_path.write_text(json.dumps(ar_summary, indent=2), encoding="utf-8")
    _write_csv(ar_by_position_path, ar_by_position_rows)
    _write_ar_target_rank_report_md(
        ar_report_md_path,
        summary=ar_summary,
        max_rows=args.max_example_rows,
    )

    print_ar_target_rank_console_summary(ar_summary)
    print(f"AR target-rank artifacts: {report_dir}")
    print(f"  - {ar_jsonl_path.name}")
    print(f"  - {ar_csv_path.name}")
    print(f"  - {ar_summary_path.name}")
    print(f"  - {ar_by_position_path.name}")
    print(f"  - {ar_report_md_path.name}")


if __name__ == "__main__":
    main()