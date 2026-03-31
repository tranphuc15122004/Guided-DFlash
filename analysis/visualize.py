import argparse
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
from model import DFlashDraftModel, apply_vcd_logits, sample, load_and_process_dataset, extract_context_feature
import distributed as dist


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

# construct negative sample by randomly replacing the batch's first token (target model generated token) 
def build_negative_block_output_random(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(0, vocab_size, (batch_size,), device=block_output_ids.device, generator=gen)
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids

def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()

    if dropout_ratio > 0.0:
        token_keep_mask = (
            torch.rand(
                negative_target_hidden.shape[:2],
                device=negative_target_hidden.device,
                generator=gen,
            ) >= dropout_ratio
        ).unsqueeze(-1)
        negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)

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
    gen: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    
    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size, gen=gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
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


def build_vcd_candidate_mask(
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)

    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def apply_vcd_candidate_filter(
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


def _compute_match_prod_curve(
    branch_tokens: torch.Tensor,
    posterior_tokens: torch.Tensor,
) -> tuple[list[int], int]:
    if branch_tokens.numel() == 0:
        return [], 0
    match = (branch_tokens == posterior_tokens).to(torch.int32)
    prod_curve = match.cumprod(dim=1)
    accept_len = int(prod_curve.sum(dim=1)[0].item())
    return [int(x) for x in prod_curve[0].tolist()], accept_len


def _require_matplotlib():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for visualization/report artifacts. "
            "Please install it (e.g., `pip install matplotlib`) and rerun."
        ) from exc


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _mean_prod_curve(step_records: list[dict[str, Any]], key: str, curve_len: int) -> list[float]:
    if curve_len <= 0:
        return []
    filtered = [r[key] for r in step_records if len(r.get(key, [])) == curve_len]
    if not filtered:
        return [0.0 for _ in range(curve_len)]
    arr = np.asarray(filtered, dtype=np.float32)
    return [float(x) for x in arr.mean(axis=0).tolist()]


def _summarize_prod_records(
    *,
    step_records: list[dict[str, Any]],
    total_step_records: int,
    block_size: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    curve_len = max(block_size - 1, 0)
    if not step_records:
        return {
            "meta": {
                "dataset": args.dataset,
                "seed": int(args.seed),
                "block_size": int(block_size),
                "vcd_alpha": float(args.vcd_alpha),
                "vcd_beta": float(args.vcd_beta),
                "negative_context_dropout": float(args.negative_context_dropout),
                "negative_context_noise_std": float(args.negative_context_noise_std),
                "max_plot_steps": int(args.max_plot_steps),
            },
            "num_step_records_total": int(total_step_records),
            "num_step_records_used": 0,
            "curve_length": int(curve_len),
            "acceptance": {
                "mean_positive_accept_len": 0.0,
                "mean_negative_accept_len": 0.0,
                "mean_final_accept_len": 0.0,
            },
            "fractions": {
                "positive_gt_negative": 0.0,
                "final_not_better_than_positive": 0.0,
            },
            "kl": {
                "avg": 0.0,
                "min": 0.0,
                "max": 0.0,
            },
            "mean_prod_curve": {
                "positive": [0.0 for _ in range(curve_len)],
                "negative": [0.0 for _ in range(curve_len)],
                "final": [0.0 for _ in range(curve_len)],
            },
        }

    positive_accept = [int(r["positive_accept_len"]) for r in step_records]
    negative_accept = [int(r["negative_accept_len"]) for r in step_records]
    final_accept = [int(r["final_accept_len"]) for r in step_records]
    kl_values = [float(r["kl_divergence"]) for r in step_records]

    positive_gt_negative = [1.0 if p > n else 0.0 for p, n in zip(positive_accept, negative_accept)]
    final_not_better_than_positive = [1.0 if f <= p else 0.0 for f, p in zip(final_accept, positive_accept)]

    return {
        "meta": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "block_size": int(block_size),
            "vcd_alpha": float(args.vcd_alpha),
            "vcd_beta": float(args.vcd_beta),
            "negative_context_dropout": float(args.negative_context_dropout),
            "negative_context_noise_std": float(args.negative_context_noise_std),
            "max_plot_steps": int(args.max_plot_steps),
        },
        "num_step_records_total": int(total_step_records),
        "num_step_records_used": int(len(step_records)),
        "curve_length": int(curve_len),
        "acceptance": {
            "mean_positive_accept_len": _mean(positive_accept),
            "mean_negative_accept_len": _mean(negative_accept),
            "mean_final_accept_len": _mean(final_accept),
        },
        "fractions": {
            "positive_gt_negative": _mean(positive_gt_negative),
            "final_not_better_than_positive": _mean(final_not_better_than_positive),
        },
        "kl": {
            "avg": _mean(kl_values),
            "min": float(min(kl_values)),
            "max": float(max(kl_values)),
        },
        "mean_prod_curve": {
            "positive": _mean_prod_curve(step_records, "positive_prod_curve", curve_len),
            "negative": _mean_prod_curve(step_records, "negative_prod_curve", curve_len),
            "final": _mean_prod_curve(step_records, "final_prod_curve", curve_len),
        },
    }


def _plot_prod_curve_mean(
    plt,
    step_records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    curve_len = int(summary["curve_length"])
    fig, ax = plt.subplots(figsize=(8, 5))
    if curve_len <= 0 or not step_records:
        ax.text(0.5, 0.5, "No speculative positions to plot", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.arange(1, curve_len + 1)
        curves = summary["mean_prod_curve"]
        ax.plot(x, curves["positive"], marker="o", label="Positive", linewidth=2)
        ax.plot(x, curves["negative"], marker="s", label="Negative", linewidth=2)
        ax.plot(x, curves["final"], marker="^", label="Final VCD", linewidth=2)
        ax.set_xlabel("Speculative token position")
        ax.set_ylabel("Mean cumprod(match)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x)
        ax.set_title(f"Mean Prod Curve by Branch (n={len(step_records)})")
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_accept_len_branch_hist(
    plt,
    step_records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    curve_len = int(summary["curve_length"])
    fig, ax = plt.subplots(figsize=(8, 5))
    if curve_len <= 0 or not step_records:
        ax.text(0.5, 0.5, "No speculative positions to plot", ha="center", va="center")
        ax.axis("off")
    else:
        pos = np.asarray([r["positive_accept_len"] for r in step_records], dtype=np.int32)
        neg = np.asarray([r["negative_accept_len"] for r in step_records], dtype=np.int32)
        fin = np.asarray([r["final_accept_len"] for r in step_records], dtype=np.int32)
        bins = np.arange(-0.5, curve_len + 1.5, 1.0)
        ax.hist(pos, bins=bins, alpha=0.45, label="Positive")
        ax.hist(neg, bins=bins, alpha=0.45, label="Negative")
        ax.hist(fin, bins=bins, alpha=0.45, label="Final VCD")
        ax.set_xlabel("Acceptance length (speculative tokens)")
        ax.set_ylabel("Count")
        ax.set_title(f"Acceptance Length Distribution by Branch (n={len(step_records)})")
        ax.set_xticks(np.arange(0, curve_len + 1))
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_accept_len_gap_vs_kl(
    plt,
    step_records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    if not step_records:
        ax.text(0.5, 0.5, "No records to plot", ha="center", va="center")
        ax.axis("off")
    else:
        x = np.asarray(
            [r["positive_accept_len"] - r["negative_accept_len"] for r in step_records],
            dtype=np.float32,
        )
        y = np.asarray([r["kl_divergence"] for r in step_records], dtype=np.float32)
        ax.scatter(x, y, s=14, alpha=0.5)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
        ax.set_xlabel("Acceptance gap: positive - negative")
        ax.set_ylabel("KL divergence (positive || negative)")
        ax.set_title(f"Acceptance Gap vs KL Divergence (n={len(step_records)})")
        ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _build_prod_report(summary: dict[str, Any]) -> str:
    n_total = int(summary["num_step_records_total"])
    n_used = int(summary["num_step_records_used"])
    acceptance = summary["acceptance"]
    fractions = summary["fractions"]
    kl_stats = summary["kl"]

    pos_mean = float(acceptance["mean_positive_accept_len"])
    neg_mean = float(acceptance["mean_negative_accept_len"])
    fin_mean = float(acceptance["mean_final_accept_len"])
    pos_gt_neg = float(fractions["positive_gt_negative"])
    final_not_better = float(fractions["final_not_better_than_positive"])

    if n_used == 0:
        interpretation = (
            "No step-level speculative diagnostics were collected. "
            "This usually means `block_size <= 1` or generation ended before speculative decoding steps."
        )
    elif final_not_better >= 0.5:
        interpretation = (
            "Final VCD branch is not better than the positive branch for most steps, "
            "so contrastive mixing is not translating into longer accepted spans."
        )
    elif pos_gt_neg <= 0.55:
        interpretation = (
            "Positive and negative branches are weakly separated in acceptance behavior, "
            "which limits how much VCD can improve acceptance length."
        )
    else:
        interpretation = (
            "Positive-vs-negative separation exists, but final branch gains are still limited, "
            "suggesting candidate filtering/mixing may dilute the positive branch benefit."
        )

    lines = [
        "# VCD Prod Diagnostics Report",
        "",
        "## Executive Summary",
        f"- Used {n_used} step records for analysis (total collected: {n_total}).",
        f"- Mean acceptance length (positive/negative/final): {pos_mean:.3f} / {neg_mean:.3f} / {fin_mean:.3f}.",
        f"- Fraction of steps with positive > negative: {pos_gt_neg * 100:.2f}%.",
        f"- Fraction of steps with final <= positive: {final_not_better * 100:.2f}%.",
        f"- KL divergence (avg/min/max): {kl_stats['avg']:.5f} / {kl_stats['min']:.5f} / {kl_stats['max']:.5f}.",
        "",
        "## Interpretation",
        interpretation,
        "",
        "## Artifacts",
        "- `prod_curve_mean.png`",
        "- `accept_len_branch_hist.png`",
        "- `accept_len_gap_vs_kl.png`",
        "- `prod_summary.json`",
        "- `prod_step_records.jsonl` (optional, based on CLI flag)",
    ]
    return "\n".join(lines) + "\n"


def _save_prod_artifacts(
    *,
    output_dir: Path,
    plt,
    step_records_all: list[dict[str, Any]],
    step_records_for_analysis: list[dict[str, Any]],
    summary: dict[str, Any],
    save_step_records: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if save_step_records:
        step_file = output_dir / "prod_step_records.jsonl"
        with step_file.open("w", encoding="utf-8") as f:
            for row in step_records_all:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = output_dir / "prod_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    _plot_prod_curve_mean(
        plt=plt,
        step_records=step_records_for_analysis,
        summary=summary,
        output_path=output_dir / "prod_curve_mean.png",
    )
    _plot_accept_len_branch_hist(
        plt=plt,
        step_records=step_records_for_analysis,
        summary=summary,
        output_path=output_dir / "accept_len_branch_hist.png",
    )
    _plot_accept_len_gap_vs_kl(
        plt=plt,
        step_records=step_records_for_analysis,
        output_path=output_dir / "accept_len_gap_vs_kl.png",
    )

    report_path = output_dir / "prod_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write(_build_prod_report(summary))

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
    vcd_alpha: float = 1.0,
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    divergence_accumulator: Optional[List[float]] = None,
    diagnostic_context: Optional[dict[str, Any]] = None,
    save_step_records: bool = True,
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
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature, gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    step_diagnostics: list[dict[str, Any]] = []
    decode_step = 0
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        positive_branch_tokens = None
        negative_branch_tokens = None
        final_branch_tokens = None
        kl_val = 0.0
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
                gen=gen,
            )
            past_key_values_draft.crop(start)
            
            # ----- Divergence witness (optional) -----
            kl_val = _compute_kl_divergence(positive_draft_logits, negative_draft_logits)
            if divergence_accumulator is not None:
                divergence_accumulator.append(kl_val)

            candidate_mask = build_vcd_candidate_mask(
                reference_logits=positive_draft_logits,
                beta=beta,
            )
                        
            final_draft_logits = apply_vcd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=vcd_alpha,
            )
            final_draft_logits = apply_vcd_candidate_filter(
                logits=final_draft_logits,
                candidate_mask=candidate_mask,
            )
            
            # Applying positive draft logits to the top candidates in the final draft logits to further reduce the chance of selecting a token that is not favored by the positive draft
            n_keep = min(6, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]

            positive_branch_tokens = torch.argmax(positive_draft_logits, dim=-1)
            negative_branch_tokens = torch.argmax(negative_draft_logits, dim=-1)
            block_output_ids[:, 1:] = sample(final_draft_logits, gen=gen)
            final_branch_tokens = block_output_ids[:, 1:].clone()
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
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)

        if block_size > 1 and save_step_records:
            posterior_tokens = posterior[:, :-1]
            positive_prod_curve, positive_accept_len = _compute_match_prod_curve(
                positive_branch_tokens,
                posterior_tokens,
            )
            negative_prod_curve, negative_accept_len = _compute_match_prod_curve(
                negative_branch_tokens,
                posterior_tokens,
            )
            final_prod_curve, final_accept_len = _compute_match_prod_curve(
                final_branch_tokens,
                posterior_tokens,
            )
            step_record: dict[str, Any] = {
                "decode_step": int(decode_step),
                "kl_divergence": float(kl_val),
                "positive_prod_curve": positive_prod_curve,
                "negative_prod_curve": negative_prod_curve,
                "final_prod_curve": final_prod_curve,
                "positive_accept_len": int(positive_accept_len),
                "negative_accept_len": int(negative_accept_len),
                "final_accept_len": int(final_accept_len),
            }
            if diagnostic_context:
                step_record.update(diagnostic_context)
            step_diagnostics.append(step_record)

        decode_step += 1
        start += acceptance_length + 1
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
        step_diagnostics=step_diagnostics,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vcd-alpha", type=float, default=0.5)
    parser.add_argument("--vcd-beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument("--analysis-dir", type=str, default="Analysis/results")
    parser.add_argument("--max-plot-steps", type=int, default=5000)
    parser.add_argument(
        "--save-step-records",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save raw step-level prod diagnostics as JSONL.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    args = parser.parse_args()
    plt = _require_matplotlib()

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
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
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = [] 

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
                    vcd_alpha=args.vcd_alpha,
                    beta=args.vcd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    divergence_accumulator=divergence_accumulator,
                    diagnostic_context={
                        "dataset_idx": int(idx),
                        "turn_index": int(turn_index),
                        "rank": int(dist.rank()),
                        "sample_seed": int(sample_seed + bs),
                        "block_size": int(bs),
                    },
                    save_step_records=args.save_step_records,
                    seed=sample_seed + bs,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        divergence_accumulator = dist.gather(divergence_accumulator, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        divergence_accumulator = list(chain(*divergence_accumulator))

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

    prod_step_records = list(chain(*[r[block_size].step_diagnostics for r in responses]))
    if args.max_plot_steps > 0 and len(prod_step_records) > args.max_plot_steps:
        prod_step_records_for_analysis = prod_step_records[: args.max_plot_steps]
    else:
        prod_step_records_for_analysis = prod_step_records

    prod_summary = _summarize_prod_records(
        step_records=prod_step_records_for_analysis,
        total_step_records=len(prod_step_records),
        block_size=block_size,
        args=args,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.analysis_dir) / f"{args.dataset}_bs{block_size}_seed{args.seed}_{timestamp}"
    _save_prod_artifacts(
        output_dir=output_dir,
        plt=plt,
        step_records_all=prod_step_records,
        step_records_for_analysis=prod_step_records_for_analysis,
        summary=prod_summary,
        save_step_records=args.save_step_records,
    )

    print(f"Saved prod diagnostics artifacts to: {output_dir}")
    print(
        "Branch mean acceptance (spec tokens) | "
        f"positive={prod_summary['acceptance']['mean_positive_accept_len']:.3f} | "
        f"negative={prod_summary['acceptance']['mean_negative_accept_len']:.3f} | "
        f"final={prod_summary['acceptance']['mean_final_accept_len']:.3f}"
    )
    print(
        "Branch ratios | "
        f"positive>negative={prod_summary['fractions']['positive_gt_negative'] * 100:.2f}% | "
        f"final<=positive={prod_summary['fractions']['final_not_better_than_positive'] * 100:.2f}%"
    )
    print(
        "KL divergence (step-level) | "
        f"avg={prod_summary['kl']['avg']:.5f} | "
        f"min={prod_summary['kl']['min']:.5f} | "
        f"max={prod_summary['kl']['max']:.5f}"
    )

if __name__ == "__main__":
    main()
