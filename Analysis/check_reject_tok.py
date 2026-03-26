import argparse
import csv
import importlib.util
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

from model import DFlashDraftModel, apply_vcd_logits, extract_context_feature, load_and_process_dataset, sample


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
    negative_block_output_ids = block_output_ids.clone()
    random_tokens = torch.randint(
        0,
        vocab_size,
        (block_output_ids.shape[0],),
        device=block_output_ids.device,
        generator=gen,
    )
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
            )
            >= dropout_ratio
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


def build_vcd_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)
    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def apply_vcd_candidate_filter(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


def _compute_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> float:
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


def _topk_token_details(
    logits_1d: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> list[dict[str, Any]]:
    log_probs = torch.log_softmax(logits_1d.float(), dim=-1)
    k = min(k, int(log_probs.shape[-1]))
    vals, ids = torch.topk(log_probs, k=k, dim=-1)
    out: list[dict[str, Any]] = []
    for token_id, logp in zip(ids.tolist(), vals.tolist()):
        out.append(
            {
                "id": int(token_id),
                "text": _token_str(tokenizer, int(token_id)),
                "logprob": float(logp),
                "prob": float(np.exp(logp)),
            }
        )
    return out


def _estimate_token_hit_probability(
    logits_1d: torch.Tensor,
    token_id: int,
    num_samples: int,
    gen: Optional[torch.Generator] = None,
) -> float:
    if num_samples <= 0:
        return 0.0
    probs = torch.softmax(logits_1d, dim=-1)
    sampled_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True, generator=gen)
    return float((sampled_ids == token_id).float().mean().item())


def _build_reject_reason(record: dict[str, Any]) -> str:
    reasons: list[str] = []
    reasons.append(
        f"Reject vì draft đề xuất `{record['sampled_draft_text']}` ({record['sampled_draft_id']}) "
        f"khác posterior `{record['target_token_text']}` ({record['target_token_id']})."
    )
    if record["pred_changed"] == 1:
        reasons.append(
            f"VCD shadow sample đã đổi token từ `{record['sampled_draft_text']}` sang `{record['vcd_shadow_sampled_text']}`."
        )
    else:
        reasons.append("VCD shadow sample không đổi token so với original ở vị trí reject.")
    if record["target_in_candidate_mask"] == 0:
        reasons.append("Token target không nằm trong candidate mask (beta filter), nên khó được VCD chọn.")
    if record["vcd_shadow_fix_hit"] == 1:
        reasons.append("VCD shadow sample trùng target token ở vị trí reject, cho thấy có khả năng cứu reject nếu rollout theo VCD.")
    if record["vcd_hit"] == 1 and record["positive_hit"] == 0:
        reasons.append("VCD đã sửa đúng token target tốt hơn positive.")
    elif record["vcd_hit"] == 0 and record["positive_hit"] == 1:
        reasons.append("VCD làm mất token target mà positive vốn chọn đúng.")
    elif record["vcd_hit"] == 0 and record["positive_hit"] == 0:
        reasons.append("Cả positive và VCD đều chưa đưa target token lên top-1.")
    return " ".join(reasons)


def _classify_reject_taxonomy(record: dict[str, Any]) -> tuple[str, str]:
    # Priority-based taxonomy: earlier rules represent stronger root-cause signals.
    if record["vcd_shadow_fix_hit"] == 1:
        return (
            "vcd_shadow_can_fix_reject",
            "Trong che do shadow, token sample tu VCD trung target tai vi tri reject cua original.",
        )
    if record["target_in_candidate_mask"] == 0:
        return (
            "target_filtered_by_candidate_mask",
            "Token target bi loai khoi candidate mask (beta filter), nen draft/VCD kho de de xuat dung target.",
        )
    if record["positive_hit"] == 1 and record["vcd_hit"] == 0:
        return (
            "vcd_regression_from_positive",
            "Positive dung target nhung VCD day target xuong, dan den reject.",
        )
    if record["positive_hit"] == 0 and record["vcd_hit"] == 1:
        return (
            "vcd_correct_but_sampled_mismatch",
            "VCD dua target len top-1 nhung token draft duoc sample van khac posterior nen van reject.",
        )
    if record["pred_changed"] == 0 and record["positive_hit"] == 0:
        return (
            "no_vcd_effect_positive_wrong",
            "VCD khong doi huong du doan va positive da sai target tu dau.",
        )
    if record["pred_changed"] == 1 and record["vcd_hit"] == 0:
        return (
            "vcd_shifted_but_still_wrong",
            "VCD co doi huong du doan nhung chua dua target len top-1.",
        )
    return (
        "posterior_mismatch_other",
        "Reject do posterior uu tien token khac voi de xuat draft tai buoc verify.",
    )


def _safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    if report_dir:
        path = Path(report_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("Analysis") / "results" / f"reject_tok_{_safe_dataset_name(dataset_name)}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        with path.open("w", encoding="utf-8") as f:
            f.write("\n")
        return
    fieldnames = list(records[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _maybe_create_plots(report_dir: Path, records: list[dict[str, Any]]) -> list[str]:
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
    pos_hit_rate = float(np.mean([r["positive_hit"] for r in records]))
    vcd_hit_rate = float(np.mean([r["vcd_hit"] for r in records]))
    unchanged_rate = float(np.mean([r["pred_changed"] == 0 for r in records]))

    plt.figure(figsize=(7, 4))
    plt.bar(["Positive hit", "VCD hit", "Unchanged pred"], [pos_hit_rate, vcd_hit_rate, unchanged_rate])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Rate")
    plt.title("Reject Position Accuracy / Change Rate")
    p1 = report_dir / "reject_accuracy_overview.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()
    plot_files.append(p1.name)

    delta_lp = np.asarray([r["delta_target_logprob"] for r in records], dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.hist(delta_lp, bins=60)
    plt.axvline(0.0, linestyle="--")
    plt.title("Delta Target LogProb (VCD - Positive) at Reject Position")
    plt.xlabel("Delta log-prob")
    plt.ylabel("Count")
    p2 = report_dir / "reject_delta_logprob_hist.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()
    plot_files.append(p2.name)

    delta_rank = np.asarray([r["delta_target_rank"] for r in records], dtype=np.float32)
    plt.figure(figsize=(7, 4))
    plt.hist(delta_rank, bins=60)
    plt.axvline(0.0, linestyle="--")
    plt.title("Delta Target Rank (Positive - VCD) at Reject Position")
    plt.xlabel("Positive rank - VCD rank (higher is better)")
    plt.ylabel("Count")
    p3 = report_dir / "reject_delta_rank_hist.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=160)
    plt.close()
    plot_files.append(p3.name)

    conf = np.zeros((2, 2), dtype=np.int64)
    for r in records:
        conf[int(r["positive_hit"])][int(r["vcd_hit"])] += 1
    plt.figure(figsize=(5, 4))
    plt.imshow(conf, cmap="Blues")
    plt.xticks([0, 1], ["VCD wrong", "VCD correct"])
    plt.yticks([0, 1], ["Positive wrong", "Positive correct"])
    plt.title("Hit/Hit Confusion at Reject Position")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(conf[i, j]), ha="center", va="center")
    p4 = report_dir / "reject_hit_confusion.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=160)
    plt.close()
    plot_files.append(p4.name)

    taxonomy_counts: dict[str, int] = {}
    for r in records:
        k = r.get("reject_taxonomy", "unknown")
        taxonomy_counts[k] = taxonomy_counts.get(k, 0) + 1
    labels = sorted(taxonomy_counts.keys(), key=lambda x: taxonomy_counts[x], reverse=True)
    values = [taxonomy_counts[k] for k in labels]
    if values:
        plt.figure(figsize=(max(8, len(labels) * 1.2), 4.8))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=30, ha="right")
        plt.ylabel("Count")
        plt.title("Reject Taxonomy Distribution")
        p5 = report_dir / "reject_taxonomy_distribution.png"
        plt.tight_layout()
        plt.savefig(p5, dpi=160)
        plt.close()
        plot_files.append(p5.name)

    return plot_files


def _build_summary(records: list[dict[str, Any]], args: argparse.Namespace, block_size: int) -> dict[str, Any]:
    if not records:
        return {
            "meta": {
                "dataset": args.dataset,
                "seed": int(args.seed),
                "block_size": int(block_size),
                "vcd_alpha": float(args.vcd_alpha),
                "vcd_beta": float(args.vcd_beta),
                "evaluation_mode": "positive_rollout_with_vcd_shadow",
                "vcd_shadow_num_samples": int(args.vcd_shadow_num_samples),
                "negative_context_dropout": float(args.negative_context_dropout),
                "negative_context_noise_std": float(args.negative_context_noise_std),
            },
            "reject_events": 0,
        }

    positive_hit = np.asarray([r["positive_hit"] for r in records], dtype=np.float32)
    vcd_hit = np.asarray([r["vcd_hit"] for r in records], dtype=np.float32)
    vcd_shadow_fix = np.asarray([r["vcd_shadow_fix_hit"] for r in records], dtype=np.float32)
    pred_changed = np.asarray([r["pred_changed"] for r in records], dtype=np.float32)
    delta_lp = np.asarray([r["delta_target_logprob"] for r in records], dtype=np.float32)
    delta_rank = np.asarray([r["delta_target_rank"] for r in records], dtype=np.float32)
    target_in_mask = np.asarray([r["target_in_candidate_mask"] for r in records], dtype=np.float32)
    sampled_in_mask = np.asarray([r["sampled_in_candidate_mask"] for r in records], dtype=np.float32)
    sampled_rank_posterior = np.asarray([r["sampled_rank_in_posterior"] for r in records], dtype=np.float32)
    target_is_posterior_argmax = np.asarray([r["target_is_posterior_argmax"] for r in records], dtype=np.float32)
    posterior_prefers_target = np.asarray([r["posterior_prefers_target_over_sampled"] for r in records], dtype=np.float32)
    posterior_gap = np.asarray([r["target_minus_sampled_posterior_logprob"] for r in records], dtype=np.float32)
    positive_rescue_prob_est = np.asarray([r["positive_rescue_prob_est"] for r in records], dtype=np.float32)
    vcd_shadow_rescue_prob_est = np.asarray([r["vcd_shadow_rescue_prob_est"] for r in records], dtype=np.float32)
    rescue_prob_gain = np.asarray([r["rescue_prob_gain_vcd_minus_positive"] for r in records], dtype=np.float32)
    taxonomy_counts: dict[str, int] = {}
    for r in records:
        label = r.get("reject_taxonomy", "unknown")
        taxonomy_counts[label] = taxonomy_counts.get(label, 0) + 1
    taxonomy_rates = {k: float(v / len(records)) for k, v in taxonomy_counts.items()}

    return {
        "meta": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "block_size": int(block_size),
            "vcd_alpha": float(args.vcd_alpha),
            "vcd_beta": float(args.vcd_beta),
            "evaluation_mode": "positive_rollout_with_vcd_shadow",
            "vcd_shadow_num_samples": int(args.vcd_shadow_num_samples),
            "negative_context_dropout": float(args.negative_context_dropout),
            "negative_context_noise_std": float(args.negative_context_noise_std),
        },
        "reject_events": int(len(records)),
        "positive_hit_rate": float(positive_hit.mean()),
        "vcd_hit_rate": float(vcd_hit.mean()),
        "hit_rate_gain": float(vcd_hit.mean() - positive_hit.mean()),
        "vcd_shadow_fix_rate": float(vcd_shadow_fix.mean()),
        "positive_rescue_prob_est": {
            "mean": float(positive_rescue_prob_est.mean()),
            "median": float(np.median(positive_rescue_prob_est)),
        },
        "vcd_shadow_rescue_prob_est": {
            "mean": float(vcd_shadow_rescue_prob_est.mean()),
            "median": float(np.median(vcd_shadow_rescue_prob_est)),
        },
        "rescue_prob_gain_vcd_minus_positive": {
            "mean": float(rescue_prob_gain.mean()),
            "median": float(np.median(rescue_prob_gain)),
            "p10": float(np.quantile(rescue_prob_gain, 0.1)),
            "p90": float(np.quantile(rescue_prob_gain, 0.9)),
            "improved_rate": float((rescue_prob_gain > 0).mean()),
        },
        "pred_changed_rate": float(pred_changed.mean()),
        "delta_target_logprob": {
            "mean": float(delta_lp.mean()),
            "median": float(np.median(delta_lp)),
            "p10": float(np.quantile(delta_lp, 0.1)),
            "p90": float(np.quantile(delta_lp, 0.9)),
            "improved_rate": float((delta_lp > 0).mean()),
        },
        "delta_target_rank": {
            "mean": float(delta_rank.mean()),
            "median": float(np.median(delta_rank)),
            "p10": float(np.quantile(delta_rank, 0.1)),
            "p90": float(np.quantile(delta_rank, 0.9)),
            "improved_rate": float((delta_rank > 0).mean()),
        },
        "reject_cause_indicators": {
            "target_in_candidate_mask_rate": float(target_in_mask.mean()),
            "sampled_in_candidate_mask_rate": float(sampled_in_mask.mean()),
            "target_is_posterior_argmax_rate": float(target_is_posterior_argmax.mean()),
            "posterior_prefers_target_over_sampled_rate": float(posterior_prefers_target.mean()),
            "target_minus_sampled_posterior_logprob": {
                "mean": float(posterior_gap.mean()),
                "median": float(np.median(posterior_gap)),
                "p10": float(np.quantile(posterior_gap, 0.1)),
                "p90": float(np.quantile(posterior_gap, 0.9)),
            },
            "sampled_rank_in_posterior": {
                "mean": float(sampled_rank_posterior.mean()),
                "median": float(np.median(sampled_rank_posterior)),
                "p10": float(np.quantile(sampled_rank_posterior, 0.1)),
                "p90": float(np.quantile(sampled_rank_posterior, 0.9)),
            },
            "taxonomy_counts": taxonomy_counts,
            "taxonomy_rates": taxonomy_rates,
        },
        "confusion": {
            "positive_wrong_vcd_wrong": int(np.sum((positive_hit == 0) & (vcd_hit == 0))),
            "positive_wrong_vcd_correct": int(np.sum((positive_hit == 0) & (vcd_hit == 1))),
            "positive_correct_vcd_wrong": int(np.sum((positive_hit == 1) & (vcd_hit == 0))),
            "positive_correct_vcd_correct": int(np.sum((positive_hit == 1) & (vcd_hit == 1))),
        },
    }


def _write_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
) -> None:
    def _fmt_topk(topk: list[dict[str, Any]], k: int = 3) -> str:
        items = topk[:k]
        return "; ".join([f"{x['text']} ({x['id']}, p={x['prob']:.4f})" for x in items])

    lines: list[str] = []
    lines.append("# Reject Token Analysis: Positive vs VCD")
    lines.append("")

    if summary.get("reject_events", 0) == 0:
        lines.append("No reject-token event was recorded.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Overall")
    lines.append(f"- Reject events: **{summary['reject_events']}**")
    lines.append(f"- Positive hit rate: **{summary['positive_hit_rate'] * 100:.2f}%**")
    lines.append(f"- VCD hit rate: **{summary['vcd_hit_rate'] * 100:.2f}%**")
    lines.append(f"- Hit-rate gain (VCD - Positive): **{summary['hit_rate_gain'] * 100:+.2f}%**")
    lines.append(f"- VCD shadow fix rate @reject: **{summary['vcd_shadow_fix_rate'] * 100:.2f}%**")
    lines.append(
        f"- Mean rescue-prob (Positive / VCD-shadow): **{summary['positive_rescue_prob_est']['mean']:.4f} / {summary['vcd_shadow_rescue_prob_est']['mean']:.4f}**"
    )
    lines.append(
        f"- Mean rescue-prob gain (VCD - Positive): **{summary['rescue_prob_gain_vcd_minus_positive']['mean']:+.4f}**"
    )
    lines.append(
        f"- Rescue-prob improved-rate (>0): **{summary['rescue_prob_gain_vcd_minus_positive']['improved_rate'] * 100:.2f}%**"
    )
    lines.append(f"- Prediction changed rate: **{summary['pred_changed_rate'] * 100:.2f}%**")
    lines.append("")

    lines.append("## Target LogProb Delta (VCD - Positive)")
    lines.append(f"- Mean: {summary['delta_target_logprob']['mean']:.5f}")
    lines.append(f"- Median: {summary['delta_target_logprob']['median']:.5f}")
    lines.append(f"- P10 / P90: {summary['delta_target_logprob']['p10']:.5f} / {summary['delta_target_logprob']['p90']:.5f}")
    lines.append(f"- Improved rate (>0): {summary['delta_target_logprob']['improved_rate'] * 100:.2f}%")
    lines.append("")

    lines.append("## Target Rank Delta (Positive - VCD)")
    lines.append(f"- Mean: {summary['delta_target_rank']['mean']:.3f}")
    lines.append(f"- Median: {summary['delta_target_rank']['median']:.3f}")
    lines.append(f"- P10 / P90: {summary['delta_target_rank']['p10']:.3f} / {summary['delta_target_rank']['p90']:.3f}")
    lines.append(f"- Improved rate (>0): {summary['delta_target_rank']['improved_rate'] * 100:.2f}%")
    lines.append("")

    lines.append("## Why Rejected")
    rc = summary["reject_cause_indicators"]
    lines.append(
        f"- Target in candidate mask rate: **{rc['target_in_candidate_mask_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Sampled token in candidate mask rate: **{rc['sampled_in_candidate_mask_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Sampled token rank in posterior (mean/median): **{rc['sampled_rank_in_posterior']['mean']:.2f} / {rc['sampled_rank_in_posterior']['median']:.2f}**"
    )
    lines.append(
        f"- Target is posterior argmax rate: **{rc['target_is_posterior_argmax_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Posterior prefers target over sampled rate: **{rc['posterior_prefers_target_over_sampled_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Target minus sampled posterior logprob (mean/median): **{rc['target_minus_sampled_posterior_logprob']['mean']:.4f} / {rc['target_minus_sampled_posterior_logprob']['median']:.4f}**"
    )
    lines.append("")

    lines.append("## Reject Taxonomy")
    lines.append("| Taxonomy | Count | Rate |")
    lines.append("|---|---:|---:|")
    taxonomy_counts = rc.get("taxonomy_counts", {})
    taxonomy_rates = rc.get("taxonomy_rates", {})
    for key in sorted(taxonomy_counts.keys(), key=lambda k: taxonomy_counts[k], reverse=True):
        lines.append(f"| {key} | {taxonomy_counts[key]} | {taxonomy_rates.get(key, 0.0) * 100:.2f}% |")
    lines.append("")

    lines.append("## Confusion (Positive Hit vs VCD Hit)")
    lines.append("| Positive\\VCD | VCD wrong | VCD correct |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Positive wrong | {summary['confusion']['positive_wrong_vcd_wrong']} | {summary['confusion']['positive_wrong_vcd_correct']} |"
    )
    lines.append(
        f"| Positive correct | {summary['confusion']['positive_correct_vcd_wrong']} | {summary['confusion']['positive_correct_vcd_correct']} |"
    )
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    lines.append("## Top Improved Reject Events")
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | vcd_pred | d_logprob | d_rank | why_reject |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|")
    improved = sorted(records, key=lambda x: x["delta_target_logprob"], reverse=True)[:max_rows]
    for rec in improved:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {reject_taxonomy} | {target_token_text} ({target_token_id}) "
            "| {sampled_draft_text} ({sampled_draft_id}) "
            "| {positive_pred_text} ({positive_pred_id}) | {vcd_pred_text} ({vcd_pred_id}) "
            "| {delta_target_logprob:.4f} | {delta_target_rank} | {reject_reason} |".format(**rec)
        )
    lines.append("")

    lines.append("## Top Worsened Reject Events")
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | vcd_pred | d_logprob | d_rank | why_reject |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|")
    worsened = sorted(records, key=lambda x: x["delta_target_logprob"])[:max_rows]
    for rec in worsened:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {reject_taxonomy} | {target_token_text} ({target_token_id}) "
            "| {sampled_draft_text} ({sampled_draft_id}) "
            "| {positive_pred_text} ({positive_pred_id}) | {vcd_pred_text} ({vcd_pred_id}) "
            "| {delta_target_logprob:.4f} | {delta_target_rank} | {reject_reason} |".format(**rec)
        )

    lines.append("")
    lines.append("## Detailed Case Explanations")
    cases = improved[: min(5, len(improved))] + worsened[: min(5, len(worsened))]
    for i, rec in enumerate(cases, start=1):
        lines.append(
            f"### Case {i}: sample={rec['sample_idx']}, turn={rec['turn_idx']}, step={rec['decode_step']}, pos={rec['absolute_position']}"
        )
        lines.append(f"- Taxonomy: `{rec['reject_taxonomy']}` - {rec['reject_taxonomy_desc']}")
        lines.append(
            f"- Proposed draft token: `{rec['sampled_draft_text']}` ({rec['sampled_draft_id']}); posterior token: `{rec['target_token_text']}` ({rec['target_token_id']})"
        )
        lines.append(
            f"- Positive pred: `{rec['positive_pred_text']}` ({rec['positive_pred_id']}), VCD pred: `{rec['vcd_pred_text']}` ({rec['vcd_pred_id']})"
        )
        lines.append(f"- Reason: {rec['reject_reason']}")
        lines.append(
            f"- Candidate mask: target_in={rec['target_in_candidate_mask']}, sampled_in={rec['sampled_in_candidate_mask']}"
        )
        lines.append(
            f"- Delta target logprob/rank: {rec['delta_target_logprob']:.4f} / {rec['delta_target_rank']}"
        )
        lines.append(f"- Positive top-3: {_fmt_topk(rec['positive_top5'], 3)}")
        lines.append(f"- VCD top-3: {_fmt_topk(rec['vcd_top5'], 3)}")
        lines.append(f"- Posterior top-3: {_fmt_topk(rec['posterior_top5'], 3)}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


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
    vcd_alpha: float = 1.0,
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    divergence_accumulator: Optional[List[float]] = None,
    reject_records: Optional[list[dict[str, Any]]] = None,
    vcd_shadow_num_samples: int = 1,
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
                gen=gen,
            )
            past_key_values_draft.crop(start)

            if divergence_accumulator is not None:
                divergence_accumulator.append(_compute_kl_divergence(positive_draft_logits, negative_draft_logits))

            candidate_mask = build_vcd_candidate_mask(reference_logits=positive_draft_logits, beta=beta)
            final_draft_logits = apply_vcd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=vcd_alpha,
            )
            final_draft_logits = apply_vcd_candidate_filter(logits=final_draft_logits, candidate_mask=candidate_mask)
            positive_sampled_tokens = sample(positive_draft_logits, gen=gen)
            vcd_shadow_sampled_tokens = sample(final_draft_logits, gen=gen)

            # Keep original rollout trajectory; VCD is evaluated in shadow mode.
            block_output_ids[:, 1:] = positive_sampled_tokens
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

        if block_size > 1 and reject_records is not None and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            target_token_id = int(posterior[0, reject_offset].item())
            sampled_draft_id = int(block_output_ids[0, reject_offset + 1].item())
            vcd_shadow_sampled_id = int(vcd_shadow_sampled_tokens[0, reject_offset].item())

            positive_step_logits = positive_draft_logits[0, reject_offset, :].float()
            vcd_step_logits = final_draft_logits[0, reject_offset, :].float()
            posterior_step_logits = output.logits[0, reject_offset, :].float()

            positive_pred_id = int(torch.argmax(positive_step_logits).item())
            vcd_pred_id = int(torch.argmax(vcd_step_logits).item())
            posterior_pred_id = int(torch.argmax(posterior_step_logits).item())

            positive_logprob = torch.log_softmax(positive_step_logits, dim=-1)
            vcd_logprob = torch.log_softmax(vcd_step_logits, dim=-1)
            posterior_logprob = torch.log_softmax(posterior_step_logits, dim=-1)

            pos_target_logprob = float(positive_logprob[target_token_id].item())
            vcd_target_logprob = float(vcd_logprob[target_token_id].item())
            posterior_target_logprob = float(posterior_logprob[target_token_id].item())
            posterior_sampled_logprob = float(posterior_logprob[sampled_draft_id].item())
            posterior_vcd_shadow_sampled_logprob = float(posterior_logprob[vcd_shadow_sampled_id].item())
            positive_sampled_logprob = float(positive_logprob[sampled_draft_id].item())
            vcd_sampled_logprob = float(vcd_logprob[sampled_draft_id].item())

            pos_rank = _token_rank_from_logits(positive_step_logits, target_token_id)
            vcd_rank = _token_rank_from_logits(vcd_step_logits, target_token_id)
            sampled_rank_posterior = _token_rank_from_logits(posterior_step_logits, sampled_draft_id)
            vcd_shadow_sampled_rank_posterior = _token_rank_from_logits(posterior_step_logits, vcd_shadow_sampled_id)
            target_in_candidate_mask = int(candidate_mask[0, reject_offset, target_token_id].item())
            sampled_in_candidate_mask = int(candidate_mask[0, reject_offset, sampled_draft_id].item())
            vcd_shadow_sampled_in_candidate_mask = int(candidate_mask[0, reject_offset, vcd_shadow_sampled_id].item())
            positive_rescue_prob_est = _estimate_token_hit_probability(
                positive_step_logits,
                target_token_id,
                num_samples=vcd_shadow_num_samples,
                gen=gen,
            )
            vcd_shadow_rescue_prob_est = _estimate_token_hit_probability(
                vcd_step_logits,
                target_token_id,
                num_samples=vcd_shadow_num_samples,
                gen=gen,
            )

            record = {
                "sample_idx": int(sample_idx),
                "turn_idx": int(turn_idx),
                "decode_step": int(decode_step),
                "absolute_position": int(start + reject_offset + 1),
                "reject_offset_in_block": int(reject_offset),
                "target_token_id": int(target_token_id),
                "target_token_text": _token_str(tokenizer, target_token_id),
                "sampled_draft_id": int(sampled_draft_id),
                "sampled_draft_text": _token_str(tokenizer, sampled_draft_id),
                "vcd_shadow_sampled_id": int(vcd_shadow_sampled_id),
                "vcd_shadow_sampled_text": _token_str(tokenizer, vcd_shadow_sampled_id),
                "positive_pred_id": int(positive_pred_id),
                "positive_pred_text": _token_str(tokenizer, positive_pred_id),
                "vcd_pred_id": int(vcd_pred_id),
                "vcd_pred_text": _token_str(tokenizer, vcd_pred_id),
                "posterior_pred_id": int(posterior_pred_id),
                "posterior_pred_text": _token_str(tokenizer, posterior_pred_id),
                "positive_hit": int(positive_pred_id == target_token_id),
                "vcd_hit": int(vcd_pred_id == target_token_id),
                "pred_changed": int(vcd_pred_id != positive_pred_id),
                "positive_target_logprob": float(pos_target_logprob),
                "vcd_target_logprob": float(vcd_target_logprob),
                "posterior_target_logprob": float(posterior_target_logprob),
                "posterior_sampled_logprob": float(posterior_sampled_logprob),
                "posterior_vcd_shadow_sampled_logprob": float(posterior_vcd_shadow_sampled_logprob),
                "positive_sampled_logprob": float(positive_sampled_logprob),
                "vcd_sampled_logprob": float(vcd_sampled_logprob),
                "delta_target_logprob": float(vcd_target_logprob - pos_target_logprob),
                "target_minus_sampled_posterior_logprob": float(posterior_target_logprob - posterior_sampled_logprob),
                "target_minus_vcd_shadow_sampled_posterior_logprob": float(posterior_target_logprob - posterior_vcd_shadow_sampled_logprob),
                "positive_target_rank": int(pos_rank),
                "vcd_target_rank": int(vcd_rank),
                "delta_target_rank": int(pos_rank - vcd_rank),
                "sampled_rank_in_posterior": int(sampled_rank_posterior),
                "vcd_shadow_sampled_rank_in_posterior": int(vcd_shadow_sampled_rank_posterior),
                "target_is_posterior_argmax": int(target_token_id == posterior_pred_id),
                "posterior_prefers_target_over_sampled": int(posterior_target_logprob > posterior_sampled_logprob),
                "target_in_candidate_mask": int(target_in_candidate_mask),
                "sampled_in_candidate_mask": int(sampled_in_candidate_mask),
                "vcd_shadow_sampled_in_candidate_mask": int(vcd_shadow_sampled_in_candidate_mask),
                "vcd_shadow_fix_hit": int(vcd_shadow_sampled_id == target_token_id),
                "positive_rescue_prob_est": float(positive_rescue_prob_est),
                "vcd_shadow_rescue_prob_est": float(vcd_shadow_rescue_prob_est),
                "rescue_prob_gain_vcd_minus_positive": float(vcd_shadow_rescue_prob_est - positive_rescue_prob_est),
                "positive_top5": _topk_token_details(positive_step_logits, tokenizer, k=5),
                "vcd_top5": _topk_token_details(vcd_step_logits, tokenizer, k=5),
                "posterior_top5": _topk_token_details(posterior_step_logits, tokenizer, k=5),
            }
            record["reject_reason"] = _build_reject_reason(record)
            taxonomy_label, taxonomy_desc = _classify_reject_taxonomy(record)
            record["reject_taxonomy"] = taxonomy_label
            record["reject_taxonomy_desc"] = taxonomy_desc

            reject_records.append(record)

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        decode_step += 1

        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]

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
    parser.add_argument(
        "--vcd-shadow-num-samples",
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
                    vcd_alpha=args.vcd_alpha,
                    beta=args.vcd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    divergence_accumulator=divergence_accumulator if bs == block_size else None,
                    reject_records=reject_records if bs == block_size else None,
                    vcd_shadow_num_samples=args.vcd_shadow_num_samples,
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
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        reject_records = list(chain(*reject_records))

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
        print(f"VCD hit rate @reject: {summary['vcd_hit_rate'] * 100:.2f}%")
        print(f"Hit-rate gain (VCD - Positive): {summary['hit_rate_gain'] * 100:+.2f}%")
        print(f"VCD shadow fix rate @reject: {summary['vcd_shadow_fix_rate'] * 100:.2f}%")
        print(
            "Mean rescue-prob (Positive / VCD-shadow): "
            f"{summary['positive_rescue_prob_est']['mean']:.4f} / {summary['vcd_shadow_rescue_prob_est']['mean']:.4f}"
        )
        print(
            "Mean rescue-prob gain (VCD - Positive): "
            f"{summary['rescue_prob_gain_vcd_minus_positive']['mean']:+.4f}"
        )
    print(f"Report directory: {report_dir}")
    print(f"  - {jsonl_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {report_md_path.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")


if __name__ == "__main__":
    main()