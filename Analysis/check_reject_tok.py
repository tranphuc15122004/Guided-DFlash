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

from model import DFlashDraftModel, apply_cd_logits, extract_context_feature, load_and_process_dataset, sample


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


def build_cd_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)
    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def apply_cd_candidate_filter(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
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


def _build_reject_reason(record: dict[str, Any]) -> str:
    reasons: list[str] = []
    reasons.append(
        f"Reject vì draft đề xuất `{record['sampled_draft_text']}` ({record['sampled_draft_id']}) "
        f"khác posterior `{record['target_token_text']}` ({record['target_token_id']})."
    )
    if record["pred_changed"] == 1:
        reasons.append(
            f"CD đã đổi argmax từ `{record['positive_pred_text']}` sang `{record['cd_pred_text']}`."
        )
    else:
        reasons.append("CD không đổi argmax so với positive ở vị trí reject.")
    if record["cd_overwritten_by_positive"] == 1:
        reasons.append("Vị trí này nằm trong vùng bị overwrite bởi positive logits (n_keep), nên tác động CD bị triệt tiêu.")
    if record["target_in_candidate_mask"] == 0:
        reasons.append("Token target không nằm trong candidate mask (beta filter), nên khó được CD chọn.")
    if record["cd_hit"] == 1 and record["positive_hit"] == 0:
        reasons.append("CD đã sửa đúng token target tốt hơn positive.")
    elif record["cd_hit"] == 0 and record["positive_hit"] == 1:
        reasons.append("CD làm mất token target mà positive vốn chọn đúng.")
    elif record["cd_hit"] == 0 and record["positive_hit"] == 0:
        reasons.append("Cả positive và CD đều chưa đưa target token lên top-1.")
    return " ".join(reasons)


def _classify_reject_taxonomy(record: dict[str, Any]) -> tuple[str, str]:
    # Priority-based taxonomy: earlier rules represent stronger root-cause signals.
    if record["cd_overwritten_by_positive"] == 1:
        return (
            "overwrite_by_positive_guard",
            "Vi tri reject nam trong vung n_keep bi overwrite boi positive logits, CD gan nhu khong con tac dung.",
        )
    if record["target_in_candidate_mask"] == 0:
        return (
            "target_filtered_by_candidate_mask",
            "Token target bi loai khoi candidate mask (beta filter), nen draft/CD kho de de xuat dung target.",
        )
    if record["positive_hit"] == 1 and record["cd_hit"] == 0:
        return (
            "cd_regression_from_positive",
            "Positive dung target nhung CD day target xuong, dan den reject.",
        )
    if record["positive_hit"] == 0 and record["cd_hit"] == 1:
        return (
            "cd_correct_but_sampled_mismatch",
            "CD dua target len top-1 nhung token draft duoc sample van khac posterior nen van reject.",
        )
    if record["pred_changed"] == 0 and record["positive_hit"] == 0:
        return (
            "no_cd_effect_positive_wrong",
            "CD khong doi huong du doan va positive da sai target tu dau.",
        )
    if record["pred_changed"] == 1 and record["cd_hit"] == 0:
        return (
            "cd_shifted_but_still_wrong",
            "CD co doi huong du doan nhung chua dua target len top-1.",
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
    cd_hit_rate = float(np.mean([r["cd_hit"] for r in records]))
    unchanged_rate = float(np.mean([r["pred_changed"] == 0 for r in records]))

    plt.figure(figsize=(7, 4))
    plt.bar(["Positive hit", "CD hit", "Unchanged pred"], [pos_hit_rate, cd_hit_rate, unchanged_rate])
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
    plt.title("Delta Target LogProb (CD - Positive) at Reject Position")
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
    plt.title("Delta Target Rank (Positive - CD) at Reject Position")
    plt.xlabel("Positive rank - CD rank (higher is better)")
    plt.ylabel("Count")
    p3 = report_dir / "reject_delta_rank_hist.png"
    plt.tight_layout()
    plt.savefig(p3, dpi=160)
    plt.close()
    plot_files.append(p3.name)

    conf = np.zeros((2, 2), dtype=np.int64)
    for r in records:
        conf[int(r["positive_hit"])][int(r["cd_hit"])] += 1
    plt.figure(figsize=(5, 4))
    plt.imshow(conf, cmap="Blues")
    plt.xticks([0, 1], ["CD wrong", "CD correct"])
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
                "cd_alpha": float(args.cd_alpha),
                "cd_beta": float(args.cd_beta),
                "negative_context_dropout": float(args.negative_context_dropout),
                "negative_context_noise_std": float(args.negative_context_noise_std),
            },
            "reject_events": 0,
        }

    positive_hit = np.asarray([r["positive_hit"] for r in records], dtype=np.float32)
    cd_hit = np.asarray([r["cd_hit"] for r in records], dtype=np.float32)
    pred_changed = np.asarray([r["pred_changed"] for r in records], dtype=np.float32)
    delta_lp = np.asarray([r["delta_target_logprob"] for r in records], dtype=np.float32)
    delta_rank = np.asarray([r["delta_target_rank"] for r in records], dtype=np.float32)
    target_in_mask = np.asarray([r["target_in_candidate_mask"] for r in records], dtype=np.float32)
    sampled_in_mask = np.asarray([r["sampled_in_candidate_mask"] for r in records], dtype=np.float32)
    cd_overwritten = np.asarray([r["cd_overwritten_by_positive"] for r in records], dtype=np.float32)
    sampled_rank_posterior = np.asarray([r["sampled_rank_in_posterior"] for r in records], dtype=np.float32)
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
            "cd_alpha": float(args.cd_alpha),
            "cd_beta": float(args.cd_beta),
            "negative_context_dropout": float(args.negative_context_dropout),
            "negative_context_noise_std": float(args.negative_context_noise_std),
        },
        "reject_events": int(len(records)),
        "positive_hit_rate": float(positive_hit.mean()),
        "cd_hit_rate": float(cd_hit.mean()),
        "hit_rate_gain": float(cd_hit.mean() - positive_hit.mean()),
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
            "cd_overwritten_by_positive_rate": float(cd_overwritten.mean()),
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
            "positive_wrong_cd_wrong": int(np.sum((positive_hit == 0) & (cd_hit == 0))),
            "positive_wrong_cd_correct": int(np.sum((positive_hit == 0) & (cd_hit == 1))),
            "positive_correct_cd_wrong": int(np.sum((positive_hit == 1) & (cd_hit == 0))),
            "positive_correct_cd_correct": int(np.sum((positive_hit == 1) & (cd_hit == 1))),
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
    lines.append("# Reject Token Analysis: Positive vs CD")
    lines.append("")

    if summary.get("reject_events", 0) == 0:
        lines.append("No reject-token event was recorded.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Overall")
    lines.append(f"- Reject events: **{summary['reject_events']}**")
    lines.append(f"- Positive hit rate: **{summary['positive_hit_rate'] * 100:.2f}%**")
    lines.append(f"- CD hit rate: **{summary['cd_hit_rate'] * 100:.2f}%**")
    lines.append(f"- Hit-rate gain (CD - Positive): **{summary['hit_rate_gain'] * 100:+.2f}%**")
    lines.append(f"- Prediction changed rate: **{summary['pred_changed_rate'] * 100:.2f}%**")
    lines.append("")

    lines.append("## Target LogProb Delta (CD - Positive)")
    lines.append(f"- Mean: {summary['delta_target_logprob']['mean']:.5f}")
    lines.append(f"- Median: {summary['delta_target_logprob']['median']:.5f}")
    lines.append(f"- P10 / P90: {summary['delta_target_logprob']['p10']:.5f} / {summary['delta_target_logprob']['p90']:.5f}")
    lines.append(f"- Improved rate (>0): {summary['delta_target_logprob']['improved_rate'] * 100:.2f}%")
    lines.append("")

    lines.append("## Target Rank Delta (Positive - CD)")
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
        f"- CD overwritten-by-positive rate (n_keep region): **{rc['cd_overwritten_by_positive_rate'] * 100:.2f}%**"
    )
    lines.append(
        f"- Sampled token rank in posterior (mean/median): **{rc['sampled_rank_in_posterior']['mean']:.2f} / {rc['sampled_rank_in_posterior']['median']:.2f}**"
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

    lines.append("## Confusion (Positive Hit vs CD Hit)")
    lines.append("| Positive\\CD | CD wrong | CD correct |")
    lines.append("|---|---:|---:|")
    lines.append(
        f"| Positive wrong | {summary['confusion']['positive_wrong_cd_wrong']} | {summary['confusion']['positive_wrong_cd_correct']} |"
    )
    lines.append(
        f"| Positive correct | {summary['confusion']['positive_correct_cd_wrong']} | {summary['confusion']['positive_correct_cd_correct']} |"
    )
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    lines.append("## Top Improved Reject Events")
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | cd_pred | d_logprob | d_rank | why_reject |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|")
    improved = sorted(records, key=lambda x: x["delta_target_logprob"], reverse=True)[:max_rows]
    for rec in improved:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {reject_taxonomy} | {target_token_text} ({target_token_id}) "
            "| {sampled_draft_text} ({sampled_draft_id}) "
            "| {positive_pred_text} ({positive_pred_id}) | {cd_pred_text} ({cd_pred_id}) "
            "| {delta_target_logprob:.4f} | {delta_target_rank} | {reject_reason} |".format(**rec)
        )
    lines.append("")

    lines.append("## Top Worsened Reject Events")
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | cd_pred | d_logprob | d_rank | why_reject |")
    lines.append("|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|")
    worsened = sorted(records, key=lambda x: x["delta_target_logprob"])[:max_rows]
    for rec in worsened:
        lines.append(
            "| {sample_idx} | {turn_idx} | {decode_step} | {absolute_position} | {reject_taxonomy} | {target_token_text} ({target_token_id}) "
            "| {sampled_draft_text} ({sampled_draft_id}) "
            "| {positive_pred_text} ({positive_pred_id}) | {cd_pred_text} ({cd_pred_id}) "
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
            f"- Positive pred: `{rec['positive_pred_text']}` ({rec['positive_pred_id']}), CD pred: `{rec['cd_pred_text']}` ({rec['cd_pred_id']})"
        )
        lines.append(f"- Reason: {rec['reject_reason']}")
        lines.append(
            f"- Candidate mask: target_in={rec['target_in_candidate_mask']}, sampled_in={rec['sampled_in_candidate_mask']}, cd_overwritten={rec['cd_overwritten_by_positive']}"
        )
        lines.append(
            f"- Delta target logprob/rank: {rec['delta_target_logprob']:.4f} / {rec['delta_target_rank']}"
        )
        lines.append(f"- Positive top-3: {_fmt_topk(rec['positive_top5'], 3)}")
        lines.append(f"- CD top-3: {_fmt_topk(rec['cd_top5'], 3)}")
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
    cd_alpha: float = 1.0,
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    divergence_accumulator: Optional[List[float]] = None,
    reject_records: Optional[list[dict[str, Any]]] = None,
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

            candidate_mask = build_cd_candidate_mask(reference_logits=positive_draft_logits, beta=beta)
            final_draft_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            final_draft_logits = apply_cd_candidate_filter(logits=final_draft_logits, candidate_mask=candidate_mask)

            n_keep = min(6, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]

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
        acceptance_length = int((block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())

        if block_size > 1 and reject_records is not None and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            target_token_id = int(posterior[0, reject_offset].item())
            sampled_draft_id = int(block_output_ids[0, reject_offset + 1].item())
            n_keep = min(6, final_draft_logits.size(1))

            positive_step_logits = positive_draft_logits[0, reject_offset, :].float()
            cd_step_logits = final_draft_logits[0, reject_offset, :].float()
            posterior_step_logits = output.logits[0, reject_offset, :].float()

            positive_pred_id = int(torch.argmax(positive_step_logits).item())
            cd_pred_id = int(torch.argmax(cd_step_logits).item())
            posterior_pred_id = int(torch.argmax(posterior_step_logits).item())

            positive_logprob = torch.log_softmax(positive_step_logits, dim=-1)
            cd_logprob = torch.log_softmax(cd_step_logits, dim=-1)
            posterior_logprob = torch.log_softmax(posterior_step_logits, dim=-1)

            pos_target_logprob = float(positive_logprob[target_token_id].item())
            cd_target_logprob = float(cd_logprob[target_token_id].item())
            posterior_target_logprob = float(posterior_logprob[target_token_id].item())

            pos_rank = _token_rank_from_logits(positive_step_logits, target_token_id)
            cd_rank = _token_rank_from_logits(cd_step_logits, target_token_id)
            sampled_rank_posterior = _token_rank_from_logits(posterior_step_logits, sampled_draft_id)
            target_in_candidate_mask = int(candidate_mask[0, reject_offset, target_token_id].item())
            sampled_in_candidate_mask = int(candidate_mask[0, reject_offset, sampled_draft_id].item())

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
                "positive_pred_id": int(positive_pred_id),
                "positive_pred_text": _token_str(tokenizer, positive_pred_id),
                "cd_pred_id": int(cd_pred_id),
                "cd_pred_text": _token_str(tokenizer, cd_pred_id),
                "posterior_pred_id": int(posterior_pred_id),
                "posterior_pred_text": _token_str(tokenizer, posterior_pred_id),
                "positive_hit": int(positive_pred_id == target_token_id),
                "cd_hit": int(cd_pred_id == target_token_id),
                "pred_changed": int(cd_pred_id != positive_pred_id),
                "positive_target_logprob": float(pos_target_logprob),
                "cd_target_logprob": float(cd_target_logprob),
                "posterior_target_logprob": float(posterior_target_logprob),
                "delta_target_logprob": float(cd_target_logprob - pos_target_logprob),
                "positive_target_rank": int(pos_rank),
                "cd_target_rank": int(cd_rank),
                "delta_target_rank": int(pos_rank - cd_rank),
                "sampled_rank_in_posterior": int(sampled_rank_posterior),
                "target_in_candidate_mask": int(target_in_candidate_mask),
                "sampled_in_candidate_mask": int(sampled_in_candidate_mask),
                "cd_overwritten_by_positive": int(reject_offset < n_keep),
                "positive_top5": _topk_token_details(positive_step_logits, tokenizer, k=5),
                "cd_top5": _topk_token_details(cd_step_logits, tokenizer, k=5),
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
    parser.add_argument("--cd-alpha", type=float, default=0.1)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
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
                    cd_alpha=args.cd_alpha,
                    beta=args.cd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    divergence_accumulator=divergence_accumulator if bs == block_size else None,
                    reject_records=reject_records if bs == block_size else None,
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
        print(f"CD hit rate @reject: {summary['cd_hit_rate'] * 100:.2f}%")
        print(f"Hit-rate gain (CD - Positive): {summary['hit_rate_gain'] * 100:+.2f}%")
    print(f"Report directory: {report_dir}")
    print(f"  - {jsonl_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {report_md_path.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")


if __name__ == "__main__":
    main()