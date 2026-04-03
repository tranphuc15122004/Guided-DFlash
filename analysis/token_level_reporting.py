from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from loguru import logger
from transformers import AutoTokenizer

from analysis import token_level_core


def build_reject_reason(record: dict[str, Any]) -> str:
    reasons: list[str] = []
    reasons.append(
        f"Reject vi draft de xuat `{record['sampled_draft_text']}` ({record['sampled_draft_id']}) "
        f"khac posterior `{record['target_token_text']}` ({record['target_token_id']})."
    )
    if record["pred_changed"] == 1:
        reasons.append(
            f"Contrastive shadow sample da doi token tu `{record['sampled_draft_text']}` sang `{record['vcd_shadow_sampled_text']}`."
        )
    else:
        reasons.append("Contrastive shadow sample khong doi token so voi original o vi tri reject.")
    if record["target_in_candidate_mask"] == 0:
        reasons.append("Token target khong nam trong candidate mask (beta filter), nen kho duoc contrastive chon.")
    if record["vcd_shadow_fix_hit"] == 1:
        reasons.append("Contrastive shadow sample trung target token o vi tri reject, cho thay co kha nang cuu reject.")
    if record["vcd_hit"] == 1 and record["positive_hit"] == 0:
        reasons.append("Contrastive sua dung token target tot hon positive.")
    elif record["vcd_hit"] == 0 and record["positive_hit"] == 1:
        reasons.append("Contrastive lam mat token target ma positive von chon dung.")
    elif record["vcd_hit"] == 0 and record["positive_hit"] == 0:
        reasons.append("Ca positive va contrastive deu chua dua target token len top-1.")
    return " ".join(reasons)


def classify_reject_taxonomy(record: dict[str, Any]) -> tuple[str, str]:
    if record["vcd_shadow_fix_hit"] == 1:
        return (
            "vcd_shadow_can_fix_reject",
            "Trong che do shadow, token sample tu contrastive trung target tai vi tri reject.",
        )
    if record["target_in_candidate_mask"] == 0:
        return (
            "target_filtered_by_candidate_mask",
            "Token target bi loai khoi candidate mask (beta filter), nen draft/contrastive kho de de xuat dung target.",
        )
    if record["positive_hit"] == 1 and record["vcd_hit"] == 0:
        return (
            "vcd_regression_from_positive",
            "Positive dung target nhung contrastive day target xuong, dan den reject.",
        )
    if record["positive_hit"] == 0 and record["vcd_hit"] == 1:
        return (
            "vcd_correct_but_sampled_mismatch",
            "Contrastive dua target len top-1 nhung token draft duoc sample van khac posterior nen van reject.",
        )
    if record["pred_changed"] == 0 and record["positive_hit"] == 0:
        return (
            "no_vcd_effect_positive_wrong",
            "Contrastive khong doi huong du doan va positive da sai target tu dau.",
        )
    if record["pred_changed"] == 1 and record["vcd_hit"] == 0:
        return (
            "vcd_shifted_but_still_wrong",
            "Contrastive co doi huong du doan nhung chua dua target len top-1.",
        )
    return (
        "posterior_mismatch_other",
        "Reject do posterior uu tien token khac voi de xuat draft tai buoc verify.",
    )


def build_reject_record(
    *,
    tokenizer: AutoTokenizer,
    sample_idx: int,
    turn_idx: int,
    decode_step: int,
    start: int,
    reject_offset: int,
    target_token_id: int,
    sampled_draft_id: int,
    shadow_sampled_id: int,
    positive_step_logits: torch.Tensor,
    contrastive_step_logits: torch.Tensor,
    posterior_step_logits: torch.Tensor,
    candidate_mask_step: torch.Tensor,
    shadow_num_samples: int,
    gen: Optional[torch.Generator],
) -> dict[str, Any]:
    positive_pred_id = int(torch.argmax(positive_step_logits).item())
    vcd_pred_id = int(torch.argmax(contrastive_step_logits).item())
    posterior_pred_id = int(torch.argmax(posterior_step_logits).item())

    positive_logprob = torch.log_softmax(positive_step_logits, dim=-1)
    vcd_logprob = torch.log_softmax(contrastive_step_logits, dim=-1)
    posterior_logprob = torch.log_softmax(posterior_step_logits, dim=-1)

    pos_target_logprob = float(positive_logprob[target_token_id].item())
    vcd_target_logprob = float(vcd_logprob[target_token_id].item())
    posterior_target_logprob = float(posterior_logprob[target_token_id].item())
    posterior_sampled_logprob = float(posterior_logprob[sampled_draft_id].item())
    posterior_shadow_sampled_logprob = float(posterior_logprob[shadow_sampled_id].item())
    positive_sampled_logprob = float(positive_logprob[sampled_draft_id].item())
    vcd_sampled_logprob = float(vcd_logprob[sampled_draft_id].item())

    pos_rank = token_level_core.token_rank_from_logits(positive_step_logits, target_token_id)
    vcd_rank = token_level_core.token_rank_from_logits(contrastive_step_logits, target_token_id)
    sampled_rank_posterior = token_level_core.token_rank_from_logits(posterior_step_logits, sampled_draft_id)
    shadow_sampled_rank_posterior = token_level_core.token_rank_from_logits(posterior_step_logits, shadow_sampled_id)

    target_in_candidate_mask = int(candidate_mask_step[target_token_id].item())
    sampled_in_candidate_mask = int(candidate_mask_step[sampled_draft_id].item())
    shadow_sampled_in_candidate_mask = int(candidate_mask_step[shadow_sampled_id].item())

    positive_rescue_prob_est = token_level_core.estimate_token_hit_probability(
        positive_step_logits,
        target_token_id,
        num_samples=shadow_num_samples,
        gen=gen,
    )
    shadow_rescue_prob_est = token_level_core.estimate_token_hit_probability(
        contrastive_step_logits,
        target_token_id,
        num_samples=shadow_num_samples,
        gen=gen,
    )

    record = {
        "sample_idx": int(sample_idx),
        "turn_idx": int(turn_idx),
        "decode_step": int(decode_step),
        "absolute_position": int(start + reject_offset + 1),
        "reject_offset_in_block": int(reject_offset),
        "target_token_id": int(target_token_id),
        "target_token_text": token_level_core.token_str(tokenizer, target_token_id),
        "sampled_draft_id": int(sampled_draft_id),
        "sampled_draft_text": token_level_core.token_str(tokenizer, sampled_draft_id),
        "vcd_shadow_sampled_id": int(shadow_sampled_id),
        "vcd_shadow_sampled_text": token_level_core.token_str(tokenizer, shadow_sampled_id),
        "positive_pred_id": int(positive_pred_id),
        "positive_pred_text": token_level_core.token_str(tokenizer, positive_pred_id),
        "vcd_pred_id": int(vcd_pred_id),
        "vcd_pred_text": token_level_core.token_str(tokenizer, vcd_pred_id),
        "posterior_pred_id": int(posterior_pred_id),
        "posterior_pred_text": token_level_core.token_str(tokenizer, posterior_pred_id),
        "positive_hit": int(positive_pred_id == target_token_id),
        "vcd_hit": int(vcd_pred_id == target_token_id),
        "pred_changed": int(vcd_pred_id != positive_pred_id),
        "positive_target_logprob": float(pos_target_logprob),
        "vcd_target_logprob": float(vcd_target_logprob),
        "posterior_target_logprob": float(posterior_target_logprob),
        "posterior_sampled_logprob": float(posterior_sampled_logprob),
        "posterior_vcd_shadow_sampled_logprob": float(posterior_shadow_sampled_logprob),
        "positive_sampled_logprob": float(positive_sampled_logprob),
        "vcd_sampled_logprob": float(vcd_sampled_logprob),
        "delta_target_logprob": float(vcd_target_logprob - pos_target_logprob),
        "target_minus_sampled_posterior_logprob": float(posterior_target_logprob - posterior_sampled_logprob),
        "target_minus_vcd_shadow_sampled_posterior_logprob": float(posterior_target_logprob - posterior_shadow_sampled_logprob),
        "positive_target_rank": int(pos_rank),
        "vcd_target_rank": int(vcd_rank),
        "delta_target_rank": int(pos_rank - vcd_rank),
        "sampled_rank_in_posterior": int(sampled_rank_posterior),
        "vcd_shadow_sampled_rank_in_posterior": int(shadow_sampled_rank_posterior),
        "target_is_posterior_argmax": int(target_token_id == posterior_pred_id),
        "posterior_prefers_target_over_sampled": int(posterior_target_logprob > posterior_sampled_logprob),
        "target_in_candidate_mask": int(target_in_candidate_mask),
        "sampled_in_candidate_mask": int(sampled_in_candidate_mask),
        "vcd_shadow_sampled_in_candidate_mask": int(shadow_sampled_in_candidate_mask),
        "vcd_shadow_fix_hit": int(shadow_sampled_id == target_token_id),
        "positive_rescue_prob_est": float(positive_rescue_prob_est),
        "vcd_shadow_rescue_prob_est": float(shadow_rescue_prob_est),
        "rescue_prob_gain_vcd_minus_positive": float(shadow_rescue_prob_est - positive_rescue_prob_est),
        "positive_top5": token_level_core.topk_token_details(positive_step_logits, tokenizer, k=5),
        "vcd_top5": token_level_core.topk_token_details(contrastive_step_logits, tokenizer, k=5),
        "posterior_top5": token_level_core.topk_token_details(posterior_step_logits, tokenizer, k=5),
    }
    record["reject_reason"] = build_reject_reason(record)
    taxonomy_label, taxonomy_desc = classify_reject_taxonomy(record)
    record["reject_taxonomy"] = taxonomy_label
    record["reject_taxonomy_desc"] = taxonomy_desc
    return record


def safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    if report_dir:
        path = Path(report_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path("analysis") / "results" / f"reject_tok_{safe_dataset_name(dataset_name)}_{ts}"
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


def maybe_create_plots(report_dir: Path, records: list[dict[str, Any]], compare_name: str = "VCD") -> list[str]:
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
    plt.bar(["Positive hit", f"{compare_name} hit", "Unchanged pred"], [pos_hit_rate, vcd_hit_rate, unchanged_rate])
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
    plt.title(f"Delta Target LogProb ({compare_name} - Positive) at Reject Position")
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
    plt.title(f"Delta Target Rank (Positive - {compare_name}) at Reject Position")
    plt.xlabel("Positive rank - contrastive rank (higher is better)")
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
    plt.xticks([0, 1], [f"{compare_name} wrong", f"{compare_name} correct"])
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


def build_summary(
    records: list[dict[str, Any]],
    *,
    dataset: str,
    seed: int,
    block_size: int,
    alpha: float,
    beta: float,
    shadow_num_samples: int,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    compare_name: str,
) -> dict[str, Any]:
    meta = {
        "dataset": dataset,
        "seed": int(seed),
        "block_size": int(block_size),
        "vcd_alpha": float(alpha),
        "vcd_beta": float(beta),
        "evaluation_mode": "positive_rollout_with_contrastive_shadow",
        "vcd_shadow_num_samples": int(shadow_num_samples),
        "negative_context_dropout": float(negative_context_dropout),
        "negative_context_noise_std": float(negative_context_noise_std),
        "compare_name": compare_name,
    }

    if not records:
        return {"meta": meta, "reject_events": 0}

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
        "meta": meta,
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


def write_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
    compare_name: str,
) -> None:
    def _fmt_topk(topk: list[dict[str, Any]], k: int = 3) -> str:
        items = topk[:k]
        return "; ".join([f"{x['text']} ({x['id']}, p={x['prob']:.4f})" for x in items])

    lines: list[str] = []
    lines.append(f"# Reject Token Analysis: Positive vs {compare_name}")
    lines.append("")

    if summary.get("reject_events", 0) == 0:
        lines.append("No reject-token event was recorded.")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append("## Overall")
    lines.append(f"- Reject events: **{summary['reject_events']}**")
    lines.append(f"- Positive hit rate: **{summary['positive_hit_rate'] * 100:.2f}%**")
    lines.append(f"- {compare_name} hit rate: **{summary['vcd_hit_rate'] * 100:.2f}%**")
    lines.append(f"- Hit-rate gain ({compare_name} - Positive): **{summary['hit_rate_gain'] * 100:+.2f}%**")
    lines.append(f"- {compare_name} shadow fix rate @reject: **{summary['vcd_shadow_fix_rate'] * 100:.2f}%**")
    lines.append(
        f"- Mean rescue-prob (Positive / {compare_name}-shadow): **{summary['positive_rescue_prob_est']['mean']:.4f} / {summary['vcd_shadow_rescue_prob_est']['mean']:.4f}**"
    )
    lines.append(
        f"- Mean rescue-prob gain ({compare_name} - Positive): **{summary['rescue_prob_gain_vcd_minus_positive']['mean']:+.4f}**"
    )
    lines.append(
        f"- Rescue-prob improved-rate (>0): **{summary['rescue_prob_gain_vcd_minus_positive']['improved_rate'] * 100:.2f}%**"
    )
    lines.append(f"- Prediction changed rate: **{summary['pred_changed_rate'] * 100:.2f}%**")
    lines.append("")

    lines.append(f"## Target LogProb Delta ({compare_name} - Positive)")
    lines.append(f"- Mean: {summary['delta_target_logprob']['mean']:.5f}")
    lines.append(f"- Median: {summary['delta_target_logprob']['median']:.5f}")
    lines.append(f"- P10 / P90: {summary['delta_target_logprob']['p10']:.5f} / {summary['delta_target_logprob']['p90']:.5f}")
    lines.append(f"- Improved rate (>0): {summary['delta_target_logprob']['improved_rate'] * 100:.2f}%")
    lines.append("")

    lines.append(f"## Target Rank Delta (Positive - {compare_name})")
    lines.append(f"- Mean: {summary['delta_target_rank']['mean']:.3f}")
    lines.append(f"- Median: {summary['delta_target_rank']['median']:.3f}")
    lines.append(f"- P10 / P90: {summary['delta_target_rank']['p10']:.3f} / {summary['delta_target_rank']['p90']:.3f}")
    lines.append(f"- Improved rate (>0): {summary['delta_target_rank']['improved_rate'] * 100:.2f}%")
    lines.append("")

    lines.append("## Reject Taxonomy")
    lines.append("| Taxonomy | Count | Rate |")
    lines.append("|---|---:|---:|")
    rc = summary["reject_cause_indicators"]
    taxonomy_counts = rc.get("taxonomy_counts", {})
    taxonomy_rates = rc.get("taxonomy_rates", {})
    for key in sorted(taxonomy_counts.keys(), key=lambda k: taxonomy_counts[k], reverse=True):
        lines.append(f"| {key} | {taxonomy_counts[key]} | {taxonomy_rates.get(key, 0.0) * 100:.2f}% |")
    lines.append("")

    if plot_files:
        lines.append("## Plots")
        for name in plot_files:
            lines.append(f"![{name}]({name})")
        lines.append("")

    lines.append("## Top Improved Reject Events")
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |")
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
    lines.append("| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |")
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
            f"- Positive pred: `{rec['positive_pred_text']}` ({rec['positive_pred_id']}), {compare_name} pred: `{rec['vcd_pred_text']}` ({rec['vcd_pred_id']})"
        )
        lines.append(f"- Reason: {rec['reject_reason']}")
        lines.append(
            f"- Candidate mask: target_in={rec['target_in_candidate_mask']}, sampled_in={rec['sampled_in_candidate_mask']}"
        )
        lines.append(
            f"- Delta target logprob/rank: {rec['delta_target_logprob']:.4f} / {rec['delta_target_rank']}"
        )
        lines.append(f"- Positive top-3: {_fmt_topk(rec['positive_top5'], 3)}")
        lines.append(f"- {compare_name} top-3: {_fmt_topk(rec['vcd_top5'], 3)}")
        lines.append(f"- Posterior top-3: {_fmt_topk(rec['posterior_top5'], 3)}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
