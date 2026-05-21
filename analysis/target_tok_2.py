#!/usr/bin/env python
"""
Offline analysis of collected data from target_tok.py.

Reads collected_output_ids.pt and collected_target_tok_predictions.pt,
reconstructs per-block target references, and computes statistics
about draft-target agreement, acceptance lengths, and what-if scenarios.

Usage:
    python -m analysis.target_tok_2 \\
        --output-ids-path path/to/collected_output_ids.pt \\
        --ar-preds-path path/to/collected_target_tok_predictions.pt \\
        --block-size 16 \\
        --output-dir analysis/results/my_analysis
"""

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


MASK_TOKEN_ID = -1  # placeholder; will be inferred from data


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    """Rank of token_id in logits (1 = highest)."""
    return int((logits_1d > logits_1d[token_id]).sum().item()) + 1


def _describe(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "mean": 0.0, "std": 0.0, "min": 0.0, "median": 0.0, "max": 0.0}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "median": float(np.median(values)),
        "max": float(values.max()),
    }


def reconstruct_blocks(
    output_ids_list: list[torch.Tensor],
    ar_preds_list: list[torch.Tensor],
    block_size: int,
    mask_token_id: int,
) -> list[dict[str, Any]]:
    """
    Reconstruct per-block data from saved collectors.

    Each block record contains:
      - block_start: absolute start position in output_ids
      - acceptance_length: number of accepted draft tokens
      - bonus_token: the posterior token at rejection point
      - block_output_ids[1:]: the draft-sampled tokens
      - tmp_target_tok_ids[1:]: the target reference tokens
      - draft_match[1:]: per-position whether draft == target
    """
    blocks: list[dict[str, Any]] = []

    for ar_stacked in ar_preds_list:
        # ar_stacked shape: (2, block_size+1)
        # row 0 = absolute positions, row 1 = predicted token IDs
        positions = ar_stacked[0].long()
        pred_ids = ar_stacked[1].long()

        # Count valid AR steps (non-mask entries)
        valid_mask = positions != mask_token_id
        num_ar_steps = int(valid_mask.sum().item())

        if num_ar_steps == 0 and positions.numel() > 0:
            # Fully-accepted block: no AR predictions needed.
            # Still include in stats for acceptance distribution.
            acceptance_length = block_size - 1
            block_start = int(positions[0].item()) - 1 - acceptance_length
            start = block_start + acceptance_length + 1
        elif num_ar_steps == 0:
            continue
        else:
            first_abs_pos = int(positions[0].item())
            start = first_abs_pos - 1
            acceptance_length = block_size - num_ar_steps - 2
            if acceptance_length < 0:
                acceptance_length = 0
            block_start = start - acceptance_length - 1

        bonus_token = mask_token_id  # default, set in matching block below

        # Find the output_ids tensor that contains this block
        matched_output: Optional[torch.Tensor] = None
        for out_t in output_ids_list:
            out_t = out_t.squeeze(0)  # (seq_len,)
            if block_start >= 0 and block_start + block_size <= out_t.shape[0]:
                # Check if the bonus token matches
                if start < out_t.shape[0] and int(out_t[start].item()) != mask_token_id:
                    matched_output = out_t
                    break

        if matched_output is None:
            # Try any tensor that is long enough
            for out_t in output_ids_list:
                out_t = out_t.squeeze(0)
                if block_start >= 0 and block_start + block_size <= out_t.shape[0]:
                    matched_output = out_t
                    break

        if matched_output is None:
            continue

        out = matched_output  # 1D tensor

        # Extract block_output_ids from output_ids
        block_tokens = out[block_start : block_start + block_size].clone()  # (block_size,)

        # Build tmp_target_tok_ids
        tmp_target = torch.full((block_size,), mask_token_id, dtype=torch.long)
        # Positions 0..acceptance_length: from block_output_ids (accepted)
        tmp_target[: acceptance_length + 1] = block_tokens[: acceptance_length + 1]
        # Position acceptance_length+1: bonus token (if space in block)
        bonus_token = int(out[start].item()) if start < out.shape[0] else mask_token_id
        if acceptance_length + 1 < block_size:
            tmp_target[acceptance_length + 1] = bonus_token
        # Positions acceptance_length+2..: AR predictions
        if num_ar_steps > 0 and acceptance_length + 2 < block_size:
            n_copy = min(num_ar_steps, block_size - acceptance_length - 2)
            tmp_target[acceptance_length + 2 : acceptance_length + 2 + n_copy] = pred_ids[:n_copy]

        # Per-position draft_match
        draft_match = torch.zeros(block_size, dtype=torch.bool)
        for p in range(1, block_size):
            if tmp_target[p].item() != mask_token_id:
                draft_match[p] = (block_tokens[p] == tmp_target[p])

        blocks.append({
            "block_start": int(block_start),
            "acceptance_length": int(acceptance_length),
            "num_ar_steps": int(num_ar_steps),
            "bonus_token": int(bonus_token),
            "block_tokens": block_tokens.clone(),
            "tmp_target": tmp_target.clone(),
            "draft_match": draft_match.clone(),
        })

    return blocks


def compute_statistics(
    blocks: list[dict[str, Any]],
    block_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """
    Compute per-position, per-cohort, and overall statistics.

    Returns: (flat_rows, pos_summary_rows, cohort_summary_rows)
    """
    flat_rows: list[dict[str, Any]] = []

    for rec in blocks:
        al = rec["acceptance_length"]
        block_tokens = rec["block_tokens"]
        tmp_target = rec["tmp_target"]
        draft_match = rec["draft_match"]

        for p in range(1, block_size):
            target_id = int(tmp_target[p].item())
            if target_id == MASK_TOKEN_ID:
                continue

            draft_id = int(block_tokens[p].item())

            if p <= al:
                cohort = "verify"
            elif p == al + 1:
                cohort = "bonus"
            else:
                cohort = "ar_suffix"

            flat_rows.append({
                "block_start": rec["block_start"],
                "acceptance_length": al,
                "block_position": p,
                "cohort": cohort,
                "target_id": target_id,
                "draft_id": draft_id,
                "draft_matches_target": int(draft_match[p].item()),
            })

    if not flat_rows:
        return [], [], []

    # ── Per-position summary ──
    pos_summary_rows: list[dict[str, Any]] = []
    for pos in sorted(set(r["block_position"] for r in flat_rows)):
        subset = [r for r in flat_rows if r["block_position"] == pos]
        match_rates = np.array([r["draft_matches_target"] for r in subset], dtype=float)
        cohort = subset[0]["cohort"]
        pos_summary_rows.append({
            "block_position": pos,
            "cohort": cohort,
            "count": len(subset),
            "draft_match_rate": float(match_rates.mean()),
            "match_count": int(match_rates.sum()),
        })

    # ── Per-cohort summary ──
    cohort_summary_rows: list[dict[str, Any]] = []
    for cohort in ["verify", "bonus", "ar_suffix"]:
        subset = [r for r in flat_rows if r["cohort"] == cohort]
        if not subset:
            continue
        match_rates = np.array([r["draft_matches_target"] for r in subset], dtype=float)
        cohort_summary_rows.append({
            "cohort": cohort,
            "count": len(subset),
            "draft_match_rate": float(match_rates.mean()),
            "match_count": int(match_rates.sum()),
        })

    return flat_rows, pos_summary_rows, cohort_summary_rows


def compute_acceptance_stats(
    blocks: list[dict[str, Any]],
    block_size: int,
) -> dict[str, Any]:
    """Compute acceptance length distribution and what-if analysis."""
    if not blocks:
        return {}

    al_vals = np.array([r["acceptance_length"] for r in blocks], dtype=float)
    block_counts = max(len(blocks), 1)

    # What-if: if draft matched target exactly, what would acceptance length be?
    whatif_lengths: list[int] = []
    for rec in blocks:
        dm = rec["draft_match"]
        # Cumulative product: first position where draft != target
        mismatches = (dm[1:] == 0).nonzero(as_tuple=True)[0]
        if mismatches.numel() == 0:
            whatif_lengths.append(block_size - 1)
        else:
            whatif_lengths.append(int(mismatches[0].item()))  # 0-based → acceptance_length

    whatif = np.array(whatif_lengths, dtype=float)

    return {
        "num_blocks": int(len(blocks)),
        "acceptance_length": _describe(al_vals),
        "whatif_acceptance_length": _describe(whatif),
        "acceptance_histogram": {
            str(b): float((al_vals == b).mean())
            for b in range(block_size + 1)
        },
        "mean_improvement": float(whatif.mean() - al_vals.mean()) if al_vals.size else 0.0,
    }


def build_report_md(
    acc_stats: dict[str, Any],
    pos_rows: list[dict[str, Any]],
    cohort_rows: list[dict[str, Any]],
    block_size: int,
) -> str:
    """Generate markdown report."""
    lines: list[str] = []
    lines.append("# Block Analysis Report (Offline)")
    lines.append("")
    lines.append(f"Total blocks: **{acc_stats.get('num_blocks', 0)}**")
    al = acc_stats.get("acceptance_length", {})
    lines.append(f"Acceptance length: mean **{al.get('mean', 0.0):.2f}**, median **{al.get('median', 0.0):.2f}**")
    wa = acc_stats.get("whatif_acceptance_length", {})
    lines.append(f"What-if acceptance length: mean **{wa.get('mean', 0.0):.2f}**, median **{wa.get('median', 0.0):.2f}**")
    lines.append(f"Mean improvement with AR target: **{acc_stats.get('mean_improvement', 0.0):+.3f}** tokens/block")
    lines.append("")

    # Histogram
    lines.append("### Acceptance Length Histogram")
    lines.append("| len | rate |")
    lines.append("|---:|---:|")
    hist = acc_stats.get("acceptance_histogram", {})
    for b in range(block_size + 1):
        pct = float(hist.get(str(b), 0.0)) * 100
        if pct > 0.5:
            lines.append(f"| {b} | {pct:.1f}% |")
    lines.append("")

    # Cohort summary
    if cohort_rows:
        lines.append("## Per-Cohort Summary")
        lines.append("| cohort | count | draft_match_rate | matches |")
        lines.append("|---|---:|---:|---:|")
        for r in cohort_rows:
            lines.append(
                f"| {r['cohort']} | {r['count']} | "
                f"{r['draft_match_rate'] * 100:.2f}% | {r['match_count']} |"
            )
        lines.append("")

    # Per-position profile
    if pos_rows:
        lines.append("## Per-Position Draft Match Profile")
        lines.append("| pos | cohort | count | draft_match_rate | matches |")
        lines.append("|---:|---|---:|---:|---:|")
        for r in pos_rows:
            lines.append(
                f"| {r['block_position']} | {r['cohort']} | {r['count']} | "
                f"{r['draft_match_rate'] * 100:.2f}% | {r['match_count']} |"
            )
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Offline analysis of target token collection data."
    )
    parser.add_argument(
        "--output-ids-path",
        type=str,
        required=True,
        help="Path to collected_output_ids.pt",
    )
    parser.add_argument(
        "--ar-preds-path",
        type=str,
        required=True,
        help="Path to collected_target_tok_predictions.pt",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="Block size used during generation.",
    )
    parser.add_argument(
        "--mask-token-id",
        type=int,
        default=None,
        help="Mask token ID. If not set, inferred from data.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for analysis artifacts. "
             "If omitted, uses the parent of output-ids-path.",
    )
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading output IDs from: {args.output_ids_path}")
    output_ids_list = torch.load(args.output_ids_path, map_location="cpu")
    if not isinstance(output_ids_list, list):
        output_ids_list = [output_ids_list]
    print(f"  → {len(output_ids_list)} sample(s) loaded")

    print(f"Loading AR predictions from: {args.ar_preds_path}")
    ar_preds_list = torch.load(args.ar_preds_path, map_location="cpu")
    if not isinstance(ar_preds_list, list):
        ar_preds_list = [ar_preds_list]
    print(f"  → {len(ar_preds_list)} AR prediction(s) loaded")

    # Infer mask_token_id from data
    global MASK_TOKEN_ID
    if args.mask_token_id is not None:
        MASK_TOKEN_ID = args.mask_token_id
    else:
        # Find the most common value that appears as "unfilled" in AR preds
        all_vals = torch.cat([t.flatten() for t in ar_preds_list])
        # The mask token is likely the most frequent value in row 0
        # (positions row = 0 in stacked tensor)
        row0_vals = torch.cat([t[0].flatten() for t in ar_preds_list])
        unique, counts = torch.unique(row0_vals, return_counts=True)
        if unique.numel() > 0:
            MASK_TOKEN_ID = int(unique[counts.argmax()].item())
        else:
            MASK_TOKEN_ID = -1
        print(f"  → Inferred mask_token_id: {MASK_TOKEN_ID}")

    block_size = args.block_size

    # ── Reconstruct blocks ──
    print("Reconstructing block data...")
    blocks = reconstruct_blocks(
        output_ids_list=output_ids_list,
        ar_preds_list=ar_preds_list,
        block_size=block_size,
        mask_token_id=MASK_TOKEN_ID,
    )
    print(f"  → {len(blocks)} block(s) reconstructed")

    if not blocks:
        print("No blocks could be reconstructed. Check your data paths and block_size.")
        return

    # ── Compute statistics ──
    print("Computing statistics...")
    flat_rows, pos_rows, cohort_rows = compute_statistics(blocks, block_size)
    acc_stats = compute_acceptance_stats(blocks, block_size)

    # ── Determine output directory ──
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.output_ids_path).parent
    os.makedirs(output_dir, exist_ok=True)

    # ── Write artifacts ──
    # Flat records
    jsonl_path = output_dir / "offline_block_analysis.jsonl"
    with open(jsonl_path, "w") as f:
        for row in flat_rows:
            f.write(json.dumps(row) + "\n")
    print(f"Written: {jsonl_path}")

    csv_path = output_dir / "offline_block_analysis.csv"
    if flat_rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(flat_rows[0].keys()))
            w.writeheader()
            w.writerows(flat_rows)
        print(f"Written: {csv_path}")

    # Per-position CSV
    pos_csv_path = output_dir / "offline_block_analysis_by_position.csv"
    if pos_rows:
        with open(pos_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(pos_rows[0].keys()))
            w.writeheader()
            w.writerows(pos_rows)
        print(f"Written: {pos_csv_path}")

    # Per-cohort CSV
    cohort_csv_path = output_dir / "offline_block_analysis_by_cohort.csv"
    if cohort_rows:
        with open(cohort_csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cohort_rows[0].keys()))
            w.writeheader()
            w.writerows(cohort_rows)
        print(f"Written: {cohort_csv_path}")

    # Summary JSON
    summary = {
        "meta": {
            "block_size": block_size,
            "mask_token_id": MASK_TOKEN_ID,
            "num_output_samples": len(output_ids_list),
            "num_ar_predictions": len(ar_preds_list),
            "num_reconstructed_blocks": len(blocks),
        },
        "acceptance_stats": {k: v for k, v in acc_stats.items() if k != "acceptance_histogram"},
        "acceptance_histogram": acc_stats.get("acceptance_histogram", {}),
        "cohorts": {r["cohort"]: dict(r) for r in cohort_rows},
        "by_position": pos_rows,
    }
    summary_path = output_dir / "offline_block_analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Written: {summary_path}")

    # Report markdown
    report_md = build_report_md(acc_stats, pos_rows, cohort_rows, block_size)
    report_path = output_dir / "offline_block_analysis_report.md"
    with open(report_path, "w") as f:
        f.write(report_md)
    print(f"Written: {report_path}")

    # ── Console summary ──
    print("\n── Console Summary ──")
    print(f"Blocks reconstructed: {len(blocks)}")
    al = acc_stats.get("acceptance_length", {})
    print(f"Acceptance length: mean {al.get('mean', 0.0):.2f}, median {al.get('median', 0.0):.2f}")
    wa = acc_stats.get("whatif_acceptance_length", {})
    print(f"What-if acceptance length: mean {wa.get('mean', 0.0):.2f}, median {wa.get('median', 0.0):.2f}")
    print(f"Mean improvement with AR target: {acc_stats.get('mean_improvement', 0.0):+.3f} tokens/block")
    for r in cohort_rows:
        print(f"  {r['cohort']}: draft_match_rate={r['draft_match_rate']*100:.2f}% (count={r['count']})")
    print(f"\nDone. All artifacts in: {output_dir}")


if __name__ == "__main__":
    main()
