"""
Alpha analysis script: evaluate different alpha values on collected logits and report statistics.

Loads collected reject-event logits shards and tests a range of alpha values to find
the best one for contrastive decoding, with detailed per-sample and aggregate metrics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import torch
import torch.nn.functional as F
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    plt = None
    HAS_MATPLOTLIB = False

from rich import print as rprint
from rich.console import Console
from rich.table import Table
from tqdm import tqdm


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def _discover_rank_dirs(collect_dir: Path) -> list[Path]:
    if (collect_dir / "logit_shards").exists():
        return [collect_dir]
    rank_dirs = sorted([p for p in collect_dir.glob("rank_*") if p.is_dir()])
    return rank_dirs


def _load_collection_meta(rank_dirs: list[Path]) -> dict[str, Any]:
    for rank_dir in rank_dirs:
        summary_path = rank_dir / "reject_token_summary.json"
        if summary_path.exists():
            return json.loads(summary_path.read_text(encoding="utf-8"))
    return {}


def _collect_shards(rank_dirs: list[Path]) -> list[Path]:
    shards: list[Path] = []
    for rank_dir in rank_dirs:
        shard_dir = rank_dir / "logit_shards"
        if not shard_dir.exists():
            continue
        shards.extend(sorted(shard_dir.glob("shard_*.npz")))
    return shards


def _count_records_in_shards(shards: list[Path]) -> int:
    total = 0
    for path in shards:
        with np.load(path) as data:
            total += int(data["target_token_id"].shape[0])
    return total


def _estimate_num_batches(shards: list[Path], batch_size: int, max_batches: int) -> int:
    total_batches = 0
    for path in shards:
        with np.load(path) as data:
            num_rows = int(data["target_token_id"].shape[0])
            total_batches += int(math.ceil(num_rows / batch_size))
            if max_batches > 0 and total_batches >= max_batches:
                return max_batches
    return total_batches


def _save_visualizations(
    results: list[dict[str, float]],
    output_dir: Path,
    console: Console,
) -> list[Path]:
    if not results:
        return []
    if not HAS_MATPLOTLIB or plt is None:
        console.print("[yellow]matplotlib is not available; skip visualization.[/yellow]")
        return []

    arr = sorted(results, key=lambda x: x["alpha"])
    alphas = np.array([float(x["alpha"]) for x in arr], dtype=np.float64)
    ce_mean = np.array([float(x["ce_mean"]) for x in arr], dtype=np.float64)
    ce_std = np.array([float(x["ce_std"]) for x in arr], dtype=np.float64)
    top1 = np.array([float(x["top1"]) * 100.0 for x in arr], dtype=np.float64)
    top5 = np.array([float(x["top5"]) * 100.0 for x in arr], dtype=np.float64)
    top10 = np.array([float(x["top10"]) * 100.0 for x in arr], dtype=np.float64)
    tbd = np.array([float(x["target_beats_draft"]) * 100.0 for x in arr], dtype=np.float64)
    lifted_rate = np.array([float(x["lifted_to_top1_rate"]) * 100.0 for x in arr], dtype=np.float64)
    lifted_count = np.array([float(x["lifted_to_top1_count"]) for x in arr], dtype=np.float64)

    saved_paths: list[Path] = []

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=150)

    ax = axes[0, 0]
    ax.plot(alphas, ce_mean, marker="o", linewidth=1.8, color="#1f77b4", label="CE mean")
    ax.fill_between(
        alphas,
        ce_mean - ce_std,
        ce_mean + ce_std,
        color="#1f77b4",
        alpha=0.2,
        label="CE std",
    )
    ax.set_title("Cross-Entropy vs Alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("CE loss")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[0, 1]
    ax.plot(alphas, top1, marker="o", linewidth=1.8, label="Top-1")
    ax.plot(alphas, top5, marker="o", linewidth=1.8, label="Top-5")
    ax.plot(alphas, top10, marker="o", linewidth=1.8, label="Top-10")
    ax.set_title("Top-k Accuracy vs Alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")

    ax = axes[1, 0]
    ax.plot(alphas, tbd, marker="o", linewidth=1.8, color="#2ca02c")
    ax.set_title("Target Beats Draft vs Alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Rate (%)")
    ax.grid(alpha=0.25)

    ax = axes[1, 1]
    ax.plot(alphas, lifted_rate, marker="o", linewidth=1.8, color="#9467bd", label="Lifted rate (%)")
    ax2 = ax.twinx()
    ax2.bar(alphas, lifted_count, alpha=0.25, color="#8c564b", width=0.04, label="Lifted count")
    ax.set_title("Lifted Target-to-Top1 vs Alpha")
    ax.set_xlabel("alpha")
    ax.set_ylabel("Lifted rate (%)")
    ax2.set_ylabel("Lifted count")
    ax.grid(alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")

    fig.tight_layout()
    summary_plot_path = output_dir / "alpha_analysis_plots.png"
    fig.savefig(summary_plot_path, bbox_inches="tight")
    plt.close(fig)
    saved_paths.append(summary_plot_path)

    fig2, ax2 = plt.subplots(figsize=(10, 4), dpi=150)
    width = 0.38
    ax2.bar(alphas - width / 2, ce_mean, width=width, label="CE mean")
    ax2.bar(alphas + width / 2, tbd, width=width, label="Target>Draft (%)")
    ax2.set_title("CE Mean and Target>Draft by Alpha")
    ax2.set_xlabel("alpha")
    ax2.grid(alpha=0.25, axis="y")
    ax2.legend(loc="best")
    fig2.tight_layout()
    compare_plot_path = output_dir / "alpha_analysis_ce_vs_tbd.png"
    fig2.savefig(compare_plot_path, bbox_inches="tight")
    plt.close(fig2)
    saved_paths.append(compare_plot_path)

    return saved_paths


def _iter_npz_batches(
    shards: list[Path],
    *,
    batch_size: int,
    seed: int,
    max_batches: int,
) -> Iterator[dict[str, np.ndarray]]:
    if not shards:
        return

    shard_paths = list(shards)
    rng = random.Random(seed)
    rng.shuffle(shard_paths)

    yielded = 0
    for path in shard_paths:
        with np.load(path) as data:
            num_rows = int(data["target_token_id"].shape[0])
            indices = np.arange(num_rows, dtype=np.int64)
            np_rng = np.random.default_rng(seed + yielded)
            np_rng.shuffle(indices)

            for start in range(0, num_rows, batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = {
                    "positive_logits": data["positive_logits"][batch_idx],
                    "negative_logits": data["negative_logits"][batch_idx],
                    "target_logits": data["target_logits"][batch_idx],
                    "target_token_id": data["target_token_id"][batch_idx],
                    "draft_token_id": data["draft_token_id"][batch_idx],
                }
                yield batch
                yielded += 1
                if max_batches > 0 and yielded >= max_batches:
                    return


def _build_cd_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)
    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def _build_cd_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    alpha: float,
    beta: float,
    apply_mask: bool,
) -> torch.Tensor:
    log_p1 = torch.log_softmax(positive_logits, dim=-1)
    log_p2 = torch.log_softmax(negative_logits, dim=-1)
    cd_logits = log_p1 - alpha * log_p2
    if apply_mask:
        candidate_mask = _build_cd_candidate_mask(positive_logits, beta)
        cd_logits = cd_logits.masked_fill(~candidate_mask, torch.finfo(cd_logits.dtype).min)
    return cd_logits


def evaluate_alpha(
    *,
    shards: list[Path],
    batch_size: int,
    device: torch.device,
    alpha_value: float,
    beta: float,
    apply_mask: bool,
    seed: int,
    max_batches: int,
    console: Console | None = None,
    log_every_batches: int = 0,
) -> dict[str, float]:
    """Evaluate a single alpha value on all shards."""
    stats_list: list[dict[str, float]] = []
    count = 0
    processed_batches = 0

    with torch.no_grad():
        for batch in _iter_npz_batches(
            shards,
            batch_size=batch_size,
            seed=seed,
            max_batches=max_batches,
        ):
            positive_logits = torch.from_numpy(batch["positive_logits"]).to(
                device=device, dtype=torch.float32
            )
            negative_logits = torch.from_numpy(batch["negative_logits"]).to(
                device=device, dtype=torch.float32
            )
            target_logits = torch.from_numpy(batch["target_logits"]).to(
                device=device, dtype=torch.float32
            )
            target_token_id = torch.from_numpy(batch["target_token_id"]).to(
                device=device, dtype=torch.long
            )
            draft_token_id = torch.from_numpy(batch["draft_token_id"]).to(
                device=device, dtype=torch.long
            )

            cd_logits = _build_cd_logits(
                positive_logits=positive_logits,
                negative_logits=negative_logits,
                alpha=alpha_value,
                beta=beta,
                apply_mask=apply_mask,
            )

            # Cross-entropy loss
            ce = F.cross_entropy(cd_logits, target_token_id, reduction="none")

            # Top-k accuracy
            pred = cd_logits.argmax(dim=-1)
            top1 = (pred == target_token_id).float()
            positive_top1_pred = positive_logits.argmax(dim=-1)
            positive_top1_match = positive_top1_pred == target_token_id
            lifted_to_top1 = ((~positive_top1_match) & (pred == target_token_id)).float()

            k5 = min(5, cd_logits.shape[-1])
            top5_idx = torch.topk(cd_logits, k=k5, dim=-1).indices
            top5 = (top5_idx == target_token_id.unsqueeze(-1)).any(dim=-1).float()

            k10 = min(10, cd_logits.shape[-1])
            top10_idx = torch.topk(cd_logits, k=k10, dim=-1).indices
            top10 = (top10_idx == target_token_id.unsqueeze(-1)).any(dim=-1).float()

            # Target beats draft
            target_score = cd_logits.gather(1, target_token_id.unsqueeze(-1)).squeeze(-1)
            draft_score = cd_logits.gather(1, draft_token_id.unsqueeze(-1)).squeeze(-1)
            target_beats_draft = (target_score > draft_score).float()

            for i in range(len(target_token_id)):
                stats_list.append(
                    {
                        "ce": float(ce[i].item()),
                        "top1": float(top1[i].item()),
                        "top5": float(top5[i].item()),
                        "top10": float(top10[i].item()),
                        "target_beats_draft": float(target_beats_draft[i].item()),
                        "lifted_to_top1": float(lifted_to_top1[i].item()),
                    }
                )
                count += 1

            processed_batches += 1
            if (
                console is not None
                and log_every_batches > 0
                and processed_batches % log_every_batches == 0
            ):
                console.print(
                    f"[dim]alpha={alpha_value:.4f} | processed_batches={processed_batches} | processed_samples={count}[/dim]"
                )

    # Aggregate statistics
    if not stats_list:
        return {
            "alpha": alpha_value,
            "num_samples": 0,
            "ce_mean": 0.0,
            "ce_std": 0.0,
            "ce_min": 0.0,
            "ce_max": 0.0,
            "top1": 0.0,
            "top5": 0.0,
            "top10": 0.0,
            "target_beats_draft": 0.0,
            "lifted_to_top1_count": 0.0,
            "lifted_to_top1_rate": 0.0,
        }

    ce_values = np.array([s["ce"] for s in stats_list])
    top1_values = np.array([s["top1"] for s in stats_list])
    top5_values = np.array([s["top5"] for s in stats_list])
    top10_values = np.array([s["top10"] for s in stats_list])
    tbd_values = np.array([s["target_beats_draft"] for s in stats_list])
    lifted_values = np.array([s["lifted_to_top1"] for s in stats_list])

    return {
        "alpha": float(alpha_value),
        "num_samples": int(count),
        "ce_mean": float(ce_values.mean()),
        "ce_std": float(ce_values.std()),
        "ce_min": float(ce_values.min()),
        "ce_max": float(ce_values.max()),
        "top1": float(top1_values.mean()),
        "top5": float(top5_values.mean()),
        "top10": float(top10_values.mean()),
        "target_beats_draft": float(tbd_values.mean()),
        "lifted_to_top1_count": int(lifted_values.sum()),
        "lifted_to_top1_rate": float(lifted_values.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze alpha values on collected logits.")
    parser.add_argument(
        "--collect-dir", type=str, required=True, help="Path to alpha collection root."
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Path to write analysis results."
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", help="auto/cuda/cpu")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument(
        "--no-candidate-mask",
        action="store_true",
        help="Disable candidate mask when computing CD logits.",
    )
    parser.add_argument(
        "--alpha-values",
        type=str,
        default="0.0,0.5,1.0,1.5,2.0",
        help="Comma-separated alpha values to test, or 'start:end:step' range.",
    )
    parser.add_argument("--max-batches", type=int, default=0, help="Max batches to evaluate.")
    parser.add_argument(
        "--log-every-batches",
        type=int,
        default=100,
        help="Print progress every N batches for each alpha (<=0 to disable).",
    )
    parser.add_argument(
        "--optimize-by",
        type=str,
        default="ce",
        choices=[
            "ce",
            "top1",
            "target_beats_draft",
            "lifted_top1_count",
            "lifted_top1_rate",
        ],
        help="Primary objective for selecting recommended alpha.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed)
    device = _resolve_device(args.device)

    collect_dir = Path(args.collect_dir)
    if not collect_dir.exists():
        raise FileNotFoundError(f"collect-dir does not exist: {collect_dir}")

    rank_dirs = _discover_rank_dirs(collect_dir)
    if not rank_dirs:
        raise FileNotFoundError(f"No rank directories found in {collect_dir}")

    meta_summary = _load_collection_meta(rank_dirs)
    meta = meta_summary.get("meta", {})
    shards = _collect_shards(rank_dirs)
    if not shards:
        raise FileNotFoundError(f"No shard_*.npz found under {collect_dir}")

    beta = float(args.beta if args.beta is not None else meta.get("cd_beta", 0.0))

    # Parse alpha values
    alpha_spec = args.alpha_values.strip()
    if ":" in alpha_spec:
        chunks = [x.strip() for x in alpha_spec.split(":")]
        if len(chunks) != 3:
            raise ValueError("Alpha range format must be start:end:step")
        start = float(chunks[0])
        end = float(chunks[1])
        step = float(chunks[2])
        if step <= 0:
            raise ValueError("Alpha step must be > 0")
        alpha_values = []
        val = start
        while val <= end + 1e-12:
            alpha_values.append(float(val))
            val += step
    else:
        alpha_values = [float(x.strip()) for x in alpha_spec.split(",") if x.strip()]

    dataset_name = str(meta.get("dataset", collect_dir.name))
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = (
            Path("alpha_adjusting")
            / "results"
            / f"alpha_analysis_{_safe_dataset_name(dataset_name)}_{ts}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    console = Console()
    console.print(f"Device: {device}")
    console.print(f"Total shards: {len(shards)}")
    console.print(f"Total examples: {_count_records_in_shards(shards)}")
    est_batches = _estimate_num_batches(shards, args.batch_size, args.max_batches)
    console.print(f"Estimated batches per alpha: {est_batches}")
    console.print(f"Testing {len(alpha_values)} alpha values: {alpha_values[:5]}...")
    console.print()

    results = []
    for alpha_value in tqdm(alpha_values, desc="Evaluating alphas"):
        start_t = time.perf_counter()
        console.print(f"[bold cyan]Start alpha[/bold cyan] {alpha_value:.4f}")
        metrics = evaluate_alpha(
            shards=shards,
            batch_size=args.batch_size,
            device=device,
            alpha_value=alpha_value,
            beta=beta,
            apply_mask=(not args.no_candidate_mask),
            seed=args.seed,
            max_batches=args.max_batches,
            console=console,
            log_every_batches=args.log_every_batches,
        )
        elapsed = time.perf_counter() - start_t
        console.print(
            f"[green]Done alpha[/green] {alpha_value:.4f} | "
            f"samples={metrics['num_samples']} | ce={metrics['ce_mean']:.4f} | "
            f"top1={metrics['top1'] * 100:.2f}% | t>d={metrics['target_beats_draft'] * 100:.2f}% | "
            f"lift={int(metrics['lifted_to_top1_count'])} ({metrics['lifted_to_top1_rate'] * 100:.2f}%) | "
            f"time={elapsed:.1f}s"
        )
        results.append(metrics)

    # Sort by CE loss
    results_by_loss = sorted(results, key=lambda x: x["ce_mean"])
    best_alpha_loss = results_by_loss[0]
    best_alpha_tbd = max(results, key=lambda x: x["target_beats_draft"])
    best_alpha_top1 = max(results, key=lambda x: x["top1"])
    best_alpha_lift_count = max(results, key=lambda x: (x["lifted_to_top1_count"], x["top1"]))
    best_alpha_lift_rate = max(results, key=lambda x: (x["lifted_to_top1_rate"], x["top1"]))

    if args.optimize_by == "ce":
        best_alpha_primary = best_alpha_loss
        primary_desc = "lowest CE loss"
    elif args.optimize_by == "top1":
        best_alpha_primary = best_alpha_top1
        primary_desc = "highest Top-1 accuracy"
    elif args.optimize_by == "target_beats_draft":
        best_alpha_primary = best_alpha_tbd
        primary_desc = "highest Target>Draft rate"
    elif args.optimize_by == "lifted_top1_count":
        best_alpha_primary = best_alpha_lift_count
        primary_desc = "max lifted-to-top1 count"
    else:
        best_alpha_primary = best_alpha_lift_rate
        primary_desc = "max lifted-to-top1 rate"

    # Write CSV
    csv_path = output_dir / "alpha_analysis_results.csv"
    if results:
        fieldnames = list(results[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

    # Display table
    table = Table(title="Alpha Analysis Results")
    table.add_column("Alpha", style="cyan")
    table.add_column("CE Loss", justify="right")
    table.add_column("Top-1 Acc", justify="right")
    table.add_column("Top-5 Acc", justify="right")
    table.add_column("Top-10 Acc", justify="right")
    table.add_column("Target > Draft", justify="right")
    table.add_column("Lifted->Top1", justify="right")
    table.add_column("# Samples", justify="right")

    for res in results_by_loss:
        table.add_row(
            f"{res['alpha']:.4f}",
            f"{res['ce_mean']:.4f} ± {res['ce_std']:.4f}",
            f"{res['top1'] * 100:.2f}%",
            f"{res['top5'] * 100:.2f}%",
            f"{res['top10'] * 100:.2f}%",
            f"{res['target_beats_draft'] * 100:.2f}%",
            f"{int(res['lifted_to_top1_count'])} ({res['lifted_to_top1_rate'] * 100:.2f}%)",
            str(res["num_samples"]),
        )

    console.print(table)
    console.print()

    # Summary report
    summary = {
        "meta": {
            "collect_dir": str(collect_dir),
            "dataset": dataset_name,
            "beta": float(beta),
            "candidate_mask_enabled": bool(not args.no_candidate_mask),
            "optimize_by": args.optimize_by,
            "num_shards": int(len(shards)),
            "num_samples": int(best_alpha_loss["num_samples"]),
            "alpha_values_tested": [float(x) for x in alpha_values],
        },
        "best_by_primary_objective": {
            "objective": args.optimize_by,
            "description": primary_desc,
            "result": best_alpha_primary,
        },
        "best_by_ce_loss": best_alpha_loss,
        "best_by_target_beats_draft": best_alpha_tbd,
        "best_by_top1_accuracy": best_alpha_top1,
        "best_by_lifted_to_top1_count": best_alpha_lift_count,
        "best_by_lifted_to_top1_rate": best_alpha_lift_rate,
        "all_results": results,
    }

    summary_path = output_dir / "alpha_analysis_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_paths = _save_visualizations(results=results, output_dir=output_dir, console=console)
    for p in plot_paths:
        console.print(f"Saved plot: {p}")

    # Print recommendations
    console.print("[bold green]Recommendations:[/bold green]")
    console.print(
        f"  Primary ({primary_desc}): alpha = {best_alpha_primary['alpha']:.4f}"
    )
    console.print(f"  Best CE Loss:        alpha = {best_alpha_loss['alpha']:.4f} (loss = {best_alpha_loss['ce_mean']:.4f})")
    console.print(
        f"  Best Target > Draft: alpha = {best_alpha_tbd['alpha']:.4f} (rate = {best_alpha_tbd['target_beats_draft'] * 100:.2f}%)"
    )
    console.print(
        f"  Best Top-1 Accuracy: alpha = {best_alpha_top1['alpha']:.4f} (acc = {best_alpha_top1['top1'] * 100:.2f}%)"
    )
    console.print(
        f"  Best Lifted->Top1:   alpha = {best_alpha_lift_count['alpha']:.4f} (count = {int(best_alpha_lift_count['lifted_to_top1_count'])}, rate = {best_alpha_lift_count['lifted_to_top1_rate'] * 100:.2f}%)"
    )
    console.print()
    console.print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
