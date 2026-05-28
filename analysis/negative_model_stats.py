"""Analyze NegativeLogitPredictor top-k stats collected by CD_negative_model."""

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


EPS = 1e-12


class MetricAccumulator:
    def __init__(self) -> None:
        self.values: dict[str, list[np.ndarray]] = defaultdict(list)

    def add(self, metrics: dict[str, torch.Tensor], mask: torch.Tensor | None = None) -> None:
        if mask is not None:
            mask_np = mask.detach().cpu().numpy().astype(bool).reshape(-1)
        else:
            mask_np = None

        for name, tensor in metrics.items():
            arr = tensor.detach().cpu().float().numpy().reshape(-1)
            if mask_np is not None:
                arr = arr[mask_np]
            arr = arr[np.isfinite(arr)]
            if arr.size:
                self.values[name].append(arr.astype(np.float64, copy=False))

    def all_values(self, name: str) -> np.ndarray:
        chunks = self.values.get(name, [])
        if not chunks:
            return np.asarray([], dtype=np.float64)
        return np.concatenate(chunks)

    def summary(self) -> dict[str, dict[str, float | int]]:
        return {name: describe_distribution(self.all_values(name)) for name in sorted(self.values)}


def describe_distribution(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
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


def find_shards(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    shards = sorted(input_path.glob("negative_stats_rank*_shard*.pt"))
    if not shards:
        shards = sorted(input_path.glob("*.pt"))
    return shards


def load_torch_file(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_manifests(input_path: Path) -> list[dict[str, Any]]:
    if input_path.is_file():
        manifest_files: list[Path] = []
    else:
        manifest_files = sorted(input_path.glob("manifest_rank*.json"))
    manifests = []
    for path in manifest_files:
        try:
            manifests.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            continue
    return manifests


def kl_div(p: torch.Tensor, log_p: torch.Tensor, log_q: torch.Tensor) -> torch.Tensor:
    return (p * (log_p - log_q)).sum(dim=-1)


def js_div(
    p: torch.Tensor,
    log_p: torch.Tensor,
    q: torch.Tensor,
    log_q: torch.Tensor,
) -> torch.Tensor:
    m = 0.5 * (p + q)
    log_m = torch.log(m.clamp_min(EPS))
    return 0.5 * kl_div(p, log_p, log_m) + 0.5 * kl_div(q, log_q, log_m)


def entropy_from_probs(probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    return -(probs * log_probs).sum(dim=-1)


def vector_pearson(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x_centered = x - x.mean(dim=-1, keepdim=True)
    y_centered = y - y.mean(dim=-1, keepdim=True)
    numerator = (x_centered * y_centered).sum(dim=-1)
    denominator = torch.sqrt(
        (x_centered.square().sum(dim=-1) * y_centered.square().sum(dim=-1)).clamp_min(EPS)
    )
    return numerator / denominator


def target_rank(logits: torch.Tensor, target_idx: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    gathered = torch.gather(logits, dim=-1, index=target_idx.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    ranks = (logits > gathered.unsqueeze(-1)).sum(dim=-1).float()
    return torch.where(valid, ranks, torch.full_like(ranks, float("nan")))


def tensor_from_record(record: dict[str, Any], key: str, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    value = record[key]
    if not torch.is_tensor(value):
        value = torch.as_tensor(value)
    if dtype is not None:
        value = value.to(dtype=dtype)
    return value


def compute_record_metrics(record: dict[str, Any], uniform_threshold: float) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    pos_logits = tensor_from_record(record, "positive_topk_logits", dtype=torch.float32)
    old_logits = tensor_from_record(record, "negative_old_topk_logits", dtype=torch.float32)
    pred_logits = tensor_from_record(record, "negative_pred_topk_logits", dtype=torch.float32)
    topk_token_ids = tensor_from_record(record, "topk_token_ids")
    target_token_ids = tensor_from_record(record, "target_token_ids")
    candidate_mask = tensor_from_record(record, "candidate_topk_mask").bool()

    pos_log_probs = F.log_softmax(pos_logits, dim=-1)
    old_log_probs = F.log_softmax(old_logits, dim=-1)
    pred_log_probs = F.log_softmax(pred_logits, dim=-1)
    pos_probs = pos_log_probs.exp()
    old_probs = old_log_probs.exp()
    pred_probs = pred_log_probs.exp()

    if "cd_topk_logits" in record:
        cd_logits = tensor_from_record(record, "cd_topk_logits", dtype=torch.float32)
    else:
        cd_logits = pos_log_probs - pred_log_probs
    if "static_cd_topk_logits" in record:
        static_cd_logits = tensor_from_record(record, "static_cd_topk_logits", dtype=torch.float32)
    else:
        static_cd_logits = pos_log_probs - old_log_probs
    cd_log_probs = F.log_softmax(cd_logits, dim=-1)
    static_cd_log_probs = F.log_softmax(static_cd_logits, dim=-1)
    cd_probs = cd_log_probs.exp()
    static_cd_probs = static_cd_log_probs.exp()

    k = pos_logits.size(-1)
    log_k = math.log(k)

    old_entropy = entropy_from_probs(old_probs, old_log_probs)
    pred_entropy = entropy_from_probs(pred_probs, pred_log_probs)
    pos_entropy = entropy_from_probs(pos_probs, pos_log_probs)
    cd_entropy = entropy_from_probs(cd_probs, cd_log_probs)

    target_match = topk_token_ids.eq(target_token_ids.unsqueeze(-1))
    target_in_topk = target_match.any(dim=-1)
    target_idx = target_match.long().argmax(dim=-1)
    positive_rank = target_rank(pos_logits, target_idx, target_in_topk)
    cd_rank = target_rank(cd_logits, target_idx, target_in_topk)
    static_cd_rank = target_rank(static_cd_logits, target_idx, target_in_topk)
    target_in_candidate = torch.gather(
        candidate_mask,
        dim=-1,
        index=target_idx.unsqueeze(-1),
    ).squeeze(-1) & target_in_topk

    positive_wrong = target_in_topk & (positive_rank > 0)
    cd_top1_hit = target_in_topk & (cd_rank == 0)
    cd_top5_hit = target_in_topk & (cd_rank < 5)
    cd_top10_hit = target_in_topk & (cd_rank < 10)
    rank_improvement = positive_rank - cd_rank

    seq_len = pos_logits.size(1)
    accepted_offsets = torch.arange(seq_len).view(1, seq_len)
    accepted_by_target = accepted_offsets < int(record.get("acceptance_length", 0))

    old_pred_logit_delta = pred_logits - old_logits
    old_pred_prob_delta = pred_probs - old_probs
    cd_static_delta = cd_logits - static_cd_logits

    metrics = {
        "old_pred_kl": kl_div(old_probs, old_log_probs, pred_log_probs),
        "pred_old_kl": kl_div(pred_probs, pred_log_probs, old_log_probs),
        "old_pred_js": js_div(old_probs, old_log_probs, pred_probs, pred_log_probs),
        "pred_pos_kl": kl_div(pred_probs, pred_log_probs, pos_log_probs),
        "pos_pred_kl": kl_div(pos_probs, pos_log_probs, pred_log_probs),
        "pred_pos_js": js_div(pred_probs, pred_log_probs, pos_probs, pos_log_probs),
        "old_entropy": old_entropy,
        "old_norm_entropy": old_entropy / log_k,
        "old_effective_support": old_entropy.exp(),
        "old_kl_to_uniform": log_k - old_entropy,
        "pred_entropy": pred_entropy,
        "pred_norm_entropy": pred_entropy / log_k,
        "pred_effective_support": pred_entropy.exp(),
        "pred_kl_to_uniform": log_k - pred_entropy,
        "pred_logit_mean": pred_logits.mean(dim=-1),
        "pred_logit_std": pred_logits.std(dim=-1),
        "pred_logit_range": pred_logits.amax(dim=-1) - pred_logits.amin(dim=-1),
        "pred_top1_prob": pred_probs.amax(dim=-1),
        "old_pred_logit_l1": old_pred_logit_delta.abs().mean(dim=-1),
        "old_pred_logit_l2": old_pred_logit_delta.square().mean(dim=-1).sqrt(),
        "old_pred_logit_max_abs": old_pred_logit_delta.abs().amax(dim=-1),
        "old_pred_prob_l1": old_pred_prob_delta.abs().sum(dim=-1),
        "old_pred_prob_l2": old_pred_prob_delta.square().sum(dim=-1).sqrt(),
        "old_pred_logit_pearson": vector_pearson(old_logits, pred_logits),
        "old_pred_logit_cosine": F.cosine_similarity(old_logits, pred_logits, dim=-1, eps=EPS),
        "pos_entropy": pos_entropy,
        "cd_entropy": cd_entropy,
        "pos_cd_kl": kl_div(pos_probs, pos_log_probs, cd_log_probs),
        "cd_pos_kl": kl_div(cd_probs, cd_log_probs, pos_log_probs),
        "pos_cd_js": js_div(pos_probs, pos_log_probs, cd_probs, cd_log_probs),
        "cd_static_logit_l1": cd_static_delta.abs().mean(dim=-1),
        "cd_static_logit_l2": cd_static_delta.square().mean(dim=-1).sqrt(),
        "cd_static_logit_max_abs": cd_static_delta.abs().amax(dim=-1),
        "cd_static_prob_l1": (cd_probs - static_cd_probs).abs().sum(dim=-1),
        "cd_static_prob_l2": (cd_probs - static_cd_probs).square().sum(dim=-1).sqrt(),
        "cd_static_logit_cosine": F.cosine_similarity(cd_logits, static_cd_logits, dim=-1, eps=EPS),
        "cd_static_argmax_changed": (cd_logits.argmax(dim=-1) != static_cd_logits.argmax(dim=-1)).float(),
        "positive_cd_argmax_changed": (cd_logits.argmax(dim=-1) != pos_logits.argmax(dim=-1)).float(),
        "target_in_top32": target_in_topk.float(),
        "target_not_in_top32": (~target_in_topk).float(),
        "target_in_candidate_top32": target_in_candidate.float(),
        "positive_rank": positive_rank,
        "cd_rank": cd_rank,
        "static_cd_rank": static_cd_rank,
        "rank_improvement": rank_improvement,
        "rank_improved_flag": (target_in_topk & (cd_rank < positive_rank)).float(),
        "rank_unchanged_flag": (target_in_topk & (cd_rank == positive_rank)).float(),
        "rank_worsened_flag": (target_in_topk & (cd_rank > positive_rank)).float(),
        "positive_top1_hit": (target_in_topk & (positive_rank == 0)).float(),
        "cd_top1_hit": cd_top1_hit.float(),
        "cd_top5_hit": cd_top5_hit.float(),
        "cd_top10_hit": cd_top10_hit.float(),
        "positive_wrong_top1": positive_wrong.float(),
        "positive_wrong_fixed_top1": (positive_wrong & cd_top1_hit).float(),
        "positive_wrong_fixed_top5": (positive_wrong & cd_top5_hit).float(),
        "positive_wrong_fixed_top10": (positive_wrong & cd_top10_hit).float(),
        "accepted_by_target": accepted_by_target.expand_as(target_in_topk).float(),
        "uniform_like_pred": ((pred_entropy / log_k) >= uniform_threshold).float(),
    }

    masks = {
        "all": torch.ones_like(target_in_topk, dtype=torch.bool),
        "target_in_top32": target_in_topk,
        "target_not_in_top32": ~target_in_topk,
        "positive_top1_correct": target_in_topk & (positive_rank == 0),
        "positive_wrong_target_in_top32": positive_wrong,
        "fixed_positive_wrong_to_top1": positive_wrong & cd_top1_hit,
        "improved_positive_wrong": positive_wrong & (cd_rank < positive_rank),
        "worsened_target_rank": target_in_topk & (cd_rank > positive_rank),
        "accepted_positions": accepted_by_target.expand_as(target_in_topk),
        "rejected_positions": ~accepted_by_target.expand_as(target_in_topk),
    }
    return metrics, masks


def write_metric_rows(path: Path, rows: list[dict[str, Any]], group_field: str) -> None:
    fieldnames = [
        group_field,
        "metric",
        "count",
        "mean",
        "std",
        "min",
        "p10",
        "median",
        "p90",
        "max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rows_from_summary(summary: dict[str, dict[str, float | int]], group_field: str, group_value: Any) -> list[dict[str, Any]]:
    rows = []
    for metric, stats in sorted(summary.items()):
        row = {group_field: group_value, "metric": metric}
        row.update(stats)
        rows.append(row)
    return rows


def write_histogram(values: np.ndarray, path: Path, title: str, xlabel: str, bins: int) -> bool:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return False
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(values, bins=bins, color="#2f6f9f", alpha=0.88)
    ax.axvline(float(values.mean()), color="#c43b31", linestyle="--", linewidth=1.5, label="mean")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def mean_metric(summary: dict[str, dict[str, float | int]], name: str) -> float:
    return float(summary.get(name, {}).get("mean", 0.0))


def count_metric(summary: dict[str, dict[str, float | int]], name: str) -> int:
    return int(summary.get(name, {}).get("count", 0))


def build_report(summary: dict[str, Any]) -> str:
    overall = summary["overall"]
    cohorts = summary["cohorts"]
    positive_wrong = cohorts.get("positive_wrong_target_in_top32", {})

    lines = [
        "# Negative Model Stats Report",
        "",
        "## Dataset",
        "",
        f"- Shards: {summary['meta']['num_shards']}",
        f"- Records: {summary['meta']['num_records']}",
        f"- Token positions: {summary['meta']['num_positions']}",
        f"- Top-k support: {summary['meta']['top_k']}",
        "",
        "## Key Signals",
        "",
        "| Metric | Mean |",
        "|---|---:|",
        f"| KL(old negative || predicted negative) | {mean_metric(overall, 'old_pred_kl'):.6f} |",
        f"| JS(old negative, predicted negative) | {mean_metric(overall, 'old_pred_js'):.6f} |",
        f"| KL(predicted negative || positive) | {mean_metric(overall, 'pred_pos_kl'):.6f} |",
        f"| KL(positive || predicted negative) | {mean_metric(overall, 'pos_pred_kl'):.6f} |",
        f"| JS(predicted negative, positive) | {mean_metric(overall, 'pred_pos_js'):.6f} |",
        f"| Predicted negative normalized entropy | {mean_metric(overall, 'pred_norm_entropy'):.6f} |",
        f"| Predicted negative KL to uniform | {mean_metric(overall, 'pred_kl_to_uniform'):.6f} |",
        f"| Uniform-like predicted negative rate | {mean_metric(overall, 'uniform_like_pred') * 100:.2f}% |",
        f"| KL(positive || CD predicted) | {mean_metric(overall, 'pos_cd_kl'):.6f} |",
        f"| JS(positive, CD predicted) | {mean_metric(overall, 'pos_cd_js'):.6f} |",
        f"| CD predicted vs static CD argmax change rate | {mean_metric(overall, 'cd_static_argmax_changed') * 100:.2f}% |",
        f"| CD predicted vs static CD logit L2 | {mean_metric(overall, 'cd_static_logit_l2'):.6f} |",
        f"| Target not in top-32 rate | {mean_metric(overall, 'target_not_in_top32') * 100:.2f}% |",
        "",
        "## Positive-Wrong Target Cohort",
        "",
        f"- Count: {count_metric(positive_wrong, 'positive_rank')}",
        f"- Mean rank improvement: {mean_metric(positive_wrong, 'rank_improvement'):.4f}",
        f"- Improved rate: {mean_metric(positive_wrong, 'rank_improved_flag') * 100:.2f}%",
        f"- Recovered to top-1 rate: {mean_metric(positive_wrong, 'cd_top1_hit') * 100:.2f}%",
        f"- Recovered to top-5 rate: {mean_metric(positive_wrong, 'cd_top5_hit') * 100:.2f}%",
        f"- Recovered to top-10 rate: {mean_metric(positive_wrong, 'cd_top10_hit') * 100:.2f}%",
        f"- Worsened rate: {mean_metric(positive_wrong, 'rank_worsened_flag') * 100:.2f}%",
        "",
        "## Artifacts",
        "",
    ]
    lines.extend(f"- `{name}`" for name in summary["artifacts"])
    lines.append("")
    return "\n".join(lines)


def analyze(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    shards = find_shards(input_path)
    if not shards:
        raise SystemExit(f"No .pt stats shards found in {input_path}")

    overall_acc = MetricAccumulator()
    by_position: dict[int, MetricAccumulator] = defaultdict(MetricAccumulator)
    by_cohort: dict[str, MetricAccumulator] = defaultdict(MetricAccumulator)

    num_records = 0
    num_positions = 0
    top_k = None

    for shard in shards:
        records = load_torch_file(shard)
        if isinstance(records, dict):
            records = records.get("records", [])
        for record in records:
            if args.limit_records is not None and num_records >= args.limit_records:
                break
            metrics, masks = compute_record_metrics(record, args.uniform_threshold)
            overall_acc.add(metrics)
            all_mask = masks["all"]
            num_records += 1
            num_positions += int(all_mask.numel())
            if top_k is None:
                top_k = int(tensor_from_record(record, "positive_topk_logits").shape[-1])

            seq_len = all_mask.shape[1]
            for pos in range(seq_len):
                position_mask = torch.zeros_like(all_mask, dtype=torch.bool)
                position_mask[:, pos] = True
                by_position[pos].add(metrics, mask=position_mask)
            for cohort, mask in masks.items():
                by_cohort[cohort].add(metrics, mask=mask)
        if args.limit_records is not None and num_records >= args.limit_records:
            break

    overall_summary = overall_acc.summary()
    position_summary = {str(pos): acc.summary() for pos, acc in sorted(by_position.items())}
    cohort_summary = {cohort: acc.summary() for cohort, acc in sorted(by_cohort.items())}

    by_position_rows = []
    for pos, metric_summary in position_summary.items():
        by_position_rows.extend(rows_from_summary(metric_summary, "block_position", int(pos)))

    by_cohort_rows = []
    for cohort, metric_summary in cohort_summary.items():
        by_cohort_rows.extend(rows_from_summary(metric_summary, "cohort", cohort))

    overall_rows = rows_from_summary(overall_summary, "group", "overall")

    write_metric_rows(output_dir / "negative_model_by_position.csv", by_position_rows, "block_position")
    write_metric_rows(output_dir / "negative_model_by_cohort.csv", by_cohort_rows, "cohort")
    write_metric_rows(output_dir / "negative_model_metric_summary.csv", overall_rows, "group")

    plot_specs = [
        ("old_pred_kl", "KL(old negative || predicted negative)", "KL"),
        ("pred_pos_kl", "KL(predicted negative || positive)", "KL"),
        ("pred_pos_js", "JS(predicted negative, positive)", "JS"),
        ("pred_norm_entropy", "Predicted negative normalized entropy", "normalized entropy"),
        ("pred_kl_to_uniform", "Predicted negative KL to uniform", "KL to uniform"),
        ("pos_cd_kl", "KL(positive || CD predicted)", "KL"),
        ("cd_static_logit_l2", "CD predicted vs static CD logit L2", "L2"),
        ("rank_improvement", "Target rank improvement", "positive rank - CD rank"),
    ]
    plot_files = []
    for metric_name, title, xlabel in plot_specs:
        out_name = f"hist_{metric_name}.png"
        if write_histogram(
            overall_acc.all_values(metric_name),
            output_dir / out_name,
            title,
            xlabel,
            bins=args.hist_bins,
        ):
            plot_files.append(out_name)

    artifacts = [
        "negative_model_summary.json",
        "negative_model_report.md",
        "negative_model_metric_summary.csv",
        "negative_model_by_position.csv",
        "negative_model_by_cohort.csv",
        *plot_files,
    ]

    summary = {
        "meta": {
            "input_dir": str(input_path),
            "num_shards": len(shards),
            "num_records": int(num_records),
            "num_positions": int(num_positions),
            "top_k": int(top_k or 0),
            "uniform_threshold": float(args.uniform_threshold),
            "manifests": load_manifests(input_path),
        },
        "overall": overall_summary,
        "positions": position_summary,
        "cohorts": cohort_summary,
        "artifacts": artifacts,
    }

    (output_dir / "negative_model_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "negative_model_report.md").write_text(
        build_report(summary),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze CD_negative_model negative predictor stats shards.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory or .pt shard produced by CD_negative_model.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for summary JSON/CSV/Markdown/PNG artifacts.")
    parser.add_argument("--uniform-threshold", type=float, default=0.95, help="Normalized entropy threshold for uniform-like predictions.")
    parser.add_argument("--hist-bins", type=int, default=60, help="Number of bins for histogram PNGs.")
    parser.add_argument("--limit-records", type=int, default=None, help="Optional cap for quick local analysis.")
    return parser.parse_args()


def main() -> None:
    summary = analyze(parse_args())
    print(f"Wrote negative model stats for {summary['meta']['num_positions']} token positions.")


if __name__ == "__main__":
    main()
