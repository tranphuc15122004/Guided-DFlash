#!/usr/bin/env python3
"""Parse alpha training PBS output logs into CSV reports.

This script reads files like `train_alpha_h200.pbs.o168218529`, extracts the
training configuration plus per-epoch metrics, and writes two CSV files:

- `training_files.csv`: one row per log file
- `training_epochs.csv`: one row per epoch

Example:
  python script/parse_alpha_training_logs.py train_alpha_h200.pbs.o*
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
from pathlib import Path
from typing import Any


NUM = r"(?:[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf)"

EPOCH_START_RE = re.compile(rf"\u2500+\s*Epoch\s+(?P<epoch>\d+)/(?:\d+)\s+\u2500+")
SUMMARY_RE = re.compile(
    rf"\u2714\s*Epoch\s+(?P<epoch>\d+)\s*\|\s*"
    rf"actor_loss=(?P<actor_loss>{NUM})\s+"
    rf"critic_loss=(?P<critic_loss>{NUM})\s+"
    rf"reward=(?P<reward>{NUM})\s+"
    rf"r1=(?P<r1>{NUM})\s+"
    rf"r2=(?P<r2>{NUM})\s+"
    rf"r3=(?P<r3>{NUM})"
)
BASELINE_RE = re.compile(
    rf"baseline_acc=(?P<baseline_acc>{NUM})\s+model_acc=(?P<model_acc>{NUM})"
)
DETAIL_RE = re.compile(
    rf"(?P<name>[A-Za-z0-9_]+):\s*"
    rf"(?P<mean>{NUM})\s*\u00b1\s*"
    rf"(?P<std>{NUM})\s*"
    rf"\[(?P<min>{NUM}),\s*(?P<max>{NUM})\]"
)

JOB_ID_RE = re.compile(r"^Job ID:\s*(?P<value>.+)$")
NODE_RE = re.compile(r"^Node:\s*(?P<value>.+)$")
TORCH_RE = re.compile(
    r"^torch\s+(?P<version>[^,]+),\s+CUDA:\s+(?P<cuda>True|False),\s+devices:\s+(?P<devices>\d+)\s*$"
)
H5PY_RE = re.compile(r"^h5py\s+(?P<version>.+)$")
TRAIN_DEVICE_RE = re.compile(r"^\[train_AC\]\s+device:\s*(?P<value>.+)$")
BUCKET_RE = re.compile(
    r"^\[train_AC\]\s+bucket config:\s+bucket_size=(?P<bucket_size>\d+),\s+num_buckets=(?P<num_buckets>\d+)\s*$"
)
LOAD_RE = re.compile(r"^Loading\s+(?P<records>\d+)\s+records into memory\.\.\.$")
LOADED_RE = re.compile(
    r"^Loaded\s+(?P<records>\d+)\s+records,\s+S=(?P<sequence_length>\d+),\s+K=(?P<k>\d+),\s+memory=(?P<memory>.+)$"
)
LIMITED_RE = re.compile(r"^Limited to\s+(?P<limited>\d+)\s*/\s*(?P<original>\d+)\s+records$")
CONFIG_RE = re.compile(r"^(?P<key>[A-Z0-9_]+):\s*(?P<value>.+)$")
W_RE = re.compile(rf"^W1=(?P<w1>{NUM})\s+W2=(?P<w2>{NUM})\s+W3=(?P<w3>{NUM})$")

CONFIG_NAME_MAP = {
    "H5_PATH": "h5_path",
    "OUTPUT_DIR": "output_dir",
    "BATCH_SIZE": "batch_size",
    "EPOCHS": "epochs_configured",
    "LR": "lr",
    "TOP_K": "top_k",
    "HIDDEN_DIM": "hidden_dim",
    "BUCKET_SIZE": "bucket_size",
    "MAX_ALPHA": "max_alpha",
    "GAMMA_DECAY": "gamma_decay",
    "LAMBDA_R3": "lambda_r3",
    "ENTROPY_COEF": "entropy_coef",
    "MAX_GRAD_NORM": "max_grad_norm",
    "SAVE_INTERVAL": "save_interval",
    "MAX_RECORDS": "max_records",
}

METRIC_ALIASES = {
    "alpha_gap_mean": "alpha_gap",
}

SUMMARY_FIELDS = [
    "source_file",
    "source_path",
    "job_id",
    "node",
    "train_device",
    "torch_version",
    "cuda",
    "devices",
    "h5py_version",
    "h5_path",
    "output_dir",
    "batch_size",
    "epochs_configured",
    "lr",
    "top_k",
    "hidden_dim",
    "bucket_size",
    "num_buckets",
    "max_alpha",
    "gamma_decay",
    "lambda_r3",
    "w1",
    "w2",
    "w3",
    "entropy_coef",
    "max_grad_norm",
    "save_interval",
    "max_records",
    "records_loading",
    "records_loaded",
    "sequence_length",
    "loaded_top_k",
    "memory_estimate",
    "limited_records",
    "original_records",
    "epochs_found",
    "last_epoch",
    "last_reward",
    "best_reward",
    "best_reward_epoch",
    "training_completed",
]


def parse_scalar(text: str) -> Any:
    value = text.strip()
    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    try:
        return float(value)
    except ValueError:
        return value


def normalize_metric_name(name: str) -> str:
    return METRIC_ALIASES.get(name, name)


def resolve_inputs(items: list[str]) -> list[Path]:
    if not items:
        items = ["train_alpha_h200.pbs.o*"]

    files: list[Path] = []
    seen: set[Path] = set()

    for item in items:
        candidate = Path(item)
        matches: list[Path] = []

        if candidate.is_dir():
            matches = [Path(p) for p in glob.glob(str(candidate / "train_alpha_h200.pbs.o*"), recursive=True)]
        elif any(ch in item for ch in "*?[]"):
            matches = [Path(p) for p in glob.glob(item, recursive=True)]
        elif candidate.is_file():
            matches = [candidate]
        else:
            matches = [Path(p) for p in glob.glob(item, recursive=True)]

        for path in sorted(matches):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    seen.add(resolved)
                    files.append(resolved)

    return sorted(files, key=lambda p: p.name)


def ensure_current_epoch(current: dict[str, Any] | None, epoch: int, total: int | None = None) -> dict[str, Any]:
    if current is None or current.get("epoch") != epoch:
        if current is not None:
            current.setdefault("finalized", False)
        current = {
            "epoch": epoch,
            "epoch_total": total,
            "metrics": {},
        }
    elif total is not None and current.get("epoch_total") is None:
        current["epoch_total"] = total
    return current


def set_metric_mean(current: dict[str, Any], metric: str, value: Any) -> None:
    metrics = current.setdefault("metrics", {})
    stats = metrics.setdefault(metric, {})
    stats.setdefault("mean", value)


def set_metric_stats(current: dict[str, Any], metric: str, mean: Any, std: Any, low: Any, high: Any) -> None:
    metrics = current.setdefault("metrics", {})
    metrics[metric] = {
        "mean": mean,
        "std": std,
        "min": low,
        "max": high,
    }


def parse_log_file(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    info: dict[str, Any] = {
        "source_file": path.name,
        "source_path": str(path),
        "training_completed": False,
    }
    epoch_rows: list[dict[str, Any]] = []

    text = path.read_text(encoding="utf-8", errors="replace").replace("\r", "\n")
    lines = text.splitlines()

    in_config = False
    current: dict[str, Any] | None = None

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line == "=== Training Configuration ===":
            in_config = True
            continue
        if line == "=== Starting training ===":
            in_config = False
            continue
        if line == "=== Environment ===":
            continue
        if line == "Training completed!":
            info["training_completed"] = True
            continue

        if m := JOB_ID_RE.match(line):
            info["job_id"] = m.group("value")
            continue
        if m := NODE_RE.match(line):
            info["node"] = m.group("value")
            continue
        if m := TORCH_RE.match(line):
            info["torch_version"] = m.group("version")
            info["cuda"] = m.group("cuda") == "True"
            info["devices"] = int(m.group("devices"))
            continue
        if m := H5PY_RE.match(line):
            info["h5py_version"] = m.group("version")
            continue
        if m := TRAIN_DEVICE_RE.match(line):
            info["train_device"] = m.group("value")
            continue
        if m := BUCKET_RE.match(line):
            info["bucket_size"] = int(m.group("bucket_size"))
            info["num_buckets"] = int(m.group("num_buckets"))
            continue
        if m := LOAD_RE.match(line):
            info["records_loading"] = int(m.group("records"))
            continue
        if m := LOADED_RE.match(line):
            info["records_loaded"] = int(m.group("records"))
            info["sequence_length"] = int(m.group("sequence_length"))
            info["loaded_top_k"] = int(m.group("k"))
            info["memory_estimate"] = m.group("memory")
            continue
        if m := LIMITED_RE.match(line):
            info["limited_records"] = int(m.group("limited"))
            info["original_records"] = int(m.group("original"))
            continue

        if in_config:
            if m := W_RE.match(line):
                info["w1"] = parse_scalar(m.group("w1"))
                info["w2"] = parse_scalar(m.group("w2"))
                info["w3"] = parse_scalar(m.group("w3"))
                continue

            if m := CONFIG_RE.match(line):
                raw_key = m.group("key")
                key = CONFIG_NAME_MAP.get(raw_key, raw_key.lower())
                info[key] = parse_scalar(m.group("value"))
                continue

        if m := EPOCH_START_RE.match(line):
            epoch = int(m.group("epoch"))
            total = None
            # The total count is part of the matched string; extract from the original line.
            total_match = re.search(r"/(\d+)", line)
            if total_match:
                total = int(total_match.group(1))
            if current is not None:
                epoch_rows.append(current)
            current = ensure_current_epoch(None, epoch, total)
            continue

        if m := SUMMARY_RE.search(line):
            epoch = int(m.group("epoch"))
            current = ensure_current_epoch(current, epoch)
            for metric_name in ("actor_loss", "critic_loss", "reward", "r1", "r2", "r3"):
                set_metric_mean(current, metric_name, parse_scalar(m.group(metric_name)))
            continue

        if m := BASELINE_RE.search(line):
            if current is None:
                continue
            set_metric_mean(current, "baseline_acc", parse_scalar(m.group("baseline_acc")))
            set_metric_mean(current, "model_acc", parse_scalar(m.group("model_acc")))
            continue

        if current is None:
            continue

        for m in DETAIL_RE.finditer(line):
            metric = normalize_metric_name(m.group("name"))
            set_metric_stats(
                current,
                metric,
                parse_scalar(m.group("mean")),
                parse_scalar(m.group("std")),
                parse_scalar(m.group("min")),
                parse_scalar(m.group("max")),
            )

    if current is not None:
        epoch_rows.append(current)

    return info, epoch_rows


def flatten_epoch_rows(file_info: dict[str, Any], epoch_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flat_rows: list[dict[str, Any]] = []
    for row in epoch_rows:
        flat: dict[str, Any] = {
            "source_file": file_info.get("source_file"),
            "source_path": file_info.get("source_path"),
            "job_id": file_info.get("job_id"),
            "epoch": row.get("epoch"),
            "epoch_total": row.get("epoch_total"),
        }
        for metric, stats in row.get("metrics", {}).items():
            for stat_name in ("mean", "std", "min", "max"):
                flat[f"{metric}_{stat_name}"] = stats.get(stat_name)
        flat_rows.append(flat)
    return flat_rows


def compute_file_summary(file_info: dict[str, Any], epoch_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary = {field: file_info.get(field) for field in SUMMARY_FIELDS}
    summary["epochs_found"] = len(epoch_rows)

    reward_values: list[tuple[int, float]] = []
    for row in epoch_rows:
        reward = row.get("metrics", {}).get("reward", {}).get("mean")
        if isinstance(reward, (int, float)) and not (isinstance(reward, float) and math.isnan(reward)):
            reward_values.append((int(row.get("epoch", 0)), float(reward)))

    if epoch_rows:
        summary["last_epoch"] = epoch_rows[-1].get("epoch")
        last_reward = epoch_rows[-1].get("metrics", {}).get("reward", {}).get("mean")
        summary["last_reward"] = last_reward

    if reward_values:
        best_epoch, best_reward = max(reward_values, key=lambda item: item[1])
        summary["best_reward"] = best_reward
        summary["best_reward_epoch"] = best_epoch

    return summary


def collect_metric_order(epoch_rows: list[dict[str, Any]]) -> list[str]:
    order: list[str] = []
    seen: set[str] = set()
    for row in epoch_rows:
        for metric in row.get("metrics", {}).keys():
            if metric not in seen:
                seen.add(metric)
                order.append(metric)
    return order


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse alpha PBS training logs into CSV reports")
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Log files, directories, or glob patterns. Defaults to train_alpha_h200.pbs.o*",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/alpha_training_reports",
        help="Directory for the generated CSV files",
    )
    args = parser.parse_args()

    files = resolve_inputs(args.inputs)
    if not files:
        raise SystemExit("No log files matched the provided inputs.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_summaries: list[dict[str, Any]] = []
    all_epoch_rows: list[dict[str, Any]] = []

    for path in files:
        file_info, epoch_rows = parse_log_file(path)
        file_summaries.append(compute_file_summary(file_info, epoch_rows))
        all_epoch_rows.extend(flatten_epoch_rows(file_info, epoch_rows))

    metric_order: list[str] = []
    seen_metrics: set[str] = set()
    for row in all_epoch_rows:
        for key in row.keys():
            if not key.endswith("_mean"):
                continue
            metric = key[: -len("_mean")]
            if metric in seen_metrics:
                continue
            seen_metrics.add(metric)
            metric_order.append(metric)

    epoch_fieldnames = ["source_file", "source_path", "job_id", "epoch", "epoch_total"]
    for metric in metric_order:
        epoch_fieldnames.extend([f"{metric}_mean", f"{metric}_std", f"{metric}_min", f"{metric}_max"])

    summary_csv = output_dir / "training_files.csv"
    epoch_csv = output_dir / "training_epochs.csv"

    write_csv(summary_csv, file_summaries, SUMMARY_FIELDS)
    write_csv(epoch_csv, all_epoch_rows, epoch_fieldnames)

    print(f"Parsed {len(files)} log file(s)")
    for row in file_summaries:
        reward = row.get("best_reward")
        epoch = row.get("best_reward_epoch")
        epochs = row.get("epochs_found")
        completed = "yes" if row.get("training_completed") else "no"
        print(f"- {row.get('source_file')}: epochs={epochs}, best_reward={reward} @ {epoch}, completed={completed}")
    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {epoch_csv}")


if __name__ == "__main__":
    main()
