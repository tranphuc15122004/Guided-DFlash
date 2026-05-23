"""
Training Logger — structured logging, CLI output, checkpoint management, and metrics tracking.

Provides a unified `TrainingLogger` class used by both Phase 1 (supervised)
and Phase 2 (RL/A2C) training scripts.

Outputs:
  - config.json               Saved experiment arguments
  - training_history.csv      Machine-readable metrics per epoch
  - training.log              Human-readable structured log
  - epoch_metrics/epoch_N.json  Per-epoch metric snapshots
  - checkpoints/              Model checkpoints with metadata
"""

from tqdm import tqdm
import csv
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Union

import torch
import torch.nn as nn


def _format_metric(value: float, fmt: str = '.4f') -> str:
    """Format a metric value consistently."""
    if abs(value) >= 1e4 or abs(value) < 1e-4:
        return f'{value:.4e}'
    return f'{value:{fmt}}'


def _format_duration(seconds: float) -> str:
    """Format a duration in human-readable form."""
    if seconds < 60:
        return f'{seconds:.1f}s'
    elif seconds < 3600:
        m, s = divmod(seconds, 60)
        return f'{int(m)}m {s:.1f}s'
    else:
        h, m = divmod(seconds, 3600)
        return f'{int(h)}h {int(m)}m'


class TrainingLogger:
    """
    Unified training logger with file I/O, CLI output, and checkpoint management.

    Usage:
        logger = TrainingLogger(args, output_dir)
        logger.log_config(args)
        logger.log_model_info(actor, critic)
        logger.log_dataset_info(dataset)

        for epoch in range(1, epochs + 1):
            logger.start_epoch(epoch, epochs)
            ... training loop ...
            logger.log_epoch_metrics(metrics_dict, epoch, model, optimizer)
    """

    def __init__(
        self,
        args,
        output_dir: Union[str, Path],
        phase_name: str = "training",
        writer: Optional[TextIO] = None,
    ):
        self.args = args
        self.phase_name = phase_name
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Sub-directories
        self.epoch_metrics_dir = self.output_dir / "epoch_metrics"
        self.epoch_metrics_dir.mkdir(exist_ok=True)

        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)

        # File handles
        self.csv_path = self.output_dir / "training_history.csv"
        self.log_path = self.output_dir / "training.log"
        self.config_path = self.output_dir / "config.json"

        self.csv_file: Optional[TextIO] = None
        self.csv_writer = None
        self.log_file: Optional[TextIO] = None

        # State
        self.cached_metrics: Dict[str, List[float]] = {}
        self._epoch_start_time = 0.0
        self._global_step = 0
        self._best_metric = float('-inf')
        self._best_metric_name = 'reward'

        # Open log file
        self._open_log()

    def _open_log(self):
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        self._log_separator('═')
        self._write_log(f"  Training Logger — {self.phase_name}")
        self._write_log(f"  Started: {datetime.now(timezone.utc).isoformat()}")
        self._write_log(f"  Host:    {platform.node()}")
        self._write_log(f"  Python:  {platform.python_version()}")
        self._write_log(f"  PyTorch: {torch.__version__}")
        self._write_log(f"  CUDA:    {torch.version.cuda or 'N/A'}")
        self._write_log(f"  Output:  {self.output_dir}")
        self._log_separator('═')

    def close(self):
        """Close all open file handles."""
        if self.log_file:
            self._log_separator('═')
            self._write_log(f"  Finished: {datetime.now(timezone.utc).isoformat()}")
            self.log_file.close()
            self.log_file = None
        if self.csv_file:
            self.csv_file.close()
            self.csv_file = None

    # ── Internal helpers ──

    def _write_log(self, msg: str):
        """Write a line to the log file with timestamp."""
        if self.log_file:
            ts = datetime.now().strftime('%H:%M:%S')
            self.log_file.write(f"[{ts}] {msg}\n")
            self.log_file.flush()

    def _log_separator(self, char: str = '─', width: int = 72):
        self._write_log(f"  {char * width}")

    def _cli_separator(self, char: str = '─', width: int = 64):
        print(f"  {char * width}")

    # ── Configuration & Metadata ──

    def log_config(self, args):
        """Save experiment configuration to config.json and log."""
        cfg = {}
        for k, v in sorted(vars(args).items()):
            cfg[k] = str(v) if not isinstance(v, (int, float, bool, str, type(None))) else v
        with open(self.config_path, 'w') as f:
            json.dump(cfg, f, indent=2)
        self._write_log("  Configuration saved to config.json")
        self._cli_separator('─')
        print("  ⚙  Configuration")
        self._cli_separator('─')
        for k, v in cfg.items():
            print(f"     {k:25s} = {v}")
        self._cli_separator('─')
        print()

    def log_model_info(self, actor: nn.Module, critic: Optional[nn.Module] = None):
        """Log model parameter counts and architecture info."""
        def count_params(m):
            total = sum(p.numel() for p in m.parameters())
            trainable = sum(p.numel() for p in m.parameters() if p.requires_grad)
            return total, trainable

        a_total, a_train = count_params(actor)
        self._write_log(f"  Actor:  {a_total:,} params ({a_train:,} trainable)")
        self._write_log(f"  Actor input dim: {getattr(actor, 'input_dim', '?')}")

        print(f"  ┌─ Model Info ──────────────────────────────────────────")
        print(f"  │ Actor:  {a_total:>10,} params  ({a_train:,} trainable)")
        if hasattr(actor, 'input_dim'):
            print(f"  │         Input dim: {actor.input_dim}")

        if critic is not None:
            c_total, c_train = count_params(critic)
            self._write_log(f"  Critic: {c_total:,} params ({c_train:,} trainable)")
            print(f"  │ Critic: {c_total:>10,} params  ({c_train:,} trainable)")

        print(f"  └──────────────────────────────────────────────────────")
        print()

    def log_dataset_info(self, dataset):
        """Log dataset statistics."""
        num_records = len(dataset)
        S = getattr(dataset, 'S', None)
        K = getattr(dataset, 'K', None)
        self._write_log(f"  Dataset: {num_records:,} records, S={S}, K={K}")
        print(f"  ┌─ Dataset Info ───────────────────────────────────────")
        print(f"  │ Records: {num_records:>12,}")
        if S:
            print(f"  │ S:       {S:>12}")
        if K:
            print(f"  │ K:       {K:>12}")
        print(f"  └──────────────────────────────────────────────────────")
        print()

    # ── CSV Setup ──

    def init_csv(self, header: List[str]):
        """Initialize CSV file with header row."""
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(header)
        self.csv_file.flush()
        self.csv_header = header
        self._write_log(f"  CSV initialized: {len(header)} columns")

    def write_csv_row(self, row: List[Union[str, float, int]]):
        """Write a row to the CSV file."""
        if self.csv_writer is not None:
            self.csv_writer.writerow(row)
            if self.csv_file:
                self.csv_file.flush()

    # ── Epoch Lifecycle ──

    def start_epoch(self, epoch: int, total_epochs: int):
        """Mark the beginning of an epoch."""
        self._epoch_start_time = time.time()
        self._log_separator()
        self._write_log(f"  Epoch {epoch}/{total_epochs}")
        print()
        self._cli_separator('━')
        print(f"  Epoch {epoch:>3d} / {total_epochs}")
        self._cli_separator('━')

    def end_epoch(
        self,
        epoch: int,
        total_epochs: int,
        metrics: Dict[str, float],
        output_format: str = "phase1",
    ):
        """
        Finalise epoch: log metrics, write CSV, save checkpoint.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total epochs
            metrics: Dict of metric_name -> scalar value
            output_format: 'phase1' or 'phase2' for CLI display style
        """
        elapsed = time.time() - self._epoch_start_time
        metrics['epoch_duration_s'] = elapsed

        # ── Write to log file ──
        self._write_log(f"  Epoch {epoch} done — {_format_duration(elapsed)}")
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float)):
                self._write_log(f"    {k:25s} = {_format_metric(v)}")

        # ── Save per-epoch metrics JSON ──
        epoch_file = self.epoch_metrics_dir / f"epoch_{epoch:04d}.json"
        with open(epoch_file, 'w') as f:
            json.dump({
                'epoch': epoch,
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'duration_s': elapsed,
                'metrics': {k: v for k, v in metrics.items() if isinstance(v, (int, float))},
            }, f, indent=2)

        # ── CLI output ──
        self._print_cli_summary(metrics, output_format, elapsed)

        # ── Checkpoint ──
        self._save_checkpoint_if_best(epoch, metrics)

        # ── Periodic checkpoint ──
        if hasattr(self.args, 'save_interval') and epoch % getattr(self.args, 'save_interval', 5) == 0:
            self._write_ckpt_metadata({}, epoch)

        print()

    def _print_cli_summary(self, metrics: Dict[str, float], fmt: str, elapsed: float):
        """Print a formatted CLI summary of epoch metrics."""
        dur_str = _format_duration(elapsed)

        if fmt == 'phase1':
            loss = metrics.get('loss', float('nan'))
            ce = metrics.get('ce', float('nan'))
            bce = metrics.get('bce', float('nan'))
            top1 = metrics.get('top1', float('nan'))
            top5 = metrics.get('top5', float('nan'))
            top10 = metrics.get('top10', float('nan'))
            rank = metrics.get('mean_rank', float('nan'))
            bce_acc = metrics.get('bce_acc', float('nan'))
            sep = metrics.get('action_sep', float('nan'))
            ent = metrics.get('cd_entropy', float('nan'))
            kl = metrics.get('pos_neg_kl', float('nan'))
            gap = metrics.get('early_late_gap', float('nan'))

            print(f"  ┌─────────────────────────────────────────────────────────────────┐")
            print(f"  │ {'Loss & Acc':45s} │")
            print(f"  │   loss={_format_metric(loss):>10s}   ce={_format_metric(ce):>8s}   bce={_format_metric(bce):>8s}   │")
            print(f"  │   top1={_format_metric(top1, '.3f'):>10s}   top5={_format_metric(top5, '.3f'):>8s}   top10={_format_metric(top10, '.3f'):>8s}   │")
            print(f"  │   rank={_format_metric(rank):>10s}   bce_acc={_format_metric(bce_acc):>8s}   │")
            print(f"  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤")
            print(f"  │ {'Distribution & Actions':41s} │")
            print(f"  │   cd_entropy={_format_metric(ent):>9s}   pos_neg_kl={_format_metric(kl):>9s}   │")
            print(f"  │   action_sep={_format_metric(sep):>9s}   early_late_gap={_format_metric(gap):>9s}   │")
            print(f"  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤")
            print(f"  │ {'Duration':>57s} │")
            print(f"  │ {dur_str:>57s} │")
            print(f"  └─────────────────────────────────────────────────────────────────┘")

        elif fmt == 'phase2':
            reward = metrics.get('reward', float('nan'))
            r1 = metrics.get('r1', float('nan'))
            r3 = metrics.get('r3', float('nan'))
            top1 = metrics.get('top1', float('nan'))
            model_acc = metrics.get('model_acc', float('nan'))
            delta = metrics.get('acc_len_delta', float('nan'))
            sep = metrics.get('action_sep', float('nan'))
            entropy = metrics.get('entropy', float('nan'))
            log_std = metrics.get('log_std_mean', float('nan'))
            adv = metrics.get('advantage_mean', float('nan'))
            v_err = metrics.get('value_error', float('nan'))
            act_grad = metrics.get('actor_grad_norm', float('nan'))

            print(f"  ┌─────────────────────────────────────────────────────────────────┐")
            print(f"  │ {'Reward & Rank':47s} │")
            print(f"  │   reward={_format_metric(reward):>10s}   r1={_format_metric(r1):>8s}   r3={_format_metric(r3):>8s}   │")
            print(f"  │   top1={_format_metric(top1, '.3f'):>10s}   model_acc={_format_metric(model_acc):>8s}   delta={_format_metric(delta):>8s}   │")
            print(f"  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤")
            print(f"  │ {'Policy':53s} │")
            print(f"  │   entropy={_format_metric(entropy):>10s}   log_std={_format_metric(log_std):>8s}   action_sep={_format_metric(sep):>8s}   │")
            print(f"  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤")
            print(f"  │ {'Value & Dynamics':43s} │")
            print(f"  │   advantage={_format_metric(adv):>8s}   value_error={_format_metric(v_err):>8s}   grad={_format_metric(act_grad):>8s}   │")
            print(f"  ├─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ┤")
            print(f"  │ {'Duration':>57s} │")
            print(f"  │ {dur_str:>57s} │")
            print(f"  └─────────────────────────────────────────────────────────────────┘")

    # ── Checkpoint Management ──

    def _save_checkpoint_if_best(self, epoch: int, metrics: Dict[str, float]):
        """Save checkpoint if the target metric improved."""
        metric_name = self._best_metric_name
        current = metrics.get(metric_name, None)
        if current is None:
            return

        if current > self._best_metric:
            self._best_metric = current
            self._write_log(f"  Best {metric_name}: {current:.4f} (epoch {epoch})")
            # Return the values so the caller can actually save the model
            self._latest_best_epoch = epoch
            self._latest_best_value = current

    def save_checkpoint(
        self,
        epoch: int,
        actor_state: Dict,
        critic_state: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
        extra: Optional[Dict] = None,
        is_best: bool = False,
        tag: str = "",
    ):
        """Save a model checkpoint with metadata."""
        suffix = f"_{tag}" if tag else ""
        filename = f"checkpoint{suffix}_epoch{epoch:04d}.pt"
        path = self.checkpoint_dir / filename

        ckpt = {
            'epoch': epoch,
            'phase': self.phase_name,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'actor_state_dict': actor_state,
            'critic_state_dict': critic_state,
            'optimizer_state_dict': optimizer_state,
            'best_metric': self._best_metric,
            'best_metric_name': self._best_metric_name,
        }
        if extra:
            ckpt.update(extra)

        torch.save(ckpt, path)
        self._write_log(f"  Checkpoint saved: {filename}")

        if is_best:
            best_path = self.checkpoint_dir / f"checkpoint_best.pt"
            torch.save(ckpt, best_path)
            self._write_log(f"  Best checkpoint saved: checkpoint_best.pt")
            print(f"     ★ Best {self._best_metric_name}={self._best_metric:.4f} saved")

        return path

    def _write_ckpt_metadata(self, extra: Dict, epoch: int):
        """Lightweight metadata logging for periodic checkpoints."""
        meta_file = self.output_dir / "checkpoint_registry.json"
        registry = []
        if meta_file.exists():
            with open(meta_file) as f:
                registry = json.load(f)
        registry.append({
            'epoch': epoch,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'best_metric': self._best_metric,
            'best_metric_name': self._best_metric_name,
        })
        # Keep only the last 20 entries
        if len(registry) > 20:
            registry = registry[-20:]
        with open(meta_file, 'w') as f:
            json.dump(registry, f, indent=2)

    # ── Progress Bar Helper ──

    def make_progress_bar(self, dataloader, **kwargs):
        """Create a tqdm progress bar with good defaults."""
        return tqdm(
            dataloader,
            bar_format='{l_bar}{bar:20}{r_bar}',
            ncols=80,
            **kwargs,
        )

    # ── Final Summary ──

    def log_final_summary(self, total_time: float):
        """Print and log the final training summary."""
        self._log_separator('═')
        self._write_log(f"  Training complete!")
        self._write_log(f"  Total duration: {_format_duration(total_time)}")
        self._write_log(f"  Best {self._best_metric_name}: {self._best_metric:.4f}")

        print()
        print(f"  ╔══════════════════════════════════════════════════════════╗")
        print(f"  ║              Training Complete!                         ║")
        print(f"  ╠══════════════════════════════════════════════════════════╣")
        print(f"  ║  Total time:  {_format_duration(total_time):>42s} ║")
        print(f"  ║  Best {self._best_metric_name}: {self._best_metric:.4f}{'':>31s} ║")
        print(f"  ║  Output:      {str(self.output_dir):>42s} ║")
        print(f"  ╚══════════════════════════════════════════════════════════╝")
        print()


class NullLogger:
    """No-op logger for debugging or when logging is not needed."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None
    def close(self):
        pass
