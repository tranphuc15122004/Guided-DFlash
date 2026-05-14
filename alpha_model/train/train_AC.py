"""
Task T3: Huấn luyện RL bandit (A2C) cho điều chỉnh α động.

Hỗ trợ batch processing – không lặp per-sample.
Có gradient clipping, entropy coef configurable.
"""

import argparse
import csv
import json
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Import từ package (chạy với `python -m alpha_model.train.train_AC`) ──
try:
    from ..model.alpha_model import ContextualBanditAlpha_Dense, Critic
    from .alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
        simulate_acceptance_length_vectorized,
        check_alpha_value,
    )
    from .offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
except (ImportError, ValueError):
    # Fallback khi chạy trực tiếp: thêm project root vào sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alpha_model.model.alpha_model import ContextualBanditAlpha_Dense, Critic
    from alpha_model.train.alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
        simulate_acceptance_length_vectorized,
        check_alpha_value,
    )
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch


def infer_num_buckets(top_k: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    return max((top_k + bucket_size - 1) // bucket_size, 1)


def get_git_commit_short() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip() or None


def write_run_metadata(output_dir: Path, args, device: torch.device, bucket_count: int) -> None:
    metadata = {
        "run_started_utc": datetime.now(timezone.utc).isoformat(),
        "command": " ".join(sys.argv),
        "cwd": os.getcwd(),
        "hostname": socket.gethostname(),
        "username": os.environ.get("USER") or os.environ.get("USERNAME"),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "device_requested": args.device,
        "device_used": str(device),
        "git_commit_short": get_git_commit_short(),
        "data_path": args.data_path,
        "output_dir": str(output_dir),
        "max_records": args.max_records,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "lr": args.lr,
        "top_k": args.top_k,
        "bucket_size": args.bucket_size,
        "bucket_count": bucket_count,
        "max_alpha": args.max_alpha,
        "gamma_decay": args.gamma_decay,
        "lambda_r3": args.lambda_r3,
        "w1": args.w1,
        "w2": args.w2,
        "w3": args.w3,
        "entropy_coef": args.entropy_coef,
        "max_grad_norm": args.max_grad_norm,
        "save_interval": args.save_interval,
        "hidden_dim": args.hidden_dim,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def write_model_config(output_dir: Path, actor: nn.Module, critic: nn.Module, args, bucket_count: int) -> None:
    def describe_sequential(module: nn.Sequential):
        layers = []
        for index, child in enumerate(module):
            layer_info = {
                "index": index,
                "class_name": child.__class__.__name__,
            }
            if isinstance(child, nn.Linear):
                layer_info.update({
                    "in_features": child.in_features,
                    "out_features": child.out_features,
                    "weight_shape": list(child.weight.shape),
                    "bias_shape": list(child.bias.shape) if child.bias is not None else None,
                })
            elif isinstance(child, nn.LayerNorm):
                layer_info.update({
                    "normalized_shape": list(child.normalized_shape) if isinstance(child.normalized_shape, tuple) else [child.normalized_shape],
                    "eps": child.eps,
                })
            elif isinstance(child, nn.Dropout):
                layer_info["p"] = child.p
            layers.append(layer_info)
        return layers

    def summarize_feature_extractor(extractor: nn.Module):
        return {
            "class_name": extractor.__class__.__name__,
            "top_k": getattr(extractor, "top_k", None),
            "ks": list(getattr(extractor, "ks", [])),
            "normalize_position": getattr(extractor, "normalize_position", None),
            "alpha_prev_dim": getattr(extractor, "alpha_prev_dim", None),
            "max_block_pos_shape": list(extractor.max_block_pos.shape) if hasattr(extractor, "max_block_pos") else None,
            "max_abs_pos_shape": list(extractor.max_abs_pos.shape) if hasattr(extractor, "max_abs_pos") else None,
        }

    actor_params = sum(p.numel() for p in actor.parameters())
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in critic.parameters())
    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)

    model_config = {
        "actor": {
            "class_name": actor.__class__.__name__,
            "base_class_name": actor.__class__.__mro__[1].__name__ if len(actor.__class__.__mro__) > 1 else None,
            "parameters": {
                "total": actor_params,
                "trainable": actor_trainable,
                "non_trainable": actor_params - actor_trainable,
            },
            "architecture": {
                "top_k": getattr(actor, "top_k", args.top_k),
                "hidden_dim": getattr(actor, "hidden_dim", args.hidden_dim),
                "num_alpha_buckets": getattr(actor, "num_alpha_buckets", bucket_count),
                "max_alpha": getattr(actor, "max_alpha", args.max_alpha),
                "max_seq_len": getattr(actor, "max_seq_len", 15),
                "input_dim": getattr(actor, "input_dim", None),
                "feature_extractor": summarize_feature_extractor(actor.feature_extractor),
                "token_projector": describe_sequential(actor.token_projector),
                "block_mlp": describe_sequential(actor.block_mlp),
                "output_shape_per_forward": ["batch_size", "seq_len", getattr(actor, "num_alpha_buckets", bucket_count)],
                "log_std": {
                    "shape": list(actor.log_std.shape) if hasattr(actor, "log_std") else None,
                    "init_value": float(actor.log_std.detach().flatten()[0].item()) if hasattr(actor, "log_std") else None,
                },
            },
        },
        "critic": {
            "class_name": critic.__class__.__name__,
            "parameters": {
                "total": critic_params,
                "trainable": critic_trainable,
                "non_trainable": critic_params - critic_trainable,
            },
            "architecture": {
                "top_k": getattr(critic.feature_extractor, "top_k", args.top_k) if getattr(critic, "feature_extractor", None) is not None else None,
                "hidden_dim": args.hidden_dim,
                "num_alpha_buckets": getattr(critic, "num_alpha_buckets", bucket_count),
                "feature_extractor": summarize_feature_extractor(critic.feature_extractor) if getattr(critic, "feature_extractor", None) is not None else None,
                "encoder": describe_sequential(critic.encoder.net),
                "value_head": describe_sequential(critic.value_head),
                "output_shape_per_forward": ["batch_size", "seq_len"],
                "reduction_note": "critic.forward returns per-token values; training code reduces with mean(dim=1)",
            },
        },
        "shared_hyperparameters": {
            "bucket_size": args.bucket_size,
            "bucket_count": bucket_count,
            "max_alpha": args.max_alpha,
            "ks": list(actor.feature_extractor.ks),
        },
    }
    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)


def write_run_summary(output_dir: Path, args, device: torch.device, bucket_count: int, actor: nn.Module, critic: nn.Module) -> None:
    actor_params = sum(p.numel() for p in actor.parameters())
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    critic_params = sum(p.numel() for p in critic.parameters())
    critic_trainable = sum(p.numel() for p in critic.parameters() if p.requires_grad)

    summary_lines = [
        "# Alpha Training Run Summary",
        "",
        "## Run",
        f"- Output dir: `{output_dir}`",
        f"- Data path: `{args.data_path}`",
        f"- Device requested: `{args.device}`",
        f"- Device used: `{device}`",
        f"- Git commit: `{get_git_commit_short() or 'unknown'}`",
        f"- Command: `{ ' '.join(sys.argv) }`",
        "",
        "## Training Hyperparameters",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.lr}`",
        f"- Top-k: `{args.top_k}`",
        f"- Bucket size: `{args.bucket_size}`",
        f"- Bucket count: `{bucket_count}`",
        f"- Max alpha: `{args.max_alpha}`",
        f"- Entropy coef: `{args.entropy_coef}`",
        f"- Max grad norm: `{args.max_grad_norm}`",
        f"- Save interval: `{args.save_interval}`",
        "",
        "## Actor",
        f"- Class: `{actor.__class__.__name__}`",
        f"- Total params: `{actor_params}`",
        f"- Trainable params: `{actor_trainable}`",
        f"- Input dim: `{getattr(actor, 'input_dim', 'n/a')}`",
        f"- Hidden dim: `{getattr(actor, 'hidden_dim', args.hidden_dim)}`",
        f"- Max seq len: `{getattr(actor, 'max_seq_len', 'n/a')}`",
        f"- Alpha bucket output shape: `[batch_size, seq_len, {bucket_count}]`",
        "",
        "## Critic",
        f"- Class: `{critic.__class__.__name__}`",
        f"- Total params: `{critic_params}`",
        f"- Trainable params: `{critic_trainable}`",
        f"- Output shape: `[batch_size, seq_len]`",
        f"- Reduction in training: `mean(dim=1)`",
        "",
        "## Generated Files",
        "- `run_metadata.json`",
        "- `model_config.json`",
        "- `run_summary.md`",
        "- `training_history.csv`",
        "- `actor_best.pt` / `critic_best.pt`",
    ]

    with (output_dir / "run_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


# -------------------- Gaussian Policy (extends actor) --------------------
class GaussianPolicy(ContextualBanditAlpha_Dense):
    """Actor với đầu ra mean và log_std học được (global cho num_alpha_buckets buckets)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std = nn.Parameter(torch.full((self.num_alpha_buckets,), -0.5, dtype=torch.float32))  # std ≈ 0.6

    def forward(self, *args, **kwargs):
        mean = super().forward(*args, **kwargs)          # (B, S, num_alpha_buckets)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std


# -------------------- Một epoch huấn luyện (full batch) --------------------
def train_one_epoch(dataloader, actor, critic, opt_actor, opt_critic, args, device):
    actor.train()
    critic.train()
    bucket_count = infer_num_buckets(args.top_k, args.bucket_size)

    # Thu thập metrics từ tất cả batches để tính statistics cuối epoch
    hist_actor_loss = []
    hist_critic_loss = []
    hist_reward = []
    hist_r1 = []
    hist_r2 = []
    hist_r3 = []
    hist_entropy = []
    hist_baseline_acc = []
    hist_model_acc = []
    hist_min_alpha_bucket_acc = []
    hist_strict_min_alpha_bucket_acc = []
    hist_target_in_topk_rate = []
    hist_alpha_gap_mean = []
    hist_per_bucket_acc = [[] for _ in range(bucket_count)]
    hist_pred_min_bucket_rate = [[] for _ in range(bucket_count)]
    hist_true_bucket_rate = [[] for _ in range(bucket_count)]

    bucket_thresholds = compute_bucket_thresholds(args.top_k, bucket_count, args.bucket_size)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # ── Chuyển dữ liệu lên device ──
        pos = batch["pos_logits"].to(device)           # (B, S, K)
        neg = batch["neg_logits"].to(device)           # (B, S, K)
        topk_ids = batch["topk_token_ids"].to(device)  # (B, S, K)
        tgt = batch["target_token_ids"].to(device)     # (B, S)
        bpos = batch["block_pos"].to(device)           # (B, S)
        apos = batch["abs_pos"].to(device)             # (B, S)
        alpha_p = batch["alpha_prev"].to(device)       # (B, S, D) - được pad/truncate về num_buckets trong model
        baseline = batch["baseline_acc_len"].to(device)  # (B,) float32

        # ── Actor forward toàn batch ──
        mean, log_std = actor(pos, neg, bpos, apos, alpha_p)  # (B, S, 3)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()                                # reparameterised sample
        action = torch.clamp(action, -args.max_alpha, args.max_alpha)

        # ── Tính reward toàn batch ──
        r1, r2, r3, model_acc_len = compute_reward_components_vectorized(
            pos, neg, topk_ids, tgt, action, baseline,
            topk=args.top_k, bucket_thresholds=bucket_thresholds,
            gamma=args.gamma_decay, lambda_=args.lambda_r3,
        )  # mỗi (B,)
        reward = total_reward(r1, r2, r3, w1=args.w1, w2=args.w2, w3=args.w3)  # (B,)

        alpha_check = check_alpha_value(pos, topk_ids, tgt, action, bucket_thresholds)

        # ── Critic forward ──
        value = critic(pos, neg, bpos, apos, alpha_p).mean(dim=1)  # (B,)
        advantage = reward - value.detach()

        # ── Actor loss ──
        log_prob = dist.log_prob(action).sum(dim=-1).sum(dim=-1)  # (B,)
        actor_loss = -(log_prob * advantage).mean()
        entropy = dist.entropy().sum(dim=-1).sum(dim=-1).mean()
        actor_loss = actor_loss - args.entropy_coef * entropy

        # ── Critic loss ──
        critic_loss = F.mse_loss(value, reward)

        # ── Cập nhật actor ──
        opt_actor.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt_actor.step()

        # ── Cập nhật critic ──
        opt_critic.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
        opt_critic.step()

        # Lưu metrics để cuối epoch tính statistics
        hist_actor_loss.append(actor_loss.item())
        hist_critic_loss.append(critic_loss.item())
        hist_reward.append(reward.mean().item())
        hist_r1.append(r1.mean().item())
        hist_r2.append(r2.mean().item())
        hist_r3.append(r3.mean().item())
        hist_entropy.append(entropy.item())
        hist_baseline_acc.append(baseline.mean().item())
        hist_model_acc.append(model_acc_len.float().mean().item())
        hist_min_alpha_bucket_acc.append(alpha_check["min_alpha_bucket_acc"])
        hist_strict_min_alpha_bucket_acc.append(alpha_check["strict_min_alpha_bucket_acc"])
        hist_target_in_topk_rate.append(alpha_check["target_in_topk_rate"])
        hist_alpha_gap_mean.append(alpha_check["alpha_gap_mean"])
        for bucket_idx in range(bucket_count):
            hist_per_bucket_acc[bucket_idx].append(alpha_check["per_bucket_acc"].get(bucket_idx, 0.0))
            hist_pred_min_bucket_rate[bucket_idx].append(alpha_check["pred_min_bucket_rates"][bucket_idx])
            hist_true_bucket_rate[bucket_idx].append(alpha_check["bucket_rates"][bucket_idx])

    # ── Tính statistics cuối epoch ──
    # Các metric được gom theo 4 nhóm để dễ đọc khi debug:
    # - optimization: actor_loss, critic_loss, entropy
    # - reward: reward, r1, r2, r3
    # - acceptance quality: baseline_acc, model_acc
    # - bucket diagnostics: target nằm ở bucket nào, alpha nhỏ nhất nằm ở bucket nào,
    #   và alpha của bucket chứa target có thực sự nhỏ nhất hay không.
    def stats(values):
        t = torch.tensor(values)
        if t.numel() == 1:
            value = t.item()
            return value, 0.0, value, value
        return t.mean().item(), t.std(unbiased=False).item(), t.min().item(), t.max().item()

    def fmt_stats(name, values):
        m, s, lo, hi = stats(values)
        return f"{name}: {m:.4f}±{s:.4f} [{lo:.4f}, {hi:.4f}]"

    # actor_loss / critic_loss: mức độ tối ưu của policy và value function.
    # reward: reward tổng hợp theo công thức w1*r1 + w2*r2 + w3*r3.
    # r1/r2/r3: ba thành phần reward riêng lẻ.
    # entropy: mức exploration của Gaussian policy; cao hơn nghĩa là policy còn phân tán hơn.
    # baseline_acc: độ dài accept trước khi áp dụng alpha động.
    # model_acc: độ dài accept sau khi sample alpha từ model.
    # min_alpha_bucket_acc: bucket chứa target token có alpha nhỏ nhất không (tie-aware).
    # strict_min_alpha_bucket_acc: cùng câu hỏi nhưng dùng argmin cứng.
    # target_in_topk_rate: target token có nằm trong top-k hay không.
    # alpha_gap_mean: alpha(target bucket) - alpha nhỏ nhất trong block.
    # per_bucket_*: accuracy theo từng bucket thật sự của target token.
    # true_bucket_rate_* / pred_min_bucket_rate_*: phân bố bucket thật và bucket model chọn là nhỏ nhất.
    print(f"  {fmt_stats('actor_loss', hist_actor_loss)}")
    print(f"  {fmt_stats('critic_loss', hist_critic_loss)}")
    print(f"  {fmt_stats('reward', hist_reward)}")
    print(f"  {fmt_stats('r1', hist_r1)}")
    print(f"  {fmt_stats('r2', hist_r2)}")
    print(f"  {fmt_stats('r3', hist_r3)}")
    print(f"  {fmt_stats('entropy', hist_entropy)}")
    print(f"  {fmt_stats('baseline_acc', hist_baseline_acc)}")
    print(f"  {fmt_stats('model_acc', hist_model_acc)}")
    print(f"  {fmt_stats('min_alpha_bucket_acc', hist_min_alpha_bucket_acc)}")
    print(f"  {fmt_stats('strict_min_alpha_bucket_acc', hist_strict_min_alpha_bucket_acc)}")
    print(f"  {fmt_stats('target_in_topk_rate', hist_target_in_topk_rate)}")
    print(f"  {fmt_stats('alpha_gap_mean', hist_alpha_gap_mean)}")
    for bucket_idx in range(bucket_count):
        print(f"  {fmt_stats(f'per_bucket_{bucket_idx}', hist_per_bucket_acc[bucket_idx])}")
    for bucket_idx in range(bucket_count):
        print(f"  {fmt_stats(f'true_bucket_rate_{bucket_idx}', hist_true_bucket_rate[bucket_idx])}")
    for bucket_idx in range(bucket_count):
        print(f"  {fmt_stats(f'pred_min_bucket_rate_{bucket_idx}', hist_pred_min_bucket_rate[bucket_idx])}")

    return (hist_actor_loss, hist_critic_loss, hist_reward, hist_r1, hist_r2, hist_r3,
            hist_entropy, hist_baseline_acc, hist_model_acc,
            hist_min_alpha_bucket_acc, hist_strict_min_alpha_bucket_acc,
            hist_target_in_topk_rate, hist_alpha_gap_mean,
            hist_per_bucket_acc, hist_true_bucket_rate, hist_pred_min_bucket_rate)


# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train alpha RL agent for DFlash")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-records", type=int, default=0,
                        help="limit to N records for debugging (0 = use all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--bucket-size",
        type=int,
        default=8,
        help="bucket width; num_buckets is inferred as ceil(top_k / bucket_size)",
    )
    parser.add_argument("--max-alpha", type=float, default=2.0)
    parser.add_argument("--gamma-decay", type=float, default=7.0, help="positional decay γ")
    parser.add_argument("--lambda-r3", type=float, default=3.0, help="penalty for shorter acceptance")
    parser.add_argument("--w1", type=float, default=0.1, help="weight for r1 (rank improvement)")
    parser.add_argument("--w2", type=float, default=0.1, help="weight for r2 (top-1 bonus)")
    parser.add_argument("--w3", type=float, default=1.0, help="weight for r3 (acceptance length)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--save-interval", type=int, default=5, help="Epoch interval to save periodic checkpoints")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    if args.bucket_size <= 0:
        parser.error("--bucket-size must be positive")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train_AC] device: {device}")
    bucket_count = infer_num_buckets(args.top_k, args.bucket_size)
    print(f"[train_AC] bucket config: bucket_size={args.bucket_size}, num_buckets={bucket_count}")

    dataset = HDF5BanditDataset(args.data_path)

    if args.max_records > 0:
        original_len = len(dataset)
        dataset.num_records = min(args.max_records, original_len)
        print(f"Limited to {dataset.num_records} / {original_len} records")
        # Also cap the underlying h5 references via slicing in __getitem__
        class SubsetHDF5BanditDataset(HDF5BanditDataset):
            def __len__(self):
                return dataset.num_records
        dataset.__class__ = SubsetHDF5BanditDataset

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_bandit_batch,
        num_workers=0,   # h5py không fork-safe
    )

    actor = GaussianPolicy(
        top_k=args.top_k, hidden_dim=args.hidden_dim,
        num_alpha_buckets=bucket_count, max_alpha=args.max_alpha,
    ).to(device)
    critic = Critic(top_k=args.top_k, hidden_dim=args.hidden_dim, num_alpha_buckets=bucket_count).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_run_metadata(out, args, device, bucket_count)
    write_model_config(out, actor, critic, args, bucket_count)
    write_run_summary(out, args, device, bucket_count, actor, critic)

    best_reward = -float("inf")
    
    # Mở file CSV để ghi log chi tiết
    csv_path = out / "training_history.csv"
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    
    # Xây dựng header: các cột cốt lõi + bucket diagnostics
    csv_header = [
        "epoch",
        # Optimization metrics.
        "actor_loss_mean", "actor_loss_std",
        "critic_loss_mean", "critic_loss_std",
        # Reward decomposition.
        "reward_mean", "reward_std",
        "r1_mean", "r1_std",
        "r2_mean", "r2_std",
        "r3_mean", "r3_std",
        # Exploration / acceptance quality.
        "entropy_mean", "entropy_std",
        "baseline_acc_mean", "baseline_acc_std",
        "model_acc_mean", "model_acc_std",
        # Bucket diagnostics tổng quát.
        "min_alpha_bucket_acc_mean", "min_alpha_bucket_acc_std",
        "strict_min_alpha_bucket_acc_mean", "strict_min_alpha_bucket_acc_std",
        "target_in_topk_rate_mean", "target_in_topk_rate_std",
        "alpha_gap_mean", "alpha_gap_std",
    ]
    # Thêm các cột per-bucket: accuracy, true rate, pred min rate
    for bucket_idx in range(bucket_count):
        csv_header.append(f"per_bucket_{bucket_idx}_mean")
        csv_header.append(f"per_bucket_{bucket_idx}_std")
    for bucket_idx in range(bucket_count):
        csv_header.append(f"true_bucket_rate_{bucket_idx}_mean")
        csv_header.append(f"true_bucket_rate_{bucket_idx}_std")
    for bucket_idx in range(bucket_count):
        csv_header.append(f"pred_min_bucket_rate_{bucket_idx}_mean")
        csv_header.append(f"pred_min_bucket_rate_{bucket_idx}_std")
    
    csv_writer.writerow(csv_header)

    for epoch in range(1, args.epochs + 1):
        print(f"─── Epoch {epoch}/{args.epochs} ───")
        hist = train_one_epoch(dataloader, actor, critic, opt_actor, opt_critic, args, device)

        # Từ lịch sử theo batch, gom lại thành một summary cho toàn epoch.
        # std được đặt 0.0 khi chỉ có một batch để tránh nan trong smoke test.
        def stats(vals):
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            t = torch.tensor(vals)
            if t.numel() == 1:
                value = t.item()
                return value, 0.0, value, value
            return t.mean().item(), t.std(unbiased=False).item(), t.min().item(), t.max().item()

        act_mean, act_std, act_min, act_max = stats(hist[0])
        crit_mean, crit_std, crit_min, crit_max = stats(hist[1])
        rew_mean, rew_std, rew_min, rew_max = stats(hist[2])
        r1_mean, r1_std, r1_min, r1_max = stats(hist[3])
        r2_mean, r2_std, r2_min, r2_max = stats(hist[4])
        r3_mean, r3_std, r3_min, r3_max = stats(hist[5])
        ent_mean, ent_std, ent_min, ent_max = stats(hist[6] if len(hist) > 6 else [])
        base_acc_mean, base_acc_std, base_acc_min, base_acc_max = stats(hist[7] if len(hist) > 7 else [])
        mod_acc_mean, mod_acc_std, mod_acc_min, mod_acc_max = stats(hist[8] if len(hist) > 8 else [])
        # Bucket diagnostics: hist[9..12] là các list batch-level đơn
        min_a_mean, min_a_std, _, _ = stats(hist[9] if len(hist) > 9 else [])
        strict_min_mean, strict_min_std, _, _ = stats(hist[10] if len(hist) > 10 else [])
        topk_rate_mean, topk_rate_std, _, _ = stats(hist[11] if len(hist) > 11 else [])
        alpha_gap_mean, alpha_gap_std, _, _ = stats(hist[12] if len(hist) > 12 else [])

        # Per-bucket metrics: hist[13..15] là list-of-lists [bucket_idx][batch_values]
        per_bucket_means_stds = []
        if len(hist) > 13 and hist[13]:
            for bucket_vals in hist[13]:
                m, s, _, _ = stats(bucket_vals)
                per_bucket_means_stds.extend([m, s])
        else:
            per_bucket_means_stds = [0.0, 0.0] * bucket_count

        true_rate_means_stds = []
        if len(hist) > 14 and hist[14]:
            for bucket_vals in hist[14]:
                m, s, _, _ = stats(bucket_vals)
                true_rate_means_stds.extend([m, s])
        else:
            true_rate_means_stds = [0.0, 0.0] * bucket_count

        pred_rate_means_stds = []
        if len(hist) > 15 and hist[15]:
            for bucket_vals in hist[15]:
                m, s, _, _ = stats(bucket_vals)
                pred_rate_means_stds.extend([m, s])
        else:
            pred_rate_means_stds = [0.0, 0.0] * bucket_count

        print(f"✔ Epoch {epoch:3d} | "
              f"actor_loss={act_mean:.4f}  "
              f"critic_loss={crit_mean:.4f}  "
              f"reward={rew_mean:.4f}  "
              f"r1={r1_mean:.4f}  "
              f"r2={r2_mean:.4f}  "
              f"r3={r3_mean:.4f}")
        print(f"            | "
              f"baseline_acc={base_acc_mean:.4f}  "
              f"model_acc={mod_acc_mean:.4f}  "
              f"min_alpha_acc={min_a_mean:.4f}")
        print()
        
        csv_writer.writerow([
            epoch,
            act_mean, act_std,
            crit_mean, crit_std,
            rew_mean, rew_std,
            r1_mean, r1_std,
            r2_mean, r2_std,
            r3_mean, r3_std,
            ent_mean, ent_std,
            base_acc_mean, base_acc_std,
            mod_acc_mean, mod_acc_std,
            min_a_mean, min_a_std,
            strict_min_mean, strict_min_std,
            topk_rate_mean, topk_rate_std,
            alpha_gap_mean, alpha_gap_std,
            *per_bucket_means_stds,
            *true_rate_means_stds,
            *pred_rate_means_stds,
        ])
        csv_file.flush()

        # Lưu best checkpoint dựa trên reward
        if rew_mean > best_reward:
            best_reward = rew_mean
            torch.save(actor.state_dict(), out / "actor_best.pt")
            torch.save(critic.state_dict(), out / "critic_best.pt")
            print(f"  [+] New best reward: {best_reward:.4f} - Saved best checkpoint.\n")

        # Lưu checkpoint định kỳ
        if epoch % args.save_interval == 0:
            torch.save(actor.state_dict(), out / f"actor_epoch{epoch}.pt")
            torch.save(critic.state_dict(), out / f"critic_epoch{epoch}.pt")

    csv_file.close()
    print("Training completed!")


if __name__ == "__main__":
    main()