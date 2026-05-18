"""
GRPO (Group Relative Policy Optimization) for dynamic alpha adjustment.

GRPO eliminates the critic network used in A2C and estimates advantages
from a group of sampled actions per state:
  A_i = (R_i - mean(R_group)) / std(R_group)

The loss combines a clipped surrogate objective with a KL penalty:
  L = -E[ min(r * A, clip(r) * A) ] + beta * KL(pi_theta || pi_ref)

Supports both fixed-beta and adaptive-beta KL scheduling.

Reference: DeepSeekMath (https://arxiv.org/abs/2402.03300)
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
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.utils.data import DataLoader
from tqdm import tqdm

try:
    from ..model.alpha_model import ContextualBanditAlpha_Dense
    from .alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
    )
    from .offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alpha_model.model.alpha_model import ContextualBanditAlpha_Dense
    from alpha_model.train.alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
    )
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch


def infer_num_buckets(top_k: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    return max((top_k + bucket_size - 1) // bucket_size, 1)


# ─── Utilities ───────────────────────────────────────────────

def get_git_commit_short() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


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
        "hidden_dim": args.hidden_dim,
        "bucket_size": args.bucket_size,
        "bucket_count": bucket_count,
        "max_alpha": args.max_alpha,
        "max_grad_norm": args.max_grad_norm,
        "save_interval": args.save_interval,
        "w1": args.w1, "w2": args.w2, "w3": args.w3,
        "gamma_decay": args.gamma_decay,
        "lambda_r3": args.lambda_r3,
        "group_size": args.group_size,
        "clip_epsilon": args.clip_epsilon,
        "kl_type": args.kl_type,
        "kl_coef": args.kl_coef,
        "target_kl": args.target_kl,
        "kl_horizon": args.kl_horizon,
        "entropy_coef": args.entropy_coef,
        "update_old_policy_every": args.update_old_policy_every,
    }
    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


def write_model_config(output_dir: Path, actor: nn.Module, args, bucket_count: int) -> None:
    def describe_sequential(module: nn.Sequential):
        layers = []
        for index, child in enumerate(module):
            layer_info = {"index": index, "class_name": child.__class__.__name__}
            if isinstance(child, nn.Linear):
                layer_info.update({
                    "in_features": child.in_features,
                    "out_features": child.out_features,
                    "weight_shape": list(child.weight.shape),
                    "bias_shape": list(child.bias.shape) if child.bias is not None else None,
                })
            elif isinstance(child, nn.LayerNorm):
                layer_info.update({
                    "normalized_shape": list(child.normalized_shape),
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
        "shared_hyperparameters": {
            "bucket_size": args.bucket_size,
            "bucket_count": bucket_count,
            "max_alpha": args.max_alpha,
            "ks": list(actor.feature_extractor.ks),
        },
    }

    with (output_dir / "model_config.json").open("w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2, sort_keys=True)


def write_run_summary(output_dir: Path, args, device: torch.device, bucket_count: int, actor: nn.Module) -> None:
    actor_params = sum(p.numel() for p in actor.parameters())
    actor_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)

    summary_lines = [
        "# Alpha GRPO Training Run Summary",
        "",
        "## Run",
        f"- Output dir: `{output_dir}`",
        f"- Data path: `{args.data_path}`",
        f"- Device requested: `{args.device}`",
        f"- Device used: `{device}`",
        f"- Git commit: `{get_git_commit_short() or 'unknown'}`",
        f"- Command: `{' '.join(sys.argv)}`",
        "",
        "## Training Hyperparameters",
        f"- Epochs: `{args.epochs}`",
        f"- Batch size: `{args.batch_size}`",
        f"- Learning rate: `{args.lr}`",
        f"- Top-k: `{args.top_k}`",
        f"- Bucket size: `{args.bucket_size}`",
        f"- Bucket count: `{bucket_count}`",
        f"- Max alpha: `{args.max_alpha}`",
        f"- Max grad norm: `{args.max_grad_norm}`",
        f"- Save interval: `{args.save_interval}`",
        "",
        "## GRPO",
        f"- Group size: `{args.group_size}`",
        f"- Gamma decay (γ): `{args.gamma_decay}`",
        f"- Lambda r3: `{args.lambda_r3}`",
        f"- Clip epsilon: `{args.clip_epsilon}`",
        f"- KL type: `{args.kl_type}`",
        f"- KL coef: `{args.kl_coef}`",
        f"- Target KL: `{args.target_kl}`",
        f"- KL horizon: `{args.kl_horizon}`",
        f"- Entropy coef: `{args.entropy_coef}`",
        f"- Update old policy every: `{args.update_old_policy_every}`",
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
        "## Generated Files",
        "- `run_metadata.json`",
        "- `model_config.json`",
        "- `run_summary.md`",
        "- `training_history.csv`",
        "- `actor_best.pt`",
    ]

    with (output_dir / "run_summary.md").open("w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")


# ─── Gaussian Policy (GRPO variant) ─────────────────────────

class GaussianPolicyGRPO(ContextualBanditAlpha_Dense):
    """Gaussian policy with group sampling and KL divergence for GRPO."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std = nn.Parameter(torch.full((self.num_alpha_buckets,), -0.5, dtype=torch.float32))

    def forward(self, *args, **kwargs):
        mean = super().forward(*args, **kwargs)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std

    def get_dist(self, pos_logits, neg_logits, block_pos, abs_pos, alpha_prev=None):
        mean, log_std = self.forward(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
        return Normal(mean, torch.exp(log_std)), mean, log_std

    def kl_divergence(self, mean, log_std, mean_ref, log_std_ref):
        r"""KL(pi_theta || pi_ref) in closed form for diagonal Gaussians.

        KL(N(mu1, sigma1^2) || N(mu2, sigma2^2))
          = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 0.5

        Returns scalar (mean over batch, sum over buckets and positions).
        """
        var = torch.exp(2 * log_std)
        var_ref = torch.exp(2 * log_std_ref)
        kl = log_std_ref - log_std + (var + (mean - mean_ref) ** 2) / (2 * var_ref) - 0.5
        return kl.sum(dim=(-1, -2)).mean()


# ─── Group Advantage ─────────────────────────────────────────

def compute_group_advantages(rewards: torch.Tensor, batch_size: int, group_size: int) -> torch.Tensor:
    """Normalize rewards within each state's action group.

    Input:  rewards (G*B,) — in [state0 x G, state1 x G, …] order
    Output: advantages (G*B,)
    """
    r2d = rewards.view(batch_size, group_size)
    mean = r2d.mean(dim=1, keepdim=True)
    std = r2d.std(dim=1, keepdim=True) + 1e-8
    return ((r2d - mean) / std).reshape(-1)


# ─── Adaptive KL Controller ─────────────────────────────────

class AdaptiveKLController:
    """Adjusts beta so that recent KL stays near target_kl (PPO-kl style)."""

    def __init__(self, init_kl_coef: float = 0.01, target_kl: float = 0.02, horizon: int = 100):
        self.kl_coef = init_kl_coef
        self.target_kl = target_kl
        self.horizon = max(horizon, 1)
        self.history = []

    def step(self, kl_value: float) -> float:
        self.history.append(kl_value)
        if len(self.history) >= min(self.horizon, 10):
            recent = torch.tensor(self.history[-self.horizon:]).mean().item()
            if recent > 1.5 * self.target_kl:
                self.kl_coef *= 1.2
            elif recent < 0.5 * self.target_kl:
                self.kl_coef *= 0.8
        return self.kl_coef


# ─── Training ────────────────────────────────────────────────

def train_one_epoch(dataloader, actor, actor_ref, opt, args, device, kl_ctrl=None, epoch=0):
    actor.train()
    bucket_count = infer_num_buckets(args.top_k, args.bucket_size)
    bucket_thresholds = compute_bucket_thresholds(args.top_k, bucket_count, args.bucket_size)
    G = args.group_size

    hist = {
        "policy_loss": [], "kl_loss": [], "total_loss": [],
        "reward": [], "r1": [], "r2": [], "r3": [],
        "approx_kl": [], "clip_frac": [], "entropy": [],
        "kl_value": [], "kl_coef": [],
        "baseline_acc": [], "model_acc": [], "group_reward_std": [],
    }

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}", leave=False)):
        pos = batch["pos_logits"].to(device)
        neg = batch["neg_logits"].to(device)
        topk_ids = batch["topk_token_ids"].to(device)
        tgt = batch["target_token_ids"].to(device)
        bpos = batch["block_pos"].to(device)
        apos = batch["abs_pos"].to(device)
        alpha_p = batch["alpha_prev"].to(device)
        baseline = batch["baseline_acc_len"].to(device)
        B = pos.size(0)

        def tile(x):
            return x.repeat_interleave(G, dim=0)

        pos_g = tile(pos)
        neg_g = tile(neg)
        topk_ids_g = tile(topk_ids)
        tgt_g = tile(tgt)
        bpos_g = tile(bpos)
        apos_g = tile(apos)
        alpha_p_g = tile(alpha_p)
        baseline_g = tile(baseline)

        # ── reference policy: sample + log_prob (no grad) ──
        with torch.no_grad():
            dist_ref, mean_ref, log_std_ref = actor_ref.get_dist(
                pos_g, neg_g, bpos_g, apos_g, alpha_p_g
            )
            actions = dist_ref.rsample()
            actions = torch.clamp(actions, -args.max_alpha, args.max_alpha)
            log_prob_ref = dist_ref.log_prob(actions).sum(dim=(-1, -2))

        # ── current policy ──
        dist, mean, log_std = actor.get_dist(
            pos_g, neg_g, bpos_g, apos_g, alpha_p_g
        )
        log_prob = dist.log_prob(actions).sum(dim=(-1, -2))
        entropy = dist.entropy().sum(dim=(-1, -2)).mean()

        # ── rewards ──
        r1, r2, r3, acc_len = compute_reward_components_vectorized(
            pos_g, neg_g, topk_ids_g, tgt_g, actions, baseline_g,
            topk=args.top_k, bucket_thresholds=bucket_thresholds,
            gamma=args.gamma_decay, lambda_=args.lambda_r3,
        )
        rewards = total_reward(r1, r2, r3, w1=args.w1, w2=args.w2, w3=args.w3)

        # ── group advantages ──
        advantages = compute_group_advantages(rewards, B, G)

        # ── importance ratio ──
        ratio = torch.exp(log_prob - log_prob_ref)

        # ── clipped surrogate ──
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - args.clip_epsilon, 1.0 + args.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # ── KL penalty ──
        kl = actor.kl_divergence(mean, log_std, mean_ref, log_std_ref)
        if kl_ctrl is not None:
            kl_coef = kl_ctrl.step(kl.item())
        else:
            kl_coef = args.kl_coef
        kl_loss = kl_coef * kl
        total_loss = policy_loss + kl_loss

        # optional entropy bonus
        if args.entropy_coef > 0.0:
            total_loss = total_loss - args.entropy_coef * entropy

        # ── backward ──
        opt.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
        opt.step()

        # ── periodically sync reference ──
        if args.update_old_policy_every > 0 and (batch_idx + 1) % args.update_old_policy_every == 0:
            actor_ref.load_state_dict(actor.state_dict())

        # ── logging ──
        with torch.no_grad():
            hist["policy_loss"].append(policy_loss.item())
            hist["kl_loss"].append(kl_loss.item())
            hist["total_loss"].append(total_loss.item())
            hist["reward"].append(rewards.mean().item())
            hist["r1"].append(r1.mean().item())
            hist["r2"].append(r2.mean().item())
            hist["r3"].append(r3.mean().item())
            hist["approx_kl"].append(((log_prob_ref - log_prob).mean()).item())
            hist["clip_frac"].append(
                ((ratio < 1.0 - args.clip_epsilon) | (ratio > 1.0 + args.clip_epsilon))
                .float().mean().item()
            )
            hist["entropy"].append(entropy.item())
            hist["kl_value"].append(kl.item())
            hist["kl_coef"].append(kl_coef)
            hist["baseline_acc"].append(baseline.mean().item())
            hist["model_acc"].append(acc_len.float().mean().item())
            hist["group_reward_std"].append(
                rewards.reshape(-1, G).std(dim=1).mean().item()
            )

    # ── print epoch summary ──
    def stats(values):
        t = torch.tensor(values)
        if t.numel() <= 1:
            return t.item(), 0.0, t.item(), t.item()
        return t.mean().item(), t.std(unbiased=False).item(), t.min().item(), t.max().item()

    def fmt(name, values):
        m, s, lo, hi = stats(values)
        return f"{name}: {m:.4f}±{s:.4f} [{lo:.4f}, {hi:.4f}]"

    for k in hist:
        print(f"  {fmt(k, hist[k])}")

    return {k: stats(v)[:2] for k, v in hist.items()}


# ─── CLI ─────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="GRPO training for DFlash alpha model")

    p.add_argument("--data-path", required=True)
    p.add_argument("--output-dir", default="checkpoints_grpo")
    p.add_argument("--max-records", type=int, default=0,
                   help="limit to N records for debugging (0 = all)")

    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--save-interval", type=int, default=5)

    p.add_argument("--top-k", type=int, default=32)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--bucket-size", type=int, default=8)
    p.add_argument("--max-alpha", type=float, default=2.0)

    p.add_argument("--gamma-decay", type=float, default=7.0,
                   help="positional decay γ for reward")
    p.add_argument("--lambda-r3", type=float, default=3.0,
                   help="penalty multiplier for shorter acceptance")
    p.add_argument("--w1", type=float, default=0.1)
    p.add_argument("--w2", type=float, default=0.1)
    p.add_argument("--w3", type=float, default=1.0)

    # GRPO
    p.add_argument("--group-size", type=int, default=8,
                   help="action samples per state (G)")
    p.add_argument("--clip-epsilon", type=float, default=0.2,
                   help="clip range for surrogate objective")
    p.add_argument("--kl-type", choices=["fixed", "adaptive"], default="fixed",
                   help="KL penalty type: fixed beta or adaptive beta")
    p.add_argument("--kl-coef", type=float, default=0.01,
                   help="KL penalty coefficient (fixed mode) / initial (adaptive)")
    p.add_argument("--target-kl", type=float, default=0.02,
                   help="target KL for adaptive scheduler")
    p.add_argument("--kl-horizon", type=int, default=100,
                   help="KL adaptation window (batches)")
    p.add_argument("--entropy-coef", type=float, default=0.0,
                   help="optional entropy bonus (default 0 = disabled)")
    p.add_argument("--update-old-policy-every", type=int, default=10,
                   help="sync reference policy every N batches (0 = once per epoch)")

    p.add_argument("--device", default="cuda")

    args = p.parse_args()
    if args.bucket_size <= 0:
        p.error("--bucket-size must be positive")
    if args.group_size < 2:
        p.error("--group-size must be >= 2 (need at least 2 for group std)")
    return args


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    bucket_count = infer_num_buckets(args.top_k, args.bucket_size)
    print(f"[train_GRPO] device={device}  buckets={bucket_count}  G={args.group_size}")

    # dataset
    ds = HDF5BanditDataset(args.data_path)
    if args.max_records > 0:
        ds.num_records = min(args.max_records, len(ds))
        print(f"  using {ds.num_records} / {len(ds)} records")

    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_bandit_batch, num_workers=0,
    )

    # actor + reference
    actor = GaussianPolicyGRPO(
        top_k=args.top_k, hidden_dim=args.hidden_dim,
        num_alpha_buckets=bucket_count, max_alpha=args.max_alpha,
    ).to(device)
    actor_ref = GaussianPolicyGRPO(
        top_k=args.top_k, hidden_dim=args.hidden_dim,
        num_alpha_buckets=bucket_count, max_alpha=args.max_alpha,
    ).to(device)
    actor_ref.load_state_dict(actor.state_dict())
    actor_ref.eval()

    opt = torch.optim.Adam(actor.parameters(), lr=args.lr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    write_run_metadata(out, args, device, bucket_count)
    write_model_config(out, actor, args, bucket_count)
    write_run_summary(out, args, device, bucket_count, actor)

    # KL controller
    kl_ctrl = None
    if args.kl_type == "adaptive":
        kl_ctrl = AdaptiveKLController(args.kl_coef, args.target_kl, args.kl_horizon)

    # CSV
    keys = [
        "policy_loss", "kl_loss", "total_loss",
        "reward", "r1", "r2", "r3",
        "approx_kl", "clip_frac", "entropy",
        "kl_value", "kl_coef",
        "baseline_acc", "model_acc", "group_reward_std",
    ]
    csv_path = out / "training_history.csv"
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["epoch"] + [f"{k}_mean" for k in keys] + [f"{k}_std" for k in keys])

    best_reward = -float("inf")

    for epoch in range(1, args.epochs + 1):
        print(f"─── Epoch {epoch}/{args.epochs} ───")
        ep_stats = train_one_epoch(loader, actor, actor_ref, opt, args, device, kl_ctrl, epoch)

        row = [epoch]
        for k in keys:
            row.extend([ep_stats[k][0], ep_stats[k][1]])
        csv_writer.writerow(row)
        csv_file.flush()

        rew_mean = ep_stats["reward"][0]
        if rew_mean > best_reward:
            best_reward = rew_mean
            torch.save(actor.state_dict(), out / "actor_best.pt")
            print(f"  [+] new best reward: {best_reward:.4f}")

        if epoch % args.save_interval == 0:
            torch.save(actor.state_dict(), out / f"actor_epoch{epoch}.pt")

        # sync reference at epoch end (if not doing batch-level sync)
        if args.update_old_policy_every <= 0:
            actor_ref.load_state_dict(actor.state_dict())

    csv_file.close()
    print("GRPO training completed.")
    print(f"  Best reward: {best_reward:.4f}")
    print(f"  Output: {out}")


if __name__ == "__main__":
    main()
