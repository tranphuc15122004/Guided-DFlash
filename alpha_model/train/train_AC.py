"""
Task T3: Huấn luyện RL bandit (A2C) cho điều chỉnh α động.

Hỗ trợ batch processing – không lặp per-sample.
Có gradient clipping, entropy coef configurable.
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

# ── Import từ package (chạy với `python -m alpha_model.train.train_AC`) ──
try:
    from ..model.alpha_model import ContextualBanditAlpha, Critic
    from .alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
    )
    from .offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
except (ImportError, ValueError):
    # Fallback khi chạy trực tiếp: thêm project root vào sys.path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alpha_model.model.alpha_model import ContextualBanditAlpha, Critic
    from alpha_model.train.alpha_simulate import (
        compute_reward_components_vectorized,
        total_reward,
        compute_bucket_thresholds,
    )
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch


# -------------------- Gaussian Policy (extends actor) --------------------
class GaussianPolicy(ContextualBanditAlpha):
    """Actor với đầu ra mean và log_std học được (global cho 3 buckets)."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log_std = nn.Parameter(torch.full((3,), -0.5, dtype=torch.float32))  # std ≈ 0.6

    def forward(self, *args, **kwargs):
        mean = super().forward(*args, **kwargs)          # (B, S, 3)
        log_std = self.log_std.view(1, 1, -1).expand_as(mean)
        return mean, log_std


# -------------------- Một epoch huấn luyện (full batch) --------------------
def train_one_epoch(dataloader, actor, critic, opt_actor, opt_critic, args, device):
    actor.train()
    critic.train()
    total_actor_loss = 0.0
    total_critic_loss = 0.0
    total_rew = 0.0
    batches = 0

    bucket_thresholds = compute_bucket_thresholds(args.top_k, args.num_buckets)

    for batch in tqdm(dataloader, desc="Training", leave=False):
        # ── Chuyển dữ liệu lên device ──
        pos = batch["pos_logits"].to(device)           # (B, S, K)
        neg = batch["neg_logits"].to(device)           # (B, S, K)
        topk_ids = batch["topk_token_ids"].to(device)  # (B, S, K)
        tgt = batch["target_token_ids"].to(device)     # (B, S)
        bpos = batch["block_pos"].to(device)           # (B, S)
        apos = batch["abs_pos"].to(device)             # (B, S)
        alpha_p = batch["alpha_prev"].to(device)       # (B, S, 3)
        baseline = batch["baseline_acc_len"].to(device)  # (B,) float32

        # ── Actor forward toàn batch ──
        mean, log_std = actor(pos, neg, bpos, apos, alpha_p)  # (B, S, 3)
        std = torch.exp(log_std)
        dist = torch.distributions.Normal(mean, std)
        action = dist.rsample()                                # reparameterised sample
        action = torch.clamp(action, -args.max_alpha, args.max_alpha)

        # ── Tính reward toàn batch ──
        r1, r2, r3 = compute_reward_components_vectorized(
            pos, neg, topk_ids, tgt, action, baseline,
            topk=args.top_k, bucket_thresholds=bucket_thresholds,
            gamma=args.gamma_decay, lambda_=args.lambda_r3,
        )  # mỗi (B,)
        reward = total_reward(r1, r2, r3, w1=args.w1, w2=args.w2, w3=args.w3)  # (B,)

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

        total_actor_loss += actor_loss.item()
        total_critic_loss += critic_loss.item()
        total_rew += reward.mean().item()
        batches += 1

    return total_actor_loss / batches, total_critic_loss / batches, total_rew / batches


# -------------------- CLI --------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train alpha RL agent for DFlash")
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-buckets", type=int, default=3)
    parser.add_argument("--max-alpha", type=float, default=2.0)
    parser.add_argument("--gamma-decay", type=float, default=7.0, help="positional decay γ")
    parser.add_argument("--lambda-r3", type=float, default=3.0, help="penalty for shorter acceptance")
    parser.add_argument("--w1", type=float, default=0.1, help="weight for r1 (rank improvement)")
    parser.add_argument("--w2", type=float, default=0.1, help="weight for r2 (top-1 bonus)")
    parser.add_argument("--w3", type=float, default=1.0, help="weight for r3 (acceptance length)")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="gradient clipping")
    parser.add_argument("--save-interval", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train_AC] device: {device}")

    dataset = HDF5BanditDataset(args.data_path)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_bandit_batch,
        num_workers=0,   # h5py không fork-safe
    )

    actor = GaussianPolicy(
        top_k=args.top_k, hidden_dim=args.hidden_dim,
        num_alpha_buckets=args.num_buckets, max_alpha=args.max_alpha,
    ).to(device)
    critic = Critic(top_k=args.top_k, hidden_dim=args.hidden_dim).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        a_loss, c_loss, r = train_one_epoch(dataloader, actor, critic, opt_actor, opt_critic, args, device)
        print(f"Epoch {epoch:3d} | actor_loss={a_loss:.4f}  critic_loss={c_loss:.4f}  avg_reward={r:.4f}")

        if epoch % args.save_interval == 0:
            torch.save(actor.state_dict(), out / f"actor_epoch{epoch}.pt")
            torch.save(critic.state_dict(), out / f"critic_epoch{epoch}.pt")

    print("Training completed!")


if __name__ == "__main__":
    main()