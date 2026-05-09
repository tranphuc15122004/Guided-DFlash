"""
Task T3: Huấn luyện RL bandit (A2C) cho điều chỉnh α động.

Hỗ trợ batch processing – không lặp per-sample.
Có gradient clipping, entropy coef configurable.
"""

import argparse
import csv
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
        simulate_acceptance_length_vectorized,
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
        simulate_acceptance_length_vectorized,
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
        r1, r2, r3, model_acc_len = compute_reward_components_vectorized(
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

    # ── Tính statistics cuối epoch ──
    def stats(values):
        t = torch.tensor(values)
        return t.mean().item(), t.std().item(), t.min().item(), t.max().item()

    def fmt_stats(name, values):
        m, s, lo, hi = stats(values)
        return f"{name}: {m:.4f}±{s:.4f} [{lo:.4f}, {hi:.4f}]"

    print(f"  {fmt_stats('actor_loss', hist_actor_loss)}")
    print(f"  {fmt_stats('critic_loss', hist_critic_loss)}")
    print(f"  {fmt_stats('reward', hist_reward)}")
    print(f"  {fmt_stats('r1', hist_r1)}")
    print(f"  {fmt_stats('r2', hist_r2)}")
    print(f"  {fmt_stats('r3', hist_r3)}")
    print(f"  {fmt_stats('entropy', hist_entropy)}")
    print(f"  {fmt_stats('baseline_acc', hist_baseline_acc)}")
    print(f"  {fmt_stats('model_acc', hist_model_acc)}")

    return hist_actor_loss, hist_critic_loss, hist_reward, hist_r1, hist_r2, hist_r3, hist_entropy, hist_baseline_acc, hist_model_acc


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
    parser.add_argument("--num-buckets", type=int, default=3)
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
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[train_AC] device: {device}")

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
        num_alpha_buckets=args.num_buckets, max_alpha=args.max_alpha,
    ).to(device)
    critic = Critic(top_k=args.top_k, hidden_dim=args.hidden_dim).to(device)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    best_reward = -float("inf")
    
    # Mở file CSV để ghi log
    csv_path = out / "training_history.csv"
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "epoch", 
        "actor_loss_mean", "actor_loss_std", 
        "critic_loss_mean", "critic_loss_std", 
        "reward_mean", "reward_std", 
        "r1_mean", "r1_std", 
        "r2_mean", "r2_std", 
        "r3_mean", "r3_std", 
        "entropy_mean", "entropy_std",
        "baseline_acc_mean", "baseline_acc_std",
        "model_acc_mean", "model_acc_std"
    ])

    for epoch in range(1, args.epochs + 1):
        print(f"─── Epoch {epoch}/{args.epochs} ───")
        hist = train_one_epoch(dataloader, actor, critic, opt_actor, opt_critic, args, device)

        # Tổng hợp epoch: mean và std của tất cả batches
        def stats(vals):
            if not vals:
                return 0.0, 0.0
            t = torch.tensor(vals)
            return t.mean().item(), t.std().item()
            
        act_mean, act_std = stats(hist[0])
        crit_mean, crit_std = stats(hist[1])
        rew_mean, rew_std = stats(hist[2])
        r1_mean, r1_std = stats(hist[3])
        r2_mean, r2_std = stats(hist[4])
        r3_mean, r3_std = stats(hist[5])
        ent_mean, ent_std = stats(hist[6] if len(hist) > 6 else []) # Compatibility for entropy
        base_acc_mean, base_acc_std = stats(hist[7] if len(hist) > 7 else [])
        mod_acc_mean, mod_acc_std = stats(hist[8] if len(hist) > 8 else [])

        print(f"✔ Epoch {epoch:3d} | "
              f"actor_loss={act_mean:.4f}  "
              f"critic_loss={crit_mean:.4f}  "
              f"reward={rew_mean:.4f}  "
              f"r1={r1_mean:.4f}  "
              f"r2={r2_mean:.4f}  "
              f"r3={r3_mean:.4f}")
        print(f"            | "
              f"baseline_acc={base_acc_mean:.4f}  "
              f"model_acc={mod_acc_mean:.4f}")
        print()
        
        # Ghi vào CSV
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
            mod_acc_mean, mod_acc_std
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