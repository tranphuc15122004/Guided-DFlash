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
        check_alpha_value,
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
        check_alpha_value,
    )
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch


def infer_num_buckets(top_k: int, bucket_size: int) -> int:
    if bucket_size <= 0:
        raise ValueError("bucket_size must be positive")
    return max((top_k + bucket_size - 1) // bucket_size, 1)


# -------------------- Gaussian Policy (extends actor) --------------------
class GaussianPolicy(ContextualBanditAlpha):
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

    best_reward = -float("inf")
    
    # Mở file CSV để ghi log
    # CSV này lưu các đường cong huấn luyện cốt lõi theo từng epoch.
    # Các bucket diagnostics chi tiết (min_alpha_bucket_acc, alpha_gap, phân bố bucket)
    # được in ở console trong train_one_epoch, còn CSV giữ các metric dễ plot nhất.
    csv_path = out / "training_history.csv"
    csv_file = open(csv_path, mode="w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
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
        "model_acc_mean", "model_acc_std"
    ])

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