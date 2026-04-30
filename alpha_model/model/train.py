import argparse
from dataclasses import dataclass
from typing import Optional

import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from alpha_model import ContextualBanditAlpha, Critic


@dataclass
class RewardConfig:
    w1: float = 0.1
    w2: float = 0.1
    w3: float = 1.0
    gamma: float = 7.0
    lambda_: float = 3.0
    bucket_size: int = 10
    r1_clip_min: Optional[float] = None
    r2_only_improve: bool = False


def _load_acceptance_baseline_map(h5_path: str) -> dict:
    baseline = {}
    with h5py.File(h5_path, "r") as h5f:
        sample_ids = h5f["sample_id"][:]
        turn_indices = h5f["turn_index"][:]
        block_indices = h5f["block_index"][:]
        acc_lengths = h5f["acceptance_length"][:]
        for sample_id, turn_index, block_index, acc_len in zip(
            sample_ids,
            turn_indices,
            block_indices,
            acc_lengths,
        ):
            baseline[(int(sample_id), int(turn_index), int(block_index))] = float(acc_len)
    return baseline


class AlphaHDF5Dataset(Dataset):
    def __init__(self, h5_path: str, baseline_h5_path: str, preload: bool = False) -> None:
        super().__init__()
        self.h5_path = h5_path
        self.h5_file: Optional[h5py.File] = None
        self.in_memory: Optional[dict] = None
        self.preload = preload
        self.acceptance_baseline = _load_acceptance_baseline_map(baseline_h5_path)

        with h5py.File(h5_path, "r") as h5f:
            self.num_records = int(h5f.attrs.get("num_records", len(h5f["draft_topk_logits"])))
            self.has_alpha_prev = "alpha_prev" in h5f
            self.has_alpha_applied = "alpha_applied" in h5f
            self.top_k = int(h5f["draft_topk_logits"].shape[-1])
            self.seq_len = int(h5f["draft_topk_logits"].shape[-2])
            if self.preload:
                self.in_memory = {
                    "sample_id": h5f["sample_id"][:],
                    "turn_index": h5f["turn_index"][:],
                    "block_index": h5f["block_index"][:],
                    "draft_topk_token_ids": h5f["draft_topk_token_ids"][:],
                    "draft_topk_logits": h5f["draft_topk_logits"][:],
                    "neg_logits_on_draft_topk_ids": h5f["neg_logits_on_draft_topk_ids"][:],
                    "block_position": h5f["block_position"][:],
                    "absolute_position": h5f["absolute_position"][:],
                    "target_token_id": h5f["target_token_id"][:],
                    "acceptance_length": h5f["acceptance_length"][:],
                }
                if self.has_alpha_prev:
                    self.in_memory["alpha_prev"] = h5f["alpha_prev"][:]
                if self.has_alpha_applied:
                    self.in_memory["alpha_applied"] = h5f["alpha_applied"][:]

    def _ensure_open(self) -> None:
        if self.h5_file is None and self.in_memory is None:
            self.h5_file = h5py.File(self.h5_path, "r")

    def __len__(self) -> int:
        return self.num_records

    def __getitem__(self, idx: int) -> dict:
        if self.in_memory is not None:
            source = self.in_memory
        else:
            self._ensure_open()
            source = self.h5_file

        sample_id = int(source["sample_id"][idx])
        turn_index = int(source["turn_index"][idx])
        block_index = int(source["block_index"][idx])
        key = (sample_id, turn_index, block_index)
        if key not in self.acceptance_baseline:
            raise KeyError(f"Missing baseline acceptance length for key {key}")

        pos_logits = torch.tensor(source["draft_topk_logits"][idx], dtype=torch.float32)
        neg_logits = torch.tensor(source["neg_logits_on_draft_topk_ids"][idx], dtype=torch.float32)
        block_pos = torch.tensor(source["block_position"][idx], dtype=torch.float32)
        abs_pos = torch.tensor(source["absolute_position"][idx], dtype=torch.float32)
        draft_topk_ids = torch.tensor(source["draft_topk_token_ids"][idx], dtype=torch.int64)
        target_token_id = torch.tensor(source["target_token_id"][idx], dtype=torch.int64)
        acceptance_length = torch.tensor(source["acceptance_length"][idx], dtype=torch.float32)
        acceptance_length_base = torch.tensor(self.acceptance_baseline[key], dtype=torch.float32)

        batch = {
            "pos_logits": pos_logits,
            "neg_logits": neg_logits,
            "block_pos": block_pos,
            "abs_pos": abs_pos,
            "draft_topk_token_ids": draft_topk_ids,
            "target_token_id": target_token_id,
            "acceptance_length": acceptance_length,
            "acceptance_length_base": acceptance_length_base,
        }

        if self.has_alpha_prev:
            batch["alpha_prev"] = torch.tensor(source["alpha_prev"][idx], dtype=torch.float32)

        if self.has_alpha_applied:
            batch["alpha_applied"] = torch.tensor(source["alpha_applied"][idx], dtype=torch.float32)

        return batch

    def __del__(self) -> None:
        if self.h5_file is not None:
            try:
                if getattr(self.h5_file, "id", None) is not None and self.h5_file.id.valid:
                    self.h5_file.close()
            except Exception:
                pass
            finally:
                self.h5_file = None
        self.in_memory = None


def _bucket_indices(top_k: int, bucket_size: int, num_buckets: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(top_k, device=device)
    indices = torch.div(indices, bucket_size, rounding_mode="floor")
    return torch.clamp(indices, max=num_buckets - 1)


def _target_rank(
    draft_topk_ids: torch.Tensor,
    target_ids: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    top_k = logits.size(-1)
    match = draft_topk_ids == target_ids.unsqueeze(-1)
    has_match = match.any(dim=-1)
    match_idx = match.to(torch.int64).argmax(dim=-1)
    target_logit = logits.gather(-1, match_idx.unsqueeze(-1)).squeeze(-1)
    rank = (logits > target_logit.unsqueeze(-1)).sum(dim=-1) + 1
    fallback_rank = torch.full_like(rank, top_k + 1)
    return torch.where(has_match, rank, fallback_rank)


def compute_rewards(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    draft_topk_ids: torch.Tensor,
    target_ids: torch.Tensor,
    alpha: torch.Tensor,
    acceptance_length: torch.Tensor,
    acceptance_length_base: torch.Tensor,
    reward_cfg: RewardConfig,
) -> torch.Tensor:
    batch_size, seq_len, top_k = pos_logits.shape
    num_buckets = alpha.size(-1)

    bucket_idx = _bucket_indices(top_k, reward_cfg.bucket_size, num_buckets, pos_logits.device)
    alpha_per_token = alpha.gather(-1, bucket_idx.view(1, 1, top_k).expand(batch_size, seq_len, top_k))
    cd_logits = pos_logits - alpha_per_token * neg_logits

    pos_rank = _target_rank(draft_topk_ids, target_ids, pos_logits)
    cd_rank = _target_rank(draft_topk_ids, target_ids, cd_logits)

    delta_rank = pos_rank - cd_rank
    if reward_cfg.r1_clip_min is not None:
        delta_rank = torch.clamp(delta_rank, min=reward_cfg.r1_clip_min)
    decay = torch.exp(-torch.arange(seq_len, device=pos_logits.device, dtype=torch.float32) / reward_cfg.gamma)
    decay = decay.view(1, seq_len)

    r1 = delta_rank * decay
    is_top1 = cd_rank == 1
    if reward_cfg.r2_only_improve:
        is_top1 = is_top1 & (pos_rank > 1)
    r2 = 2.0 * decay * is_top1.to(torch.float32)

    delta_acc = acceptance_length.to(torch.float32) - acceptance_length_base.to(torch.float32)
    r3 = torch.where(delta_acc >= 0, delta_acc, -reward_cfg.lambda_ * (-delta_acc))
    r3 = (r3 / seq_len).unsqueeze(-1)

    return reward_cfg.w1 * r1 + reward_cfg.w2 * r2 + reward_cfg.w3 * r3


def train_step(
    actor: ContextualBanditAlpha,
    critic: Critic,
    optimizer_actor: optim.Optimizer,
    optimizer_critic: optim.Optimizer,
    batch: dict,
    reward_cfg: RewardConfig,
    policy_std: float,
    entropy_coef: float,
    device: torch.device,
    use_logged_actions: bool,
) -> tuple[float, float]:
    pos_logits = batch["pos_logits"].to(device)
    neg_logits = batch["neg_logits"].to(device)
    block_pos = batch["block_pos"].to(device)
    abs_pos = batch["abs_pos"].to(device)
    draft_topk_ids = batch["draft_topk_token_ids"].to(device)
    target_ids = batch["target_token_id"].to(device)
    acceptance_length = batch["acceptance_length"].to(device)
    acceptance_length_base = batch["acceptance_length_base"].to(device)

    alpha_prev = batch.get("alpha_prev")
    if alpha_prev is not None:
        alpha_prev = alpha_prev.to(device)

    alpha_applied = batch.get("alpha_applied") if use_logged_actions else None
    if alpha_applied is not None:
        alpha_applied = alpha_applied.to(device)

    alphas_mean = actor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev)
    std = torch.full_like(alphas_mean, policy_std)
    dist = torch.distributions.Normal(alphas_mean, std)

    if alpha_applied is None:
        action = dist.rsample()
    else:
        action = alpha_applied

    action = torch.clamp(action, 0.0, actor.max_alpha)
    log_probs = dist.log_prob(action).sum(dim=-1)

    rewards = compute_rewards(
        pos_logits=pos_logits,
        neg_logits=neg_logits,
        draft_topk_ids=draft_topk_ids,
        target_ids=target_ids,
        alpha=action,
        acceptance_length=acceptance_length,
        acceptance_length_base=acceptance_length_base,
        reward_cfg=reward_cfg,
    )

    state_features = actor.feature_extractor(pos_logits, neg_logits, block_pos, abs_pos, alpha_prev).detach()
    values = critic(state_features)

    advantages = rewards - values.detach()
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-6)

    actor_loss = -(log_probs * advantages).mean()
    if entropy_coef > 0:
        actor_loss -= entropy_coef * dist.entropy().sum(dim=-1).mean()

    critic_loss = nn.functional.mse_loss(values, rewards)

    optimizer_actor.zero_grad()
    actor_loss.backward()
    optimizer_actor.step()

    optimizer_critic.zero_grad()
    critic_loss.backward()
    optimizer_critic.step()

    return actor_loss.item(), critic_loss.item()


def train(
    actor: ContextualBanditAlpha,
    critic: Critic,
    dataloader: DataLoader,
    reward_cfg: RewardConfig,
    num_epochs: int,
    lr: float,
    policy_std: float,
    entropy_coef: float,
    device: torch.device,
    use_logged_actions: bool,
) -> None:
    optimizer_actor = optim.Adam(actor.parameters(), lr=lr)
    optimizer_critic = optim.Adam(critic.parameters(), lr=lr)

    actor.train()
    critic.train()

    for epoch in range(num_epochs):
        total_actor_loss = 0.0
        total_critic_loss = 0.0

        for batch in dataloader:
            a_loss, c_loss = train_step(
                actor=actor,
                critic=critic,
                optimizer_actor=optimizer_actor,
                optimizer_critic=optimizer_critic,
                batch=batch,
                reward_cfg=reward_cfg,
                policy_std=policy_std,
                entropy_coef=entropy_coef,
                device=device,
                use_logged_actions=use_logged_actions,
            )
            total_actor_loss += a_loss
            total_critic_loss += c_loss

        print(
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"Actor Loss: {total_actor_loss:.4f} | "
            f"Critic Loss: {total_critic_loss:.4f}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Alpha Contextual Bandit")
    parser.add_argument("--h5-path", type=str, required=True, help="Path to aggregated HDF5 dataset")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--policy-std", type=float, default=0.1)
    parser.add_argument("--entropy-coef", type=float, default=0.0)
    parser.add_argument("--max-alpha", type=float, default=2.0)
    parser.add_argument("--use-logged-actions", action="store_true")
    parser.add_argument("--baseline-h5-path", type=str, required=True)
    parser.add_argument("--preload-h5", action="store_true")
    parser.add_argument("--w1", type=float, default=0.1)
    parser.add_argument("--w2", type=float, default=0.1)
    parser.add_argument("--w3", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=7.0)
    parser.add_argument("--lambda", dest="lambda_", type=float, default=3.0)
    parser.add_argument("--bucket-size", type=int, default=10)
    parser.add_argument("--r1-clip-min", type=float, default=None)
    parser.add_argument("--r2-only-improve", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.use_logged_actions:
        raise ValueError("Offline training requires --use-logged-actions to align rewards with alpha_applied.")

    dataset = AlphaHDF5Dataset(
        args.h5_path,
        baseline_h5_path=args.baseline_h5_path,
        preload=args.preload_h5,
    )

    if args.use_logged_actions and not dataset.has_alpha_applied:
        raise ValueError("--use-logged-actions was set but alpha_applied is missing in the dataset.")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    actor = ContextualBanditAlpha(
        top_k=dataset.top_k,
        has_alpha_prev=dataset.has_alpha_prev,
        max_alpha=args.max_alpha,
    ).to(device)
    critic = Critic(input_dim=actor.encoder.net[0].in_features).to(device)

    reward_cfg = RewardConfig(
        w1=args.w1,
        w2=args.w2,
        w3=args.w3,
        gamma=args.gamma,
        lambda_=args.lambda_,
        bucket_size=args.bucket_size,
        r1_clip_min=args.r1_clip_min,
        r2_only_improve=args.r2_only_improve,
    )

    print(
        "Training config: "
        f"records={len(dataset)}, top_k={dataset.top_k}, seq_len={dataset.seq_len}, device={device}"
    )
    train(
        actor=actor,
        critic=critic,
        dataloader=dataloader,
        reward_cfg=reward_cfg,
        num_epochs=args.epochs,
        lr=args.lr,
        policy_std=args.policy_std,
        entropy_coef=args.entropy_coef,
        device=device,
        use_logged_actions=args.use_logged_actions,
    )