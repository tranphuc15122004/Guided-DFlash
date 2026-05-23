"""
Train NegativeLogitPredictor using Actor-Critic (A2C) RL.

Unlike the alpha model (which learns 3 scalar alpha values), this trains
a model to directly output 32 negative logit values for the top-32 positive
tokens, with fixed alpha=1.0 in the CD formula.

Key differences from train_AC.py:
  - Action space: 32 continuous logit values (not 3 bucket alphas)
  - No bucket logic (thresholds, num_buckets, bucket_size)
  - Uses compute_reward_components_with_predicted_neg_batch()
  - Output model: NegativeLogitPredictor / GaussianNegativePolicy
"""

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Import from package ──
try:
    from ..model.negative_predictor import (
        NegativeLogitPredictor,
        GaussianNegativePolicy,
        NegativePredictorCritic,
        count_parameters,
    )
    from .alpha_simulate import (
        compute_reward_components_with_predicted_neg_batch,
        compute_reward_with_predicted_neg_total,
    )
    from .offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
    from .metrics import (
        compute_rank_metrics,
        compute_distribution_metrics,
        compute_action_metrics,
        compute_acceptance_metrics,
        compute_rl_metrics,
        compute_training_dynamics,
    )
    from .logging_utils import TrainingLogger
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alpha_model.model.negative_predictor import (
        NegativeLogitPredictor,
        GaussianNegativePolicy,
        NegativePredictorCritic,
        count_parameters,
    )
    from alpha_model.train.alpha_simulate import (
        compute_reward_components_with_predicted_neg_batch,
        compute_reward_with_predicted_neg_total,
    )
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
    from alpha_model.train.metrics import (
        compute_rank_metrics,
        compute_distribution_metrics,
        compute_action_metrics,
        compute_acceptance_metrics,
        compute_rl_metrics,
        compute_training_dynamics,
    )
    from alpha_model.train.logging_utils import TrainingLogger


def average_metrics(hist_dict: dict) -> dict:
    """Reduce per-batch metric lists to epoch-level scalar means."""
    return {
        key: torch.tensor(vals).mean().item()
        for key, vals in hist_dict.items()
        if vals and isinstance(vals, list)
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train NegativeLogitPredictor for DFlash CD"
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints")
    parser.add_argument("--max-records", type=int, default=0,
                        help="limit to N records for debugging (0 = use all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--predict-delta", action="store_true",
                        help="Predict delta to add to original neg_logits instead of replacement")
    parser.add_argument("--from-phase1", type=str, default=None,
                        help="Path to Phase 1 checkpoint to initialize actor weights")
    parser.add_argument("--gamma-decay", type=float, default=7.0,
                        help="positional decay γ for r1/r2 weights")
    parser.add_argument("--lambda-r3", type=float, default=3.0,
                        help="penalty multiplier for shorter acceptance")
    parser.add_argument("--w1", type=float, default=0.1,
                        help="weight for r1 (rank improvement)")
    parser.add_argument("--w2", type=float, default=0.1,
                        help="weight for r2 (top-1 bonus)")
    parser.add_argument("--w3", type=float, default=1.0,
                        help="weight for r3 (acceptance length)")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                        help="entropy bonus coefficient")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="gradient clipping")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Epoch interval to save periodic checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume training from (overrides --from-phase1)")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "plateau", "none"],
                        help="Learning rate scheduler type (cosine: auto cosine decay from lr to lr_min over epochs)")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                        help="Minimum LR for cosine/linear/plateau schedulers")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Logger ──
    logger = TrainingLogger(args, args.output_dir, phase_name="Phase2_RL_A2C")
    logger.log_config(args)
    logger._write_log(f"  Device: {device}")

    # ── Dataset ──
    dataset = HDF5BanditDataset(args.data_path)
    if args.max_records > 0:
        original_len = len(dataset)
        dataset.num_records = min(args.max_records, original_len)
        logger._write_log(f"  Dataset limited: {dataset.num_records} / {original_len} records")
    logger.log_dataset_info(dataset)

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_bandit_batch,
        num_workers=0,
    )

    # ── Models ──
    actor = GaussianNegativePolicy(
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        predict_delta=args.predict_delta,
    ).to(device)

    critic = NegativePredictorCritic(
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
    ).to(device)

    if args.from_phase1:
        logger._write_log(f"  Loading Phase 1 checkpoint: {args.from_phase1}")
        state = torch.load(args.from_phase1, map_location=device)
        # Handle both full checkpoint and state_dict formats
        if isinstance(state, dict) and 'actor_state_dict' in state:
            actor.load_state_dict(state['actor_state_dict'], strict=False)
        else:
            actor.load_state_dict(state, strict=False)
        logger._write_log(f"  Phase 1 weights loaded successfully (strict=False)")

    logger.log_model_info(actor, critic)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=args.lr)

    # ── LR Scheduler ──
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
    if args.lr_scheduler == "cosine":
        scheduler = CosineAnnealingLR(opt_actor, T_max=args.epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "linear":
        final_ratio = args.lr_min / args.lr if args.lr > 0 else 0.0
        scheduler = LambdaLR(
            opt_actor,
            lr_lambda=lambda epoch: (1.0 - final_ratio) * max(0.0, 1.0 - epoch / args.epochs) + final_ratio,
        )
    elif args.lr_scheduler == "plateau":
        scheduler = ReduceLROnPlateau(opt_actor, mode='min', factor=0.5,
                                       patience=3, min_lr=args.lr_min)
    else:
        scheduler = None

    # ── Resume from checkpoint ──
    start_epoch = 1
    if args.resume:
        logger._write_log(f"  Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        if 'actor_state_dict' in ckpt:
            actor.load_state_dict(ckpt['actor_state_dict'], strict=False)
            logger._write_log(f"    Actor weights loaded (strict=False)")
        if 'critic_state_dict' in ckpt and ckpt['critic_state_dict'] is not None:
            critic.load_state_dict(ckpt['critic_state_dict'])
            logger._write_log(f"    Critic weights loaded")
        if 'optimizer_state_dict' in ckpt:
            opt_actor.load_state_dict(ckpt['optimizer_state_dict'])
            logger._write_log(f"    Actor optimizer state restored")
        if 'critic_optimizer' in ckpt:
            opt_critic.load_state_dict(ckpt['critic_optimizer'])
            logger._write_log(f"    Critic optimizer state restored")
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if isinstance(scheduler, LambdaLR) and hasattr(scheduler, 'last_epoch'):
                pass  # last_epoch correctly restored
            logger._write_log(f"    Scheduler state restored")
        start_epoch = ckpt.get('epoch', 0) + 1
        rest_best = ckpt.get('best_metric', None)
        if rest_best is not None:
            logger._best_metric = rest_best
            logger._write_log(f"    Best metric restored: {logger._best_metric_name}={rest_best:.4f}")
        logger._write_log(f"    Resuming from epoch {start_epoch}")

    # ── CSV header ──
    csv_header = [
        "epoch",
        "lr",
        "actor_loss_mean", "actor_loss_std",
        "critic_loss_mean", "critic_loss_std",
        "reward_mean", "reward_std",
        "r1_mean", "r1_std",
        "r2_mean", "r2_std",
        "r3_mean", "r3_std",
        "entropy_mean", "entropy_std",
        "baseline_acc_mean", "baseline_acc_std",
        "model_acc_mean", "model_acc_std",
        "acc_len_delta_mean", "acc_len_delta_std",
        "acc_len_improvement_rate_mean", "acc_len_improvement_rate_std",
        "top1_acc_mean", "top1_acc_std",
        "top5_acc_mean", "top5_acc_std",
        "mean_rank_mean", "mean_rank_std",
        "target_in_topk_mean", "target_in_topk_std",
        "cd_entropy_mean", "cd_entropy_std",
        "target_cd_score_mean", "target_cd_score_std",
        "action_separation_mean", "action_separation_std",
        "action_magnitude_mean", "action_magnitude_std",
        "log_std_mean", "log_std_std",
        "advantage_mean", "advantage_std",
        "value_error_mean", "value_error_std",
        "actor_grad_norm_mean", "actor_grad_norm_std",
        "critic_grad_norm_mean", "critic_grad_norm_std",
    ]
    logger.init_csv(csv_header)
    logger._best_metric_name = 'reward'
    logger._best_metric = float('-inf')

    total_start = time.time()

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        logger.start_epoch(epoch, args.epochs)
        actor.train()
        critic.train()

        batch_metrics = {
            'hist_actor_loss': [], 'hist_critic_loss': [],
            'hist_reward': [], 'hist_r1': [], 'hist_r2': [], 'hist_r3': [],
            'hist_entropy': [],
            'hist_baseline_acc': [], 'hist_model_acc': [],
            'hist_top1': [], 'hist_top5': [], 'hist_mean_rank': [],
            'hist_target_in_topk': [],
            'hist_cd_entropy': [], 'hist_target_cd_score': [],
            'hist_action_separation': [], 'hist_action_magnitude': [],
            'hist_actor_grad_norm': [], 'hist_critic_grad_norm': [],
            'hist_advantage_mean': [], 'hist_advantage_std': [],
            'hist_value_error': [],
            'hist_log_std_mean': [],
            'hist_acc_len_delta': [], 'hist_acc_len_impr': [],
        }

        pbar = logger.make_progress_bar(dataloader, desc=f"  Batch")
        for batch in pbar:
            pos = batch["pos_logits"].to(device)
            neg = batch["neg_logits"].to(device)
            topk_ids = batch["topk_token_ids"].to(device)
            tgt = batch["target_token_ids"].to(device)
            bpos = batch["block_pos"].to(device)
            apos = batch["abs_pos"].to(device)
            baseline = batch["baseline_acc_len"].to(device)

            # Actor forward
            mean, log_std = actor(pos, neg, bpos, apos)
            std = torch.exp(log_std)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample()

            # Reward
            r1, r2, r3, model_acc_len = compute_reward_components_with_predicted_neg_batch(
                pos, action, topk_ids, tgt, baseline,
                gamma=args.gamma_decay, lambda_=args.lambda_r3,
            )
            reward = compute_reward_with_predicted_neg_total(
                r1, r2, r3, w1=args.w1, w2=args.w2, w3=args.w3,
            )

            # Critic forward
            value = critic(pos, neg, bpos, apos).mean(dim=1)
            advantage = reward - value.detach()

            # Losses
            log_prob = dist.log_prob(action).sum(dim=-1).sum(dim=-1)
            actor_loss = -(log_prob * advantage).mean()
            entropy = dist.entropy().sum(dim=-1).sum(dim=-1).mean()
            actor_loss = actor_loss - args.entropy_coef * entropy
            critic_loss = F.mse_loss(value, reward)

            # Update
            opt_actor.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            opt_critic.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(critic.parameters(), args.max_grad_norm)
            opt_critic.step()

            # Core metrics
            batch_metrics['hist_actor_loss'].append(actor_loss.item())
            batch_metrics['hist_critic_loss'].append(critic_loss.item())
            batch_metrics['hist_reward'].append(reward.mean().item())
            batch_metrics['hist_r1'].append(r1.mean().item())
            batch_metrics['hist_r2'].append(r2.mean().item())
            batch_metrics['hist_r3'].append(r3.mean().item())
            batch_metrics['hist_entropy'].append(entropy.item())
            batch_metrics['hist_baseline_acc'].append(baseline.mean().item())
            batch_metrics['hist_model_acc'].append(model_acc_len.float().mean().item())

            K_val = topk_ids.size(-1)
            tgt_expanded = tgt.unsqueeze(-1).expand(-1, -1, K_val)
            in_topk = (tgt_expanded == topk_ids).any(dim=-1).float().mean().item()
            batch_metrics['hist_target_in_topk'].append(in_topk)

            # Advanced metrics
            with torch.no_grad():
                target_mask = (topk_ids == tgt.unsqueeze(-1)).float()
                target_in_topk_bool = target_mask.sum(dim=-1) > 0.5
                target_idx = target_mask.argmax(dim=-1)
                cd_topk = F.log_softmax(pos, dim=-1) - F.log_softmax(action, dim=-1)

                rank_m = compute_rank_metrics(cd_topk, target_idx, target_in_topk_bool, K_val)
                dist_m = compute_distribution_metrics(pos, action, target_idx, target_in_topk_bool)
                act_m = compute_action_metrics(action, pos, topk_ids, tgt)
                rl_m = compute_rl_metrics(log_std, torch.tensor(entropy.item()))
                dyn_m = compute_training_dynamics(actor, critic, advantage, value, reward)
                pred_tokens_greedy = cd_topk.argmax(dim=-1)
                acc_m = compute_acceptance_metrics(pred_tokens_greedy, tgt, baseline)

                batch_metrics['hist_top1'].append(rank_m['top1_acc'].item())
                batch_metrics['hist_top5'].append(rank_m['top5_acc'].item())
                batch_metrics['hist_mean_rank'].append(rank_m['target_mean_rank'].item())
                batch_metrics['hist_cd_entropy'].append(dist_m['cd_entropy'].item())
                batch_metrics['hist_target_cd_score'].append(dist_m['target_cd_score'].item())
                batch_metrics['hist_action_separation'].append(act_m['action_separation'].item())
                batch_metrics['hist_action_magnitude'].append(act_m['action_magnitude'].item())
                batch_metrics['hist_log_std_mean'].append(rl_m.get('log_std_mean', torch.tensor(0.0)).item())
                batch_metrics['hist_actor_grad_norm'].append(
                    dyn_m.get('actor_grad_norm', torch.tensor(0.0)).item())
                batch_metrics['hist_critic_grad_norm'].append(
                    dyn_m.get('critic_grad_norm', torch.tensor(0.0)).item())
                batch_metrics['hist_advantage_mean'].append(
                    dyn_m.get('advantage_mean', torch.tensor(0.0)).item())
                batch_metrics['hist_advantage_std'].append(
                    dyn_m.get('advantage_std', torch.tensor(0.0)).item())
                batch_metrics['hist_value_error'].append(
                    dyn_m.get('value_error', torch.tensor(0.0)).item())
                batch_metrics['hist_acc_len_delta'].append(
                    acc_m.get('acc_len_delta', torch.tensor(0.0)).item())
                batch_metrics['hist_acc_len_impr'].append(
                    acc_m.get('acc_len_improvement_rate', torch.tensor(0.0)).item())

        # Aggregate metrics
        agg = average_metrics(batch_metrics)

        epoch_metrics = {
            'actor_loss': agg['hist_actor_loss'],
            'critic_loss': agg['hist_critic_loss'],
            'reward': agg['hist_reward'],
            'r1': agg['hist_r1'], 'r3': agg['hist_r3'],
            'top1': agg['hist_top1'], 'top5': agg['hist_top5'],
            'mean_rank': agg['hist_mean_rank'],
            'model_acc': agg['hist_model_acc'],
            'baseline_acc': agg['hist_baseline_acc'],
            'acc_len_delta': agg['hist_acc_len_delta'],
            'entropy': agg['hist_entropy'],
            'log_std_mean': agg['hist_log_std_mean'],
            'action_sep': agg['hist_action_separation'],
            'cd_entropy': agg['hist_cd_entropy'],
            'advantage_mean': agg['hist_advantage_mean'],
            'value_error': agg['hist_value_error'],
            'actor_grad_norm': agg['hist_actor_grad_norm'],
        }

        # LR scheduler step
        current_lr = opt_actor.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(agg.get('hist_loss', torch.tensor(0.0)))
            else:
                scheduler.step()
            current_lr = opt_actor.param_groups[0]['lr']

        # CSV row
        def stats(vals):
            t = torch.tensor(batch_metrics.get(f'hist_{vals}', []))
            m = t.mean().item() if t.numel() > 0 else 0.0
            s = t.std().item() if t.numel() > 1 else 0.0
            return m, s

        row = [epoch, f"{current_lr:.8f}"]
        for key in ['actor_loss', 'critic_loss', 'reward', 'r1', 'r2', 'r3',
                     'entropy', 'baseline_acc', 'model_acc',
                     'acc_len_delta', 'acc_len_impr',
                     'top1', 'top5', 'mean_rank', 'target_in_topk',
                     'cd_entropy', 'target_cd_score',
                     'action_separation', 'action_magnitude',
                     'log_std_mean',
                     'advantage_mean', 'value_error',
                     'actor_grad_norm', 'critic_grad_norm']:
            m, s = stats(key)
            row.extend([f"{m:.6f}", f"{s:.6f}"])
        logger.write_csv_row(row)

        # End epoch
        logger.end_epoch(epoch, args.epochs, epoch_metrics, output_format='phase2')

        # Save checkpoint
        if hasattr(logger, '_latest_best_epoch') and logger._latest_best_epoch == epoch:
            logger.save_checkpoint(
                epoch=epoch,
                actor_state=actor.state_dict(),
                critic_state=critic.state_dict(),
                optimizer_state=opt_actor.state_dict(),
                is_best=True,
                extra={'critic_optimizer': opt_critic.state_dict(),
                       'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
            )
        if epoch % args.save_interval == 0:
            logger.save_checkpoint(
                epoch=epoch,
                actor_state=actor.state_dict(),
                critic_state=critic.state_dict(),
                extra={'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
                tag=f"epoch{epoch}",
            )

    # Final
    total_duration = time.time() - total_start
    logger.log_final_summary(total_duration)
    logger.save_checkpoint(
        epoch=args.epochs,
        actor_state=actor.state_dict(),
        critic_state=critic.state_dict(),
        extra={'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
        tag="final",
    )
    logger.close()


if __name__ == "__main__":
    main()
