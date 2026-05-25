"""
Phase 1: Supervised Training for NegativeLogitPredictor.

Train the model to recognize target tokens via:
  - CE loss: cross-entropy on CD output (target should rank #1 in top-K)
  - BCE auxiliary: binary classification for each token (is this the target?)
  - Both weighted by position (early positions matter more)

After Phase 1 converges, the model is used as initialization for Phase 2 (RL/A2C).
"""

import argparse
import os
import sys
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import WeightedRandomSampler
from tqdm import tqdm

# ── Imports ──
try:
    from ..model.negative_predictor import (
        NegativeLogitPredictor,
        NegativeLogitPredictor_Dense,
        count_parameters,
    )
    from .alpha_simulate import compute_phase1_loss
    from .offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
    from .metrics import (
        compute_rank_metrics,
        compute_distribution_metrics,
        compute_action_metrics,
    )
    from .logging_utils import TrainingLogger
except (ImportError, ValueError):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from alpha_model.model.negative_predictor import (
        NegativeLogitPredictor,
        NegativeLogitPredictor_Dense,
        count_parameters,
    )
    from alpha_model.train.alpha_simulate import compute_phase1_loss
    from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
    from alpha_model.train.metrics import (
        compute_rank_metrics,
        compute_distribution_metrics,
        compute_action_metrics,
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
        description="Phase 1: Supervised Training for NegativeLogitPredictor"
    )
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="checkpoints/phase1")
    parser.add_argument("--max-records", type=int, default=0,
                        help="limit to N records for debugging (0 = use all)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--predict-delta", action="store_true",
                        help="Use delta mode (output is added to original neg_logits)")
    parser.add_argument("--dense", action="store_true",
                        help="Use Dense (MLP) variant instead of Transformer")
    parser.add_argument("--lambda-bce", type=float, default=0.2,
                        help="Weight for BCE auxiliary loss")
    parser.add_argument("--gamma-decay", type=float, default=7.0,
                        help="Positional weight decay γ (higher = flatter weights)")
    parser.add_argument("--early-boost-n", type=int, default=6,
                        help="Number of earliest positions to boost (default: 6)")
    parser.add_argument("--early-boost-weight", type=float, default=2.0,
                        help="Extra weight multiplier for first early_boost_n positions (default: 2.0)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0,
                        help="Gradient clipping")
    parser.add_argument("--save-interval", type=int, default=5,
                        help="Epoch interval for periodic checkpoints")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint (.pt) to resume training from")
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["cosine", "linear", "plateau", "none"],
                        help="Learning rate scheduler type (cosine: auto cosine decay from lr to lr_min over epochs)")
    parser.add_argument("--lr-min", type=float, default=1e-6,
                        help="Minimum LR for cosine/linear/plateau schedulers")
    parser.add_argument("--augment-mode", type=str, default="none",
                        choices=["none", "weighted", "noise", "combined"],
                        help="Data augmentation mode: weighted sampling, noise, or both")
    parser.add_argument("--augment-neg-std", type=float, default=0.0,
                        help="Std of Gaussian noise added to neg_logits during collate")
    parser.add_argument("--augment-pos-std", type=float, default=0.0,
                        help="Std of Gaussian noise added to pos_logits during collate")
    parser.add_argument("--augment-prob", type=float, default=1.0,
                        help="Probability of applying batch noise augmentation")
    parser.add_argument("--sampler-alpha", type=float, default=1.0,
                        help="Strength of difficulty-based oversampling; higher = more biased toward hard samples")
    parser.add_argument("--sampler-min-weight", type=float, default=0.05,
                        help="Lower bound for per-sample weights after normalization")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def build_sampling_weights(dataset: HDF5BanditDataset, sampler_alpha: float, sampler_min_weight: float) -> torch.Tensor:
    """Build per-record weights that oversample harder samples using baseline acceptance length."""
    baseline = dataset.baseline_acc_len.float()
    inv_difficulty = 1.0 / (baseline + 1.0)
    weights = inv_difficulty.pow(max(sampler_alpha, 0.0))
    weights = weights / weights.mean().clamp_min(1e-8)
    weights = weights.clamp_min(sampler_min_weight)
    return weights


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model_type = "NegativeLogitPredictor_Dense" if args.dense else "NegativeLogitPredictor"

    # ── Logger ──
    logger = TrainingLogger(args, args.output_dir, phase_name="Phase1_Supervised")
    logger.log_config(args)

    logger._write_log(f"  Device: {device}")
    logger._write_log(f"  Model:  {model_type}")

    # ── Dataset ──
    dataset = HDF5BanditDataset(args.data_path)
    if args.max_records > 0:
        original_len = len(dataset)
        dataset.num_records = min(args.max_records, original_len)
        logger._write_log(f"  Dataset limited: {dataset.num_records} / {original_len} records")
    logger.log_dataset_info(dataset)

    sampler = None
    shuffle = True
    if args.augment_mode in {"weighted", "combined"}:
        sampler_weights = build_sampling_weights(dataset, args.sampler_alpha, args.sampler_min_weight)
        if args.max_records > 0:
            sampler_weights = sampler_weights[:dataset.num_records]
        sampler = WeightedRandomSampler(
            weights=sampler_weights,
            num_samples=len(sampler_weights),
            replacement=True,
        )
        shuffle = False
        logger._write_log(
            f"  Weighted sampler enabled: alpha={args.sampler_alpha}, min_weight={args.sampler_min_weight}"
        )

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        collate_fn=partial(
            collate_bandit_batch,
            augment_neg_std=args.augment_neg_std if args.augment_mode in {"noise", "combined"} else 0.0,
            augment_pos_std=args.augment_pos_std if args.augment_mode in {"noise", "combined"} else 0.0,
            augment_prob=args.augment_prob,
        ),
        num_workers=0,
    )

    # ── Model ──
    actor_cls = NegativeLogitPredictor_Dense if args.dense else NegativeLogitPredictor
    actor = actor_cls(
        top_k=args.top_k,
        hidden_dim=args.hidden_dim,
        predict_delta=args.predict_delta,
    ).to(device)

    logger.log_model_info(actor)

    opt_actor = torch.optim.Adam(actor.parameters(), lr=args.lr)

    # ── LR Scheduler ──
    from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LambdaLR
    if args.lr_scheduler == "cosine":
        # Cosine decay: smoothly from lr → lr_min over args.epochs
        scheduler = CosineAnnealingLR(opt_actor, T_max=args.epochs, eta_min=args.lr_min)
    elif args.lr_scheduler == "linear":
        # Linear decay: lr → lr_min over args.epochs
        final_ratio = args.lr_min / args.lr if args.lr > 0 else 0.0
        scheduler = LambdaLR(
            opt_actor,
            lr_lambda=lambda epoch: (1.0 - final_ratio) * max(0.0, 1.0 - epoch / args.epochs) + final_ratio,
        )
    elif args.lr_scheduler == "plateau":
        # Reduce on plateau: giảm LR × 0.5 nếu loss không giảm sau 3 epochs
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
        if 'optimizer_state_dict' in ckpt:
            opt_actor.load_state_dict(ckpt['optimizer_state_dict'])
            logger._write_log(f"    Optimizer state restored")
        # Restore scheduler state if available
        if scheduler is not None and 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            # LambdaLR uses last_epoch internally — when resuming, set last_epoch = start_epoch - 1
            # so the lambda receives the correct epoch index
            if isinstance(scheduler, LambdaLR) and hasattr(scheduler, 'last_epoch'):
                # The scheduler was stepped at end of each epoch, so after loading it's at last_epoch = start_epoch - 1
                pass  # Correct as-is
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
        "loss_mean", "loss_std",
        "ce_loss_mean", "ce_loss_std",
        "bce_loss_mean", "bce_loss_std",
        "top1_acc_mean", "top1_acc_std",
        "top5_acc_mean", "top5_acc_std",
        "top10_acc_mean", "top10_acc_std",
        "mean_rank_mean", "mean_rank_std",
        "median_rank_mean", "median_rank_std",
        "rank_std_mean", "rank_std_std",
        "bce_acc_mean", "bce_acc_std",
        "cd_entropy_mean", "cd_entropy_std",
        "entropy_reduction_mean", "entropy_reduction_std",
        "pos_neg_kl_mean", "pos_neg_kl_std",
        "target_cd_score_mean", "target_cd_score_std",
        "action_separation_mean", "action_separation_std",
        "action_magnitude_mean", "action_magnitude_std",
        "early_late_gap_mean", "early_late_gap_std",
    ]
    logger.init_csv(csv_header)
    logger._best_metric_name = 'top1'
    logger._best_metric = float('-inf')

    total_start = time.time()

    # ── Training loop ──
    for epoch in range(start_epoch, args.epochs + 1):
        logger.start_epoch(epoch, args.epochs)
        actor.train()

        batch_metrics = {
            'hist_loss': [], 'hist_ce': [], 'hist_bce': [],
            'hist_top1': [], 'hist_top5': [], 'hist_top10': [],
            'hist_mean_rank': [], 'hist_median_rank': [], 'hist_rank_std': [],
            'hist_bce_acc': [],
            'hist_cd_entropy': [], 'hist_entropy_reduction': [],
            'hist_pos_neg_kl': [], 'hist_target_cd_score': [],
            'hist_action_separation': [], 'hist_action_magnitude': [],
            'hist_early_late_gap': [],
        }

        pbar = logger.make_progress_bar(dataloader, desc=f"  Batch")
        for batch in pbar:
            pos = batch["pos_logits"].to(device)
            neg = batch["neg_logits"].to(device)
            topk_ids = batch["topk_token_ids"].to(device)
            tgt = batch["target_token_ids"].to(device)
            bpos = batch["block_pos"].to(device)
            apos = batch["abs_pos"].to(device)

            predicted_neg = actor(pos, neg, bpos, apos)
            result = compute_phase1_loss(
                pos, predicted_neg, topk_ids, tgt,
                lambda_bce=args.lambda_bce, gamma=args.gamma_decay,
                early_boost_n=args.early_boost_n,
                early_boost_weight=args.early_boost_weight,
            )

            opt_actor.zero_grad()
            result['loss'].backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), args.max_grad_norm)
            opt_actor.step()

            # Collect core metrics
            batch_metrics['hist_loss'].append(result['loss'].item())
            batch_metrics['hist_ce'].append(result['ce_loss'].item())
            batch_metrics['hist_bce'].append(result['bce_loss'].item())
            batch_metrics['hist_bce_acc'].append(result['bce_accuracy'].item())

            # Advanced metrics
            with torch.no_grad():
                target_mask = (topk_ids == tgt.unsqueeze(-1)).float()
                target_in_topk = target_mask.sum(dim=-1) > 0.5
                target_idx = target_mask.argmax(dim=-1)
                cd_topk = F.log_softmax(pos, dim=-1) - F.log_softmax(predicted_neg, dim=-1)

                rank_m = compute_rank_metrics(cd_topk, target_idx, target_in_topk, args.top_k)
                dist_m = compute_distribution_metrics(pos, predicted_neg, target_idx, target_in_topk)
                act_m = compute_action_metrics(predicted_neg, pos, topk_ids, tgt)

                batch_metrics['hist_top1'].append(rank_m['top1_acc'].item())
                batch_metrics['hist_top5'].append(rank_m['top5_acc'].item())
                batch_metrics['hist_top10'].append(rank_m['top10_acc'].item())
                batch_metrics['hist_mean_rank'].append(rank_m['target_mean_rank'].item())
                batch_metrics['hist_median_rank'].append(rank_m['target_median_rank'].item())
                batch_metrics['hist_rank_std'].append(rank_m['target_rank_std'].item())
                batch_metrics['hist_cd_entropy'].append(dist_m['cd_entropy'].item())
                batch_metrics['hist_entropy_reduction'].append(dist_m['entropy_reduction'].item())
                batch_metrics['hist_pos_neg_kl'].append(dist_m['pos_neg_kl'].item())
                batch_metrics['hist_target_cd_score'].append(dist_m['target_cd_score'].item())
                batch_metrics['hist_action_separation'].append(act_m['action_separation'].item())
                batch_metrics['hist_action_magnitude'].append(act_m['action_magnitude'].item())

        # Aggregate metrics
        agg = average_metrics(batch_metrics)

        epoch_metrics = {
            'loss': agg['hist_loss'], 'ce': agg['hist_ce'], 'bce': agg['hist_bce'],
            'top1': agg['hist_top1'], 'top5': agg['hist_top5'], 'top10': agg['hist_top10'],
            'mean_rank': agg['hist_mean_rank'],
            'median_rank': agg['hist_median_rank'],
            'rank_std': agg['hist_rank_std'],
            'bce_acc': agg['hist_bce_acc'],
            'cd_entropy': agg['hist_cd_entropy'],
            'entropy_red': agg['hist_entropy_reduction'],
            'pos_neg_kl': agg['hist_pos_neg_kl'],
            'target_cd_score': agg['hist_target_cd_score'],
            'action_sep': agg['hist_action_separation'],
            'action_mag': agg['hist_action_magnitude'],
        }

        # LR scheduler step
        current_lr = opt_actor.param_groups[0]['lr']
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(agg['hist_loss'])
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
        for key in ['loss', 'ce', 'bce', 'top1', 'top5', 'top10',
                     'mean_rank', 'median_rank', 'rank_std', 'bce_acc',
                     'cd_entropy', 'entropy_reduction', 'pos_neg_kl',
                     'target_cd_score', 'action_separation', 'action_magnitude',
                     'early_late_gap']:
            m, s = stats(key)
            row.extend([f"{m:.6f}", f"{s:.6f}"])
        logger.write_csv_row(row)

        # End epoch
        logger.end_epoch(epoch, args.epochs, epoch_metrics, output_format='phase1')

        # Save checkpoint
        if hasattr(logger, '_latest_best_epoch') and logger._latest_best_epoch == epoch:
            logger.save_checkpoint(
                epoch=epoch,
                actor_state=actor.state_dict(),
                extra={'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
                is_best=True,
            )
        if epoch % args.save_interval == 0:
            logger.save_checkpoint(
                epoch=epoch,
                actor_state=actor.state_dict(),
                optimizer_state=opt_actor.state_dict(),
                extra={'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
                tag=f"epoch{epoch}",
            )

    # Final
    total_duration = time.time() - total_start
    logger.log_final_summary(total_duration)
    logger.save_checkpoint(
        epoch=args.epochs,
        actor_state=actor.state_dict(),
        optimizer_state=opt_actor.state_dict(),
        extra={'scheduler_state_dict': scheduler.state_dict() if scheduler else None},
        tag="final",
    )
    logger.close()


if __name__ == "__main__":
    main()
