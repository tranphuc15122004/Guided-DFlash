"""
Comprehensive metrics for NegativeLogitPredictor training.

Provides metric computation functions for both Phase 1 (supervised) and
Phase 2 (RL/A2C) training. All functions are purely functional: they take
tensors and return dictionaries of scalar metrics (or arrays for per-position
metrics).

Metric categories:
  - Rank & Acceptance: target rank, top-1/5, acceptance length
  - Distributional: entropy, KL divergence, probability mass
  - Action Analysis: predicted_neg statistics, target vs non-target separation
  - Positional: per-position breakdowns for all key metrics
  - Training Dynamics: gradient norms, value estimation quality
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union


# ═══════════════════════════════════════════════════════════
#  Rank & Acceptance Metrics
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rank_metrics(
    cd_logits: torch.Tensor,           # (B, S, K) — CD output in log-prob space
    target_idx_in_topk: torch.Tensor,  # (B, S) — index of target in top-K
    target_in_topk_mask: torch.Tensor, # (B, S) — boolean mask
    K: int = 32,
) -> Dict[str, torch.Tensor]:
    """
    Compute rank-based metrics after CD.

    Returns dict with:
      - top1_acc, top5_acc, top10_acc: accuracy at various thresholds
      - target_mean_rank: mean rank of target token (0 = best)
      - rank_histogram: count per rank position (K bins)
      - position_buckets: top1_acc split into [0-5), [5-10), [10-15) position ranges
      - rank_improvement: how many positions target gained vs random
    """
    B, S = target_idx_in_topk.shape
    cd_probs = F.softmax(cd_logits, dim=-1)  # (B, S, K)

    # Gather target probability
    gather_idx = target_idx_in_topk.unsqueeze(-1).clamp(0, K - 1)
    target_prob = torch.gather(cd_probs, dim=-1, index=gather_idx).squeeze(-1)

    # Rank = number of tokens with higher probability than target
    rank_after = (cd_probs > target_prob.unsqueeze(-1)).sum(dim=-1)  # (B, S)
    rank_after[~target_in_topk_mask] = K

    num_valid = target_in_topk_mask.float().sum().clamp(min=1)

    return {
        'top1_acc': (rank_after == 0).float().sum() / num_valid,
        'top5_acc': (rank_after < 5).float().sum() / num_valid,
        'top10_acc': (rank_after < 10).float().sum() / num_valid,
        'target_mean_rank': (rank_after * target_in_topk_mask.float()).sum() / num_valid,
        'target_median_rank': rank_after[target_in_topk_mask].float().median(),
        'target_rank_std': rank_after[target_in_topk_mask].float().std(),
    }


@torch.no_grad()
def compute_acceptance_metrics(
    pred_tokens: torch.Tensor,      # (B, S) — predicted token IDs after CD
    target_token_ids: torch.Tensor, # (B, S) — ground truth tokens
    baseline_acc_len: torch.Tensor, # (B,) — baseline acceptance length
) -> Dict[str, torch.Tensor]:
    """
    Compute acceptance length metrics.

    Returns dict with:
      - model_acc_len: acceptance length with CD
      - acc_len_delta: model - baseline
      - acc_len_improvement_rate: % of samples where acc_len improved
      - acc_len_regression_rate: % where acc_len got worse
      - per_position_acceptance: acceptance rate at each position
      - first_reject_position: mean position of first rejection
    """
    B, S = pred_tokens.shape
    correct = (pred_tokens == target_token_ids)  # (B, S)
    cumprod_correct = torch.cumprod(correct.int(), dim=1)
    model_acc_len = cumprod_correct.sum(dim=1).float()  # (B,)

    # Per-position acceptance rate
    per_pos_acc = correct.float().mean(dim=0)  # (S,)

    # First rejection position
    first_reject = torch.argmax((~correct).int(), dim=1).float()  # (B,)
    # For positions where all correct, first_reject = S
    all_correct = correct.all(dim=1)
    first_reject[all_correct] = S

    delta = model_acc_len - baseline_acc_len

    return {
        'model_acc_len': model_acc_len.mean(),
        'baseline_acc_len': baseline_acc_len.mean(),
        'acc_len_delta': delta.mean(),
        'acc_len_improvement_rate': (delta > 0).float().mean(),
        'acc_len_regression_rate': (delta < 0).float().mean(),
        'acc_len_unchanged_rate': (delta == 0).float().mean(),
        'per_position_acceptance': per_pos_acc,
        'first_reject_position': first_reject.mean(),
    }


# ═══════════════════════════════════════════════════════════
#  Distributional Metrics
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_distribution_metrics(
    pos_logits: torch.Tensor,          # (B, S, K)
    predicted_neg_logits: torch.Tensor, # (B, S, K)
    target_idx_in_topk: torch.Tensor,   # (B, S)
    target_in_topk_mask: torch.Tensor,  # (B, S)
) -> Dict[str, torch.Tensor]:
    """
    Compute distributional statistics for CD, positive, and negative distributions.

    Returns dict with:
      - pos_entropy, neg_entropy, cd_entropy: entropy of each distribution
      - pos_neg_kl: KL(pos || neg)
      - pos_cd_kl: KL(pos || CD)
      - neg_cd_kl: KL(neg || CD)
      - target_pos_logprob: log-prob of target under positive
      - target_neg_logprob: log-prob of target under negative
      - target_cd_score: CD score for target (positive signal strength)
    """
    pos_probs = F.softmax(pos_logits, dim=-1)
    pos_log_probs = F.log_softmax(pos_logits, dim=-1)
    neg_probs = F.softmax(predicted_neg_logits, dim=-1)
    neg_log_probs = F.log_softmax(predicted_neg_logits, dim=-1)

    # CD distribution
    cd_logits = pos_log_probs - neg_log_probs  # (B, S, K)
    cd_probs = F.softmax(cd_logits, dim=-1)
    cd_log_probs = F.log_softmax(cd_logits, dim=-1)  # recompute for accuracy

    # Entropies
    pos_entropy = -(pos_probs * pos_log_probs).sum(dim=-1)  # (B, S)
    neg_entropy = -(neg_probs * neg_log_probs).sum(dim=-1)
    cd_entropy = -(cd_probs * cd_log_probs).sum(dim=-1)

    # KL divergences
    pos_neg_kl = (pos_probs * (pos_log_probs - neg_log_probs)).sum(dim=-1)
    pos_cd_kl = (pos_probs * (pos_log_probs - cd_log_probs)).sum(dim=-1)
    neg_cd_kl = (neg_probs * (neg_log_probs - cd_log_probs)).sum(dim=-1)

    # Target token statistics
    gather_idx = target_idx_in_topk.unsqueeze(-1).clamp(0, pos_logits.size(-1) - 1)
    target_pos_lp = torch.gather(pos_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    target_neg_lp = torch.gather(neg_log_probs, dim=-1, index=gather_idx).squeeze(-1)
    target_cd_score = target_pos_lp - target_neg_lp  # CD signal for target

    # Mask to valid positions
    mask = target_in_topk_mask.float()

    def masked_mean(t):
        return (t * mask).sum() / mask.sum().clamp(min=1)

    def masked_std(t):
        m = masked_mean(t)
        var = ((t - m) ** 2 * mask).sum() / mask.sum().clamp(min=1)
        return var.sqrt()

    return {
        'pos_entropy': masked_mean(pos_entropy),
        'neg_entropy': masked_mean(neg_entropy),
        'cd_entropy': masked_mean(cd_entropy),
        'entropy_reduction': masked_mean(pos_entropy - cd_entropy),
        'pos_neg_kl': masked_mean(pos_neg_kl),
        'pos_cd_kl': masked_mean(pos_cd_kl),
        'target_pos_logprob': masked_mean(target_pos_lp),
        'target_neg_logprob': masked_mean(target_neg_lp),
        'target_cd_score': masked_mean(target_cd_score),
        'target_cd_score_std': masked_std(target_cd_score),
    }


# ═══════════════════════════════════════════════════════════
#  Action (Predicted Negative Logits) Analysis
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_action_metrics(
    predicted_neg_logits: torch.Tensor,  # (B, S, K)
    pos_logits: torch.Tensor,            # (B, S, K)
    topk_token_ids: torch.Tensor,        # (B, S, K)
    target_token_ids: torch.Tensor,      # (B, S)
) -> Dict[str, torch.Tensor]:
    """
    Analyze the predicted negative logits (the action).

    Returns dict with:
      - action_mean, action_std, action_min, action_max: overall stats
      - action_target_mean: predicted_neg for target token
      - action_non_target_mean: predicted_neg for non-target tokens
      - action_separation: non_target - target (higher = better separation)
      - action_variance_ratio: var_non_target / var_target
      - action_top1_boost: how much lower predicted_neg is for the argmax token
      - action_vs_neg_correlation: correlation with original negative logits
      - per_dimension_std: std of each of the 32 output dimensions
    """
    B, S, K = predicted_neg_logits.shape
    device = predicted_neg_logits.device

    # Target mask
    target_mask = (topk_token_ids == target_token_ids.unsqueeze(-1)).float()  # (B, S, K)
    non_target_mask = 1.0 - target_mask

    # Per-sample statistics
    act = predicted_neg_logits  # (B, S, K)

    # Target vs non-target analysis
    target_vals = (act * target_mask).sum(dim=-1) / target_mask.sum(dim=-1).clamp(min=1)  # (B, S)
    non_target_vals = (act * non_target_mask).sum(dim=-1) / non_target_mask.sum(dim=-1).clamp(min=1)

    # For positions where target is not in top-K, we still have non-target stats
    valid_target = target_mask.sum(dim=-1) > 0  # (B, S)

    # Separation: non_target - target (positive = model penalizes non-targets more)
    separation = torch.full_like(target_vals, float('nan'))
    separation[valid_target] = non_target_vals[valid_target] - target_vals[valid_target]

    # Variance ratio
    target_var = ((act - target_vals.unsqueeze(-1)) ** 2 * target_mask).sum(dim=-1) / target_mask.sum(dim=-1).clamp(min=1)
    non_target_var = ((act - non_target_vals.unsqueeze(-1)) ** 2 * non_target_mask).sum(dim=-1) / non_target_mask.sum(dim=-1).clamp(min=1)
    var_ratio = target_var / non_target_var.clamp(min=1e-8)

    # Per-dimension statistics (over B×S)
    per_dim_std = act.reshape(-1, K).std(dim=0)  # (K,)

    # Correlation with original negative logits (not available here, pass explicitly)
    # We compute action magnitude instead
    action_magnitude = act.abs().mean()

    return {
        'action_mean': act.mean(),
        'action_std': act.std(),
        'action_min': act.min(),
        'action_max': act.max(),
        'action_magnitude': action_magnitude,
        'action_target_mean': target_vals[valid_target].mean(),
        'action_non_target_mean': non_target_vals.mean(),
        'action_separation': separation[valid_target].mean(),
        'action_separation_std': separation[valid_target].std(),
        'action_var_ratio': var_ratio[valid_target].mean(),
        'per_dim_std_min': per_dim_std.min(),
        'per_dim_std_max': per_dim_std.max(),
        'per_dim_std_mean': per_dim_std.mean(),
        'per_dim_std_std': per_dim_std.std(),
    }


@torch.no_grad()
def compute_action_correlation(
    predicted_neg_logits: torch.Tensor,  # (B, S, K)
    original_neg_logits: torch.Tensor,   # (B, S, K)
) -> Dict[str, torch.Tensor]:
    """
    Compute correlation between predicted negative logits and original negatives.

    Returns dict with:
      - pearson_r: Pearson correlation per sample
      - l2_distance: ||predicted - original||₂
      - cosine_similarity: cosine sim between predicted and original
      - sign_agreement: % of dimensions where sign matches
    """
    B, S, K = predicted_neg_logits.shape
    device = predicted_neg_logits.device

    p = predicted_neg_logits.reshape(-1, K)
    o = original_neg_logits.reshape(-1, K)

    # Pearson correlation per sample
    p_centered = p - p.mean(dim=-1, keepdim=True)
    o_centered = o - o.mean(dim=-1, keepdim=True)
    cov = (p_centered * o_centered).sum(dim=-1)
    p_std = p_centered.std(dim=-1).clamp(min=1e-8)
    o_std = o_centered.std(dim=-1).clamp(min=1e-8)
    pearson_r = (cov / (p_std * o_std)).mean()

    # L2 distance
    l2 = (p - o).norm(p=2, dim=-1).mean()

    # Cosine similarity
    cos_sim = F.cosine_similarity(p, o, dim=-1).mean()

    # Sign agreement
    sign_agree = ((p * o) > 0).float().mean()

    return {
        'action_neg_pearson_r': pearson_r,
        'action_neg_l2_distance': l2,
        'action_neg_cosine_sim': cos_sim,
        'action_neg_sign_agreement': sign_agree,
    }


# ═══════════════════════════════════════════════════════════
#  Positional Analysis
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_positional_metrics(
    metric_per_position: torch.Tensor,  # (S,) or (B, S) — per-position values
    position_groups: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Break down a per-position metric into position groups.

    Default groups:
      - very_early: positions 0-1  (initial tokens)
      - early: positions 2-4       (early tokens)
      - mid: positions 5-9         (mid block)
      - late: positions 10-14      (late tokens)

    Returns:
        Dict with group means and the full positional breakdown
    """
    if position_groups is None:
        position_groups = [
            (0, 2, 'very_early'),
            (2, 5, 'early'),
            (5, 10, 'mid'),
            (10, 15, 'late'),
        ]

    if metric_per_position.dim() == 2:
        metric_per_position = metric_per_position.mean(dim=0)

    S = metric_per_position.size(0)
    result = {}

    for start, end, name in position_groups:
        end = min(end, S)
        if end > start:
            result[f'{name}_mean'] = metric_per_position[start:end].mean()
        else:
            result[f'{name}_mean'] = torch.tensor(0.0, device=metric_per_position.device)

    result['positional_std'] = metric_per_position.std()
    result['positional_range'] = metric_per_position.max() - metric_per_position.min()
    result['early_late_gap'] = result.get('very_early_mean', torch.tensor(0.0)) - result.get('late_mean', torch.tensor(0.0))

    return result


# ═══════════════════════════════════════════════════════════
#  Training Dynamics
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_training_dynamics(
    actor: torch.nn.Module,
    critic: Optional[torch.nn.Module] = None,
    advantage: Optional[torch.Tensor] = None,  # (B,)
    value: Optional[torch.Tensor] = None,       # (B,)
    reward: Optional[torch.Tensor] = None,      # (B,)
) -> Dict[str, torch.Tensor]:
    """
    Compute training dynamics metrics.

    Returns dict with:
      - actor_grad_norm: ||∇θ_actor||₂
      - critic_grad_norm: ||∇θ_critic||₂ (if critic provided)
      - grad_norm_ratio: ||∇θ_actor|| / ||∇θ_critic||
      - advantage_mean, advantage_std (if provided)
      - value_reward_correlation: Pearson r(value, reward) (if both provided)
      - value_error: |value - reward| mean
    """
    result = {}

    # Gradient norms
    actor_total_norm = 0.0
    for p in actor.parameters():
        if p.grad is not None:
            actor_total_norm += p.grad.norm(2).item() ** 2
    actor_norm = actor_total_norm ** 0.5
    result['actor_grad_norm'] = torch.tensor(actor_norm, dtype=torch.float32)

    if critic is not None:
        critic_total_norm = 0.0
        for p in critic.parameters():
            if p.grad is not None:
                critic_total_norm += p.grad.norm(2).item() ** 2
        critic_norm = critic_total_norm ** 0.5
        result['critic_grad_norm'] = torch.tensor(critic_norm, dtype=torch.float32)
        if actor_norm > 0 and critic_norm > 0:
            result['grad_norm_ratio'] = torch.tensor(actor_norm / critic_norm, dtype=torch.float32)

    # Advantage statistics
    if advantage is not None:
        result['advantage_mean'] = advantage.mean()
        result['advantage_std'] = advantage.std()

    # Value-Reward correlation
    if value is not None and reward is not None:
        v_centered = value - value.mean()
        r_centered = reward - reward.mean()
        cov = (v_centered * r_centered).sum()
        denom = value.std() * reward.std()
        result['value_reward_r'] = cov / denom.clamp(min=1e-8)
        result['value_error'] = (value - reward).abs().mean()

    return result


# ═══════════════════════════════════════════════════════════
#  RL-Specific Metrics
# ═══════════════════════════════════════════════════════════

@torch.no_grad()
def compute_rl_metrics(
    log_std: torch.Tensor,         # (K,) — learnable log_std parameters
    entropy: torch.Tensor,         # scalar — policy entropy
    dist: Optional[torch.distributions.Normal] = None,
) -> Dict[str, torch.Tensor]:
    """
    RL-specific metrics for policy behavior.

    Returns dict with:
      - log_std_mean, log_std_std: summary of log_std across action dims
      - std_mean, std_min, std_max: standard deviation statistics
      - policy_entropy: total entropy (already passed in)
      - exploration_rate: estimated exploration rate (std at action scale)
    """
    std = torch.exp(log_std)
    return {
        'log_std_mean': log_std.mean(),
        'log_std_std': log_std.std(),
        'std_mean': std.mean(),
        'std_min': std.min(),
        'std_max': std.max(),
        'policy_entropy': entropy,
        'exploration_std': std.mean(),  # synonym for convenience
    }


# ═══════════════════════════════════════════════════════════
#  Helper: merge metric dicts
# ═══════════════════════════════════════════════════════════

def merge_metrics(*dicts: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """
    Merge multiple metric dictionaries into one with scalar Python values.

    Lists/arrays are converted to comma-separated strings.
    """
    result = {}
    for d in dicts:
        for k, v in d.items():
            if torch.is_tensor(v):
                if v.numel() == 1:
                    result[k] = v.item()
                elif v.numel() < 50:  # small arrays: serialize
                    result[k] = ','.join(f'{x:.4f}' for x in v.flatten().tolist())
                else:  # large arrays: store stats
                    result[k] = v.mean().item()
            else:
                result[k] = v
    return result
