import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

def compute_bucket_thresholds(topk: int, num_buckets: int, bucket_size: Optional[int] = None) -> List[int]:
    if num_buckets <= 1:
        return []

    if bucket_size is None or bucket_size <= 0:
        bucket_size = max(topk // num_buckets, 1)

    thresholds = [min((i + 1) * bucket_size, topk) for i in range(num_buckets - 1)]
    return thresholds

def get_rank_in_topk_batch(
    target_token_ids: torch.Tensor,   # (B, S)
    topk_token_ids: torch.Tensor,     # (B, S, K)
) -> torch.Tensor:
    B, S, K = topk_token_ids.shape
    expanded_target = target_token_ids.unsqueeze(-1).expand(-1, -1, K)  # (B, S, K)
    mask = (expanded_target == topk_token_ids)  # (B, S, K)
    rank = torch.argmax(mask.int(), dim=-1)  # (B, S)
    has_token = mask.any(dim=-1)
    rank = torch.where(has_token, rank, torch.full_like(rank, K))
    return rank  # (B, S)

def get_bucket_from_rank_batch(
    ranks: torch.Tensor,        # (B, S)
    thresholds: List[int],
) -> torch.Tensor:
    if len(thresholds) == 0:
        return torch.zeros_like(ranks, dtype=torch.long)

    boundary = torch.as_tensor(thresholds, device=ranks.device, dtype=ranks.dtype)
    buckets = torch.bucketize(ranks, boundary, right=False)
    return buckets.to(torch.long)  # (B, S)

def apply_contrastive_batch(
    pos_logits: torch.Tensor,   # (B, S, K)
    neg_logits: torch.Tensor,   # (B, S, K)
    alphas: torch.Tensor,       # (B, S)
) -> torch.Tensor:
    return pos_logits - alphas.unsqueeze(-1) * neg_logits

def greedy_sample_batch(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(logits, dim=-1)  # (B, S)

def simulate_acceptance_length_batch(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    topk_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    alpha_per_token: torch.Tensor,   # (B, S, num_buckets)
    topk: Optional[int] = None,
    bucket_thresholds: Optional[List[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
        acc_len: (B,) – số token được accept liên tiếp từ đầu (0..S)
        correct_mask: (B, S) – token i có được accept không (chỉ trong đoạn accept)
    """
    pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token, _, single_sample = _ensure_batched_inputs(
        pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token
    )
    if topk is None:
        topk = topk_token_ids.size(-1)
    num_buckets = alpha_per_token.size(-1)
    if bucket_thresholds is None:
        bucket_thresholds = compute_bucket_thresholds(topk, num_buckets)
    B, S, K = pos_logits.shape

    ranks = get_rank_in_topk_batch(target_token_ids, topk_token_ids)  # (B, S)
    buckets = get_bucket_from_rank_batch(ranks, bucket_thresholds)    # (B, S)
    # Lấy alpha cho từng token: gather từ alpha_per_token (B,S,num_buckets) theo bucket
    batch_idx = torch.arange(B, device=alpha_per_token.device).unsqueeze(-1).expand(-1, S)
    token_idx = torch.arange(S, device=alpha_per_token.device).unsqueeze(0).expand(B, -1)
    alphas = alpha_per_token[batch_idx, token_idx, buckets]  # (B, S)

    contrastive_logits = apply_contrastive_batch(pos_logits, neg_logits, alphas)  # (B, S, K)
    pred_tokens = greedy_sample_batch(contrastive_logits)  # (B, S)
    correct = (pred_tokens == target_token_ids)  # (B, S)

    cumprod_correct = torch.cumprod(correct.int(), dim=1)  # (B, S)
    acc_len = cumprod_correct.sum(dim=1)  # (B,)
    correct_mask = (cumprod_correct == 1)
    if single_sample:
        return acc_len.squeeze(0), correct_mask.squeeze(0)
    return acc_len, correct_mask

def compute_reward_components_batch(
    pos_logits: torch.Tensor,           # (B, S, K)
    neg_logits: torch.Tensor,           # (B, S, K)
    topk_token_ids: torch.Tensor,       # (B, S, K)
    target_token_ids: torch.Tensor,     # (B, S)
    alpha_per_token: torch.Tensor,      # (B, S, num_buckets)
    baseline_acc_len: torch.Tensor,     # (B,) baseline acceptance length
    topk: Optional[int] = None,
    bucket_thresholds: Optional[List[int]] = None,
    gamma: float = 7.0,
    lambda_: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
        r1: (B,)
        r2: (B,)
        r3: (B,)
    """
    pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token, baseline_acc_len, single_sample = _ensure_batched_inputs(
        pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token, baseline_acc_len
    )
    if topk is None:
        topk = topk_token_ids.size(-1)
    num_buckets = alpha_per_token.size(-1)
    if bucket_thresholds is None:
        bucket_thresholds = compute_bucket_thresholds(topk, num_buckets)
    B, S, K = pos_logits.shape
    device = pos_logits.device

    # Ranks before
    ranks_before = get_rank_in_topk_batch(target_token_ids, topk_token_ids)  # (B, S)
    buckets = get_bucket_from_rank_batch(ranks_before, bucket_thresholds)    # (B, S)
    # Gather alphas
    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, S)
    token_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    alphas = alpha_per_token[batch_idx, token_idx, buckets]  # (B, S)

    contrastive_logits = apply_contrastive_batch(pos_logits, neg_logits, alphas)  # (B, S, K)
    contrastive_probs = F.softmax(contrastive_logits, dim=-1)

    # Ranks after: số token có prob > prob của target
    # Tìm xác suất của target token trong topK
    rank_before_indices = get_rank_in_topk_batch(target_token_ids, topk_token_ids)  # (B, S)
    valid = rank_before_indices < topk
    # Tạo gather index: nếu không valid thì set tạm 0
    gather_idx = rank_before_indices.clone()
    gather_idx[~valid] = 0
    target_probs = torch.gather(contrastive_probs, dim=-1, index=gather_idx.unsqueeze(-1)).squeeze(-1)  # (B, S)
    target_probs[~valid] = 0.0
    # Đếm số token có prob > target_probs
    rank_after = (contrastive_probs > target_probs.unsqueeze(-1)).sum(dim=-1)  # (B, S)
    rank_after[~valid] = topk

    delta_rank = ranks_before - rank_after  # (B, S)
    # Weights decay theo vị trí
    i = torch.arange(S, device=device, dtype=torch.float32)  # (S,)
    weights = torch.exp(-i / gamma)  # (S,)

    # r1, r2 trên toàn bộ token
    r1 = (delta_rank * weights).sum(dim=1)  # (B,)
    r2 = ((rank_after == 0).float() * 2.0 * weights).sum(dim=1)  # (B,)

    # Tính acceptance length mới (dùng simulate)
    acc_len_contrastive, _ = simulate_acceptance_length_batch(
        pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token,
        topk=topk, bucket_thresholds=bucket_thresholds
    )  # (B,)
    if baseline_acc_len is None:
        raise ValueError("baseline_acc_len is required")
    delta = acc_len_contrastive - baseline_acc_len
    r3 = torch.max(delta, torch.zeros_like(delta)) - lambda_ * torch.max(-delta, torch.zeros_like(delta))

    if single_sample:
        return r1.squeeze(0), r2.squeeze(0), r3.squeeze(0), acc_len_contrastive.squeeze(0)

    return r1, r2, r3, acc_len_contrastive

def total_reward_batch(r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor,
                       w1=0.1, w2=0.1, w3=1.0) -> torch.Tensor:
    return w1 * r1 + w2 * r2 + w3 * r3


def _check_alpha_value(
    pos_logits: torch.Tensor,           # (B, S, K)
    topk_token_ids: torch.Tensor,       # (B, S, K)
    target_token_ids: torch.Tensor,     # (B, S)
    alpha_per_token: torch.Tensor,      # (B, S, num_buckets)
    bucket_thresholds: Optional[List[int]] = None,
    atol: float = 1e-6,
):
    B, S, num_buckets = alpha_per_token.shape
    K = topk_token_ids.shape[-1]
    device = alpha_per_token.device

    if bucket_thresholds is None:
        bucket_thresholds = compute_bucket_thresholds(K, num_buckets)

    ranks = get_rank_in_topk_batch(target_token_ids, topk_token_ids)
    true_buckets = get_bucket_from_rank_batch(ranks, bucket_thresholds)

    batch_idx = torch.arange(B, device=device).unsqueeze(-1).expand(-1, S)
    token_idx = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    target_alpha = alpha_per_token[batch_idx, token_idx, true_buckets]

    min_alpha, min_alpha_bucket = alpha_per_token.min(dim=-1)
    sorted_alpha = torch.sort(alpha_per_token, dim=-1).values
    second_min_alpha = sorted_alpha[..., 1] if num_buckets > 1 else min_alpha

    strict_correct = (min_alpha_bucket == true_buckets)
    tie_correct = target_alpha <= (min_alpha + atol)

    bucket_counts = torch.bincount(true_buckets.reshape(-1), minlength=num_buckets)
    pred_min_bucket_counts = torch.bincount(min_alpha_bucket.reshape(-1), minlength=num_buckets)

    total_count = bucket_counts.sum().clamp_min(1).to(torch.float32)
    bucket_rates = (bucket_counts.to(torch.float32) / total_count).tolist()
    pred_min_bucket_rates = (
        pred_min_bucket_counts.to(torch.float32)
        / pred_min_bucket_counts.sum().clamp_min(1).to(torch.float32)
    ).tolist()

    alpha_gap = target_alpha - min_alpha
    alpha_margin = second_min_alpha - min_alpha

    strict_min_alpha_bucket_acc = strict_correct.float().mean().item()
    min_alpha_bucket_acc = tie_correct.float().mean().item()
    target_in_topk_rate = (ranks < K).float().mean().item()

    per_bucket_acc = {}
    per_bucket_tie_acc = {}
    for bucket in range(num_buckets):
        mask = (true_buckets == bucket)
        if mask.sum() > 0:
            per_bucket_acc[bucket] = strict_correct[mask].float().mean().item()
            per_bucket_tie_acc[bucket] = tie_correct[mask].float().mean().item()
        else:
            per_bucket_acc[bucket] = 0.0
            per_bucket_tie_acc[bucket] = 0.0

    return {
        "min_alpha_bucket_acc": min_alpha_bucket_acc,
        "strict_min_alpha_bucket_acc": strict_min_alpha_bucket_acc,
        "target_bucket_tie_acc": min_alpha_bucket_acc,
        "per_bucket_acc": per_bucket_acc,
        "per_bucket_tie_acc": per_bucket_tie_acc,
        "bucket_counts": bucket_counts.tolist(),
        "bucket_rates": bucket_rates,
        "pred_min_bucket_counts": pred_min_bucket_counts.tolist(),
        "pred_min_bucket_rates": pred_min_bucket_rates,
        "target_in_topk_rate": target_in_topk_rate,
        "target_outside_topk_rate": 1.0 - target_in_topk_rate,
        "alpha_gap_mean": alpha_gap.mean().item(),
        "alpha_gap_std": alpha_gap.std(unbiased=False).item() if alpha_gap.numel() > 1 else 0.0,
        "alpha_margin_mean": alpha_margin.mean().item(),
        "alpha_margin_std": alpha_margin.std(unbiased=False).item() if alpha_margin.numel() > 1 else 0.0,
    }



# ──────────────────────────────────────────────────────
# Reward simulation for NegativeLogitPredictor
# Unlike the alpha model, this uses predicted_neg_logits directly
# (not alpha-scaled neg_logits) with fixed alpha=1.0
# ──────────────────────────────────────────────────────

def apply_contrastive_with_predicted_neg_batch(
    pos_logits: torch.Tensor,               # (B, S, K)
    predicted_neg_logits: torch.Tensor,       # (B, S, K)
) -> torch.Tensor:
    """
    Apply CD with predicted negative logits and fixed alpha=1.0.

    CD = log_softmax(pos) - log_softmax(predicted_neg)
    """
    log_p_pos = F.log_softmax(pos_logits, dim=-1)
    log_p_neg = F.log_softmax(predicted_neg_logits, dim=-1)
    return log_p_pos - log_p_neg  # alpha = 1.0 fixed


def simulate_acceptance_length_with_predicted_neg_batch(
    pos_logits: torch.Tensor,               # (B, S, K)
    predicted_neg_logits: torch.Tensor,       # (B, S, K)
    topk_token_ids: torch.Tensor,             # (B, S, K)
    target_token_ids: torch.Tensor,           # (B, S)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Simulate acceptance length using predicted negative logits.

    Returns:
        acc_len: (B,) — number of consecutive accepted tokens from start
        correct_mask: (B, S) — which tokens were accepted (within accepted segment)
    """
    pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids, _, single_sample = (
        _ensure_batched_inputs_predicted_neg(
            pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids,
        )
    )

    contrastive_logits = apply_contrastive_with_predicted_neg_batch(
        pos_logits, predicted_neg_logits,
    )  # (B, S, K) in log-prob space
    pred_tokens = greedy_sample_batch(contrastive_logits)  # (B, S)
    correct = (pred_tokens == target_token_ids)  # (B, S)

    cumprod_correct = torch.cumprod(correct.int(), dim=1)  # (B, S)
    acc_len = cumprod_correct.sum(dim=1)  # (B,)
    correct_mask = (cumprod_correct == 1)

    if single_sample:
        return acc_len.squeeze(0), correct_mask.squeeze(0)
    return acc_len, correct_mask


def compute_reward_components_with_predicted_neg_batch(
    pos_logits: torch.Tensor,               # (B, S, K)
    predicted_neg_logits: torch.Tensor,       # (B, S, K)
    topk_token_ids: torch.Tensor,             # (B, S, K)
    target_token_ids: torch.Tensor,           # (B, S)
    baseline_acc_len: torch.Tensor,           # (B,) baseline acceptance length
    gamma: float = 7.0,
    lambda_: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute reward components for negative logit prediction.

    r1: rank improvement — how many positions the target token moves up
    r2: top-1 bonus — extra reward when target reaches rank 0
    r3: acceptance length change (vs baseline)

    Returns:
        r1, r2, r3, acc_len_contrastive: each (B,)
    """
    pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids, baseline_acc_len, single_sample = (
        _ensure_batched_inputs_predicted_neg(
            pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids,
            baseline_acc_len=baseline_acc_len,
        )
    )

    B, S, K = pos_logits.shape
    device = pos_logits.device

    # ── Rank before CD ──
    ranks_before = get_rank_in_topk_batch(target_token_ids, topk_token_ids)  # (B, S)

    # ── Apply CD with predicted negative logits ──
    contrastive_logits = apply_contrastive_with_predicted_neg_batch(
        pos_logits, predicted_neg_logits,
    )  # (B, S, K) in log-prob space (from log_softmax)
    contrastive_probs = F.softmax(contrastive_logits, dim=-1)  # convert to probs

    # ── Rank after CD ──
    # Find probability of target token in the CD distribution
    rank_before_indices = get_rank_in_topk_batch(target_token_ids, topk_token_ids)  # (B, S)
    valid = rank_before_indices < K
    gather_idx = rank_before_indices.clone()
    gather_idx[~valid] = 0
    target_probs = torch.gather(
        contrastive_probs, dim=-1, index=gather_idx.unsqueeze(-1)
    ).squeeze(-1)  # (B, S)
    target_probs[~valid] = 0.0

    # Count how many tokens have higher probability than target
    rank_after = (contrastive_probs > target_probs.unsqueeze(-1)).sum(dim=-1)  # (B, S)
    rank_after[~valid] = K

    # ── r1: Rank improvement ──
    delta_rank = ranks_before - rank_after  # (B, S)
    i = torch.arange(S, device=device, dtype=torch.float32)  # (S,)
    weights = torch.exp(-i / gamma)  # (S,)
    r1 = (delta_rank * weights).sum(dim=1)  # (B,)

    # ── r2: Top-1 bonus ──
    r2 = ((rank_after == 0).float() * 2.0 * weights).sum(dim=1)  # (B,)

    # ── r3: Acceptance length change ──
    acc_len_contrastive, _ = simulate_acceptance_length_with_predicted_neg_batch(
        pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids,
    )  # (B,)

    if baseline_acc_len is None:
        raise ValueError("baseline_acc_len is required")
    delta = acc_len_contrastive - baseline_acc_len
    r3 = torch.max(delta, torch.zeros_like(delta)) - lambda_ * torch.max(
        -delta, torch.zeros_like(delta)
    )

    if single_sample:
        return (r1.squeeze(0), r2.squeeze(0), r3.squeeze(0),
                acc_len_contrastive.squeeze(0))

    return r1, r2, r3, acc_len_contrastive


def compute_reward_with_predicted_neg_total(
    r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor,
    w1: float = 0.1, w2: float = 0.1, w3: float = 1.0,
) -> torch.Tensor:
    """Total reward = w1 * r1 + w2 * r2 + w3 * r3."""
    return w1 * r1 + w2 * r2 + w3 * r3


def _ensure_batched_inputs_predicted_neg(
    pos_logits: torch.Tensor,
    predicted_neg_logits: torch.Tensor,
    topk_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    baseline_acc_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
    """Ensure all tensors have batch dimension for vectorized processing."""
    single_sample = pos_logits.dim() == 2
    if single_sample:
        pos_logits = pos_logits.unsqueeze(0)
        predicted_neg_logits = predicted_neg_logits.unsqueeze(0)
        topk_token_ids = topk_token_ids.unsqueeze(0)
        target_token_ids = target_token_ids.unsqueeze(0)
        if baseline_acc_len is not None:
            if not torch.is_tensor(baseline_acc_len):
                baseline_acc_len = torch.tensor(
                    [baseline_acc_len], device=pos_logits.device, dtype=pos_logits.dtype,
                )
            elif baseline_acc_len.dim() == 0:
                baseline_acc_len = baseline_acc_len.unsqueeze(0)
    return (pos_logits, predicted_neg_logits, topk_token_ids, target_token_ids,
            baseline_acc_len, single_sample)


# Backward-compatible aliases used by training code and the local smoke test.
simulate_acceptance_length_vectorized = simulate_acceptance_length_batch
compute_reward_components_vectorized = compute_reward_components_batch
total_reward = total_reward_batch
check_alpha_value = _check_alpha_value


def _ensure_batched_inputs(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    topk_token_ids: torch.Tensor,
    target_token_ids: torch.Tensor,
    alpha_per_token: torch.Tensor,
    baseline_acc_len: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], bool]:
    single_sample = pos_logits.dim() == 2
    if single_sample:
        pos_logits = pos_logits.unsqueeze(0)
        neg_logits = neg_logits.unsqueeze(0)
        topk_token_ids = topk_token_ids.unsqueeze(0)
        target_token_ids = target_token_ids.unsqueeze(0)
        if alpha_per_token.dim() == 2:
            alpha_per_token = alpha_per_token.unsqueeze(0)
        if baseline_acc_len is not None:
            if not torch.is_tensor(baseline_acc_len):
                baseline_acc_len = torch.tensor([baseline_acc_len], device=pos_logits.device, dtype=pos_logits.dtype)
            elif baseline_acc_len.dim() == 0:
                baseline_acc_len = baseline_acc_len.unsqueeze(0)
    return (
        pos_logits,
        neg_logits,
        topk_token_ids,
        target_token_ids,
        alpha_per_token,
        baseline_acc_len,
        single_sample,
    )


# ──────────────────────────────────────────────────────
# Phase 1 Supervised Loss Functions
# Train model to recognize target token via CD + auxiliary BCE
# ──────────────────────────────────────────────────────

def compute_phase1_loss(
    pos_logits: torch.Tensor,               # (B, S, K)
    predicted_neg_logits: torch.Tensor,      # (B, S, K) — model output
    topk_token_ids: torch.Tensor,            # (B, S, K)
    target_token_ids: torch.Tensor,          # (B, S)
    lambda_bce: float = 0.2,
    gamma: float = 7.0,
    early_boost_n: int = 6,
    early_boost_weight: float = 2.0,
) -> dict:
    """
    Compute Phase 1 supervised loss: CE + auxiliary BCE.

    CE:   Cross-entropy on CD output (target token should rank #1 in top-K)
    BCE:  Binary classification — is this token the target? (uses -predicted_neg as logits)

    Positional weighting:
      - Base: exponential decay `exp(-t / gamma)` (higher weight for early tokens)
      - Boost: first `early_boost_n` positions additionally multiplied by `early_boost_weight`

    Returns dict with keys: 'loss', 'ce_loss', 'bce_loss', 'target_top1_acc',
                            'target_top5_acc', 'target_mean_rank', 'bce_accuracy'
    """
    B, S, K = pos_logits.shape
    device = pos_logits.device

    # ── Find target index in top-K ──
    # target_mask: (B, S, K) — 1 at the position of target token in top-K
    target_mask = (topk_token_ids == target_token_ids.unsqueeze(-1)).float()
    target_in_topk = target_mask.sum(dim=-1) > 0.5  # (B, S) — boolean
    # argmax gives first occurrence index; for positions where target not in top-k it gives 0 (ignored later)
    target_idx_in_topk = target_mask.argmax(dim=-1)  # (B, S)

    # Valid positions: ~90% of positions have target in top-32
    num_valid = target_in_topk.float().sum().clamp(min=1)

    # ── Positional weights (exponential decay + early boost) ──
    pos_indices = torch.arange(S, device=device, dtype=torch.float32)
    weights = torch.exp(-pos_indices / gamma)  # (S,) — base exponential decay
    # Extra boost for the first `early_boost_n` positions
    if early_boost_n > 0 and early_boost_weight != 1.0:
        boost = torch.ones(S, device=device, dtype=torch.float32)
        boost[:early_boost_n] = early_boost_weight
        weights = weights * boost
    weight_per_position = weights.unsqueeze(0)  # (1, S)

    # ── 1. CE Loss on CD output ──
    # CD = log_softmax(pos) - log_softmax(predicted_neg)
    cd_topk = F.log_softmax(pos_logits, dim=-1) - F.log_softmax(predicted_neg_logits, dim=-1)
    # cd_topk: (B, S, K) — higher = more likely under CD

    ce_per_position = F.cross_entropy(
        cd_topk.reshape(-1, K),         # (B*S, K)
        target_idx_in_topk.reshape(-1),  # (B*S,)
        reduction='none',
    ).reshape(B, S)

    # Mask invalid positions (target not in top-K)
    ce_masked = ce_per_position * target_in_topk.float() * weight_per_position
    ce_loss = ce_masked.sum() / num_valid

    # ── 2. BCE Auxiliary Loss ──
    # -predicted_neg: high logit = likely target token (should have low penalty)
    bce_logits = -predicted_neg_logits  # (B, S, K)
    bce_per_position = F.binary_cross_entropy_with_logits(
        bce_logits.reshape(-1, K),
        target_mask.reshape(-1, K),
        reduction='none',
    ).reshape(B, S, K).mean(dim=-1)  # average over K → (B, S)

    bce_masked = bce_per_position * target_in_topk.float() * weight_per_position
    bce_loss = bce_masked.sum() / num_valid

    # ── 3. Total ──
    total_loss = ce_loss + lambda_bce * bce_loss

    # ── 4. Metrics ──
    with torch.no_grad():
        # Rank of target after CD
        cd_probs = F.softmax(cd_topk, dim=-1)  # (B, S, K)
        # Gather probability of target token
        gather_idx = target_idx_in_topk.unsqueeze(-1).clamp(0, K-1)
        target_prob = torch.gather(cd_probs, dim=-1, index=gather_idx).squeeze(-1)  # (B, S)
        # Count tokens with higher probability than target
        rank_after = (cd_probs > target_prob.unsqueeze(-1)).sum(dim=-1)  # (B, S)
        rank_after[~target_in_topk] = K  # invalid positions get K

        target_top1 = (rank_after == 0).float()
        target_top5 = (rank_after < 5).float()
        target_top1_acc = (target_top1 * target_in_topk.float()).sum() / num_valid
        target_top5_acc = (target_top5 * target_in_topk.float()).sum() / num_valid
        target_mean_rank = (rank_after * target_in_topk.float()).sum() / num_valid

        # BCE accuracy
        bce_pred = (bce_logits > 0).float()  # (B, S, K)
        bce_correct = (bce_pred == target_mask).float()
        bce_accuracy = (bce_correct * target_in_topk.unsqueeze(-1).float()).sum() / (num_valid * K)

    return {
        'loss': total_loss,
        'ce_loss': ce_loss.detach(),
        'bce_loss': bce_loss.detach(),
        'target_top1_acc': target_top1_acc,
        'target_top5_acc': target_top5_acc,
        'target_mean_rank': target_mean_rank,
        'bce_accuracy': bce_accuracy,
    }


# ========== Kiểm thử ==========
if __name__ == "__main__":
    torch.manual_seed(42)
    S, K = 15, 32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    pos_logits = torch.randn(1, S, K, device=device)
    neg_logits = torch.randn(1, S, K, device=device)
    topk_token_ids = torch.arange(K, device=device).unsqueeze(0).unsqueeze(0).expand(1, S, K)
    target_token_ids = torch.randint(0, K, (1, S), device=device)
    baseline_acc_len = torch.tensor([7], device=device)
    num_buckets = 3
    alpha_per_token = torch.rand(1, S, num_buckets, device=device) * 2.0
    bucket_thresholds = compute_bucket_thresholds(K, num_buckets)  # [10,20]
    
    r1, r2, r3 = compute_reward_components_vectorized(
        pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token,
        baseline_acc_len, topk=K, bucket_thresholds=bucket_thresholds
    )
    reward = total_reward(r1, r2, r3)
    print(f"r1={r1.item():.4f}, r2={r2.item():.4f}, r3={r3.item():.4f}, total_reward={reward.item():.4f}")
    
    acc_len, _ = simulate_acceptance_length_vectorized(
        pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token,
        topk=K, bucket_thresholds=bucket_thresholds
    )
    print(f"Simulated acceptance length: {acc_len.item()}")
    print("Test completed.")

    # ========== Deterministic test: ensure acc_len > 0 ==========
    print('\nRunning deterministic acceptance-length test...')
    S2, K2 = 8, 32
    pos_logits2 = torch.full((1, S2, K2), -100.0, device=device)
    neg_logits2 = torch.zeros((1, S2, K2), device=device)
    # choose target ids and make them highest-scoring for first 3 tokens
    target_ids2 = (torch.arange(0, S2, device=device) % K2).unsqueeze(0)
    for i in range(3):
        pos_logits2[0, i, target_ids2[0, i]] = 100.0
    # other positions random but lower than 100
    for i in range(3, S2):
        pos_logits2[0, i, (i+1) % K2] = 10.0

    topk_token_ids2 = torch.arange(K2, device=device).unsqueeze(0).unsqueeze(0).expand(1, S2, K2)
    alpha_per_token2 = torch.zeros(1, S2, 3, device=device)  # zero alpha -> contrastive = pos
    baseline_acc_len2 = torch.tensor([1], device=device)
    bucket_thresholds2 = compute_bucket_thresholds(K2, 3)

    acc_len2, mask2 = simulate_acceptance_length_vectorized(
        pos_logits2, neg_logits2, topk_token_ids2, target_ids2, alpha_per_token2,
        topk=K2, bucket_thresholds=bucket_thresholds2
    )
    print(f"Deterministic simulated acceptance length: {acc_len2.item()} (expected >= 3)")
    r1b, r2b, r3b = compute_reward_components_vectorized(
        pos_logits2, neg_logits2, topk_token_ids2, target_ids2, alpha_per_token2,
        baseline_acc_len2, topk=K2, bucket_thresholds=bucket_thresholds2
    )
    print(f"Deterministic rewards: r1={r1b.item():.4f}, r2={r2b.item():.4f}, r3={r3b.item():.4f}, total={total_reward(r1b,r2b,r3b).item():.4f}")