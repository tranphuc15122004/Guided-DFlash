import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional

def compute_bucket_thresholds(topk: int, num_buckets: int) -> List[int]:
    bucket_size = topk // num_buckets
    thresholds = [(i + 1) * bucket_size for i in range(num_buckets - 1)]
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
    buckets = torch.zeros_like(ranks, dtype=torch.long)
    for i, th in enumerate(thresholds):
        buckets = torch.where(ranks < th, torch.full_like(buckets, i), buckets)
    buckets = torch.where(ranks >= thresholds[-1], torch.full_like(buckets, len(thresholds)), buckets)
    return buckets  # (B, S)

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
        return r1.squeeze(0), r2.squeeze(0), r3.squeeze(0)

    return r1, r2, r3

def total_reward_batch(r1: torch.Tensor, r2: torch.Tensor, r3: torch.Tensor,
                       w1=0.1, w2=0.1, w3=1.0) -> torch.Tensor:
    return w1 * r1 + w2 * r2 + w3 * r3


# Backward-compatible aliases used by training code and the local smoke test.
simulate_acceptance_length_vectorized = simulate_acceptance_length_batch
compute_reward_components_vectorized = compute_reward_components_batch
total_reward = total_reward_batch


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