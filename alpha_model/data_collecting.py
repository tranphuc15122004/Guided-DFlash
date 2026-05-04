import argparse
import os
import time
import random
from pathlib import Path
from itertools import chain
from types import SimpleNamespace
from typing import Any, Dict, Iterator, List, Optional, Tuple
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import *
from alpha_model import load_and_process_dataset_ALPHA
import distributed as dist


"""
python alpha_model/data_collecting.py \
  --model-name-or-path <target_model> \
  --draft-name-or-path <draft_model> \
  --dataset <dataset_name_or_path> \
  --block-size 16 \
  --collect-dataset \
  --collector-output-dir alpha_model/collected_alpha_records \
  --collector-topk 32 \
  --collector-chunk-size 2048
"""

"""
Dataset:
- gsm8k (math): 7,473                   -> 7,473
- metamath (math): 395,000              -> 20.000
- math_instruct (math): 262,039         -> 5.000
- magicoder (code): 111,183             -> 20.000

In total: 52,473
"""

'''
- **sample_id**: int — id của sample (từ iterator / args.seed + idx).
- **turn_index**: int — thứ tự turn trong cuộc hội thoại.
- **block_index**: int — chỉ số block hiện tại trong quá trình decode.
- **draft_topk_token_ids**: int32 tensor, shape (S-1, K) — các token id top‑K do draft (positive) xếp hạng cao nhất cho mỗi vị trí diffusion trong block (lưu batch dim đã bị squeeze(0) vì generation chạy batch=1).
    - Source: torch.topk(positive_draft_logits, k=collector_topk).indices.
    - Gắn với cùng thứ tự vị trí như các trường logits/positions bên dưới.
- **draft_topk_logits**: float32 tensor, shape (S-1, K) — logits tương ứng với draft_topk_token_ids từ positive draft distribution.
    - Source: topk.values từ positive draft logits.
- **neg_logits_on_draft_topk_ids**: float32 tensor, shape (S-1, K) — logits của negative draft lấy trên chính các token id trong draft_topk_token_ids (để so sánh pos vs neg trên cùng token set).
    - Tạo bằng torch.gather(negative_logits, dim=-1, index=draft_topk_token_ids).
- **block_position**: float32 tensor, shape (S-1,) — vị trí nội bộ trong block, normalized (0..1).
    - Tạo bằng torch.arange(block_size-1) / (block_size-2) (denom tối thiểu 1).
- **absolute_position**: float32 tensor, shape (S-1,) — vị trí tuyệt đối trong sequence, normalized (0..1).
    - Tạo bằng block_position_ids[:,1:] / (max_length-1).
- **target_token_id**: int32 tensor, shape (S-1,) — token ids do target model sinh (posterior[:, :-1]) tương ứng với các vị trí diffusion (dùng để so sánh/verify với draft).
    - Lưu dạng CPU, đã squeeze(0).
- **acceptance_length**: int — số token được chấp nhận ở block này (giá trị lưu là int(acceptance_length + 1) theo code).
    - Cách tính:acceptance_length_var = (block_output_ids[:,1:] == posterior[:,:-1]).cumprod(dim=1).sum(dim=1)[0].item()stored = acceptance_length_var + 1
    - Ý nghĩa: số token liên tiếp đầu tiên trong block được target chấp nhận (tức block_output_ids[:, :stored] là các token accepted). Giá trị ∈ [1, block_size].
- **alpha_prev**: float32 tensor, shape (S-1, 3) — alpha từ block trước (per-position, 3 buckets), nếu có; lưu dạng CPU.
    - Trong code khởi tạo alpha_prev có kích thước `(1, max(block_size-1,1), 3)` và trước khi lưu dùng .squeeze(0).
- **alpha_applied**: float32 tensor, shape (S-1, 3) — alpha đã được áp dụng cho block hiện tại (per-position, 3 buckets), lưu CPU.
'''



NEGATIVE_HIDDEN_MODE = 'mask_zero' 
TOP64_MASK_MODE = False 


class BlockRecordWriter:
    def __init__(self, output_dir: str, rank: int, chunk_size: int = 2048) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.chunk_size = max(1, chunk_size)
        self.buffer: List[Dict[str, Any]] = []
        self.chunk_index = 0

    def add(self, record: Dict[str, Any]) -> None:
        self.buffer.append(record)
        if len(self.buffer) >= self.chunk_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        out_file = self.output_dir / f"rank_{self.rank:02d}_chunk_{self.chunk_index:05d}.pt"
        out_file.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.buffer, out_file)
        self.buffer.clear()
        self.chunk_index += 1

    def close(self) -> None:
        self.flush()


def _dataset_length(dataset: Any) -> Optional[int]:
    try:
        return len(dataset)
    except TypeError:
        return None


def _iter_dataset_for_rank(dataset: Any, rank: int, world_size: int) -> Iterator[Tuple[int, dict]]:
    for idx, instance in enumerate(dataset):
        if idx % world_size != rank:
            continue
        yield idx, instance


def _estimate_total_for_rank(
    dataset_len: Optional[int],
    num_instances: Optional[int],
    max_samples: Optional[int],
    rank: int,
    world_size: int,
) -> Optional[int]:
    total = dataset_len
    if total is None:
        if num_instances is not None:
            total = num_instances
        elif max_samples is not None:
            total = max_samples

    if total is None:
        return None

    total = max(0, int(total))
    remaining = max(0, total - rank)
    return (remaining + world_size - 1) // world_size


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        # Required by some CUDA kernels for reproducible matmul behavior.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True

# construct negative sample by randomly replacing the batch's first token (target model generated token) 
def build_negative_block_output_random(block_output_ids: torch.Tensor, vocab_size: int , gen : torch.Generator = None) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(0, vocab_size, (batch_size,), device=block_output_ids.device , generator=gen)
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids

def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    mode: str = "mask_zero",
    gen : torch.Generator = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()
    batch_size, context_len = negative_target_hidden.shape[:2]

    if mode == "mask_zero":
        if dropout_ratio > 0.0:
            token_keep_mask = (
                torch.rand(
                    negative_target_hidden.shape[:2],
                    device=negative_target_hidden.device,
                    generator=gen
                ) >= dropout_ratio
            ).unsqueeze(-1)
            negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)
    elif mode == "shuffle_tokens":
        if context_len > 1:
            per_batch_indices = []
            for _ in range(batch_size):
                perm = torch.randperm(context_len, device=negative_target_hidden.device, generator=gen)
                per_batch_indices.append(perm)
            gather_index = torch.stack(per_batch_indices, dim=0).unsqueeze(-1).expand(-1, -1, negative_target_hidden.shape[-1])
            negative_target_hidden = torch.gather(negative_target_hidden, dim=1, index=gather_index)
    else:
        raise ValueError(
            f"Unsupported negative hidden mode: {mode}. "
            "Use 'mask_zero' or 'shuffle_tokens'."
        )

    if noise_std > 0.0:
        noise = torch.randn(
            negative_target_hidden.shape,
            dtype=negative_target_hidden.dtype,
            device=negative_target_hidden.device,
            generator=gen
        )
        negative_target_hidden = negative_target_hidden + noise_std * noise

    return negative_target_hidden

def compute_contrastive_draft_logits(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    block_output_ids: torch.Tensor,
    target_hidden: torch.Tensor,
    draft_position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    block_size: int,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    negative_hidden_mode: str,
    gen : torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    
    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size , gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        mode=negative_hidden_mode,
        gen=gen,
    )
    
    possitive_noise_embedding = target.model.embed_tokens(block_output_ids)
    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)
    

    paired_noise_embedding = torch.cat([possitive_noise_embedding, negative_noise_embedding], dim=0)
    paired_hidden = torch.cat([target_hidden, negative_target_hidden], dim=0)
    
    paired_position_ids = torch.cat([draft_position_ids, draft_position_ids], dim=0)

    paired_draft_logits = target.lm_head(model(
        target_hidden=paired_hidden,
        noise_embedding=paired_noise_embedding,
        position_ids=paired_position_ids,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )[:, -block_size+1:, :])

    first_draft_logits, second_draft_logits = paired_draft_logits.split(batch_size, dim=0)
    return first_draft_logits, second_draft_logits


def _extract_topk_aligned_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k = min(max(1, top_k), positive_logits.size(-1))
    topk = torch.topk(positive_logits, k=k, dim=-1)
    draft_topk_logits = topk.values
    draft_topk_token_ids = topk.indices
    neg_logits_on_draft_topk_ids = torch.gather(negative_logits, dim=-1, index=draft_topk_token_ids)
    return draft_topk_token_ids, draft_topk_logits, neg_logits_on_draft_topk_ids


def build_cd_candidate_mask(
    reference_logits: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)

    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def build_topk_probability_mask(
    reference_logits: torch.Tensor,
    top_k: int = 64,
) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    vocab_size = reference_probs.size(-1)
    k = min(max(1, top_k), vocab_size)

    topk_indices = torch.topk(reference_probs, k=k, dim=-1).indices
    topk_mask = torch.zeros_like(reference_probs, dtype=torch.bool)
    topk_mask.scatter_(-1, topk_indices, True)
    return topk_mask


def apply_cd_candidate_filter(
    logits: torch.Tensor,
    candidate_mask: torch.Tensor,
) -> torch.Tensor:
    filtered_logits = logits.masked_fill(
        ~candidate_mask,
        torch.finfo(logits.dtype).min,
    )
    return filtered_logits

def _compute_kl_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> float:
    """
    Returns a scalar KL(P_first || P_second) averaged over the vocab dimension.
    """
    p_first = torch.softmax(first_logits, dim=-1)
    p_second = torch.softmax(second_logits, dim=-1)
    # Add a tiny epsilon to avoid log(0)
    kl = (p_first * (torch.log(p_first + 1e-10) - torch.log(p_second + 1e-10))).sum(dim=-1).mean()
    return kl.item()

def log_logits_difference(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
    step: int,
) -> None:
    # Compute KL divergence between the two logit distributions
    first_probs = torch.softmax(first_logits, dim=-1)
    second_probs = torch.softmax(second_logits, dim=-1)
    
    # KL(P_first || P_second)
    kl_div = torch.sum(first_probs * (torch.log(first_probs + 1e-10) - torch.log(second_probs + 1e-10)), dim=-1).mean()
    
    # Cosine similarity between logit vectors
    first_logits_flat = first_logits.view(-1, first_logits.size(-1))
    second_logits_flat = second_logits.view(-1, second_logits.size(-1))
    cosine_sim = torch.nn.functional.cosine_similarity(first_logits_flat, second_logits_flat, dim=-1).mean()
    
    # L2 distance between logits
    l2_dist = torch.norm(first_logits - second_logits, p=2, dim=-1).mean()
    
    logger.info(
        f"Step {step}: KL divergence: {kl_div.item():.4f}, "
        f"Cosine similarity: {cosine_sim.item():.4f}, "
        f"L2 distance: {l2_dist.item():.4f}"
    )

@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    cd_alpha: float = 1.0,
    beta: float = 0.0,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    negative_hidden_mode: str = "mask_zero",
    divergence_accumulator: Optional[List[float]] = None,
    sample_id: Optional[int] = None,
    turn_index: Optional[int] = None,
    collector_topk: int = 32,
    collector: Optional[BlockRecordWriter] = None,
    seed: int = 0,
) -> SimpleNamespace:
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)
    
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    output = target(
        input_ids,
        position_ids=position_ids[:, :num_input_tokens],
        past_key_values=past_key_values_target,
        use_cache=True,
        logits_to_keep=1,
        output_hidden_states=True if block_size > 1 else False,
    )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature , gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True
    alpha_prev = torch.zeros((1, max(block_size - 1, 1), 3), dtype=torch.float32, device=model.device)
    alpha_current = torch.full((1, max(block_size - 1, 1), 3), float(cd_alpha), dtype=torch.float32, device=model.device)
    block_index = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        block_record_payload: Optional[Dict[str, torch.Tensor]] = None
        if block_size > 1:
            draft_position_ids = position_ids[:, start - target_hidden.shape[1] : start + block_size]
            positive_draft_logits, negative_draft_logits = compute_contrastive_draft_logits(
                model=model,
                target=target,
                block_output_ids=block_output_ids,
                target_hidden=target_hidden,
                draft_position_ids=draft_position_ids,
                past_key_values_draft=past_key_values_draft,
                block_size=block_size,
                negative_context_dropout=negative_context_dropout,
                negative_context_noise_std=negative_context_noise_std,
                negative_hidden_mode=negative_hidden_mode,
                gen=gen
            )
            past_key_values_draft.crop(start)

            draft_topk_token_ids, draft_topk_logits, neg_logits_on_draft_topk_ids = _extract_topk_aligned_logits(
                positive_logits=positive_draft_logits,
                negative_logits=negative_draft_logits,
                top_k=collector_topk,
            )

            block_positions = torch.arange(
                block_size - 1,
                device=model.device,
                dtype=torch.float32,
            ).unsqueeze(0)
            denom = float(max(block_size - 2, 1))
            block_positions = block_positions / denom

            abs_positions = block_position_ids[:, 1:].to(torch.float32)
            abs_positions = abs_positions / float(max(max_length - 1, 1))

            block_record_payload = {
                "draft_topk_token_ids": draft_topk_token_ids,
                "draft_topk_logits": draft_topk_logits,
                "neg_logits_on_draft_topk_ids": neg_logits_on_draft_topk_ids,
                "block_position": block_positions,
                "absolute_position": abs_positions,
                "alpha_prev": alpha_prev,
                "alpha_applied": alpha_current,
            }
            
            # ----- Divergence witness (optional) -----
            if divergence_accumulator is not None:
                kl_val = _compute_kl_divergence(positive_draft_logits, negative_draft_logits)
                divergence_accumulator.append(kl_val)

            if not TOP64_MASK_MODE:
                candidate_mask = build_cd_candidate_mask(
                    reference_logits=positive_draft_logits,
                    beta=beta,
                )
            else:
                candidate_mask = build_topk_probability_mask(
                    reference_logits=positive_draft_logits,
                    top_k=64,
                )
            
            final_draft_logits = apply_cd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=cd_alpha,
            )
            
            final_draft_logits = apply_cd_candidate_filter(
                logits=final_draft_logits,
                candidate_mask=candidate_mask,
            )
            
            # Applying positive draft logits to the top candidates in the final draft logits to further reduce the chance of selecting a token that is not favored by the positive draft
            """ n_keep = min(6, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :] """
            
            block_output_ids[:, 1:] = sample(final_draft_logits , gen=gen)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature , gen=gen)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        if collector is not None and block_record_payload is not None:
            collector.add(
                {
                    "sample_id": int(-1 if sample_id is None else sample_id),
                    "turn_index": int(-1 if turn_index is None else turn_index),
                    "block_index": int(block_index),
                    "draft_topk_token_ids": block_record_payload["draft_topk_token_ids"].to(torch.int32).squeeze(0).detach().cpu(),
                    "draft_topk_logits": block_record_payload["draft_topk_logits"].to(torch.float32).squeeze(0).detach().cpu(),
                    "neg_logits_on_draft_topk_ids": block_record_payload["neg_logits_on_draft_topk_ids"].to(torch.float32).squeeze(0).detach().cpu(),
                    "block_position": block_record_payload["block_position"].to(torch.float32).squeeze(0).detach().cpu(),
                    "absolute_position": block_record_payload["absolute_position"].to(torch.float32).squeeze(0).detach().cpu(),
                    "target_token_id": posterior[:, :-1].to(torch.int32).squeeze(0).detach().cpu(),
                    "acceptance_length": int(acceptance_length + 1),
                    "alpha_prev": block_record_payload["alpha_prev"].to(torch.float32).squeeze(0).detach().cpu(),
                    "alpha_applied": block_record_payload["alpha_applied"].to(torch.float32).squeeze(0).detach().cpu(),
                }
            )

        start += acceptance_length + 1
        block_index += 1
        if block_size > 1:
            alpha_prev = alpha_current
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


def main() -> None:
    global NEGATIVE_HIDDEN_MODE
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Global dataset row index to start loading from.",
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=None,
        help="Number of instances to load starting from --start-index.",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", type=float, default=0.1)
    parser.add_argument("--cd-beta", type=float, default=0.0)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--negative-hidden-mode",
        type=str,
        choices=["mask_zero", "shuffle_tokens"],
        default=NEGATIVE_HIDDEN_MODE,
        help="How to construct negative hidden: mask tokens to zero or shuffle token positions.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--collect-dataset",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Collect block-level alpha-training records during speculative decoding.",
    )
    parser.add_argument(
        "--collector-output-dir",
        type=str,
        default="alpha_model/collected_alpha_records",
        help="Output directory for block record shards.",
    )
    parser.add_argument(
        "--collector-topk",
        type=int,
        default=32,
        help="Top-k draft logits/token ids to store per diffusion position.",
    )
    parser.add_argument(
        "--collector-chunk-size",
        type=int,
        default=2048,
        help="Number of block records per saved shard file.",
    )
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    args = parser.parse_args()
    NEGATIVE_HIDDEN_MODE = args.negative_hidden_mode
    print(f"Using negative hidden mode: [bold magenta]{args.negative_hidden_mode}[/bold magenta]")

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")
        model_dtype = torch.bfloat16
    else:
        logger.warning("CUDA is unavailable. Falling back to CPU inference; this benchmark will run much slower.")
        device = torch.device("cpu")
        model_dtype = torch.float32

    def has_flash_attn():
        if not use_cuda:
            return False
        if args.deterministic:
            logger.info("Deterministic mode enabled. Forcing SDPA attention backend.")
            return False
        try:
            import flash_attn
            return True
        except ImportError:
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=model_dtype,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=model_dtype,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset_ALPHA(
        args.dataset,
        streaming="auto",
        start_index=args.start_index,
        num_instances=args.num_instances,
        max_samples=args.max_samples,
    )

    divergence_accumulator: List[float] = [] 
    collector: Optional[BlockRecordWriter] = None
    if args.collect_dataset:
        collector = BlockRecordWriter(
            output_dir=args.collector_output_dir,
            rank=dist.rank(),
            chunk_size=args.collector_chunk_size,
        )

    responses = []
    world_size = dist.size()
    rank = dist.rank()
    dataset_len = _dataset_length(dataset)
    total_for_rank = _estimate_total_for_rank(
        dataset_len=dataset_len,
        num_instances=args.num_instances,
        max_samples=args.max_samples,
        rank=rank,
        world_size=world_size,
    )
    
    base_seed = args.seed
    dataset_iterator = _iter_dataset_for_rank(dataset, rank, world_size)
    progress = tqdm(
        dataset_iterator,
        total=total_for_rank,
        disable=not dist.is_main(),
        desc=f"collect[{args.dataset}] rank{rank}",
        unit="sample",
        dynamic_ncols=True,
    )
    for idx, instance in progress:
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(target.device)

            response = {}
            for bs in [1, block_size]:
                response[bs] = dflash_generate(
                    model=draft_model,
                    target=target,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    cd_alpha=args.cd_alpha,
                    beta=args.cd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    negative_hidden_mode=args.negative_hidden_mode,
                    divergence_accumulator = divergence_accumulator,
                    sample_id=idx,
                    turn_index=turn_index,
                    collector_topk=args.collector_topk,
                    collector=collector if bs == block_size else None,
                    seed = sample_seed + bs,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            if collector is not None:
                collector.close()
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")
    
    # Report the collected divergence metric (if any)
    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        # Optionally also show min/max for a quick sense of spread
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

    if collector is not None:
        collector.close()
        if dist.is_main():
            print(f"Saved alpha training records to: {args.collector_output_dir}")

if __name__ == "__main__":
    main()
