import argparse
import csv
import json
import os
import time
import random
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import *
import distributed as dist

NEGATIVE_HIDDEN_MODE = 'mask_zero' 
TOP64_MASK_MODE = False 


def _safe_dataset_name(name: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name)


def _parse_position_list(position_text: str) -> list[int]:
    out: list[int] = []
    for chunk in position_text.split(","):
        s = chunk.strip()
        if not s:
            continue
        out.append(int(s))
    if not out:
        raise ValueError("collect position list is empty")
    return sorted(set(out))


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    token_logit = logits_1d[token_id]
    return int((logits_1d > token_logit).sum().item()) + 1


def _resolve_numpy_dtype(dtype_name: str) -> np.dtype:
    name = dtype_name.lower().strip()
    if name == "float16":
        return np.float16
    if name == "float32":
        return np.float32
    raise ValueError(f"Unsupported collect logits dtype: {dtype_name}")


class RejectTokenCollector:
    def __init__(
        self,
        *,
        output_dir: Path,
        rank_id: int,
        block_size: int,
        positions_to_collect: list[int],
        shard_size: int,
        logits_dtype: np.dtype,
        meta: dict[str, Any],
    ) -> None:
        self.rank_id = int(rank_id)
        self.block_size = int(block_size)
        self.positions_to_collect = sorted(set(int(x) for x in positions_to_collect))
        self.positions_set = set(self.positions_to_collect)
        self.shard_size = max(1, int(shard_size))
        self.logits_dtype = logits_dtype

        self.rank_dir = output_dir / f"rank_{self.rank_id:02d}"
        self.rank_dir.mkdir(parents=True, exist_ok=True)
        self.shard_dir = self.rank_dir / "logit_shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        self.meta_jsonl_path = self.rank_dir / "reject_token_records.jsonl"
        self.meta_csv_path = self.rank_dir / "reject_token_records.csv"
        self.summary_path = self.rank_dir / "reject_token_summary.json"

        self.fieldnames = [
            "sample_idx",
            "turn_idx",
            "decode_step",
            "block_start_position",
            "absolute_position",
            "block_position",
            "reject_offset_in_block",
            "acceptance_length",
            "target_token_id",
            "draft_token_id",
            "draft_rank_in_positive",
            "shard_id",
            "row_in_shard",
        ]

        self.meta_jsonl_file = self.meta_jsonl_path.open("w", encoding="utf-8")
        self.meta_csv_file = self.meta_csv_path.open("w", newline="", encoding="utf-8")
        self.meta_csv_writer = csv.DictWriter(self.meta_csv_file, fieldnames=self.fieldnames)
        self.meta_csv_writer.writeheader()

        self.pos_logits_buffer: list[np.ndarray] = []
        self.neg_logits_buffer: list[np.ndarray] = []
        self.target_logits_buffer: list[np.ndarray] = []
        self.target_token_buffer: list[int] = []
        self.draft_token_buffer: list[int] = []
        self.draft_rank_buffer: list[int] = []

        self.shard_id = 0
        self.total_records = 0
        self.counts_by_position: dict[int, int] = {pos: 0 for pos in self.positions_to_collect}
        self.run_meta = meta

    def add(
        self,
        *,
        sample_idx: int,
        turn_idx: int,
        decode_step: int,
        block_start_position: int,
        block_position: int,
        reject_offset_in_block: int,
        acceptance_length: int,
        target_token_id: int,
        draft_token_id: int,
        positive_logits: torch.Tensor,
        negative_logits: torch.Tensor,
        target_logits: torch.Tensor,
    ) -> None:
        if block_position not in self.positions_set:
            return

        pos_float = positive_logits.float()
        draft_rank_in_positive = _token_rank_from_logits(pos_float, int(draft_token_id))

        pos_np = pos_float.detach().cpu().numpy().astype(self.logits_dtype, copy=False)
        neg_np = negative_logits.float().detach().cpu().numpy().astype(self.logits_dtype, copy=False)
        target_np = target_logits.float().detach().cpu().numpy().astype(self.logits_dtype, copy=False)

        row_in_shard = len(self.pos_logits_buffer)
        row = {
            "sample_idx": int(sample_idx),
            "turn_idx": int(turn_idx),
            "decode_step": int(decode_step),
            "block_start_position": int(block_start_position),
            "absolute_position": int(block_start_position + reject_offset_in_block + 1),
            "block_position": int(block_position),
            "reject_offset_in_block": int(reject_offset_in_block),
            "acceptance_length": int(acceptance_length),
            "target_token_id": int(target_token_id),
            "draft_token_id": int(draft_token_id),
            "draft_rank_in_positive": int(draft_rank_in_positive),
            "shard_id": int(self.shard_id),
            "row_in_shard": int(row_in_shard),
        }

        self.meta_jsonl_file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.meta_csv_writer.writerow(row)

        self.pos_logits_buffer.append(pos_np)
        self.neg_logits_buffer.append(neg_np)
        self.target_logits_buffer.append(target_np)
        self.target_token_buffer.append(int(target_token_id))
        self.draft_token_buffer.append(int(draft_token_id))
        self.draft_rank_buffer.append(int(draft_rank_in_positive))

        self.total_records += 1
        self.counts_by_position[int(block_position)] = self.counts_by_position.get(int(block_position), 0) + 1

        if len(self.pos_logits_buffer) >= self.shard_size:
            self.flush_shard()

    def flush_shard(self) -> None:
        if not self.pos_logits_buffer:
            return

        path = self.shard_dir / f"shard_{self.shard_id:06d}.npz"
        np.savez_compressed(
            path,
            positive_logits=np.stack(self.pos_logits_buffer, axis=0),
            negative_logits=np.stack(self.neg_logits_buffer, axis=0),
            target_logits=np.stack(self.target_logits_buffer, axis=0),
            target_token_id=np.asarray(self.target_token_buffer, dtype=np.int32),
            draft_token_id=np.asarray(self.draft_token_buffer, dtype=np.int32),
            draft_rank_in_positive=np.asarray(self.draft_rank_buffer, dtype=np.int32),
        )

        self.pos_logits_buffer.clear()
        self.neg_logits_buffer.clear()
        self.target_logits_buffer.clear()
        self.target_token_buffer.clear()
        self.draft_token_buffer.clear()
        self.draft_rank_buffer.clear()
        self.shard_id += 1

    def finalize(self) -> None:
        self.flush_shard()
        self.meta_jsonl_file.close()
        self.meta_csv_file.close()

        summary = {
            "meta": self.run_meta,
            "rank_id": int(self.rank_id),
            "block_size": int(self.block_size),
            "positions_to_collect": [int(x) for x in self.positions_to_collect],
            "logits_dtype": str(self.logits_dtype),
            "shard_size": int(self.shard_size),
            "num_records": int(self.total_records),
            "num_shards": int(self.shard_id),
            "counts_by_position": {str(k): int(v) for k, v in sorted(self.counts_by_position.items())},
            "artifacts": {
                "records_jsonl": self.meta_jsonl_path.name,
                "records_csv": self.meta_csv_path.name,
                "shard_dir": self.shard_dir.name,
            },
        }
        self.summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


def _build_collect_output_dir(dataset_name: str, output_dir: Optional[str]) -> Path:
    if output_dir:
        out = Path(output_dir)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = Path("alpha_adjusting") / "results" / f"alpha_collect_{_safe_dataset_name(dataset_name)}_{ts}"
    out.mkdir(parents=True, exist_ok=True)
    return out

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
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    negative_hidden_mode: str = "mask_zero",
    divergence_accumulator: Optional[List[float]] = None,
    reject_collector: Optional[RejectTokenCollector] = None,
    sample_idx: int = -1,
    turn_idx: int = -1,
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
    decode_step = 0

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
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
        acceptance_length = int((block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())

        if reject_collector is not None and block_size > 1 and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            reject_block_position = reject_offset + 1
            target_token_id = int(posterior[0, reject_offset].item())
            draft_token_id = int(block_output_ids[0, reject_offset + 1].item())
            reject_collector.add(
                sample_idx=sample_idx,
                turn_idx=turn_idx,
                decode_step=decode_step,
                block_start_position=start,
                block_position=reject_block_position,
                reject_offset_in_block=reject_offset,
                acceptance_length=acceptance_length,
                target_token_id=target_token_id,
                draft_token_id=draft_token_id,
                positive_logits=positive_draft_logits[0, reject_offset, :],
                negative_logits=negative_draft_logits[0, reject_offset, :],
                target_logits=output.logits[0, reject_offset, :],
            )

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        decode_step += 1
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
    parser.add_argument(
        "--collect-reject-positions",
        type=str,
        default="6,7,8",
        help="Comma-separated block positions to collect when reject happens (1-based inside draft block).",
    )
    parser.add_argument(
        "--collect-output-dir",
        type=str,
        default=None,
        help="Output directory for collected alpha-adjusting dataset; defaults to timestamped folder under alpha_adjusting/results.",
    )
    parser.add_argument(
        "--collect-shard-size",
        type=int,
        default=128,
        help="Number of records per logits shard (.npz).",
    )
    parser.add_argument(
        "--collect-logits-dtype",
        type=str,
        choices=["float16", "float32"],
        default="float16",
        help="Storage dtype for collected logits shards.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
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
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    collect_positions = _parse_position_list(args.collect_reject_positions)
    for pos in collect_positions:
        if pos < 1 or pos >= block_size:
            raise ValueError(
                f"Invalid collect position {pos} for block_size={block_size}. Valid range is [1, {block_size - 1}]"
            )

    collect_output_dir = _build_collect_output_dir(args.dataset, args.collect_output_dir)
    collector_meta = {
        "dataset": args.dataset,
        "seed": int(args.seed),
        "model_name_or_path": args.model_name_or_path,
        "draft_name_or_path": args.draft_name_or_path,
        "block_size": int(block_size),
        "cd_alpha": float(args.cd_alpha),
        "cd_beta": float(args.cd_beta),
        "negative_context_dropout": float(args.negative_context_dropout),
        "negative_context_noise_std": float(args.negative_context_noise_std),
        "negative_hidden_mode": args.negative_hidden_mode,
        "collect_reject_positions": [int(x) for x in collect_positions],
        "collect_logits_dtype": args.collect_logits_dtype,
    }
    reject_collector = RejectTokenCollector(
        output_dir=collect_output_dir,
        rank_id=dist.rank(),
        block_size=block_size,
        positions_to_collect=collect_positions,
        shard_size=args.collect_shard_size,
        logits_dtype=_resolve_numpy_dtype(args.collect_logits_dtype),
        meta=collector_meta,
    )

    divergence_accumulator: List[float] = [] 

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    
    base_seed = args.seed
    for idx in tqdm(indices, disable=not dist.is_main()):
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        instance = dataset[idx]
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
                    reject_collector=reject_collector if bs == block_size else None,
                    sample_idx=idx,
                    turn_idx=turn_index,
                    seed = sample_seed + bs,
                )
            
            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    reject_collector.finalize()
    per_rank_collect_count = dist.all_gather(reject_collector.total_records)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
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
    print(f"Collect reject positions: {collect_positions}")
    print(f"Collected records across ranks: {sum(int(x) for x in per_rank_collect_count)}")
    print(f"Collector output dir: {collect_output_dir}")
    
    # Report the collected divergence metric (if any)
    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        # Optionally also show min/max for a quick sense of spread
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

if __name__ == "__main__":
    main()
