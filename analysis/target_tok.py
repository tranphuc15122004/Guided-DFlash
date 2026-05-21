import argparse
import csv
import datetime
import json
import os
import time
import random
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional
from loguru import logger
import numpy as np
import torch
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import *
import distributed as dist
import model
from scheme.run_metadata import log_run_parameters

NEGATIVE_HIDDEN_MODE = 'mask_zero' 
TOP64_MASK_MODE = False 

def cuda_time() -> float:
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

def build_negative_block_output_mask(block_output_ids: torch.Tensor, mask_token : int , vocab_size: int , gen : torch.Generator = None) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    negative_block_output_ids[:, 0] = mask_token
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
    mask_token : int,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    negative_hidden_mode: str,
    gen : torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    
    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size , gen)
    #neg_block_output_ids = build_negative_block_output_mask (block_output_ids , mask_token , target.config.vocab_size , gen)
    
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


# =============================================================================

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
    output_ids_collector: Optional[List[torch.Tensor]] = None,
    ar_target_tok_prediction_collector: Optional[List[torch.Tensor]] = None,
    block_analysis_collector: Optional[List[dict]] = None,
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
    
    # GPU-side buffer for AR predictions — accumulates on device and flushes to CPU once at the end
    gpu_ar_buffer: List[torch.Tensor] = []

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
                mask_token = mask_token_id,
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
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]
        
        
        # Number of remaining positions after the bonus token to predict via AR
        num_ar_steps = block_size - acceptance_length - 2

        # ── Only run expensive AR prediction when needed by a collector ──
        need_ar_prediction = (
            ar_target_tok_prediction_collector is not None
            or block_analysis_collector is not None
        )
        if block_size > 1 and num_ar_steps > 0 and need_ar_prediction:
            # AR target model prediction (non-intrusive: save/restore KV cache)
            # Seed = bonus token (posterior[:, acceptance_length]) at current `start` position
            ar_seed_tok = output_ids[:, start : start + 1]
            ar_position_ids_local = position_ids[:, start : start + 1]

            ar_target_tok_pred_stacked = torch.full(
                (2, block_size + 1),
                mask_token_id,
                dtype=torch.long,
                device=model.device,
            )

            # Save a snapshot of the current KV cache length
            ar_cache_len = past_key_values_target.get_seq_length()
            
            for ar_step_idx in range(num_ar_steps):
                abs_pos = start + 1 + ar_step_idx
                ar_target_tok_pred_stacked[0, ar_step_idx] = abs_pos  # absolute position
                ar_output = target(
                    ar_seed_tok,
                    position_ids=ar_position_ids_local,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                )
                ar_next_token = sample(ar_output.logits, temperature , gen=gen)
                ar_target_tok_pred_stacked[1, ar_step_idx] = ar_next_token
                ar_seed_tok = ar_next_token
                ar_position_ids_local += 1
            
            # Restore cache to pre-AR state so speculative loop is unaffected
            past_key_values_target.crop(ar_cache_len)
        
        # Keep AR predictions on GPU, flush to CPU collector at the end of generation
        if (
            ar_target_tok_prediction_collector is not None
            and block_size > 1
            and num_ar_steps > 0
            and need_ar_prediction
        ):
            gpu_ar_buffer.append(ar_target_tok_pred_stacked)  # stays on GPU
            
        
        # Build full-block target reference (only needed by collectors)
        if block_analysis_collector is not None:
            tmp_target_tok_ids = torch.full((1, block_size), mask_token_id, dtype=torch.long, device=model.device)
            tmp_target_tok_ids[:, : acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            if acceptance_length + 1 < block_size:
                tmp_target_tok_ids[:, acceptance_length + 1] = posterior[:, acceptance_length]  # bonus token
            if num_ar_steps > 0 and need_ar_prediction:
                tmp_target_tok_ids[:, acceptance_length + 2:] = ar_target_tok_pred_stacked[1, :num_ar_steps]
        
        # ── Per-position analysis: target rank in positive & CD logits ──
        if block_analysis_collector is not None and block_size > 1:
            record = {
                "block_start": int(start - acceptance_length - 1),
                "acceptance_length": int(acceptance_length),
                "num_ar_steps": int(max(0, num_ar_steps)),
            }
            for p in range(1, block_size):
                target_id = int(tmp_target_tok_ids[0, p].item())
                if target_id == mask_token_id:
                    continue
                draft_id = int(block_output_ids[0, p].item())
                pos_logits_1d = positive_draft_logits[0, p-1, :].float()
                cd_logits_1d  = final_draft_logits[0, p-1, :].float()
                
                pos_rank = int((pos_logits_1d > pos_logits_1d[target_id]).sum().item()) + 1
                cd_rank  = int((cd_logits_1d  > cd_logits_1d[target_id]).sum().item()) + 1
                pos_logprob = float(torch.log_softmax(pos_logits_1d, dim=-1)[target_id].item())
                cd_logprob  = float(torch.log_softmax(cd_logits_1d, dim=-1)[target_id].item())
                pos_probs = torch.softmax(pos_logits_1d, dim=-1)
                entropy = float(-(pos_probs * torch.log(pos_probs + 1e-10)).sum().item())
                
                if p <= acceptance_length:
                    cohort = "verify"
                elif p == acceptance_length + 1:
                    cohort = "bonus"
                else:
                    cohort = "ar_suffix"
                
                record[f"p{p}_cohort"] = cohort
                record[f"p{p}_target_id"] = target_id
                record[f"p{p}_draft_id"] = draft_id
                record[f"p{p}_draft_matches_target"] = int(draft_id == target_id)
                record[f"p{p}_pos_rank"] = pos_rank
                record[f"p{p}_cd_rank"] = cd_rank
                record[f"p{p}_pos_logprob"] = pos_logprob
                record[f"p{p}_cd_logprob"] = cd_logprob
                record[f"p{p}_entropy"] = entropy
            block_analysis_collector.append(record)
        
        
        past_key_values_target.crop(start)
        # Check if we've hit a stop token in the accepted output
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

    # Flush GPU AR buffer to CPU collector in one go
    if ar_target_tok_prediction_collector is not None and gpu_ar_buffer:
        for t in gpu_ar_buffer:
            ar_target_tok_prediction_collector.append(t.cpu())
        gpu_ar_buffer.clear()

    if output_ids_collector is not None:
        output_ids_collector.append(output_ids.cpu())

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
    parser.add_argument("--negative-context-dropout", type=float, default=1.0)
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
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    parser.add_argument(
        "--collect-divergence",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compute and accumulate KL divergence between pos/neg draft logits each block.",
    )
    parser.add_argument(
        "--collect-outputs",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save generated output IDs to disk (requires extra CPU memory).",
    )
    parser.add_argument(
        "--collect-block-analysis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect per-position rank/logprob stats every block (triggers expensive AR prediction loop).",
    )
    parser.add_argument(
        "--collect-target-tok-preds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Collect AR target-model predictions for suffix positions (triggers expensive AR prediction loop).",
    )
    parser.add_argument(
        "--output-dir", type=str, default="./analysis/results", help="Directory to save outputs and logs."
    )
    args = parser.parse_args()
    log_run_parameters("CD_v2", args)
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

    divergence_accumulator: Optional[List[float]] = ([] if args.collect_divergence else None)
    output_ids_collector: Optional[List[torch.Tensor]] = ([] if args.collect_outputs else None)
    target_tok_prediction_collector: Optional[List[torch.Tensor]] = ([] if args.collect_target_tok_preds else None)
    block_analysis_collector: Optional[List[dict]] = ([] if args.collect_block_analysis else None)

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
                    output_ids_collector = output_ids_collector,
                    ar_target_tok_prediction_collector = target_tok_prediction_collector,
                    block_analysis_collector = block_analysis_collector,
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
    
    # save the collected output ids for further analysis
    output_dir = Path(args.output_dir) / f'target_tok_{args.dataset}_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(output_dir, exist_ok=True)
    if output_ids_collector:
        output_path = output_dir / "collected_output_ids.pt"
        torch.save(output_ids_collector, output_path)
        print(f"Collected output IDs saved to: {output_path}")
    
    if target_tok_prediction_collector:
        target_tok_output_path = output_dir / "collected_target_tok_predictions.pt"
        torch.save(target_tok_prediction_collector, target_tok_output_path)
        print(f"Collected target token predictions saved to: {target_tok_output_path}")
    
    # ── Block analysis post-processing ──
    if block_analysis_collector:
        # Flatten per-position records into rows
        flat_rows: list[dict] = []
        for rec in block_analysis_collector:
            for p in range(1, block_size):
                key = f"p{p}_cohort"
                if key not in rec:
                    continue
                row = {
                    "block_start": rec.get("block_start", 0),
                    "acceptance_length": rec.get("acceptance_length", 0),
                    "block_position": p,
                    "cohort": rec[key],
                    "target_id": rec.get(f"p{p}_target_id", -1),
                    "draft_id": rec.get(f"p{p}_draft_id", -1),
                    "draft_matches_target": rec.get(f"p{p}_draft_matches_target", 0),
                    "pos_rank": rec.get(f"p{p}_pos_rank", -1),
                    "cd_rank": rec.get(f"p{p}_cd_rank", -1),
                    "pos_logprob": rec.get(f"p{p}_pos_logprob", 0.0),
                    "cd_logprob": rec.get(f"p{p}_cd_logprob", 0.0),
                    "entropy": rec.get(f"p{p}_entropy", 0.0),
                }
                row["delta_rank"] = row["pos_rank"] - row["cd_rank"]  # >0 = CD improves
                row["delta_logprob"] = row["cd_logprob"] - row["pos_logprob"]
                flat_rows.append(row)
        
        # Write JSONL
        analysis_jsonl = output_dir / "block_analysis.jsonl"
        with open(analysis_jsonl, "w") as f:
            for row in flat_rows:
                f.write(json.dumps(row) + "\n")
        
        # Build per-position and per-cohort summaries
        pos_summary_rows: list[dict] = []
        for pos in sorted(set(r["block_position"] for r in flat_rows)):
            subset = [r for r in flat_rows if r["block_position"] == pos]
            ranks = np.array([r["pos_rank"] for r in subset], dtype=float)
            cd_ranks = np.array([r["cd_rank"] for r in subset], dtype=float)
            deltas = np.array([r["delta_rank"] for r in subset], dtype=float)
            match_rate = np.mean([r["draft_matches_target"] for r in subset])
            cohort = subset[0]["cohort"]
            pos_summary_rows.append({
                "block_position": pos,
                "cohort": cohort,
                "count": len(subset),
                "pos_rank_mean": float(ranks.mean()),
                "pos_rank_median": float(np.median(ranks)),
                "cd_rank_mean": float(cd_ranks.mean()),
                "delta_rank_mean": float(deltas.mean()),
                "draft_match_rate": float(match_rate),
                "top1_rate": float((ranks == 1).mean()),
                "cd_top1_rate": float((cd_ranks == 1).mean()),
            })
        
        cohort_summary_rows: list[dict] = []
        for cohort in ["verify", "bonus", "ar_suffix"]:
            subset = [r for r in flat_rows if r["cohort"] == cohort]
            if not subset:
                continue
            ranks = np.array([r["pos_rank"] for r in subset], dtype=float)
            cd_ranks = np.array([r["cd_rank"] for r in subset], dtype=float)
            deltas = np.array([r["delta_rank"] for r in subset], dtype=float)
            logprobs = np.array([r["pos_logprob"] for r in subset], dtype=float)
            entropies = np.array([r["entropy"] for r in subset], dtype=float)
            cohort_summary_rows.append({
                "cohort": cohort,
                "count": len(subset),
                "pos_rank_mean": float(ranks.mean()),
                "pos_rank_median": float(np.median(ranks)),
                "cd_rank_mean": float(cd_ranks.mean()),
                "delta_rank_mean": float(deltas.mean()),
                "pos_logprob_mean": float(logprobs.mean()),
                "entropy_mean": float(entropies.mean()),
                "top1_rate": float((ranks == 1).mean()),
                "top5_rate": float((ranks <= 5).mean()),
                "cd_top1_rate": float((cd_ranks == 1).mean()),
                "cd_top5_rate": float((cd_ranks <= 5).mean()),
                "draft_match_rate": float(np.mean([r["draft_matches_target"] for r in subset])),
            })
        
        # Write CSV
        analysis_csv = output_dir / "block_analysis.csv"
        if flat_rows:
            fieldnames = list(flat_rows[0].keys())
            with open(analysis_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(flat_rows)
        
        pos_csv = output_dir / "block_analysis_by_position.csv"
        if pos_summary_rows:
            fieldnames = list(pos_summary_rows[0].keys())
            with open(pos_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(pos_summary_rows)
        
        cohort_csv = output_dir / "block_analysis_by_cohort.csv"
        if cohort_summary_rows:
            fieldnames = list(cohort_summary_rows[0].keys())
            with open(cohort_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(cohort_summary_rows)
        
        # Build summary dict
        summary_data = {
            "meta": {
                "dataset": args.dataset,
                "seed": args.seed,
                "block_size": block_size,
            },
            "num_blocks": len(block_analysis_collector),
            "num_position_records": len(flat_rows),
            "cohorts": {r["cohort"]: {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for r in cohort_summary_rows},
            "by_position": [{k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in r.items()} for r in pos_summary_rows],
        }
        summary_json = output_dir / "block_analysis_summary.json"
        with open(summary_json, "w") as f:
            json.dump(summary_data, f, indent=2)
        
        # Write markdown report
        report_md = output_dir / "block_analysis_report.md"
        lines = ["# Block Analysis Report", ""]
        lines.append(f"Dataset: **{args.dataset}**")
        lines.append(f"Block size: **{block_size}**")
        lines.append(f"Total blocks: **{len(block_analysis_collector)}**")
        lines.append(f"Total position records: **{len(flat_rows)}**")
        lines.append("")
        
        lines.append("## Per-Cohort Summary")
        lines.append("| cohort | count | pos_rank | cd_rank | delta_rank | top1 | cd_top1 | top5 | cd_top5 | draft_match | entropy |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in cohort_summary_rows:
            lines.append(
                f"| {r['cohort']} | {r['count']} | "
                f"{r['pos_rank_mean']:.2f} | {r['cd_rank_mean']:.2f} | {r['delta_rank_mean']:+.3f} | "
                f"{r['top1_rate']*100:.1f}% | {r['cd_top1_rate']*100:.1f}% | "
                f"{r['top5_rate']*100:.1f}% | {r['cd_top5_rate']*100:.1f}% | "
                f"{r['draft_match_rate']*100:.1f}% | {r['entropy_mean']:.3f} |"
            )
        lines.append("")
        
        lines.append("## Per-Position Profile")
        lines.append("| pos | cohort | count | pos_rank | cd_rank | delta_rank | top1 | cd_top1 | draft_match |")
        lines.append("|---:|---|---:|---:|---:|---:|---:|---:|---:|")
        for r in pos_summary_rows:
            lines.append(
                f"| {r['block_position']} | {r['cohort']} | {r['count']} | "
                f"{r['pos_rank_mean']:.2f} | {r['cd_rank_mean']:.2f} | {r['delta_rank_mean']:+.3f} | "
                f"{r['top1_rate']*100:.1f}% | {r['cd_top1_rate']*100:.1f}% | "
                f"{r['draft_match_rate']*100:.1f}% |"
            )
        lines.append("")
        
        with open(report_md, "w") as f:
            f.write("\n".join(lines))
        
        print(f"Block analysis artifacts saved to: {output_dir}")
        print(f"  - {analysis_jsonl.name}")
        print(f"  - {analysis_csv.name}")
        print(f"  - {pos_csv.name}")
        print(f"  - {cohort_csv.name}")
        print(f"  - {summary_json.name}")
        print(f"  - {report_md.name}")


if __name__ == "__main__":
    main()
