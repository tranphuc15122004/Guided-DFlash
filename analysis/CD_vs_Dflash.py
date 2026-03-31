import argparse
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
from model import DFlashDraftModel, apply_cd_logits, sample, load_and_process_dataset, extract_context_feature
import distributed as dist


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
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


def _mean(values: list[float | int]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def _acceptance_histogram(acceptance_lengths: list[int], block_size: int) -> list[float]:
    if not acceptance_lengths:
        return [0.0 for _ in range(block_size + 1)]
    return [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]


def _input_bucket(num_input_tokens: int) -> str:
    if num_input_tokens <= 128:
        return "<=128"
    if num_input_tokens <= 512:
        return "129-512"
    if num_input_tokens <= 1024:
        return "513-1024"
    return ">1024"


def _detect_prompt_type(text: str) -> str:
    lower = text.lower()
    if "```" in text or any(k in lower for k in ["def ", "class ", "import ", "function", "bug", "code"]):
        return "coding"
    if "\\boxed" in text or any(k in lower for k in ["equation", "prove", "calculate", "integral", "math"]):
        return "math"
    return "chat"


def _build_case_record(
    *,
    dataset_idx: int,
    turn_index: int,
    rank: int,
    sample_seed: int,
    block_size: int,
    prompt_text: str,
    dflash: SimpleNamespace,
    cd_dflash: SimpleNamespace,
    dflash_text: str,
    cd_text: str,
    avg_kl_divergence: float,
) -> dict[str, Any]:
    dflash_acceptance = [int(x) for x in dflash.acceptance_lengths]
    cd_acceptance = [int(x) for x in cd_dflash.acceptance_lengths]
    dflash_mean = _mean(dflash_acceptance)
    cd_mean = _mean(cd_acceptance)
    delta = cd_mean - dflash_mean
    if delta > 1e-9:
        better = "cd"
    elif delta < -1e-9:
        better = "dflash"
    else:
        better = "tie"

    return {
        "dataset_idx": int(dataset_idx),
        "turn_index": int(turn_index),
        "rank": int(rank),
        "sample_seed": int(sample_seed),
        "block_size": int(block_size),
        "prompt_type": _detect_prompt_type(prompt_text),
        "input_bucket": _input_bucket(int(dflash.num_input_tokens)),
        "prompt_char_len": int(len(prompt_text)),
        "num_input_tokens": int(dflash.num_input_tokens),
        "avg_kl_divergence": float(avg_kl_divergence),
        "better_method": better,
        "dflash": {
            "num_output_tokens": int(dflash.num_output_tokens),
            "time_to_first_token": float(dflash.time_to_first_token),
            "time_per_output_token": float(dflash.time_per_output_token),
            "acceptance_lengths": dflash_acceptance,
            "acceptance_mean": float(dflash_mean),
            "acceptance_histogram": _acceptance_histogram(dflash_acceptance, block_size),
            "output_text": dflash_text,
        },
        "cd_dflash": {
            "num_output_tokens": int(cd_dflash.num_output_tokens),
            "time_to_first_token": float(cd_dflash.time_to_first_token),
            "time_per_output_token": float(cd_dflash.time_per_output_token),
            "acceptance_lengths": cd_acceptance,
            "acceptance_mean": float(cd_mean),
            "acceptance_histogram": _acceptance_histogram(cd_acceptance, block_size),
            "output_text": cd_text,
        },
        "acceptance_delta_cd_minus_dflash": float(delta),
    }


def _summarize_records(
    case_records: list[dict[str, Any]],
    *,
    block_size: int,
    args: argparse.Namespace,
    divergence_accumulator: list[float],
) -> dict[str, Any]:
    if not case_records:
        return {
            "meta": {
                "dataset": args.dataset,
                "seed": int(args.seed),
                "block_size": int(block_size),
            },
            "num_cases": 0,
        }

    dflash_means = [r["dflash"]["acceptance_mean"] for r in case_records]
    cd_means = [r["cd_dflash"]["acceptance_mean"] for r in case_records]
    deltas = [r["acceptance_delta_cd_minus_dflash"] for r in case_records]

    wins_cd = [r for r in case_records if r["better_method"] == "cd"]
    wins_dflash = [r for r in case_records if r["better_method"] == "dflash"]
    ties = [r for r in case_records if r["better_method"] == "tie"]

    by_pattern: dict[str, dict[str, list[float]]] = {}
    for r in case_records:
        key = f"{r['prompt_type']}|{r['input_bucket']}"
        if key not in by_pattern:
            by_pattern[key] = {"delta": [], "cd": [], "dflash": []}
        by_pattern[key]["delta"].append(r["acceptance_delta_cd_minus_dflash"])
        by_pattern[key]["cd"].append(r["cd_dflash"]["acceptance_mean"])
        by_pattern[key]["dflash"].append(r["dflash"]["acceptance_mean"])

    pattern_summary = {
        key: {
            "num_cases": len(vals["delta"]),
            "mean_delta_cd_minus_dflash": _mean(vals["delta"]),
            "mean_cd_acceptance": _mean(vals["cd"]),
            "mean_dflash_acceptance": _mean(vals["dflash"]),
        }
        for key, vals in by_pattern.items()
    }

    top_cd_better = sorted(case_records, key=lambda r: r["acceptance_delta_cd_minus_dflash"], reverse=True)[:20]
    top_dflash_better = sorted(case_records, key=lambda r: r["acceptance_delta_cd_minus_dflash"])[:20]

    return {
        "meta": {
            "dataset": args.dataset,
            "seed": int(args.seed),
            "block_size": int(block_size),
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "cd_alpha": float(args.cd_alpha),
            "cd_beta": float(args.cd_beta),
            "negative_context_dropout": float(args.negative_context_dropout),
            "negative_context_noise_std": float(args.negative_context_noise_std),
            "deterministic": bool(args.deterministic),
        },
        "num_cases": len(case_records),
        "overall": {
            "mean_dflash_acceptance": _mean(dflash_means),
            "mean_cd_acceptance": _mean(cd_means),
            "mean_delta_cd_minus_dflash": _mean(deltas),
            "cd_better_cases": len(wins_cd),
            "dflash_better_cases": len(wins_dflash),
            "tie_cases": len(ties),
            "cd_better_ratio": len(wins_cd) / len(case_records),
            "avg_kl_divergence": _mean(divergence_accumulator),
        },
        "pattern_summary": pattern_summary,
        "top_cd_better_cases": top_cd_better,
        "top_dflash_better_cases": top_dflash_better,
    }


def _save_analysis_artifacts(
    *,
    case_records: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    case_file = output_dir / "case_records.jsonl"
    with case_file.open("w", encoding="utf-8") as f:
        for row in case_records:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_file = output_dir / "summary.json"
    with summary_file.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    pattern_csv = output_dir / "pattern_summary.csv"
    with pattern_csv.open("w", encoding="utf-8") as f:
        f.write("pattern,num_cases,mean_delta_cd_minus_dflash,mean_cd_acceptance,mean_dflash_acceptance\n")
        for pattern, stats in sorted(summary.get("pattern_summary", {}).items()):
            f.write(
                f"{pattern},{stats['num_cases']},{stats['mean_delta_cd_minus_dflash']:.6f},"
                f"{stats['mean_cd_acceptance']:.6f},{stats['mean_dflash_acceptance']:.6f}\n"
            )

# construct negative sample by randomly replacing the batch's first token (target model generated token) 
def build_negative_block_output_random(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(0, vocab_size, (batch_size,), device=block_output_ids.device, generator=gen)
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids

def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()

    if dropout_ratio > 0.0:
        token_keep_mask = (
            torch.rand(
                negative_target_hidden.shape[:2],
                device=negative_target_hidden.device,
                generator=gen,
            ) >= dropout_ratio
        ).unsqueeze(-1)
        negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)

    if noise_std > 0.0:
        noise = torch.randn(
            negative_target_hidden.shape,
            dtype=negative_target_hidden.dtype,
            device=negative_target_hidden.device,
            generator=gen,
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
    gen: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    
    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size, gen=gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
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
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature, gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            noise_embedding = target.model.embed_tokens(block_output_ids)
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden,
                noise_embedding=noise_embedding,
                position_ids=position_ids[:, past_key_values_draft.get_seq_length() : start + block_size],
                past_key_values=past_key_values_draft,
                use_cache=True,
                is_causal=False,
            )[:, -block_size+1:, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits, gen=gen)
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

        posterior = sample(output.logits, temperature, gen=gen)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
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

@torch.inference_mode()
def dflash_cd_generate(
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
    divergence_accumulator: Optional[List[float]] = None,
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
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature, gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        
        # Draft phase
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
                gen=gen,
            )
            past_key_values_draft.crop(start)
            
            # ----- Divergence witness (optional) -----
            if divergence_accumulator is not None:
                kl_val = _compute_kl_divergence(positive_draft_logits, negative_draft_logits)
                divergence_accumulator.append(kl_val)

            candidate_mask = build_cd_candidate_mask(
                reference_logits=positive_draft_logits,
                beta=beta,
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
            n_keep = min(6, final_draft_logits.size(1))
            final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]
            
            block_output_ids[:, 1:] = sample(final_draft_logits, gen=gen)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        # verify phase
        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        # verified token for the next batch
        posterior = sample(output.logits, temperature, gen=gen)
        #update the output
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length+1)
        start += acceptance_length + 1
        #KV cache update
        past_key_values_target.crop(start)
        # update target hidden
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--cd-alpha", type=float, default=0.5)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument("--analysis-dir", type=str, default="Analysis/results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    args = parser.parse_args()

    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
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
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = []
    case_records: list[dict[str, Any]] = []

    indices = range(dist.rank(), len(dataset), dist.size())
    base_seed = args.seed
    for idx in tqdm(indices, disable=not dist.is_main()):
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        instance = dataset[idx]
        messages_dflash: list[dict[str, str]] = []
        messages_cd: list[dict[str, str]] = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages_dflash.append({"role": "user", "content": user_content})
            messages_cd.append({"role": "user", "content": user_content})

            dflash_input_text = tokenizer.apply_chat_template(
                messages_dflash,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            cd_input_text = tokenizer.apply_chat_template(
                messages_cd,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )

            dflash_input_ids = tokenizer.encode(dflash_input_text, return_tensors="pt").to(target.device)
            cd_input_ids = tokenizer.encode(cd_input_text, return_tensors="pt").to(target.device)

            dflash_response = dflash_generate(
                model=draft_model,
                target=target,
                input_ids=dflash_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                seed=sample_seed + turn_index * 100 + 11,
            )
            cd_response = dflash_cd_generate(
                model=draft_model,
                target=target,
                input_ids=cd_input_ids,
                mask_token_id=draft_model.mask_token_id,
                max_new_tokens=args.max_new_tokens,
                block_size=block_size,
                stop_token_ids=[tokenizer.eos_token_id],
                temperature=args.temperature,
                cd_alpha=args.cd_alpha,
                beta=args.cd_beta,
                negative_context_dropout=args.negative_context_dropout,
                negative_context_noise_std=args.negative_context_noise_std,
                divergence_accumulator=divergence_accumulator,
                seed=sample_seed + turn_index * 100 + 22,
            )

            dflash_ids = dflash_response.output_ids[0, dflash_response.num_input_tokens:]
            cd_ids = cd_response.output_ids[0, cd_response.num_input_tokens:]
            dflash_text = tokenizer.decode(dflash_ids, skip_special_tokens=True)
            cd_text = tokenizer.decode(cd_ids, skip_special_tokens=True)

            messages_dflash.append({"role": "assistant", "content": dflash_text})
            messages_cd.append({"role": "assistant", "content": cd_text})

            case_records.append(
                _build_case_record(
                    dataset_idx=idx,
                    turn_index=turn_index,
                    rank=dist.rank(),
                    sample_seed=sample_seed,
                    block_size=block_size,
                    prompt_text=user_content,
                    dflash=dflash_response,
                    cd_dflash=cd_response,
                    dflash_text=dflash_text,
                    cd_text=cd_text,
                    avg_kl_divergence=_mean(divergence_accumulator),
                )
            )

    if dist.size() > 1:
        case_records = dist.gather(case_records, dst=0)
        divergence_accumulator = dist.gather(divergence_accumulator, dst=0)
        if not dist.is_main():
            return
        case_records = list(chain(*case_records))
        divergence_accumulator = list(chain(*divergence_accumulator))

    summary = _summarize_records(
        case_records,
        block_size=block_size,
        args=args,
        divergence_accumulator=divergence_accumulator,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.analysis_dir) / f"{args.dataset}_bs{block_size}_seed{args.seed}_{timestamp}"
    _save_analysis_artifacts(
        case_records=case_records,
        summary=summary,
        output_dir=output_dir,
    )

    overall = summary.get("overall", {})
    print(f"Saved analysis artifacts to: {output_dir}")
    print(f"Total cases: {summary.get('num_cases', 0)}")
    print(
        "Mean acceptance | "
        f"DFlash={overall.get('mean_dflash_acceptance', 0.0):.3f} | "
        f"CD-DFLASH={overall.get('mean_cd_acceptance', 0.0):.3f} | "
        f"Delta={overall.get('mean_delta_cd_minus_dflash', 0.0):.3f}"
    )
    print(
        "Win ratio | "
        f"CD better={overall.get('cd_better_cases', 0)} | "
        f"DFlash better={overall.get('dflash_better_cases', 0)} | "
        f"Tie={overall.get('tie_cases', 0)}"
    )
    if divergence_accumulator:
        print(
            f"KL divergence | avg={overall.get('avg_kl_divergence', 0.0):.5f} | "
            f"min={min(divergence_accumulator):.5f} | max={max(divergence_accumulator):.5f}"
        )

if __name__ == "__main__":
    main()