"""
CD_negative_model — Contrastive Decoding with Learned Negative Logit Predictor.

Instead of learning scalar alpha values (CD_alpha_model.py), this variant uses
a NegativeLogitPredictor to directly output 32 negative logit values for the
top-32 positive tokens, with a fixed alpha=1.0 in the CD formula.

Key differences from CD_alpha_model.py:
  - Uses NegativeLogitPredictor instead of ContextualBanditAlpha
  - Output head produces 32 logit values (not 3 bucket alphas)
  - apply_cd_with_predicted_neg() replaces apply_cd_logits_dynamic()
  - build_full_neg_from_top32() replaces _build_full_alpha_from_buckets()
  - No bucket logic (thresholds, num_alpha_buckets)
"""

import argparse
import os
import time
import random
from itertools import chain
from types import SimpleNamespace
from typing import List, Optional
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import *
from alpha_model import (
    NegativeLogitPredictor,
    GaussianNegativePolicy,
)
import distributed as dist
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
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


# ── Negative context construction (identical to CD_alpha_model) ──

def build_negative_block_output_random(
    block_output_ids: torch.Tensor, vocab_size: int,
    gen: torch.Generator = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    random_tokens = torch.randint(
        0, vocab_size, (batch_size,),
        device=block_output_ids.device, generator=gen,
    )
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids


def build_negative_block_output_mask(
    block_output_ids: torch.Tensor, mask_token: int,
    vocab_size: int, gen: torch.Generator = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]
    negative_block_output_ids[:, 0] = mask_token
    return negative_block_output_ids


def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    mode: str = "mask_zero",
    gen: torch.Generator = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()
    batch_size, context_len = negative_target_hidden.shape[:2]

    if mode == "mask_zero":
        if dropout_ratio > 0.0:
            token_keep_mask = (
                torch.rand(
                    negative_target_hidden.shape[:2],
                    device=negative_target_hidden.device,
                    generator=gen,
                ) >= dropout_ratio
            ).unsqueeze(-1)
            negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)
    elif mode == "shuffle_tokens":
        if context_len > 1:
            per_batch_indices = []
            for _ in range(batch_size):
                perm = torch.randperm(context_len, device=negative_target_hidden.device, generator=gen)
                per_batch_indices.append(perm)
            gather_index = torch.stack(per_batch_indices, dim=0).unsqueeze(-1).expand(
                -1, -1, negative_target_hidden.shape[-1]
            )
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
    mask_token: int,
    negative_context_dropout: float,
    negative_context_noise_std: float,
    negative_hidden_mode: str,
    gen: torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]

    neg_block_output_ids = build_negative_block_output_random(
        block_output_ids, target.config.vocab_size, gen,
    )

    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        mode=negative_hidden_mode,
        gen=gen,
    )

    positive_noise_embedding = target.model.embed_tokens(block_output_ids)
    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)

    paired_noise_embedding = torch.cat([positive_noise_embedding, negative_noise_embedding], dim=0)
    paired_hidden = torch.cat([target_hidden, negative_target_hidden], dim=0)
    paired_position_ids = torch.cat([draft_position_ids, draft_position_ids], dim=0)

    paired_draft_logits = target.lm_head(model(
        target_hidden=paired_hidden,
        noise_embedding=paired_noise_embedding,
        position_ids=paired_position_ids,
        past_key_values=past_key_values_draft,
        use_cache=True,
        is_causal=False,
    )[:, -block_size + 1:, :])

    first_draft_logits, second_draft_logits = paired_draft_logits.split(batch_size, dim=0)
    return first_draft_logits, second_draft_logits


# ── Candidate masking (identical to CD_alpha_model) ──

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


# ── Logit extraction (identical to CD_alpha_model) ──

def _extract_topk_aligned_logits(
    positive_logits: torch.Tensor,
    negative_logits: torch.Tensor,
    top_k: int,
):
    k = min(max(1, top_k), positive_logits.size(-1))
    topk = torch.topk(positive_logits, k=k, dim=-1)
    draft_topk_logits = topk.values
    draft_topk_token_ids = topk.indices
    neg_logits_on_draft_topk_ids = torch.gather(
        negative_logits, dim=-1, index=draft_topk_token_ids,
    )
    return draft_topk_token_ids, draft_topk_logits, neg_logits_on_draft_topk_ids


def _compute_kl_divergence(
    first_logits: torch.Tensor,
    second_logits: torch.Tensor,
) -> float:
    p_first = torch.softmax(first_logits, dim=-1)
    p_second = torch.softmax(second_logits, dim=-1)
    kl = (p_first * (torch.log(p_first + 1e-10) - torch.log(p_second + 1e-10))).sum(dim=-1).mean()
    return kl.item()


# ── Main generation loop ──

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
    negative_model: Optional[NegativeLogitPredictor] = None,
    predict_delta: bool = False,
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
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(
        output.logits, temperature, gen=gen,
    )
    if block_size > 1:
        target_hidden = extract_context_feature(
            output.hidden_states, model.target_layer_ids,
        )

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    # Track previous predicted negative logits for temporal consistency
    neg_logits_prev = torch.zeros(
        (1, max(block_size - 1, 1), 32),
        dtype=torch.float32,
        device=model.device,
    )

    while start < max_length:
        block_output_ids = output_ids[:, start: start + block_size].clone()
        block_position_ids = position_ids[:, start: start + block_size]
        if block_size > 1:
            draft_position_ids = position_ids[
                :, start - target_hidden.shape[1]: start + block_size
            ]
            positive_draft_logits, negative_draft_logits = compute_contrastive_draft_logits(
                model=model,
                target=target,
                block_output_ids=block_output_ids,
                target_hidden=target_hidden,
                draft_position_ids=draft_position_ids,
                past_key_values_draft=past_key_values_draft,
                block_size=block_size,
                mask_token=mask_token_id,
                negative_context_dropout=negative_context_dropout,
                negative_context_noise_std=negative_context_noise_std,
                negative_hidden_mode=negative_hidden_mode,
                gen=gen,
            )
            past_key_values_draft.crop(start)

            if not TOP64_MASK_MODE:
                # beta-based candidate mask (default)
                candidate_mask = build_cd_candidate_mask(
                    reference_logits=positive_draft_logits,
                    beta=beta,
                )
            else:
                # apply top-k candidate mask 
                candidate_mask = build_topk_probability_mask(
                    reference_logits=positive_draft_logits,
                    top_k=64,
                )

            # ── Default: static CD (alpha=1.0) ──
            final_logits = positive_draft_logits

            # ── If negative_model is provided, use predicted negative logits ──
            if negative_model is not None:
                if divergence_accumulator is not None:
                    kl_val = _compute_kl_divergence(
                        positive_draft_logits, negative_draft_logits,
                    )
                    divergence_accumulator.append(kl_val)

                # Extract top-32 aligned logits
                draft_topk_token_ids, draft_topk_logits, neg_logits_on_draft_topk_ids = (
                    _extract_topk_aligned_logits(
                        positive_logits=positive_draft_logits,
                        negative_logits=negative_draft_logits,
                        top_k=32,
                    )
                )

                # Prepare position features
                block_positions = torch.arange(
                    block_size - 1,
                    device=model.device,
                    dtype=torch.float32,
                ).unsqueeze(0)
                denom = float(max(block_size - 2, 1))
                block_positions_norm = block_positions / denom

                abs_positions = block_position_ids[:, 1:].to(torch.float32)
                abs_positions_norm = abs_positions / float(max(max_length - 1, 1))

                # Predict negative logits for top-32 tokens
                predicted_neg_logits = negative_model(
                    draft_topk_logits,
                    neg_logits_on_draft_topk_ids,
                    block_positions_norm,
                    abs_positions_norm,
                    neg_logits_prev,
                )  # (1, block_size-1, 32)

                # Apply CD only inside the top-32 candidate set.
                # The rest of the vocabulary is ignored at selection time.
                top32_cd_logits = F.log_softmax(draft_topk_logits, dim=-1) - F.log_softmax(predicted_neg_logits, dim=-1)

                top32_best_indices = torch.argmax(top32_cd_logits, dim=-1, keepdim=True)
                draft_tokens = torch.gather(draft_topk_token_ids, dim=-1, index=top32_best_indices).squeeze(-1)

                # Save predicted neg logits for next block's temporal consistency
                neg_logits_prev = predicted_neg_logits.detach()
            else:
                # Apply candidate mask
                final_logits = apply_cd_candidate_filter(final_logits, candidate_mask)

                # Sample
                draft_tokens = torch.argmax(final_logits, dim=-1)

            # Verify with target model
            target_logits = target(
                input_ids=None,
                position_ids=block_position_ids[:, 1:],
                past_key_values=past_key_values_target,
                use_cache=True,
            ).logits

            # Update target KV cache
            if draft_prefill:
                # First decode step: reuse prefill KV cache
                past_key_values_target = DynamicCache.from_legacy(
                    past_key_values_target, 1
                )
                draft_prefill = False

            # Calculate acceptance
            posterior = sample(target_logits, temperature, gen=gen)
            acceptance_mask = (draft_tokens[:, :-1] == posterior[:, :-1])
            acceptance_length = acceptance_mask.cumprod(dim=1).sum(dim=1)

            # Write accepted tokens to output
            num_accepted = int(acceptance_length[0].item()) + 1
            output_ids[:, start + 1: start + 1 + num_accepted] = draft_tokens[
                :, :num_accepted
            ]

            if num_accepted < draft_tokens.shape[1]:
                # Write the bonus token (first non-accepted)
                output_ids[:, start + 1 + num_accepted] = posterior[
                    :, num_accepted
                ]

            acceptance_lengths.append(num_accepted)
            start += num_accepted + 1

            if any(
                output_ids[0, start - 1].item() in stop_token_ids
                for _ in range(1)
            ):
                break

    time_to_generate = cuda_time() - decode_start
    num_generated = min(start - num_input_tokens, max_new_tokens)

    return SimpleNamespace(
        output_ids=output_ids[:, :start],
        acceptance_lengths=acceptance_lengths,
        time_to_first_token=time_to_first_token,
        time_to_generate=time_to_generate,
        num_generated=num_generated,
        num_input_tokens=num_input_tokens,
    )


# ── CLI entry point ──

def parse_args():
    parser = argparse.ArgumentParser(
        description="CD with Learned Negative Logit Predictor"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument("--negative-hidden-mode", type=str, default="mask_zero")
    parser.add_argument("--cd-alpha", type=float, default=1.0)
    parser.add_argument("--negative-model-path", type=str, default=None,
                        help="Path to pretrained NegativeLogitPredictor checkpoint")
    parser.add_argument("--predict-delta", action="store_true",
                        help="Use delta mode for negative predictor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    set_global_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load models
    logger.info(f"Loading target model: {args.model_name_or_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    logger.info(f"Loading draft model: {args.draft_name_or_path}")
    draft = AutoModelForCausalLM.from_pretrained(
        args.draft_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device).eval()

    # Extract draft model from the wrapper
    from model import get_dflash_model
    draft_model = get_dflash_model(draft, target)

    # Load negative model if specified
    negative_model = None
    if args.negative_model_path:
        logger.info(f"Loading negative predictor: {args.negative_model_path}")
        checkpoint = torch.load(args.negative_model_path, map_location=device)
        if 'actor_state_dict' in checkpoint:
            state_dict = checkpoint['actor_state_dict']
            logger.info("Extracted actor_state_dict from training checkpoint.")
        else:
            state_dict = checkpoint
        negative_model = NegativeLogitPredictor(
            top_k=32,
            hidden_dim=128,
            predict_delta=args.predict_delta,
        ).to(device).eval()
        missing, unexpected = negative_model.load_state_dict(state_dict, strict=False)
        if missing:
            logger.warning(f"Missing keys: {missing}")
        if unexpected:
            logger.warning(f"Unexpected keys: {unexpected}")
        logger.info("Negative predictor loaded successfully.")

    # Load dataset
    logger.info(f"Loading dataset: {args.dataset}")
    from alpha_model import load_and_process_dataset_ALPHA
    dataset = load_and_process_dataset_ALPHA(
        data_name=args.dataset, split="test", max_samples=args.max_samples,
    )

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.draft_name_or_path, trust_remote_code=True,
    )
    mask_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    stop_token_ids = [tokenizer.eos_token_id]

    # Generation loop
    all_acceptance_lengths = []
    for idx, sample_data in enumerate(tqdm(dataset, desc="Generating")):
        input_text = sample_data["question"] if "question" in sample_data else sample_data["text"]
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(device)
        input_ids = input_ids[:, :512]  # truncate to max context

        result = dflash_generate(
            model=draft_model,
            target=target,
            input_ids=input_ids,
            mask_token_id=mask_token_id,
            max_new_tokens=args.max_new_tokens,
            block_size=args.block_size,
            stop_token_ids=stop_token_ids,
            temperature=args.temperature,
            cd_alpha=args.cd_alpha,
            beta=args.beta,
            negative_context_dropout=args.negative_context_dropout,
            negative_context_noise_std=args.negative_context_noise_std,
            negative_hidden_mode=args.negative_hidden_mode,
            negative_model=negative_model,
            predict_delta=args.predict_delta,
            seed=args.seed + idx,
        )

        all_acceptance_lengths.extend(result.acceptance_lengths)
        logger.info(
            f"Sample {idx}: generated {result.num_generated} tokens, "
            f"avg acceptance length: "
            f"{np.mean(result.acceptance_lengths):.2f}"
        )

    # Summary
    if all_acceptance_lengths:
        logger.info(
            f"Overall acceptance length: "
            f"mean={np.mean(all_acceptance_lengths):.3f}, "
            f"std={np.std(all_acceptance_lengths):.3f}"
        )


if __name__ == "__main__":
    main()
