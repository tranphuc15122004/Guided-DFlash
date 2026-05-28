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
DONE
"""

import argparse
import json
import os
import time
import random
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional
from loguru import logger
import numpy as np
import torch
import torch.nn.functional as F
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import (
    DFlashDraftModel,
    apply_cd_logits,
    extract_context_feature,
    load_and_process_dataset,
    sample,
)
from alpha_model import NegativeLogitPredictor, NegativeLogitPredictor_Dense
import distributed as dist

NEGATIVE_HIDDEN_MODE = 'mask_zero'
TOP64_MASK_MODE = False


class NegativeModelStatsCollector:
    """Collect top-k tensors for offline NegativeLogitPredictor analysis."""

    def __init__(
        self,
        output_dir: str,
        rank: int = 0,
        shard_size: int = 256,
        max_records: Optional[int] = None,
        store_float32: bool = False,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.rank = rank
        self.shard_size = max(1, shard_size)
        self.max_records = max_records
        self.float_dtype = torch.float32 if store_float32 else torch.float16
        self.buffer: list[dict[str, Any]] = []
        self.num_records = 0
        self.shard_id = 0

    def _to_cpu(self, value: Any) -> Any:
        if torch.is_tensor(value):
            value = value.detach().cpu()
            if torch.is_floating_point(value):
                value = value.to(self.float_dtype)
            return value
        return value

    def add(self, record: dict[str, Any]) -> None:
        if self.max_records is not None and self.num_records >= self.max_records:
            return
        self.buffer.append({key: self._to_cpu(value) for key, value in record.items()})
        self.num_records += 1
        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        out_file = self.output_dir / f"negative_stats_rank{self.rank:03d}_shard{self.shard_id:06d}.pt"
        torch.save(self.buffer, out_file)
        logger.info(f"Wrote {len(self.buffer)} negative model stats records to {out_file}")
        self.buffer = []
        self.shard_id += 1


def _extract_state_dict(checkpoint: dict) -> dict:
    if isinstance(checkpoint, dict):
        for state_key in ("state_dict", "model_state_dict", "actor_state_dict"):
            if state_key in checkpoint and isinstance(checkpoint[state_key], dict):
                logger.info(f"Extracted {state_key} from training checkpoint.")
                return _strip_inference_unused_keys(checkpoint[state_key])
    return checkpoint


def _strip_inference_unused_keys(state_dict: dict) -> dict:
    """Remove policy-only parameters when loading an actor for deterministic inference."""
    if not isinstance(state_dict, dict):
        return state_dict
    return {
        key: value
        for key, value in state_dict.items()
        if key not in {"log_std"}
    }


def _as_bool(value) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return None


def _as_int(value) -> Optional[int]:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_checkpoint_config(checkpoint: dict, checkpoint_path: str) -> dict:
    """Read training args saved by TrainingLogger, either in checkpoint or sibling config.json."""
    config = {}
    if isinstance(checkpoint, dict):
        for key in ("config", "args", "model_config"):
            candidate = checkpoint.get(key)
            if isinstance(candidate, dict):
                config.update(candidate)

    config_path = Path(checkpoint_path).resolve().parent.parent / "config.json"
    if config_path.exists():
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                file_config = json.load(handle)
            config.update(file_config)
            logger.info(f"Loaded negative predictor training config: {config_path}")
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning(f"Could not read negative predictor config {config_path}: {exc}")
    return config


def _detect_negative_model_type(state_dict: dict) -> str:
    if "encoder.net.0.weight" in state_dict:
        return "regular"
    if "token_projector.0.weight" in state_dict:
        return "dense"
    raise ValueError(
        "Unknown negative predictor checkpoint type. Expected keys "
        "'encoder.net.0.weight' (regular) or 'token_projector.0.weight' (dense). "
        f"Got keys: {list(state_dict.keys())[:10]}..."
    )


def infer_negative_model_config(state_dict: dict, default_ks: Optional[list[int]] = None) -> dict:
    if default_ks is None:
        default_ks = [5, 10, 20]
    len_ks = len(default_ks)
    model_type = _detect_negative_model_type(state_dict)

    if model_type == "regular":
        input_dim = state_dict["encoder.net.0.weight"].shape[1]
        hidden_dim = state_dict["encoder.net.0.weight"].shape[0]
        numerator = input_dim - 7 - 2 * len_ks
        if numerator <= 0 or numerator % 7 != 0:
            raise ValueError(f"Cannot infer regular negative predictor top_k from input_dim={input_dim}.")
        return {
            "model_type": "regular",
            "top_k": numerator // 7,
            "hidden_dim": hidden_dim,
        }

    input_dim = state_dict["token_projector.0.weight"].shape[1]
    hidden_dim = state_dict["token_projector.0.weight"].shape[0]
    numerator = input_dim - 7 - 2 * len_ks
    output_dim = state_dict["block_mlp.6.weight"].shape[0]
    if numerator <= 0 or numerator % 7 != 0:
        raise ValueError(f"Cannot infer Dense negative predictor top_k from input_dim={input_dim}.")
    top_k = numerator // 7
    if output_dim % top_k != 0:
        raise ValueError(
            f"Cannot infer Dense negative predictor max_seq_len from "
            f"output_dim={output_dim}, top_k={top_k}."
        )
    return {
        "model_type": "dense",
        "top_k": top_k,
        "hidden_dim": hidden_dim,
        "max_seq_len": output_dim // top_k,
    }


def build_negative_model_from_checkpoint(
    checkpoint: dict,
    checkpoint_path: str,
    device: torch.device,
    cli_predict_delta: bool,
) -> torch.nn.Module:
    state_dict = _extract_state_dict(checkpoint)
    inferred = infer_negative_model_config(state_dict)
    config = _load_checkpoint_config(checkpoint, checkpoint_path)

    dense_from_config = _as_bool(config.get("dense"))
    top_k_from_config = _as_int(config.get("top_k"))
    hidden_dim_from_config = _as_int(config.get("hidden_dim"))
    predict_delta_from_config = _as_bool(config.get("predict_delta"))

    if dense_from_config is not None:
        inferred["model_type"] = "dense" if dense_from_config else "regular"
    if top_k_from_config is not None:
        inferred["top_k"] = top_k_from_config
    if hidden_dim_from_config is not None:
        inferred["hidden_dim"] = hidden_dim_from_config

    predict_delta = cli_predict_delta or bool(predict_delta_from_config)
    inferred["predict_delta"] = predict_delta

    if inferred["model_type"] == "dense":
        negative_model = NegativeLogitPredictor_Dense(
            top_k=inferred["top_k"],
            hidden_dim=inferred["hidden_dim"],
            predict_delta=predict_delta,
            max_seq_len=inferred.get("max_seq_len", 15),
        )
    else:
        negative_model = NegativeLogitPredictor(
            top_k=inferred["top_k"],
            hidden_dim=inferred["hidden_dim"],
            predict_delta=predict_delta,
        )

    negative_model.to(device).eval()
    load_result = negative_model.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        logger.warning(f"Missing keys: {load_result.missing_keys}")
    if load_result.unexpected_keys:
        logger.warning(f"Unexpected keys: {load_result.unexpected_keys}")
    logger.info(f"Negative predictor loaded successfully with config: {inferred}.")
    return negative_model


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
    negative_stats_collector: Optional[NegativeModelStatsCollector] = None,
    sample_idx: Optional[int] = None,
    turn_index: Optional[int] = None,
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
    block_index = 0

    while start < max_length:
        block_output_ids = output_ids[:, start: start + block_size].clone()
        block_position_ids = position_ids[:, start: start + block_size]
        pending_stats_record = None
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

                negative_top_k = getattr(negative_model, "top_k", 32)

                # Extract top-k aligned logits expected by the negative predictor.
                draft_topk_token_ids, draft_topk_logits, neg_logits_on_draft_topk_ids = (
                    _extract_topk_aligned_logits(
                        positive_logits=positive_draft_logits,
                        negative_logits=negative_draft_logits,
                        top_k=negative_top_k,
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
                )  # (1, block_size-1, 32)

                # Apply CD inside the top-k set, then scatter back to a sparse
                # vocab-shaped tensor so sampling/verification follows benchmark.py.
                topk_cd_logits = F.log_softmax(draft_topk_logits, dim=-1) - F.log_softmax(
                    predicted_neg_logits, dim=-1
                )
                static_cd_topk_logits = F.log_softmax(draft_topk_logits, dim=-1) - F.log_softmax(
                    neg_logits_on_draft_topk_ids, dim=-1
                )
                candidate_topk_mask = torch.gather(
                    candidate_mask, dim=-1, index=draft_topk_token_ids,
                )
                final_logits = torch.full_like(
                    positive_draft_logits,
                    torch.finfo(positive_draft_logits.dtype).min,
                )
                final_logits.scatter_(-1, draft_topk_token_ids, topk_cd_logits.to(final_logits.dtype))
                final_logits = apply_cd_candidate_filter(final_logits, candidate_mask)
                draft_tokens = sample(final_logits, temperature, gen=gen)

                if negative_stats_collector is not None:
                    pending_stats_record = {
                        "sample_idx": sample_idx,
                        "turn_index": turn_index,
                        "block_index": int(block_index),
                        "start_position": int(start),
                        "seed": int(seed),
                        "topk_token_ids": draft_topk_token_ids,
                        "positive_topk_logits": draft_topk_logits,
                        "negative_old_topk_logits": neg_logits_on_draft_topk_ids,
                        "negative_pred_topk_logits": predicted_neg_logits,
                        "static_cd_topk_logits": static_cd_topk_logits,
                        "cd_topk_logits": topk_cd_logits,
                        "candidate_topk_mask": candidate_topk_mask,
                        "block_positions": block_positions_norm,
                        "absolute_positions": abs_positions_norm,
                    }
            else:
                final_logits = apply_cd_logits(
                    first_logits=positive_draft_logits,
                    second_logits=negative_draft_logits,
                    alpha=cd_alpha,
                )

                # Apply candidate mask
                final_logits = apply_cd_candidate_filter(final_logits, candidate_mask)

                # Sample
                draft_tokens = sample(final_logits, temperature, gen=gen)

            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

            block_output_ids[:, 1:] = draft_tokens

        output = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True if block_size > 1 else False,
        )

        posterior = sample(output.logits, temperature, gen=gen)
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()

        if pending_stats_record is not None and negative_stats_collector is not None:
            pending_stats_record.update(
                {
                    "acceptance_length": int(acceptance_length),
                    "target_token_ids": posterior[:, :-1],
                    "draft_token_ids": block_output_ids[:, 1:],
                }
            )
            negative_stats_collector.add(pending_stats_record)

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        block_index += 1
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
        stop_token_ids_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids_tensor
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(num_output_tokens, 1)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


# ── CLI entry point ──

def parse_args():
    parser = argparse.ArgumentParser(
        description="CD with Learned Negative Logit Predictor"
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
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
    parser.add_argument(
        "--negative-stats-output-dir",
        type=str,
        default=None,
        help="Optional directory for NegativeLogitPredictor top-k stats shards.",
    )
    parser.add_argument(
        "--negative-stats-shard-size",
        type=int,
        default=256,
        help="Number of generation blocks per negative stats shard.",
    )
    parser.add_argument(
        "--negative-stats-max-records",
        type=int,
        default=None,
        help="Optional cap on collected negative stats records per rank.",
    )
    parser.add_argument(
        "--negative-stats-float32",
        action="store_true",
        help="Store floating tensors as float32 instead of float16 in stats shards.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable fully deterministic behavior for reproducible runs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Deprecated compatibility option; device is selected from distributed local rank.",
    )
    return parser.parse_args()


@torch.inference_mode()
def main():
    args = parse_args()
    set_global_seed(args.seed, deterministic=args.deterministic)

    dist.init()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")
        supports_bf16 = torch.cuda.is_bf16_supported() if hasattr(torch.cuda, "is_bf16_supported") else False
        model_dtype = torch.bfloat16 if supports_bf16 else torch.float16
        if not supports_bf16:
            logger.info("CUDA device does not support bfloat16. Falling back to float16.")
    else:
        logger.warning("CUDA is unavailable. Falling back to CPU inference; this benchmark will run much slower.")
        device = torch.device("cpu")
        model_dtype = torch.float32
    logger.info(f"Device: {device}")

    def has_flash_attn() -> bool:
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
    attn_implementation = "flash_attention_2" if installed_flash_attn else "sdpa"

    # Load models
    logger.info(f"Loading target model: {args.model_name_or_path}")
    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation=attn_implementation,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(device).eval()

    logger.info(f"Loading draft model: {args.draft_name_or_path}")
    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation=attn_implementation,
        dtype=model_dtype,
        trust_remote_code=True,
    ).to(device).eval()

    # Load negative model if specified
    negative_model = None
    if args.negative_model_path:
        logger.info(f"Loading negative predictor: {args.negative_model_path}")
        checkpoint = torch.load(args.negative_model_path, map_location=device, weights_only=True)
        negative_model = build_negative_model_from_checkpoint(
            checkpoint=checkpoint,
            checkpoint_path=args.negative_model_path,
            device=device,
            cli_predict_delta=args.predict_delta,
        )

    negative_stats_collector = None
    if args.negative_stats_output_dir:
        negative_stats_collector = NegativeModelStatsCollector(
            output_dir=args.negative_stats_output_dir,
            rank=dist.rank(),
            shard_size=args.negative_stats_shard_size,
            max_records=args.negative_stats_max_records,
            store_float32=args.negative_stats_float32,
        )
        manifest_path = Path(args.negative_stats_output_dir) / f"manifest_rank{dist.rank():03d}.json"
        manifest = {
            "model_name_or_path": args.model_name_or_path,
            "draft_name_or_path": args.draft_name_or_path,
            "dataset": args.dataset,
            "negative_model_path": args.negative_model_path,
            "predict_delta": bool(args.predict_delta),
            "block_size_arg": args.block_size,
            "max_samples": args.max_samples,
            "max_new_tokens": int(args.max_new_tokens),
            "temperature": float(args.temperature),
            "beta": float(args.beta),
            "negative_context_dropout": float(args.negative_context_dropout),
            "negative_context_noise_std": float(args.negative_context_noise_std),
            "negative_hidden_mode": args.negative_hidden_mode,
            "seed": int(args.seed),
            "rank": int(dist.rank()),
            "world_size": int(dist.size()),
        }
        with open(manifest_path, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2)
        if negative_model is None:
            logger.warning(
                "--negative-stats-output-dir was set, but no --negative-model-path was provided. "
                "No negative predictor stats records will be collected."
            )

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    logger.info(f"Loading dataset: {args.dataset}")
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    responses = []
    indices = range(dist.rank(), len(dataset), dist.size())
    base_seed = args.seed
    for idx in tqdm(indices, disable=not dist.is_main()):
        sample_seed = base_seed + idx + dist.rank() * 10_000_000
        instance = dataset[idx]
        messages = []
        for turn_index, user_content in enumerate(instance["turns"]):
            messages.append({"role": "user", "content": user_content})
            input_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
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
                    beta=args.beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    negative_hidden_mode=args.negative_hidden_mode,
                    negative_model=negative_model if bs > 1 else None,
                    predict_delta=args.predict_delta,
                    negative_stats_collector=negative_stats_collector if bs > 1 else None,
                    sample_idx=idx,
                    turn_index=turn_index,
                    seed=sample_seed + bs,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if negative_stats_collector is not None:
        negative_stats_collector.flush()

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


if __name__ == "__main__":
    main()
