from __future__ import annotations

from typing import Any, Callable, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import DFlashDraftModel


NegativeBlockBuilder = Callable[
    [torch.Tensor, int, Optional[torch.Generator]],
    torch.Tensor,
]


def build_negative_block_output_random(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    random_tokens = torch.randint(
        0,
        vocab_size,
        (block_output_ids.shape[0],),
        device=block_output_ids.device,
        generator=gen,
    )
    negative_block_output_ids[:, 0] = random_tokens
    return negative_block_output_ids


def build_negative_block_output_neighbor(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    token_neighbor_table: Optional[torch.Tensor],
    neighbor_topk: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    negative_block_output_ids = block_output_ids.clone()
    batch_size = block_output_ids.shape[0]

    if token_neighbor_table is None:
        random_tokens = torch.randint(0, vocab_size, (batch_size,), device=block_output_ids.device, generator=gen)
        negative_block_output_ids[:, 0] = random_tokens
        return negative_block_output_ids

    source_tokens = block_output_ids[:, 0]
    neighbor_candidates = token_neighbor_table.index_select(0, source_tokens)
    topk = min(max(1, neighbor_topk), neighbor_candidates.shape[1])
    neighbor_candidates = neighbor_candidates[:, :topk]

    if topk == 1:
        replacement_tokens = neighbor_candidates[:, 0]
    else:
        sampled_col = torch.randint(0, topk, (batch_size,), device=block_output_ids.device, generator=gen)
        replacement_tokens = neighbor_candidates.gather(1, sampled_col.unsqueeze(1)).squeeze(1)

    same_token_mask = replacement_tokens == source_tokens
    if same_token_mask.any():
        replacement_tokens[same_token_mask] = (replacement_tokens[same_token_mask] + 1) % vocab_size

    negative_block_output_ids[:, 0] = replacement_tokens
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
            )
            >= dropout_ratio
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
    negative_block_builder: Optional[NegativeBlockBuilder] = None,
    gen: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = block_output_ids.shape[0]
    block_builder = negative_block_builder or build_negative_block_output_random
    neg_block_output_ids = block_builder(block_output_ids, target.config.vocab_size, gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        gen=gen,
    )

    positive_noise_embedding = target.model.embed_tokens(block_output_ids)
    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)

    paired_noise_embedding = torch.cat([positive_noise_embedding, negative_noise_embedding], dim=0)
    paired_hidden = torch.cat([target_hidden, negative_target_hidden], dim=0)
    paired_position_ids = torch.cat([draft_position_ids, draft_position_ids], dim=0)

    paired_draft_logits = target.lm_head(
        model(
            target_hidden=paired_hidden,
            noise_embedding=paired_noise_embedding,
            position_ids=paired_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -block_size + 1 :, :]
    )

    first_draft_logits, second_draft_logits = paired_draft_logits.split(batch_size, dim=0)
    return first_draft_logits, second_draft_logits


def build_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    reference_probs = torch.softmax(reference_logits, dim=-1)
    max_reference_probs = reference_probs.amax(dim=-1, keepdim=True)
    candidate_mask = reference_probs >= (beta * max_reference_probs)
    top_token_indices = reference_probs.argmax(dim=-1, keepdim=True)
    candidate_mask.scatter_(-1, top_token_indices, True)
    return candidate_mask


def apply_candidate_filter(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return logits.masked_fill(~candidate_mask, torch.finfo(logits.dtype).min)


def compute_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> float:
    p_first = torch.softmax(first_logits, dim=-1)
    p_second = torch.softmax(second_logits, dim=-1)
    kl = (p_first * (torch.log(p_first + 1e-10) - torch.log(p_second + 1e-10))).sum(dim=-1).mean()
    return float(kl.item())


def token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    token_logit = logits_1d[token_id]
    return int((logits_1d > token_logit).sum().item()) + 1


def token_str(tokenizer: AutoTokenizer, token_id: int) -> str:
    text = tokenizer.decode([token_id], skip_special_tokens=False)
    return text.replace("\n", "\\n")


def topk_token_details(
    logits_1d: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> list[dict[str, Any]]:
    log_probs = torch.log_softmax(logits_1d.float(), dim=-1)
    k = min(k, int(log_probs.shape[-1]))
    vals, ids = torch.topk(log_probs, k=k, dim=-1)
    out: list[dict[str, Any]] = []
    for token_id, logp in zip(ids.tolist(), vals.tolist()):
        out.append(
            {
                "id": int(token_id),
                "text": token_str(tokenizer, int(token_id)),
                "logprob": float(logp),
                "prob": float(np.exp(logp)),
            }
        )
    return out


def estimate_token_hit_probability(
    logits_1d: torch.Tensor,
    token_id: int,
    num_samples: int,
    gen: Optional[torch.Generator] = None,
) -> float:
    if num_samples <= 0:
        return 0.0
    probs = torch.softmax(logits_1d, dim=-1)
    sampled_ids = torch.multinomial(probs, num_samples=num_samples, replacement=True, generator=gen)
    return float((sampled_ids == token_id).float().mean().item())
