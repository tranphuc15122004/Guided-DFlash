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
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
from model import *
import distributed as dist

""" 
Contrastive decoding with CD candidate filtering and enhanced negative sampling for the draft model.
Key features:
Similar to the previous version but negative sample generation.
The Fused hidden feature insteads of random droping out a certain portion, attention rate awared dropout is implemented
A part of highest attention score token of the positive forward on target hidden is mask to make negative sample.
There are 2 draft forward in each decode step, the first one for original positive sample and attetion score collecting and the second for negative sample making

My intention is masking the tokens in hidden feature having the high attention in positive draft, in order to make a strong negative sample 
"""

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

def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    attention_scores: Optional[torch.Tensor] = None,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    gen : torch.Generator = None,
) -> torch.Tensor:
    negative_target_hidden = target_hidden.clone()

    if dropout_ratio > 0.0:
        batch_size, context_len = negative_target_hidden.shape[:2]
        if (
            attention_scores is not None
            and attention_scores.shape[0] == batch_size
            and attention_scores.shape[1] == context_len
        ):
            num_tokens_to_mask = min(
                context_len,
                max(1, int(round(context_len * dropout_ratio))),
            )
            top_indices = torch.topk(attention_scores, k=num_tokens_to_mask, dim=-1).indices
            token_keep_mask = torch.ones(
                (batch_size, context_len),
                dtype=torch.bool,
                device=negative_target_hidden.device,
            )
            token_keep_mask.scatter_(1, top_indices, False)
        else:
            token_keep_mask = (
                torch.rand(
                    (batch_size, context_len),
                    device=negative_target_hidden.device,
                    generator=gen
                ) >= dropout_ratio
            )
        token_keep_mask = token_keep_mask.unsqueeze(-1)
        negative_target_hidden = negative_target_hidden.masked_fill(~token_keep_mask, 0.0)

    if noise_std > 0.0:
        noise = torch.randn(
            negative_target_hidden.shape,
            dtype=negative_target_hidden.dtype,
            device=negative_target_hidden.device,
            generator=gen
        )
        negative_target_hidden = negative_target_hidden + noise_std * noise

    return negative_target_hidden


def extract_context_attention_scores(
    attn_weights: Optional[torch.Tensor],
    context_len: int,
) -> Optional[torch.Tensor]:
    if attn_weights is None or context_len <= 0:
        return None

    if attn_weights.dim() == 4:
        # [batch, heads, query_len, key_len] -> [batch, context_len]
        context_attention = attn_weights[..., :context_len]
        return context_attention.mean(dim=(1, 2))

    if attn_weights.dim() == 3:
        # [batch, query_len, key_len] -> [batch, context_len]
        context_attention = attn_weights[..., :context_len]
        return context_attention.mean(dim=1)

    return None


def run_draft_forward(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    target_hidden: torch.Tensor,
    noise_embedding: torch.Tensor,
    draft_position_ids: torch.Tensor,
    past_key_values_draft: DynamicCache,
    block_size: int,
    capture_attention: bool = False,
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    captured_attention = {"weights": None}
    hook_handle = None
    original_attn_impl = model.config._attn_implementation

    if capture_attention:
        if original_attn_impl != "eager":
            model.config._attn_implementation = "eager"

        def _capture_attn(_module, _inputs, output):
            if isinstance(output, tuple) and len(output) > 1:
                captured_attention["weights"] = output[1]

        hook_handle = model.layers[-1].self_attn.register_forward_hook(_capture_attn)

    try:
        draft_hidden = model(
            target_hidden=target_hidden,
            noise_embedding=noise_embedding,
            position_ids=draft_position_ids,
            past_key_values=past_key_values_draft,
            use_cache=True,
            is_causal=False,
        )[:, -block_size+1:, :]
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        if capture_attention:
            model.config._attn_implementation = original_attn_impl

    return target.lm_head(draft_hidden), captured_attention["weights"]

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
    gen : torch.Generator = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    cache_seq_len = past_key_values_draft.get_seq_length()

    positive_noise_embedding = target.model.embed_tokens(block_output_ids)
    positive_draft_logits, positive_attn_weights = run_draft_forward(
        model=model,
        target=target,
        target_hidden=target_hidden,
        noise_embedding=positive_noise_embedding,
        draft_position_ids=draft_position_ids,
        past_key_values_draft=past_key_values_draft,
        block_size=block_size,
        capture_attention=True,
    )
    past_key_values_draft.crop(cache_seq_len)

    attention_scores = extract_context_attention_scores(
        attn_weights=positive_attn_weights,
        context_len=target_hidden.shape[1],
    )

    if attention_scores is None and not getattr(compute_contrastive_draft_logits, "_warned_no_attention", False):
        logger.warning(
            "Draft attention weights are unavailable with the current attention backend. "
            "Falling back to random hidden dropout for negative sampling."
        )
        compute_contrastive_draft_logits._warned_no_attention = True

    neg_block_output_ids = build_negative_block_output_random(block_output_ids, target.config.vocab_size , gen)
    negative_target_hidden = build_negative_target_hidden(
        target_hidden=target_hidden,
        attention_scores=attention_scores,
        dropout_ratio=negative_context_dropout,
        noise_std=negative_context_noise_std,
        gen=gen,
    )

    negative_noise_embedding = target.model.embed_tokens(neg_block_output_ids)
    negative_draft_logits, _ = run_draft_forward(
        model=model,
        target=target,
        target_hidden=negative_target_hidden,
        noise_embedding=negative_noise_embedding,
        draft_position_ids=draft_position_ids,
        past_key_values_draft=past_key_values_draft,
        block_size=block_size,
        capture_attention=False,
    )
    past_key_values_draft.crop(cache_seq_len)

    return positive_draft_logits, negative_draft_logits


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
    output_ids[:, num_input_tokens:num_input_tokens+1] = sample(output.logits, temperature , gen=gen)
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
                gen=gen
            )
            
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
            n_keep = min(6, final_draft_logits.size(-1))
            top_candidate_indices = torch.topk(final_draft_logits, k=n_keep, dim=-1).indices
            final_draft_logits.scatter_(
                -1,
                top_candidate_indices,
                positive_draft_logits.gather(-1, top_candidate_indices),
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
    parser.add_argument("--cd-alpha", type=float, default=0.1)
    parser.add_argument("--cd-beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
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
                    divergence_accumulator = divergence_accumulator,
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

if __name__ == "__main__":
    main()