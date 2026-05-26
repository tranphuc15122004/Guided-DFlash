import argparse
import os
import random
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pandas as pd
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import distributed as dist
from model import DFlashDraftModel, extract_context_feature, sample


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
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.benchmark = True


class DraftLayerCapture:
    """Capture per-layer hidden states from DFlash drafter decoder layers."""

    def __init__(self, model: DFlashDraftModel):
        self.model = model
        self._hooks: list[Any] = []
        self.buffer: list[torch.Tensor] = []

    def clear(self) -> None:
        self.buffer.clear()

    def _make_hook(self):
        def hook(_module, _inputs, output):
            self.buffer.append(output.detach())
        return hook

    def __enter__(self):
        self.clear()
        for layer in self.model.layers:
            self._hooks.append(layer.register_forward_hook(self._make_hook()))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def _fix_target_tokens_for_blocks(block_records: list[dict]) -> list[torch.Tensor]:
    """
    Fix target tokens per block with the same chaining semantics used in fix_target_token.py.
    """
    if not block_records:
        return []

    seq_len = int(block_records[0]["raw_target_token_id"].numel())
    fixed: list[torch.Tensor] = [r["raw_target_token_id"].clone() for r in block_records]

    for i in range(len(block_records)):
        acc = int(block_records[i]["acceptance_length_plus1"])
        correct_upto = acc
        if correct_upto >= seq_len:
            continue

        for p in range(correct_upto, seq_len):
            offset = p - correct_upto
            found = False
            for j in range(i + 1, len(block_records)):
                next_acc = int(block_records[j]["acceptance_length_plus1"])
                if offset < next_acc:
                    fixed[i][p] = fixed[j][offset]
                    found = True
                    break
                offset -= next_acc
            if not found:
                break
    return fixed


def _project_target_metrics(
    lm_head: torch.nn.Module,
    hidden_states: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    hidden_states: (S, H), target_token_ids: (S,)
    Returns:
      target_logits: (S,)
      target_logprobs: (S,)
      target_ranks: (S,)
    """
    logits = lm_head(hidden_states)  # (S, V)
    target_logits = logits.gather(-1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    log_probs = torch.log_softmax(logits, dim=-1)
    target_logprobs = log_probs.gather(-1, target_token_ids.unsqueeze(-1)).squeeze(-1)
    target_ranks = (logits > target_logits.unsqueeze(-1)).sum(dim=-1)
    return target_logits, target_logprobs, target_ranks


def _build_summary(df: pd.DataFrame) -> pd.DataFrame:
    def agg_for(group_key: str) -> pd.DataFrame:
        grouped = df.groupby(group_key, as_index=False).agg(
            count=("target_logit", "count"),
            target_logit_mean=("target_logit", "mean"),
            target_logit_std=("target_logit", "std"),
            target_logprob_mean=("target_logprob", "mean"),
            target_logprob_std=("target_logprob", "std"),
            target_rank_mean=("target_rank", "mean"),
            target_rank_std=("target_rank", "std"),
            top1_rate=("target_rank", lambda x: float((x == 0).mean())),
        )
        grouped["scope"] = group_key
        grouped = grouped.rename(columns={group_key: "scope_value"})
        return grouped

    by_layer = agg_for("layer_index")
    by_position = agg_for("position_in_block")
    return pd.concat([by_layer, by_position], ignore_index=True)


def _run_self_checks() -> None:
    # Check target-fix chaining semantics on a synthetic case.
    records = [
        {
            "acceptance_length_plus1": 2,
            "raw_target_token_id": torch.tensor([10, 11, 99, 99], dtype=torch.long),
        },
        {
            "acceptance_length_plus1": 3,
            "raw_target_token_id": torch.tensor([20, 21, 22, 88], dtype=torch.long),
        },
        {
            "acceptance_length_plus1": 2,
            "raw_target_token_id": torch.tensor([30, 31, 77, 77], dtype=torch.long),
        },
    ]
    fixed = _fix_target_tokens_for_blocks(records)
    assert fixed[0].tolist() == [10, 11, 20, 21]
    assert fixed[1].tolist() == [20, 21, 22, 30]

    # Check projected metrics shapes and rank bounds.
    vocab = 16
    hidden = 8
    seq_len = 5
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    hs = torch.randn(seq_len, hidden)
    tgt = torch.randint(0, vocab, (seq_len,), dtype=torch.long)
    t_logit, t_logprob, t_rank = _project_target_metrics(lm_head, hs, tgt)
    assert t_logit.shape == (seq_len,)
    assert t_logprob.shape == (seq_len,)
    assert t_rank.shape == (seq_len,)
    assert int(t_rank.min().item()) >= 0
    assert int(t_rank.max().item()) <= vocab

    logger.info("Self-checks passed.")


@torch.inference_mode()
def dflash_generate_and_collect(
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
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature, gen=gen)
    target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids) if block_size > 1 else None
    time_to_first_token = cuda_time() - prefill_start

    decode_start = cuda_time()
    start = num_input_tokens
    acceptance_lengths: list[int] = []
    block_records: list[dict] = []
    draft_prefill = True

    with DraftLayerCapture(model) as capturer:
        while start < max_length:
            block_output_ids = output_ids[:, start:start + block_size].clone()
            block_position_ids = position_ids[:, start:start + block_size]

            if block_size > 1:
                noise_embedding = target.model.embed_tokens(block_output_ids)
                capturer.clear()
                draft_hidden = model(
                    target_hidden=target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                    past_key_values=past_key_values_draft,
                    use_cache=True,
                    is_causal=False,
                )
                layer_outputs = capturer.buffer
                assert len(layer_outputs) == len(model.layers), "Captured layer count mismatch."

                draft_logits = target.lm_head(draft_hidden[:, -block_size + 1:, :])
                block_output_ids[:, 1:] = sample(draft_logits, gen=gen)
                past_key_values_draft.crop(start)
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
            acceptance_plus1 = int(acceptance_length + 1)

            output_ids[:, start:start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
            output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
            acceptance_lengths.append(acceptance_plus1)

            if block_size > 1:
                per_layer = []
                for layer_hidden in layer_outputs:
                    sliced = layer_hidden[0, -block_size + 1:, :].detach().to(torch.float32).cpu()
                    per_layer.append(sliced)
                raw_target = posterior[0, :-1].detach().to(torch.long).cpu()
                assert raw_target.numel() == block_size - 1, "Target token length mismatch."
                for layer_idx, layer_h in enumerate(per_layer):
                    assert layer_h.shape[0] == block_size - 1, f"Position count mismatch at layer {layer_idx}."

                block_records.append(
                    {
                        "block_index": len(block_records),
                        "acceptance_length_plus1": acceptance_plus1,
                        "raw_target_token_id": raw_target,
                        "layer_hidden_states": per_layer,
                    }
                )

            start += acceptance_plus1
            past_key_values_target.crop(start)
            if block_size > 1:
                target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_plus1, :]

            if stop_token_ids is not None and any(
                stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
            ):
                break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids_t = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids_t).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / max(1, num_output_tokens)

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
        block_records=block_records,
    )


def _build_rows(
    target: AutoModelForCausalLM,
    block_records: list[dict],
    fixed_targets: list[torch.Tensor],
) -> list[dict]:
    rows: list[dict] = []
    num_layers = len(block_records[0]["layer_hidden_states"]) if block_records else 0

    for rec, tgt_fixed in zip(block_records, fixed_targets):
        block_index = int(rec["block_index"])
        acc = int(rec["acceptance_length_plus1"])
        raw_target = rec["raw_target_token_id"]
        is_tail = torch.arange(raw_target.numel(), dtype=torch.long) >= acc

        for layer_index in range(num_layers):
            h = rec["layer_hidden_states"][layer_index].to(next(target.parameters()).device)
            tok = tgt_fixed.to(h.device)
            tgt_logits, tgt_logprobs, tgt_ranks = _project_target_metrics(target.lm_head, h, tok)

            norms = torch.norm(h, p=2, dim=-1)
            means = h.mean(dim=-1)
            stds = h.std(dim=-1, unbiased=False)

            for pos in range(h.shape[0]):
                rows.append(
                    {
                        "sample_id": 0,
                        "block_index": block_index,
                        "position_in_block": pos,
                        "layer_index": layer_index,
                        "acceptance_length_plus1": acc,
                        "is_corrected_tail": bool(is_tail[pos].item()),
                        "target_token_id_raw": int(raw_target[pos].item()),
                        "target_token_id_fixed": int(tgt_fixed[pos].item()),
                        "hidden_l2_norm": float(norms[pos].item()),
                        "hidden_mean": float(means[pos].item()),
                        "hidden_std": float(stds[pos].item()),
                        "target_logit": float(tgt_logits[pos].item()),
                        "target_logprob": float(tgt_logprobs[pos].item()),
                        "target_rank": int(tgt_ranks[pos].item()),
                    }
                )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze per-layer DFlash drafter target-token logits on a single prompt."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="analysis/sublayer_outputs")
    parser.add_argument("--output-prefix", type=str, default="sublayer_target_logit")
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deterministic behavior.",
    )
    parser.add_argument(
        "--run-self-checks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run lightweight correctness checks before loading models.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_self_checks:
        _run_self_checks()

    set_global_seed(args.seed, deterministic=args.deterministic)
    dist.init()
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(dist.local_rank())
        device = torch.device(f"cuda:{dist.local_rank()}")
        model_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        device = torch.device("cpu")
        model_dtype = torch.float32

    if dist.size() > 1 and not dist.is_main():
        logger.info("Non-main rank exits for single-prompt analysis.")
        return

    def has_flash_attn() -> bool:
        if not use_cuda:
            return False
        if args.deterministic:
            logger.info("Deterministic mode enabled. Using SDPA.")
            return False
        try:
            import flash_attn  # noqa: F401
            return True
        except ImportError:
            logger.warning("flash_attn not installed. Using SDPA.")
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
    messages = [{"role": "user", "content": args.prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    result = dflash_generate_and_collect(
        model=draft_model,
        target=target,
        input_ids=input_ids,
        mask_token_id=draft_model.mask_token_id,
        max_new_tokens=args.max_new_tokens,
        block_size=block_size,
        stop_token_ids=[tokenizer.eos_token_id],
        temperature=args.temperature,
        seed=args.seed,
    )

    generated_ids = result.output_ids[0, result.num_input_tokens:]
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    logger.info(f"Generated output: {output_text}")
    logger.info(f"Acceptance lengths: {result.acceptance_lengths}")

    if not result.block_records:
        logger.warning("No block records captured; nothing to write.")
        return

    fixed_targets = _fix_target_tokens_for_blocks(result.block_records)
    assert len(fixed_targets) == len(result.block_records), "Fixed target block count mismatch."
    for rec, fixed in zip(result.block_records, fixed_targets):
        assert fixed.shape == rec["raw_target_token_id"].shape, "Fixed target shape mismatch."

    rows = _build_rows(target, result.block_records, fixed_targets)
    df = pd.DataFrame(rows)

    vocab_size = int(target.config.vocab_size)
    if not df.empty:
        assert int(df["target_rank"].min()) >= 0
        assert int(df["target_rank"].max()) <= vocab_size

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{args.output_prefix}.parquet"
    summary_path = output_dir / f"{args.output_prefix}_summary.csv"

    try:
        df.to_parquet(parquet_path, index=False)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to write parquet to {parquet_path}. Ensure `pyarrow` is installed."
        ) from exc

    summary_df = _build_summary(df)
    summary_df.to_csv(summary_path, index=False)

    logger.info(f"Wrote row-level parquet: {parquet_path}")
    logger.info(f"Wrote summary csv: {summary_path}")
    logger.info(f"Rows: {len(df)}, blocks: {len(result.block_records)}, layers: {len(draft_model.layers)}")


if __name__ == "__main__":
    main()
