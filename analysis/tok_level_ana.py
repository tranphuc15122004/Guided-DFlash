import argparse
import csv
import importlib.util
import json
import os
import random
import time
from datetime import datetime
from itertools import chain
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional

import distributed as dist
import numpy as np
import torch
from loguru import logger
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from analysis import token_level_core
from analysis import token_level_reporting
from model import DFlashDraftModel, apply_vcd_logits, extract_context_feature, load_and_process_dataset, sample


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


def build_negative_block_output_random(
    block_output_ids: torch.Tensor,
    vocab_size: int,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return token_level_core.build_negative_block_output_random(
        block_output_ids=block_output_ids,
        vocab_size=vocab_size,
        gen=gen,
    )


def build_negative_target_hidden(
    target_hidden: torch.Tensor,
    dropout_ratio: float = 0.3,
    noise_std: float = 0.0,
    gen: Optional[torch.Generator] = None,
) -> torch.Tensor:
    return token_level_core.build_negative_target_hidden(
        target_hidden=target_hidden,
        dropout_ratio=dropout_ratio,
        noise_std=noise_std,
        gen=gen,
    )


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
    return token_level_core.compute_contrastive_draft_logits(
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


def build_vcd_candidate_mask(reference_logits: torch.Tensor, beta: float) -> torch.Tensor:
    return token_level_core.build_candidate_mask(reference_logits=reference_logits, beta=beta)


def apply_vcd_candidate_filter(logits: torch.Tensor, candidate_mask: torch.Tensor) -> torch.Tensor:
    return token_level_core.apply_candidate_filter(logits=logits, candidate_mask=candidate_mask)


def _compute_kl_divergence(first_logits: torch.Tensor, second_logits: torch.Tensor) -> float:
    return token_level_core.compute_kl_divergence(first_logits=first_logits, second_logits=second_logits)


def _token_rank_from_logits(logits_1d: torch.Tensor, token_id: int) -> int:
    return token_level_core.token_rank_from_logits(logits_1d=logits_1d, token_id=token_id)


def _token_str(tokenizer: AutoTokenizer, token_id: int) -> str:
    return token_level_core.token_str(tokenizer=tokenizer, token_id=token_id)


def _topk_token_details(
    logits_1d: torch.Tensor,
    tokenizer: AutoTokenizer,
    k: int = 5,
) -> list[dict[str, Any]]:
    return token_level_core.topk_token_details(logits_1d=logits_1d, tokenizer=tokenizer, k=k)


def _estimate_token_hit_probability(
    logits_1d: torch.Tensor,
    token_id: int,
    num_samples: int,
    gen: Optional[torch.Generator] = None,
) -> float:
    return token_level_core.estimate_token_hit_probability(
        logits_1d=logits_1d,
        token_id=token_id,
        num_samples=num_samples,
        gen=gen,
    )


def _build_reject_reason(record: dict[str, Any]) -> str:
    return token_level_reporting.build_reject_reason(record)


def _classify_reject_taxonomy(record: dict[str, Any]) -> tuple[str, str]:
    return token_level_reporting.classify_reject_taxonomy(record)


def _safe_dataset_name(name: str) -> str:
    return token_level_reporting.safe_dataset_name(name)


def _build_report_dir(dataset_name: str, report_dir: Optional[str]) -> Path:
    return token_level_reporting.build_report_dir(dataset_name, report_dir)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    token_level_reporting.write_jsonl(path, records)


def _write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    token_level_reporting.write_csv(path, records)


def _maybe_create_plots(report_dir: Path, records: list[dict[str, Any]]) -> list[str]:
    return token_level_reporting.maybe_create_plots(report_dir, records, compare_name="VCD")


def _build_summary(records: list[dict[str, Any]], args: argparse.Namespace, block_size: int) -> dict[str, Any]:
    return token_level_reporting.build_summary(
        records,
        dataset=args.dataset,
        seed=args.seed,
        block_size=block_size,
        alpha=args.vcd_alpha,
        beta=args.vcd_beta,
        shadow_num_samples=args.vcd_shadow_num_samples,
        negative_context_dropout=args.negative_context_dropout,
        negative_context_noise_std=args.negative_context_noise_std,
        compare_name="VCD",
    )


def _write_report_md(
    report_path: Path,
    summary: dict[str, Any],
    records: list[dict[str, Any]],
    plot_files: list[str],
    max_rows: int,
) -> None:
    token_level_reporting.write_report_md(
        report_path=report_path,
        summary=summary,
        records=records,
        plot_files=plot_files,
        max_rows=max_rows,
        compare_name="VCD",
    )


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
    vcd_alpha: float = 1.0,
    beta: float = 0.1,
    negative_context_dropout: float = 0.3,
    negative_context_noise_std: float = 0.0,
    divergence_accumulator: Optional[List[float]] = None,
    reject_records: Optional[list[dict[str, Any]]] = None,
    vcd_shadow_num_samples: int = 1,
    sample_idx: int = -1,
    turn_idx: int = -1,
    seed: int = 0,
) -> SimpleNamespace:
    gen = torch.Generator(device=model.device)
    gen.manual_seed(seed)

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full((1, max_length + block_size), mask_token_id, dtype=torch.long, device=model.device)
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
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature, gen=gen)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

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
                gen=gen,
            )
            past_key_values_draft.crop(start)

            if divergence_accumulator is not None:
                divergence_accumulator.append(_compute_kl_divergence(positive_draft_logits, negative_draft_logits))

            candidate_mask = build_vcd_candidate_mask(reference_logits=positive_draft_logits, beta=beta)
            final_draft_logits = apply_vcd_logits(
                first_logits=positive_draft_logits,
                second_logits=negative_draft_logits,
                alpha=vcd_alpha,
            )
            final_draft_logits = apply_vcd_candidate_filter(logits=final_draft_logits, candidate_mask=candidate_mask)
            positive_sampled_tokens = sample(positive_draft_logits, gen=gen)
            vcd_shadow_sampled_tokens = sample(final_draft_logits, gen=gen)

            # Keep original rollout trajectory; VCD is evaluated in shadow mode.
            block_output_ids[:, 1:] = positive_sampled_tokens
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
        acceptance_length = int((block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item())

        if block_size > 1 and reject_records is not None and acceptance_length < (block_size - 1):
            reject_offset = acceptance_length
            target_token_id = int(posterior[0, reject_offset].item())
            sampled_draft_id = int(block_output_ids[0, reject_offset + 1].item())
            vcd_shadow_sampled_id = int(vcd_shadow_sampled_tokens[0, reject_offset].item())

            positive_step_logits = positive_draft_logits[0, reject_offset, :].float()
            vcd_step_logits = final_draft_logits[0, reject_offset, :].float()
            posterior_step_logits = output.logits[0, reject_offset, :].float()

            reject_records.append(
                token_level_reporting.build_reject_record(
                    tokenizer=tokenizer,
                    sample_idx=sample_idx,
                    turn_idx=turn_idx,
                    decode_step=decode_step,
                    start=start,
                    reject_offset=reject_offset,
                    target_token_id=target_token_id,
                    sampled_draft_id=sampled_draft_id,
                    shadow_sampled_id=vcd_shadow_sampled_id,
                    positive_step_logits=positive_step_logits,
                    contrastive_step_logits=vcd_step_logits,
                    posterior_step_logits=posterior_step_logits,
                    candidate_mask_step=candidate_mask[0, reject_offset, :],
                    shadow_num_samples=vcd_shadow_num_samples,
                    gen=gen,
                )
            )

        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        decode_step += 1

        past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, : acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_tensor = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_indices = torch.isin(output_ids[0][num_input_tokens:], stop_tensor).nonzero(as_tuple=True)[0]
        if stop_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_indices[0] + 1]

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vcd-alpha", type=float, default=0.5)
    parser.add_argument("--vcd-beta", type=float, default=0.1)
    parser.add_argument("--negative-context-dropout", type=float, default=0.3)
    parser.add_argument("--negative-context-noise-std", type=float, default=0.0)
    parser.add_argument(
        "--vcd-shadow-num-samples",
        type=int,
        default=1,
        help="Number of Monte Carlo samples per reject position to estimate rescue probability.",
    )
    parser.add_argument("--max-example-rows", type=int, default=30)
    parser.add_argument("--report-dir", type=str, default=None)
    parser.add_argument("--no-plots", action="store_true")
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

    def has_flash_attn() -> bool:
        if args.deterministic:
            logger.info("Deterministic mode enabled. Forcing SDPA attention backend.")
            return False
        try:
            found = importlib.util.find_spec("flash_attn") is not None
            if found:
                return True
            logger.warning("flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower.")
            return False
        except Exception:
            logger.warning("Failed to check flash_attn. Falling back to torch.sdpa.")
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
    if block_size <= 1:
        raise ValueError("Reject-token analysis requires block_size > 1.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    dataset = load_and_process_dataset(args.dataset)
    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=args.seed).select(range(args.max_samples))

    divergence_accumulator: List[float] = []
    reject_records: list[dict[str, Any]] = []
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
                    tokenizer=tokenizer,
                    input_ids=input_ids,
                    mask_token_id=draft_model.mask_token_id,
                    max_new_tokens=args.max_new_tokens,
                    block_size=bs,
                    stop_token_ids=[tokenizer.eos_token_id],
                    temperature=args.temperature,
                    vcd_alpha=args.vcd_alpha,
                    beta=args.vcd_beta,
                    negative_context_dropout=args.negative_context_dropout,
                    negative_context_noise_std=args.negative_context_noise_std,
                    divergence_accumulator=divergence_accumulator if bs == block_size else None,
                    reject_records=reject_records if bs == block_size else None,
                    vcd_shadow_num_samples=args.vcd_shadow_num_samples,
                    sample_idx=idx,
                    turn_idx=turn_index,
                    seed=sample_seed + bs,
                )

            spec_response = response[block_size]
            generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": output_text})
            responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        reject_records = dist.gather(reject_records, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))
        reject_records = list(chain(*reject_records))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [acceptance_lengths.count(b) / len(acceptance_lengths) for b in range(block_size + 1)]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    if divergence_accumulator:
        avg_divergence = sum(divergence_accumulator) / len(divergence_accumulator)
        print(f"Average KL divergence between draft logits: {avg_divergence:.5f}")
        print(f"  Min KL: {min(divergence_accumulator):.5f}   Max KL: {max(divergence_accumulator):.5f}")

    report_dir = _build_report_dir(args.dataset, args.report_dir)
    summary = _build_summary(reject_records, args, block_size)

    jsonl_path = report_dir / "reject_records.jsonl"
    csv_path = report_dir / "reject_records.csv"
    summary_path = report_dir / "summary.json"
    report_md_path = report_dir / "report.md"

    _write_jsonl(jsonl_path, reject_records)
    _write_csv(csv_path, reject_records)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    plot_files = [] if args.no_plots else _maybe_create_plots(report_dir, reject_records)
    _write_report_md(
        report_path=report_md_path,
        summary=summary,
        records=reject_records,
        plot_files=plot_files,
        max_rows=args.max_example_rows,
    )

    print(f"Reject events recorded: {summary.get('reject_events', 0)}")
    if summary.get("reject_events", 0) > 0:
        print(f"Positive hit rate @reject: {summary['positive_hit_rate'] * 100:.2f}%")
        print(f"VCD hit rate @reject: {summary['vcd_hit_rate'] * 100:.2f}%")
        print(f"Hit-rate gain (VCD - Positive): {summary['hit_rate_gain'] * 100:+.2f}%")
        print(f"VCD shadow fix rate @reject: {summary['vcd_shadow_fix_rate'] * 100:.2f}%")
        print(
            "Mean rescue-prob (Positive / VCD-shadow): "
            f"{summary['positive_rescue_prob_est']['mean']:.4f} / {summary['vcd_shadow_rescue_prob_est']['mean']:.4f}"
        )
        print(
            "Mean rescue-prob gain (VCD - Positive): "
            f"{summary['rescue_prob_gain_vcd_minus_positive']['mean']:+.4f}"
        )
    print(f"Report directory: {report_dir}")
    print(f"  - {jsonl_path.name}")
    print(f"  - {csv_path.name}")
    print(f"  - {summary_path.name}")
    print(f"  - {report_md_path.name}")
    if plot_files:
        print(f"  - plots: {', '.join(plot_files)}")


if __name__ == "__main__":
    main()