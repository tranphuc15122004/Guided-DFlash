import argparse
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

from model import DFlashDraftModel, extract_context_feature, sample


'''
Chiều 1: số mẫu.
Chiều 2: số token context hiện có ở thời điểm đó (prefill thì dài prompt, decode thì phụ thuộc verify/acceptance).
Chiều 3: độ rộng đặc trưng; với target_hidden là tổng hợp nhiều layer nên phình lên thành len(target_layer_ids) * hidden_size.
'''

'''
=== target_hidden inspection ===
device: cuda
block_size: 16
target hidden_size: 2560
draft.target_layer_ids: [1, 9, 17, 25, 33]
num selected layers: 5
source: target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)

[prefill]
len(output.hidden_states): 37
shape of one hidden state (layer 0): 1x19x2560
target_hidden shape: 1x19x12800
expected last dim = len(layer_ids)*hidden_size = 12800
actual last dim: 12800

[after first verify step]
verify hidden state shape (layer 0): 1x16x2560
acceptance_length: 0
target_hidden shape after slicing [:, :acceptance_length+1, :]: 1x1x12800
'''

def _shape_str(tensor: torch.Tensor) -> str:
    return "x".join(str(x) for x in tensor.shape)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect where target_hidden comes from and its runtime shape."
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is 23 + 19? Please explain briefly.",
        help="Single-turn user prompt used for inspection.",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="sdpa",
        dtype=dtype,
    ).to(device).eval()

    draft = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="sdpa",
        dtype=dtype,
    ).to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    block_size = args.block_size if args.block_size is not None else draft.block_size

    messages = [{"role": "user", "content": args.prompt}]
    input_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

    position_ids = torch.arange(input_ids.shape[1] + block_size + 8, device=device).unsqueeze(0)
    past_key_values_target = DynamicCache()

    with torch.inference_mode():
        # Prefill: get hidden states from target model.
        prefill = target(
            input_ids,
            position_ids=position_ids[:, : input_ids.shape[1]],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True,
        )

        hidden_states: List[torch.Tensor] = list(prefill.hidden_states)
        target_layer_ids = draft.target_layer_ids
        target_hidden = extract_context_feature(hidden_states, target_layer_ids)

        # Build one decode block and run one verify step to inspect updated target_hidden.
        max_len = input_ids.shape[1] + block_size
        output_ids = torch.full(
            (1, max_len + block_size),
            draft.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        output_ids[:, : input_ids.shape[1]] = input_ids
        output_ids[:, input_ids.shape[1] : input_ids.shape[1] + 1] = sample(
            prefill.logits, args.temperature
        )

        start = input_ids.shape[1]
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]

        verify = target(
            block_output_ids,
            position_ids=block_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            output_hidden_states=True,
        )
        posterior = sample(verify.logits, args.temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        )
        target_hidden_after_verify = extract_context_feature(
            verify.hidden_states,
            target_layer_ids,
        )[:, : acceptance_length + 1, :]

    num_selected_layers = len(target_layer_ids)
    hidden_size = target.config.hidden_size
    expected_last_dim = num_selected_layers * hidden_size

    print("=== target_hidden inspection ===")
    print(f"device: {device}")
    print(f"block_size: {block_size}")
    print(f"target hidden_size: {hidden_size}")
    print(f"draft.target_layer_ids: {target_layer_ids}")
    print(f"num selected layers: {num_selected_layers}")
    print(
        "source: target_hidden = extract_context_feature(output.hidden_states, draft.target_layer_ids)"
    )
    print()
    print("[prefill]")
    print(f"len(output.hidden_states): {len(hidden_states)}")
    print(f"shape of one hidden state (layer 0): {_shape_str(hidden_states[0])}")
    print(f"target_hidden shape: {_shape_str(target_hidden)}")
    print(f"expected last dim = len(layer_ids)*hidden_size = {expected_last_dim}")
    print(f"actual last dim: {target_hidden.shape[-1]}")
    print()
    print("[after first verify step]")
    print(f"verify hidden state shape (layer 0): {_shape_str(verify.hidden_states[0])}")
    print(f"acceptance_length: {acceptance_length}")
    print(
        "target_hidden shape after slicing [:, :acceptance_length+1, :]: "
        f"{_shape_str(target_hidden_after_verify)}"
    )


if __name__ == "__main__":
    main()
