import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a token-neighbor lookup table from target model input embeddings. "
            "Output is a .npy array with shape [vocab_size, top_k] where each row "
            "contains token ids of nearest neighbors by cosine similarity."
        )
    )
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device for similarity search.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Embedding dtype during similarity computation.",
    )
    return parser.parse_args()


def str_to_torch_dtype(dtype: str) -> torch.dtype:
    if dtype == "float16":
        return torch.float16
    if dtype == "bfloat16":
        return torch.bfloat16
    return torch.float32


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compute_dtype = str_to_torch_dtype(args.dtype)
    device = torch.device(args.device)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        dtype=compute_dtype,
    ).to(device).eval()

    embedding = model.get_input_embeddings().weight.detach().to(device=device, dtype=compute_dtype)
    vocab_size, hidden_dim = embedding.shape

    if args.top_k <= 0:
        raise ValueError("--top-k must be > 0")
    if args.top_k >= vocab_size:
        raise ValueError(f"--top-k must be < vocab_size ({vocab_size})")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    embedding = torch.nn.functional.normalize(embedding, p=2, dim=-1)
    neighbor_table = np.empty((vocab_size, args.top_k), dtype=np.int32)

    for start in tqdm(range(0, vocab_size, args.chunk_size), desc="Building neighbor table"):
        end = min(start + args.chunk_size, vocab_size)
        query = embedding[start:end]  # [chunk, hidden]

        # Cosine similarity because embeddings are L2-normalized.
        sim = query @ embedding.T  # [chunk, vocab]

        # Exclude self-token from nearest-neighbor candidates.
        row_idx = torch.arange(end - start, device=device)
        col_idx = torch.arange(start, end, device=device)
        sim[row_idx, col_idx] = -torch.inf

        _, top_idx = torch.topk(sim, k=args.top_k, dim=-1, largest=True, sorted=True)
        neighbor_table[start:end] = top_idx.to(torch.int32).cpu().numpy()

    np.save(output_path, neighbor_table)
    print(f"Saved neighbor table to {output_path} with shape {neighbor_table.shape}")
    print(f"vocab_size={vocab_size}, hidden_dim={hidden_dim}, top_k={args.top_k}")


if __name__ == "__main__":
    main()
