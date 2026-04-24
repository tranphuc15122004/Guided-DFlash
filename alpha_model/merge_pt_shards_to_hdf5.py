import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch

try:
    import h5py
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "h5py is required for HDF5 export. Install it with: pip install h5py"
    ) from exc


REQUIRED_FIELDS = [
    "sample_id",
    "turn_index",
    "block_index",
    "draft_topk_token_ids",
    "draft_topk_logits",
    "neg_logits_on_draft_topk_ids",
    "block_position",
    "absolute_position",
    "target_token_id",
    "acceptance_length",
]

OPTIONAL_FIELDS = ["alpha_prev"]


"""
python alpha_model/merge_pt_shards_to_hdf5.py \
  --input-dir alpha_model/collected_alpha_records \
  --output-file alpha_model/alpha_dataset.h5 \
  --compression gzip
"""

def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _list_shards(input_dir: Path) -> List[Path]:
    shards = sorted(input_dir.glob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No .pt shards found in: {input_dir}")
    return shards


def _iter_records(shard_paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for shard_path in shard_paths:
        records = torch.load(shard_path, map_location="cpu")
        if not isinstance(records, list):
            raise ValueError(f"Shard is not a list of records: {shard_path}")
        for i, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {i} in {shard_path} is not a dict")
            yield record


def _validate_record(record: Dict[str, Any], shard_hint: str = "") -> None:
    missing = [name for name in REQUIRED_FIELDS if name not in record]
    if missing:
        raise ValueError(f"Missing required fields {missing} in {shard_hint}")


def _infer_layout(first_record: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    layout: Dict[str, Dict[str, Any]] = {}

    for key in REQUIRED_FIELDS:
        arr = _to_numpy(first_record[key])
        if key in {
            "sample_id",
            "turn_index",
            "block_index",
            "acceptance_length",
        }:
            arr = np.asarray(arr, dtype=np.int32).reshape(())
            layout[key] = {"shape": (), "dtype": np.int32}
        elif key in {"draft_topk_token_ids", "target_token_id"}:
            layout[key] = {"shape": arr.shape, "dtype": np.int32}
        else:
            layout[key] = {"shape": arr.shape, "dtype": np.float32}

    for key in OPTIONAL_FIELDS:
        if key in first_record:
            arr = _to_numpy(first_record[key])
            layout[key] = {"shape": arr.shape, "dtype": np.float32}

    return layout


def _create_datasets(
    h5f: "h5py.File",
    total_records: int,
    layout: Dict[str, Dict[str, Any]],
    compression: Optional[str],
) -> Dict[str, "h5py.Dataset"]:
    datasets: Dict[str, "h5py.Dataset"] = {}
    for name, spec in layout.items():
        ds_shape = (total_records,) + tuple(spec["shape"])
        chunks = True if total_records > 1 else None
        datasets[name] = h5f.create_dataset(
            name,
            shape=ds_shape,
            dtype=spec["dtype"],
            chunks=chunks,
            compression=compression,
        )
    return datasets


def merge_pt_shards_to_hdf5(
    input_dir: Path,
    output_file: Path,
    compression: Optional[str] = "gzip",
) -> None:
    shard_paths = _list_shards(input_dir)

    all_records: List[Dict[str, Any]] = []
    for shard_path in shard_paths:
        records = torch.load(shard_path, map_location="cpu")
        if not isinstance(records, list):
            raise ValueError(f"Shard is not a list of records: {shard_path}")
        for idx, record in enumerate(records):
            if not isinstance(record, dict):
                raise ValueError(f"Record {idx} in {shard_path} is not a dict")
            _validate_record(record, shard_hint=f"{shard_path}#{idx}")
            all_records.append(record)

    if not all_records:
        raise ValueError(f"No records found in shards under {input_dir}")

    layout = _infer_layout(all_records[0])

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output_file, "w") as h5f:
        datasets = _create_datasets(
            h5f=h5f,
            total_records=len(all_records),
            layout=layout,
            compression=compression,
        )

        for row_idx, record in enumerate(all_records):
            for name, spec in layout.items():
                if name in record:
                    arr = _to_numpy(record[name])
                else:
                    # Optional field fallback when absent in later records.
                    arr = np.zeros(spec["shape"], dtype=spec["dtype"])

                if spec["shape"] == ():
                    casted = np.asarray(arr, dtype=spec["dtype"]).reshape(())
                else:
                    casted = np.asarray(arr, dtype=spec["dtype"])
                    if casted.shape != tuple(spec["shape"]):
                        raise ValueError(
                            f"Shape mismatch for field '{name}' at row {row_idx}: "
                            f"expected {spec['shape']}, got {casted.shape}"
                        )
                datasets[name][row_idx] = casted

        h5f.attrs["num_records"] = len(all_records)
        h5f.attrs["source_input_dir"] = str(input_dir)
        h5f.attrs["num_shards"] = len(shard_paths)
        h5f.attrs["layout_json"] = json.dumps(
            {
                name: {
                    "shape": list(spec["shape"]),
                    "dtype": np.dtype(spec["dtype"]).name,
                }
                for name, spec in layout.items()
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge block-level .pt shard files into a single HDF5 dataset."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing rank_XX_chunk_YYYYY.pt files.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Path to output HDF5 file.",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="gzip",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression algorithm.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    compression = None if args.compression == "none" else args.compression

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    merge_pt_shards_to_hdf5(
        input_dir=input_dir,
        output_file=output_file,
        compression=compression,
    )
    print(f"Merged shards from {input_dir} into {output_file}")


if __name__ == "__main__":
    main()
