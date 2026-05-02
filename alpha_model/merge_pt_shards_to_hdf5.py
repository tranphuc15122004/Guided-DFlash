import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import torch
from tqdm import tqdm

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

OPTIONAL_FIELDS = ["alpha_prev", "alpha_applied"]

SCALAR_FIELDS = {
    "sample_id",
    "turn_index",
    "block_index",
    "acceptance_length",
}

INT_FIELDS = {
    "sample_id",
    "turn_index",
    "block_index",
    "draft_topk_token_ids",
    "target_token_id",
    "acceptance_length",
}

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
    shards = sorted(input_dir.rglob("*.pt"))
    if not shards:
        raise FileNotFoundError(f"No .pt shards found in: {input_dir}")
    return shards


def _iter_records(shard_paths: Iterable[Path], show_progress: bool = False) -> Iterable[Dict[str, Any]]:
    shard_list = list(shard_paths) if show_progress else shard_paths
    iterator = tqdm(shard_list, desc="Loading shards", disable=not show_progress)
    for shard_path in iterator:
        try:
            records = torch.load(shard_path, map_location="cpu", weights_only=True)
        except TypeError:
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


def _infer_layout(first_record: Dict[str, Any], all_records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    layout: Dict[str, Dict[str, Any]] = {}
    optional_field_examples: Dict[str, Any] = {}

    for key in REQUIRED_FIELDS:
        arr = _to_numpy(first_record[key])
        if key in SCALAR_FIELDS:
            layout[key] = {"shape": (), "dtype": np.int32}
        elif key in INT_FIELDS:
            layout[key] = {"shape": arr.shape, "dtype": np.int32}
        else:
            layout[key] = {"shape": arr.shape, "dtype": np.float32}

    for key in OPTIONAL_FIELDS:
        if key in first_record:
            optional_field_examples[key] = first_record[key]
        else:
            for record in all_records[1:]:
                if key in record:
                    optional_field_examples[key] = record[key]
                    break

        if key in optional_field_examples:
            arr = _to_numpy(optional_field_examples[key])
            layout[key] = {"shape": arr.shape, "dtype": np.float32}

    return layout


def _scan_records_for_layout(
    shard_paths: Iterable[Path],
    logger: logging.Logger,
) -> tuple[Dict[str, Any], Dict[str, Dict[str, Any]], int]:
    logger.info("Scanning shards for layout inference...")
    first_record: Optional[Dict[str, Any]] = None
    optional_field_examples: Dict[str, Any] = {}
    record_count = 0

    for record in tqdm(_iter_records(shard_paths, show_progress=True), desc="Scanning records"):
        if first_record is None:
            first_record = record
        for key in OPTIONAL_FIELDS:
            if key not in optional_field_examples and key in record:
                optional_field_examples[key] = record[key]
        record_count += 1

    if first_record is None:
        raise ValueError("No records found in shards")

    logger.info(f"Found {record_count} records across shards")
    layout = _infer_layout(first_record, [first_record])
    for key, value in optional_field_examples.items():
        layout[key] = {"shape": _to_numpy(value).shape, "dtype": np.float32}

    logger.info(f"Inferred layout with {len(layout)} fields: {sorted(layout.keys())}")
    return first_record, layout, record_count


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
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    shard_paths = _list_shards(input_dir)
    logger.info(f"Found {len(shard_paths)} shards in {input_dir}")

    first_record, layout, total_records = _scan_records_for_layout(shard_paths, logger)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Creating HDF5 file at {output_file} with {total_records} records...")
    with h5py.File(output_file, "w") as h5f:
        datasets = _create_datasets(
            h5f=h5f,
            total_records=total_records,
            layout=layout,
            compression=compression,
        )

        logger.info("Writing records to HDF5...")
        for row_idx, record in tqdm(
            enumerate(_iter_records(shard_paths, show_progress=False)),
            total=total_records,
            desc="Writing records"
        ):
            _validate_record(record, shard_hint=f"row {row_idx}")
            for name, spec in layout.items():
                if name in record:
                    arr = _to_numpy(record[name])
                else:
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

        h5f.attrs["num_records"] = total_records
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
        logger.info(f"Successfully merged {total_records} records from {len(shard_paths)} shards into {output_file}")


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

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger = logging.getLogger(__name__)

    input_dir = Path(args.input_dir)
    output_file = Path(args.output_file)

    logger.info(f"Starting merge: {input_dir} -> {output_file}")
    merge_pt_shards_to_hdf5(
        input_dir=input_dir,
        output_file=output_file,
        compression=compression,
        logger=logger,
    )
    logger.info(f"Merge completed successfully: {output_file}")


if __name__ == "__main__":
    main()
