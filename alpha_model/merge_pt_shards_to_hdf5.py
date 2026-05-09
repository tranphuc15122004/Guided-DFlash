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


def _infer_shapes_from_all_records(
    shard_paths: Iterable[Path],
    logger: logging.Logger,
) -> tuple[Dict[str, Dict[str, Any]], int]:
    """
    Scan ALL records across all shards to determine the canonical shape and dtype
    for each field. This is more robust than inferring from the first record alone.
    """
    logger.info("Scanning all records for shape inference...")

    # field_shapes[key] = set of shape tuples seen across records
    field_shapes: Dict[str, set] = {}
    record_count = 0

    for record in tqdm(_iter_records(shard_paths, show_progress=True), desc="Scanning records"):
        for key, value in record.items():
            arr = _to_numpy(value)
            shape_tuple = tuple(arr.shape)
            if key not in field_shapes:
                field_shapes[key] = set()
            field_shapes[key].add(shape_tuple)
        record_count += 1

    if record_count == 0:
        raise ValueError("No records found in shards")

    logger.info(f"Found {record_count} records across shards")

    # Build layout from observed shapes
    layout: Dict[str, Dict[str, Any]] = {}
    all_expected_keys = set(REQUIRED_FIELDS) | set(OPTIONAL_FIELDS)

    for key in all_expected_keys:
        if key not in field_shapes:
            # Optional field not present in any record → skip
            if key in OPTIONAL_FIELDS:
                continue
            raise ValueError(
                f"Required field '{key}' not found in any record. "
                f"Available fields: {sorted(field_shapes.keys())}"
            )

        shapes = field_shapes[key]

        # Validate scalar fields
        if key in SCALAR_FIELDS:
            non_scalar = [s for s in shapes if s != ()]
            if non_scalar:
                raise ValueError(
                    f"Field '{key}' is declared as SCALAR but found non-scalar shapes: {non_scalar}. "
                    f"Either fix the data collection or remove '{key}' from SCALAR_FIELDS."
                )
            canonical_shape: tuple = ()
        else:
            # Non-scalar field: verify shape consistency across all records
            if len(shapes) > 1:
                raise ValueError(
                    f"Field '{key}' has inconsistent shapes across records: {shapes}. "
                    f"Expected all records to have the same shape for this field."
                )
            canonical_shape = next(iter(shapes))

        # Determine dtype
        if key in INT_FIELDS:
            canonical_dtype = np.int32
        elif key in SCALAR_FIELDS:
            canonical_dtype = np.int32
        else:
            # Float fields (logits, positions, alpha values, etc.)
            canonical_dtype = np.float32

        layout[key] = {"shape": canonical_shape, "dtype": canonical_dtype}

    logger.info(f"Inferred layout with {len(layout)} fields: {sorted(layout.keys())}")
    for name, spec in sorted(layout.items()):
        logger.info(f"  {name}: shape={spec['shape']}, dtype={np.dtype(spec['dtype']).name}")

    return layout, record_count


def merge_pt_shards_to_hdf5(
    input_dir: Path,
    output_file: Path,
    compression: Optional[str] = "lzf",
    logger: Optional[logging.Logger] = None,
) -> None:
    if logger is None:
        logger = logging.getLogger(__name__)
    
    shard_paths = _list_shards(input_dir)
    logger.info(f"Found {len(shard_paths)} shards in {input_dir}")

    # ── Pass 1: scan ALL records to infer layout & count ────────────────────
    layout, total_records = _infer_shapes_from_all_records(shard_paths, logger)

    output_file.parent.mkdir(parents=True, exist_ok=True)

    # ── Pre-allocate contiguous numpy arrays ─────────────────────────────────
    logger.info(f"Pre-allocating {len(layout)} numpy arrays for {total_records} records...")
    arrays: Dict[str, np.ndarray] = {}
    for name, spec in layout.items():
        arr_shape = (total_records,) + tuple(spec["shape"])
        arrays[name] = np.empty(arr_shape, dtype=spec["dtype"])

    # ── Pass 2: fill arrays from shards (fast, no HDF5 I/O) ─────────────────
    logger.info("Loading records from shards into arrays...")
    for row_idx, record in enumerate(
        tqdm(_iter_records(shard_paths, show_progress=False), total=total_records, desc="Loading records")
    ):
        _validate_record(record, shard_hint=f"row {row_idx}")
        for name, spec in layout.items():
            if name in record:
                value = _to_numpy(record[name])
            else:
                value = np.zeros(spec["shape"], dtype=spec["dtype"])

            arr = np.asarray(value, dtype=spec["dtype"])
            if spec["shape"] == ():
                arrays[name][row_idx] = arr.reshape(())
            else:
                if arr.shape != tuple(spec["shape"]):
                    raise ValueError(
                        f"Shape mismatch for field '{name}' at row {row_idx}: "
                        f"expected {spec['shape']}, got {arr.shape}"
                    )
                arrays[name][row_idx] = arr

    # ── Pass 3: write all arrays to HDF5 in one shot per field ──────────────
    logger.info(f"Writing {len(layout)} arrays to HDF5 (single write per field)...")
    with h5py.File(output_file, "w") as h5f:
        for name in tqdm(sorted(layout.keys()), desc="Writing HDF5 datasets"):
            h5f.create_dataset(name, data=arrays[name], compression=compression)

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
        default="lzf",
        choices=["gzip", "lzf", "none"],
        help="HDF5 compression algorithm. Use 'lzf' for speed/reliability, 'gzip' for better ratio.",
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
