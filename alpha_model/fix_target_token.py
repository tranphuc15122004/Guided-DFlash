"""
Fix target_token_id in collected DFlash alpha records.

Problem:
  Old parallel verification produced WRONG target tokens at positions
  >= acceptance_length + 1 within each block, because the target model
  was conditioned on rejected (wrong) draft tokens.

Fix:
  For block i with acceptance_length L_i:
    - Correct: target_token_id[0 : L_i+1]  (L_i accepted + 1 bonus)
    - Wrong:   target_token_id[L_i+1 : 15]
  Each wrong position p is fixed by chaining through subsequent blocks:
    offset = p - (L_i + 1)
    for j in [i+1, i+2, ...]:
        if offset < L_j + 1:  → target_i[p] = target_j[offset]; break
        offset -= (L_j + 1)   → chain to next block

Usage:
  python alpha_model/fix_target_token.py
      --input-dir alpha_model/collected_alpha_records_darft
      --backup-dir alpha_model/collected_alpha_records_darft.BACKUP
      --num-workers 8
      [--dry-run]
      [--resume]
"""

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

try:
    import torch
except ImportError:
    print("ERROR: torch is required. Activate your dflash environment.", file=sys.stderr)
    sys.exit(1)

from tqdm import tqdm


# ---------------------------------------------------------------------------
# Core fix logic
# ---------------------------------------------------------------------------

def fix_target_tokens(records: list[dict]) -> int:
    """Fix target_token_id in-place for all records.

    Groups by (sample_id, turn_index), sorts by block_index, then chains
    correct target tokens from subsequent blocks to fill wrong positions.

    Returns number of records modified.
    """
    # Group
    groups: dict[tuple[int, int], list[dict]] = {}
    for r in records:
        key = (int(r["sample_id"]), int(r["turn_index"]))
        groups.setdefault(key, []).append(r)

    # The number of diffusion positions in a block:  target_token_id shape (15,)
    S = 15  # block_size - 1
    modified = 0

    for group in groups.values():
        group.sort(key=lambda x: int(x["block_index"]))
        N = len(group)

        for i, rec in enumerate(group):
            L = int(rec["acceptance_length"])
            correct_upto = L + 1  # positions [0, correct_upto) are correct
            if correct_upto >= S:
                continue  # all positions are already correct

            tgt = rec["target_token_id"]
            changed = False

            for p in range(correct_upto, S):
                offset = p - correct_upto
                found = False

                for j in range(i + 1, N):
                    next_L = int(group[j]["acceptance_length"])
                    n_correct = next_L + 1

                    if offset < n_correct:
                        new_val = group[j]["target_token_id"][offset]
                        if int(tgt[p]) != int(new_val):
                            tgt[p] = new_val
                            changed = True
                        found = True
                        break
                    offset -= n_correct

                if not found:
                    # No more blocks to chain from — remaining positions unfixable
                    break

            if changed:
                modified += 1

    return modified


# ---------------------------------------------------------------------------
# Per-subfolder worker
# ---------------------------------------------------------------------------

def process_subfolder(
    subfolder: Path,
    dry_run: bool = False,
) -> int:
    """Load all chunks in a subfolder, fix target tokens, save back.

    Returns total record count processed.
    """
    chunk_paths = sorted(subfolder.glob("rank_*.pt"))
    if not chunk_paths:
        return 0

    # Record original chunk sizes so we can redistribute after fix
    chunk_sizes: list[int] = []
    all_records: list[dict] = []

    for cp in chunk_paths:
        recs = torch.load(cp, map_location="cpu", weights_only=True)
        chunk_sizes.append(len(recs))
        all_records.extend(recs)

    total = len(all_records)

    # Fix
    modified = fix_target_tokens(all_records)

    # Save back (only if not dry_run)
    if not dry_run:
        idx = 0
        for cp, size in zip(chunk_paths, chunk_sizes):
            torch.save(all_records[idx: idx + size], cp)
            idx += size

    return total, modified


# ---------------------------------------------------------------------------
# Backup helper (sequential, per-subfolder)
# ---------------------------------------------------------------------------

def backup_subfolder(src: Path, dst: Path, dry_run: bool = False) -> None:
    """Copy subfolder to backup location."""
    if dst.exists():
        return  # already backed up (used with --resume)
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src, dst)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fix target_token_id in DFlash alpha records by chaining correct "
                    "tokens from subsequent blocks."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Root directory containing subfolders of chunk .pt files.",
    )
    parser.add_argument(
        "--backup-dir",
        type=str,
        default=None,
        help="Backup directory. If omitted, backup is created as "
             "<input_dir>.BACKUP next to input_dir.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: os.cpu_count()).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Find and report what would be fixed without writing anything.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Skip subfolders that already have a backup (for restarting after "
             "partial completion).",
    )
    parser.add_argument(
        "--skip-backup",
        action="store_true",
        default=False,
        help="Skip the backup phase entirely.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"ERROR: input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    backup_dir = Path(args.backup_dir) if args.backup_dir else Path(str(input_dir) + ".BACKUP")

    # Support two layouts:
    #   1) input_dir itself contains rank_*.pt files (single subfolder)
    #   2) input_dir contains subfolders that each contain rank_*.pt files
    if list(input_dir.glob("rank_*.pt")):
        # Layout 1: treat input_dir as the only processing unit
        subfolders = [input_dir]
    else:
        # Layout 2: look for subdirectories with .pt shards
        subfolders = sorted(
            [p for p in input_dir.iterdir() if p.is_dir() and list(p.glob("rank_*.pt"))]
        )
        if not subfolders:
            print(f"ERROR: no subfolders with rank_*.pt files found in {input_dir}", file=sys.stderr)
            sys.exit(1)

    num_workers = args.num_workers or os.cpu_count()
    num_workers = max(1, num_workers)

    print(f"Input:    {input_dir}")
    print(f"Backup:   {backup_dir}")
    print(f"Subdirs:  {len(subfolders)}")
    print(f"Workers:  {num_workers}")
    print(f"Dry run:  {args.dry_run}")
    print(f"Resume:   {args.resume}")
    print()

    # ── Phase 1: Backup ─────────────────────────────────────────────
    if not args.skip_backup and not args.dry_run:
        print("=" * 60)
        print("Phase 1: Backup")
        print("=" * 60)
        t0 = time.time()
        for sf in tqdm(subfolders, desc="Backup", unit="subdir"):
            dst = backup_dir / sf.relative_to(input_dir) if sf != input_dir else backup_dir
            backup_subfolder(sf, dst, dry_run=args.dry_run)
        print(f"Backup complete in {time.time() - t0:.1f}s\n")

    # ── Phase 2: Fix (parallel) ─────────────────────────────────────
    print("=" * 60)
    print("Phase 2: Fix target_token_id")
    print("=" * 60)
    t0 = time.time()
    total_records = 0
    total_modified = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        future_to_sf = {
            pool.submit(process_subfolder, sf, args.dry_run): sf
            for sf in subfolders
        }
        for f in tqdm(as_completed(future_to_sf), total=len(subfolders), desc="Fix", unit="subdir"):
            sf = future_to_sf[f]
            try:
                n_records, n_modified = f.result()
                total_records += n_records
                total_modified += n_modified
            except Exception as exc:
                print(f"\n  ✗ {sf.name}: FAILED — {exc}")
                failed += 1

    elapsed = time.time() - t0
    print()
    print(f"Processed: {total_records} records across {len(subfolders)} subfolders")
    print(f"Modified:  {total_modified} records (blocks with at least 1 fixed position)")
    if failed:
        print(f"Failed:    {failed} subfolders")
    print(f"Duration:  {elapsed:.1f}s ({elapsed / max(1, len(subfolders)):.1f}s/subdir)")
    print()

    if args.dry_run:
        print("Dry run complete — no files were modified.")
    else:
        print("All files updated in-place.")


if __name__ == "__main__":
    main()
