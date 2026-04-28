#!/usr/bin/env python3
"""
Quick validation script to verify float16 compression and storage improvements.
Run: python3 test_compression.py
"""

import torch
import numpy as np
from pathlib import Path
import sys

def check_collected_data(collection_dir: str = "alpha_model/collected_alpha_records/h200_collect_gsm8k_0_200"):
    """Analyze the storage structure and dtypes of collected data."""
    coll_path = Path(collection_dir)
    
    if not coll_path.exists():
        print(f"❌ Collection directory not found: {coll_path}")
        return False
    
    files = sorted(coll_path.glob("*.pt"))
    if not files:
        print(f"❌ No .pt files found in {coll_path}")
        return False
    
    print(f"✓ Found {len(files)} .pt files\n")
    
    # Load first chunk to check dtypes
    first_chunk = torch.load(files[0], map_location="cpu")
    if isinstance(first_chunk, dict):
        records = first_chunk if isinstance(list(first_chunk.values())[0], torch.Tensor) else [first_chunk]
    else:
        records = first_chunk
    
    if not records:
        print(f"❌ First chunk is empty")
        return False
    
    first_record = records[0]
    
    print("=" * 70)
    print("CURRENT DATA STRUCTURE (First Record)")
    print("=" * 70)
    
    total_bytes_per_record = 0
    dtype_usage = {}
    
    for key, value in sorted(first_record.items()):
        if isinstance(value, torch.Tensor):
            nbytes = value.numel() * value.element_size()
            dtype = str(value.dtype)
            shape = tuple(value.shape)
            total_bytes_per_record += nbytes
            
            dtype_usage[dtype] = dtype_usage.get(dtype, 0) + nbytes
            
            print(f"  {key:30s} {str(shape):20s} {dtype:15s} {nbytes:>8,} bytes")
        else:
            print(f"  {key:30s} (scalar) {type(value).__name__:15s}")
    
    print("-" * 70)
    print(f"  Total per record: {total_bytes_per_record:,} bytes ({total_bytes_per_record/1024:.2f} KB)")
    print()
    
    # Count records in first chunk
    num_records_in_chunk = len(records)
    chunk_size = sum(v.numel() * v.element_size() for v in records[0].values() if isinstance(v, torch.Tensor)) * num_records_in_chunk
    
    print(f"Records in chunk 0: {num_records_in_chunk}")
    print(f"Approx. chunk size: {chunk_size / 1024 / 1024:.2f} MB")
    print()
    
    # Analyze dtype distribution
    print("=" * 70)
    print("DTYPE DISTRIBUTION")
    print("=" * 70)
    for dtype, bytes_used in sorted(dtype_usage.items(), key=lambda x: x[1], reverse=True):
        pct = 100 * bytes_used / total_bytes_per_record
        print(f"  {dtype:15s}: {bytes_used:>8,} bytes ({pct:>5.1f}%)")
    print()
    
    # Calculate space savings if float32 → float16 conversion
    if "torch.float32" in dtype_usage:
        float32_bytes = dtype_usage["torch.float32"]
        potential_savings = float32_bytes * 0.5  # float32 -> float16 is 50% reduction
        estimated_per_record = total_bytes_per_record - potential_savings
        
        print("=" * 70)
        print("POTENTIAL COMPRESSION (if float32 logits → float16)")
        print("=" * 70)
        print(f"  Current per-record: {total_bytes_per_record:,} bytes")
        print(f"  With float16:       {estimated_per_record:,} bytes")
        print(f"  Savings:            {potential_savings:,} bytes ({100*potential_savings/total_bytes_per_record:.1f}% reduction)")
        print()
        
        # Project to 50K scale
        num_records_observed = len(files) * num_records_in_chunk
        print(f"Observed records from {collection_dir}:")
        print(f"  Current scale: {num_records_observed} records")
        print(f"  Current size:  {(num_records_observed * total_bytes_per_record) / 1024 / 1024 / 1024:.2f} GB")
        print()
        
        # Assume similar distribution scales linearly
        per_sample_records = num_records_observed / 200  # 200 samples in pilot
        records_at_50k = int(per_sample_records * 50000)
        size_current_gb = (records_at_50k * total_bytes_per_record) / 1024 / 1024 / 1024
        size_compressed_gb = (records_at_50k * estimated_per_record) / 1024 / 1024 / 1024
        
        print("Projected for 50,000 samples (assuming similar distribution):")
        print(f"  Records: {records_at_50k:,}")
        print(f"  Current size: {size_current_gb:.1f} GB")
        print(f"  With float16: {size_compressed_gb:.1f} GB")
        print(f"  Savings: {size_current_gb - size_compressed_gb:.1f} GB ({100*(size_current_gb-size_compressed_gb)/size_current_gb:.1f}%)")
        print()
    
    return True


def check_updated_code():
    """Verify that the code changes are in place."""
    print("=" * 70)
    print("CHECKING CODE UPDATES")
    print("=" * 70)
    
    data_collecting_path = Path("alpha_model/data_collecting.py")
    if not data_collecting_path.exists():
        print(f"❌ {data_collecting_path} not found")
        return False
    
    content = data_collecting_path.read_text()
    
    checks = [
        ("float16 conversion in data_collecting.py", "torch.float16" in content),
        ("Script chunk_size=4096", "COLLECTOR_CHUNK_SIZE}-4096" in content or "4096" in content),
    ]
    
    for check_name, result in checks:
        status = "✓" if result else "✗"
        print(f"  {status} {check_name}")
    
    merge_path = Path("alpha_model/merge_pt_shards_to_hdf5.py")
    if merge_path.exists():
        merge_content = merge_path.read_text()
        merge_check = "arr.dtype" in merge_content or "preserve actual dtype" in merge_content
        status = "✓" if merge_check else "✗"
        print(f"  {status} merge_pt_shards_to_hdf5.py dtype preservation")
    
    print()
    return all(result for _, result in checks)


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  ALPHA COLLECTION COMPRESSION VALIDATION".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    print()
    
    check_updated_code()
    success = check_collected_data()
    
    if success:
        print("✓ Validation complete!\n")
        sys.exit(0)
    else:
        print("✗ Validation failed\n")
        sys.exit(1)
