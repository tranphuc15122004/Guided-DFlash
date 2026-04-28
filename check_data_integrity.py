#!/usr/bin/env python3
"""Kiểm tra tính toàn vẹn và phát hiện trùng lặp dữ liệu trong các thư mục collected_alpha_records.

Script này thực hiện:
1. Kiểm tra xem các file .pt có load được không (không bị corrupt)
2. Kiểm tra cấu trúc dữ liệu có đúng định dạng không
3. Phát hiện các bản ghi bị trùng lặp giữa các thư mục (dựa trên các trường khóa)
4. Báo cáo thống kê tổng hợp

Usage:
    python3 check_data_integrity.py
"""

from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import torch
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "alpha_model" / "collected_alpha_records"


def get_record_key(record: Dict[str, Any]) -> str:
    """Tạo một khóa duy nhất cho mỗi record dựa trên các trường quan trọng.
    
    Các trường được dùng để xác định trùng lặp:
    - prompt_ids: ID của prompt (dãy token đầu vào)
    - target_token_ids: ID token được sinh bởi target model
    - draft_token_ids: ID token được sinh bởi draft model
    """
    parts = []
    # Sử dụng prompt_ids làm phần chính của khóa
    if "prompt_ids" in record:
        parts.append("p" + "_".join(str(x) for x in record["prompt_ids"]))
    # Thêm target_token_ids
    if "target_token_ids" in record:
        parts.append("t" + "_".join(str(x) for x in record["target_token_ids"]))
    # Thêm draft_token_ids
    if "draft_token_ids" in record:
        parts.append("d" + "_".join(str(x) for x in record["draft_token_ids"]))
    
    if parts:
        key_str = "|".join(parts)
        return hashlib.md5(key_str.encode()).hexdigest()
    return ""


def check_file(filepath: Path) -> Tuple[bool, str, List[Dict[str, Any]]]:
    """Kiểm tra một file .pt có load được và định dạng đúng không.
    
    Returns:
        (is_valid, error_message, data)
    """
    try:
        data = torch.load(filepath, map_location="cpu", weights_only=True)
    except Exception as e:
        return False, f"Lỗi load file: {e}", []
    
    if not isinstance(data, list):
        return False, f"Dữ liệu không phải list mà là {type(data)}", []
    
    if len(data) == 0:
        return False, "File rỗng (list trống)", []
    
    # Kiểm tra từng record
    for i, record in enumerate(data):
        if not isinstance(record, dict):
            return False, f"Record {i} không phải dict mà là {type(record)}", []
        
        # Kiểm tra các trường bắt buộc
        required_fields = ["prompt_ids", "target_token_ids", "draft_token_ids"]
        for field in required_fields:
            if field not in record:
                return False, f"Record {i} thiếu trường '{field}'", []
    
    return True, "OK", data


def check_all_directories() -> Dict[str, Any]:
    """Kiểm tra toàn bộ thư mục collected_alpha_records.
    
    Returns:
        Dict chứa kết quả kiểm tra
    """
    if not DATA_DIR.exists():
        console.print(f"[red]Thư mục {DATA_DIR} không tồn tại![/red]")
        return {}
    
    subdirs = sorted([d for d in DATA_DIR.iterdir() if d.is_dir()])
    
    if not subdirs:
        console.print(f"[yellow]Không tìm thấy thư mục con trong {DATA_DIR}[/yellow]")
        return {}
    
    results = {
        "dirs": {},
        "global_record_keys": defaultdict(list),  # key -> [(dir, file, idx)]
        "all_keys": set(),
    }
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console,
    ) as progress:
        for subdir in subdirs:
            task = progress.add_task(f"Đang kiểm {subdir.name}...", total=None)
            
            dir_result = {
                "path": subdir,
                "total_files": 0,
                "valid_files": 0,
                "invalid_files": [],
                "total_records": 0,
                "records": [],  # Tất cả records hợp lệ
                "file_records": {},  # file -> [records]
            }
            
            pt_files = sorted(subdir.glob("*.pt"))
            dir_result["total_files"] = len(pt_files)
            
            for filepath in pt_files:
                is_valid, error_msg, data = check_file(filepath)
                
                if is_valid:
                    dir_result["valid_files"] += 1
                    dir_result["total_records"] += len(data)
                    dir_result["records"].extend(data)
                    dir_result["file_records"][filepath.name] = data
                    
                    # Thu thập keys để kiểm tra trùng lặp
                    for idx, record in enumerate(data):
                        key = get_record_key(record)
                        if key:
                            results["global_record_keys"][key].append(
                                (subdir.name, filepath.name, idx)
                            )
                            results["all_keys"].add(key)
                else:
                    dir_result["invalid_files"].append({
                        "file": filepath.name,
                        "error": error_msg,
                    })
            
            results["dirs"][subdir.name] = dir_result
            progress.update(task, completed=1)
    
    return results


def print_summary(results: Dict[str, Any]) -> None:
    """In bảng tổng hợp kết quả kiểm tra."""
    console.print("\n[bold cyan]=== TỔNG HỢP KIỂM TRA DỮ LIỆU ===[/bold cyan]\n")
    
    # Bảng tổng hợp từng thư mục
    table = Table(title="Tình trạng các thư mục dữ liệu")
    table.add_column("Thư mục", style="cyan")
    table.add_column("Files", justify="right")
    table.add_column("Valid", justify="right", style="green")
    table.add_column("Invalid", justify="right", style="red")
    table.add_column("Records", justify="right", style="yellow")
    
    total_files = 0
    total_valid = 0
    total_invalid = 0
    total_records = 0
    
    for dirname, info in results["dirs"].items():
        n_files = info["total_files"]
        n_valid = info["valid_files"]
        n_invalid = len(info["invalid_files"])
        n_records = info["total_records"]
        
        table.add_row(
            dirname,
            str(n_files),
            str(n_valid),
            str(n_invalid) if n_invalid > 0 else "0",
            str(n_records),
        )
        
        total_files += n_files
        total_valid += n_valid
        total_invalid += n_invalid
        total_records += n_records
    
    table.add_row(
        "[bold]TỔNG[/bold]",
        f"[bold]{total_files}[/bold]",
        f"[bold green]{total_valid}[/bold green]",
        f"[bold red]{total_invalid}[/bold red]",
        f"[bold yellow]{total_records}[/bold yellow]",
    )
    
    console.print(table)
    
    # In lỗi nếu có
    has_errors = any(len(info["invalid_files"]) > 0 for info in results["dirs"].values())
    if has_errors:
        console.print("\n[bold red]=== CÁC FILE LỖI ===[/bold red]")
        for dirname, info in results["dirs"].items():
            if info["invalid_files"]:
                console.print(f"\n[red]Thư mục: {dirname}[/red]")
                for err in info["invalid_files"]:
                    console.print(f"  - {err['file']}: {err['error']}")
    
    # Kiểm tra trùng lặp
    duplicates = {k: v for k, v in results["global_record_keys"].items() if len(v) > 1}
    
    console.print("\n[bold cyan]=== KIỂM TRA TRÙNG LẶP ===[/bold cyan]\n")
    
    if duplicates:
        console.print(f"[red]Tìm thấy {len(duplicates)} record bị trùng lặp giữa các thư mục![/red]\n")
        
        dup_table = Table(title="Các record bị trùng lặp")
        dup_table.add_column("STT", justify="right")
        dup_table.add_column("Số lần xuất hiện", justify="center")
        dup_table.add_column("Các vị trí", style="yellow")
        
        for i, (key, locations) in enumerate(sorted(duplicates.items(), key=lambda x: -len(x[1])))[:20]:
            loc_str = ", ".join(
                f"{loc[0]}/{loc[1]}[#{loc[2]}]" for loc in locations
            )
            dup_table.add_row(
                str(i + 1),
                str(len(locations)),
                loc_str,
            )
        
        console.print(dup_table)
        
        if len(duplicates) > 20:
            console.print(f"\n[yellow]... và {len(duplicates) - 20} record trùng khác[/yellow]")
        
        # Thống kê trùng lặp theo cặp thư mục
        console.print("\n[bold]=== TRÙNG LẶP THEO CÁP THƯ MỤC ===[/bold]")
        pair_counts = defaultdict(int)
        for key, locations in duplicates.items():
            dirs_involved = sorted(set(loc[0] for loc in locations))
            if len(dirs_involved) > 1:
                for i in range(len(dirs_involved)):
                    for j in range(i + 1, len(dirs_involved)):
                        pair = (dirs_involved[i], dirs_involved[j])
                        pair_counts[pair] += 1
        
        if pair_counts:
            pair_table = Table()
            pair_table.add_column("Thư mục 1", style="cyan")
            pair_table.add_column("Thư mục 2", style="cyan")
            pair_table.add_column("Số record trùng", justify="right", style="red")
            
            for (d1, d2), count in sorted(pair_counts.items(), key=lambda x: -x[1]):
                pair_table.add_row(d1, d2, str(count))
            
            console.print(pair_table)
        else:
            console.print("[yellow]Không có record trùng lặp giữa các thư mục khác nhau (chỉ trùng trong cùng 1 thư mục)[/yellow]")
    else:
        console.print("[green]✓ Không tìm thấy record bị trùng lặp![/green]")
    
    # Thống kê unique records
    unique_keys = {k: v for k, v in results["global_record_keys"].items() if len(v) == 1}
    console.print(f"\n[bold]Số record duy nhất: {len(unique_keys)}[/bold]")
    console.print(f"[bold]Tổng số record (có tính trùng): {sum(len(v) for v in results['global_record_keys'].values())}[/bold]")


def main() -> None:
    console.print("[bold green]Bắt đầu kiểm tra dữ liệu...[/bold green]\n")
    
    results = check_all_directories()
    
    if not results:
        return
    
    print_summary(results)
    
    console.print("\n[bold green]✓ Hoàn thành kiểm tra![/bold green]")


if __name__ == "__main__":
    main()
