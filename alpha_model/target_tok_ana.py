import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch

try:
	from tqdm import tqdm
except ImportError:  # pragma: no cover
	tqdm = None


def _iter_shards(input_dir: Path, max_shards: int | None = None) -> List[Path]:
	shards = sorted(input_dir.rglob("*.pt"))
	if max_shards is not None:
		shards = shards[: max(0, max_shards)]
	return shards


def _load_records(shard_path: Path) -> List[Dict[str, Any]]:
	try:
		records = torch.load(shard_path, map_location="cpu", weights_only=True)
	except TypeError:
		records = torch.load(shard_path, map_location="cpu")
	if not isinstance(records, list):
		raise ValueError(f"Shard is not a list of records: {shard_path}")
	return records


def _to_int_array(value: Any) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		arr = value.detach().cpu().numpy()
	elif isinstance(value, np.ndarray):
		arr = value
	else:
		arr = np.asarray(value)
	return arr.astype(np.int64, copy=False).reshape(-1)


def _to_2d_array(value: Any, dtype: np.dtype) -> np.ndarray:
	if isinstance(value, torch.Tensor):
		arr = value.detach().cpu().numpy()
	elif isinstance(value, np.ndarray):
		arr = value
	else:
		arr = np.asarray(value)

	arr = arr.astype(dtype, copy=False)
	if arr.ndim == 3 and arr.shape[0] == 1:
		arr = arr[0]
	if arr.ndim == 1:
		arr = arr.reshape(1, -1)
	if arr.ndim != 2:
		raise ValueError(f"Expected 2D array, got shape={arr.shape}")
	return arr


def _rank_of_index_desc(logits_row: np.ndarray, index: int) -> int:
	order = np.argsort(-logits_row, kind="stable")
	return int(np.where(order == index)[0][0]) + 1


def _top_items(counter: Counter, top_n: int) -> List[Tuple[int, int, float]]:
	total = sum(counter.values())
	if total == 0:
		return []
	out: List[Tuple[int, int, float]] = []
	for token_id, count in counter.most_common(top_n):
		out.append((int(token_id), int(count), float(count / total)))
	return out


def analyze_target_tokens(
	input_dir: Path,
	top_n: int,
	per_pos_top_n: int,
	max_shards: int | None = None,
) -> Dict[str, Any]:
	shards = _iter_shards(input_dir, max_shards=max_shards)
	if not shards:
		raise FileNotFoundError(f"No .pt shards found under: {input_dir}")

	shard_iter: Iterable[Path]
	if tqdm is not None:
		shard_iter = tqdm(shards, desc="Reading shards", unit="shard")
	else:
		shard_iter = shards

	token_counter: Counter = Counter()
	token_counter_by_position: Dict[int, Counter] = defaultdict(Counter)
	acceptance_counter: Counter = Counter()
	target_length_counter: Counter = Counter()
	target_rank_pos_counter: Counter = Counter()
	target_rank_neg_counter: Counter = Counter()
	target_topk_index_counter: Counter = Counter()
	target_rank_pos_by_position: Dict[int, Counter] = defaultdict(Counter)
	target_rank_neg_by_position: Dict[int, Counter] = defaultdict(Counter)
	target_topk_index_by_position: Dict[int, Counter] = defaultdict(Counter)
	rank_delta_counter: Counter = Counter()
	rank_delta_by_position: Dict[int, Counter] = defaultdict(Counter)
	rank_compare_by_position: Dict[int, Dict[str, int]] = defaultdict(
		lambda: {"total": 0, "pos_better": 0, "neg_better": 0, "tie": 0}
	)
	accepted_rank_pos_counter: Counter = Counter()
	accepted_rank_neg_counter: Counter = Counter()
	rejected_rank_pos_counter: Counter = Counter()
	rejected_rank_neg_counter: Counter = Counter()
	accepted_rank_delta_counter: Counter = Counter()
	rejected_rank_delta_counter: Counter = Counter()
	accepted_rank_pos_top1 = 0
	accepted_rank_neg_top1 = 0
	rejected_rank_pos_top1 = 0
	rejected_rank_neg_top1 = 0
	rejected_pos_in_topk: Dict[int, int] = defaultdict(int)
	rejected_pos_top1: Dict[int, int] = defaultdict(int)
	accepted_rank_compared = 0
	rejected_rank_compared = 0
	accepted_target_in_topk = 0
	rejected_target_in_topk = 0
	accepted_target_not_in_topk = 0
	rejected_target_not_in_topk = 0
	accepted_pos_better = 0
	accepted_neg_better = 0
	accepted_tie = 0
	rejected_pos_better = 0
	rejected_neg_better = 0
	rejected_tie = 0

	num_shards = 0
	num_records = 0
	num_missing_target = 0
	num_missing_acceptance = 0
	num_missing_topk_or_logits = 0
	total_target_tokens = 0
	total_rankable_positions = 0
	total_target_in_topk = 0
	total_target_not_in_topk = 0
	total_rank_compared = 0
	total_pos_better = 0
	total_neg_better = 0
	total_tie = 0

	for shard_path in shard_iter:
		records = _load_records(shard_path)
		num_shards += 1

		for rec in records:
			num_records += 1
			if "target_token_id" not in rec:
				num_missing_target += 1
				continue

			target_tokens = _to_int_array(rec["target_token_id"])
			if target_tokens.size == 0:
				continue

			length = int(target_tokens.shape[0])
			target_length_counter[length] += 1

			total_target_tokens += length
			token_counter.update(int(x) for x in target_tokens)
			for pos, tok in enumerate(target_tokens.tolist()):
				token_counter_by_position[pos].update([int(tok)])

			if "acceptance_length" in rec:
				acceptance_counter[int(rec["acceptance_length"])] += 1
			else:
				num_missing_acceptance += 1

			if (
				"draft_topk_token_ids" not in rec
				or "draft_topk_logits" not in rec
				or "neg_logits_on_draft_topk_ids" not in rec
			):
				num_missing_topk_or_logits += 1
				continue

			try:
				topk_ids = _to_2d_array(rec["draft_topk_token_ids"], np.int64)
				pos_logits = _to_2d_array(rec["draft_topk_logits"], np.float32)
				neg_logits = _to_2d_array(rec["neg_logits_on_draft_topk_ids"], np.float32)
			except ValueError:
				num_missing_topk_or_logits += 1
				continue

			seq_len = min(target_tokens.shape[0], topk_ids.shape[0], pos_logits.shape[0], neg_logits.shape[0])
			if seq_len <= 0:
				continue

			for pos in range(seq_len):
				total_rankable_positions += 1
				tok = int(target_tokens[pos])
				candidates = topk_ids[pos]
				matches = np.where(candidates == tok)[0]
				is_accepted = pos < int(rec.get("acceptance_length", 0))
				if matches.size == 0:
					total_target_not_in_topk += 1
					if is_accepted:
						accepted_target_not_in_topk += 1
					else:
						rejected_target_not_in_topk += 1
					continue

				total_target_in_topk += 1
				if is_accepted:
					accepted_target_in_topk += 1
				else:
					rejected_target_in_topk += 1
				topk_idx = int(matches[0])
				target_topk_index_counter[topk_idx] += 1
				target_topk_index_by_position[pos][topk_idx] += 1

				pos_rank = _rank_of_index_desc(pos_logits[pos], topk_idx)
				neg_rank = _rank_of_index_desc(neg_logits[pos], topk_idx)
				rank_delta = int(neg_rank - pos_rank)

				target_rank_pos_counter[pos_rank] += 1
				target_rank_neg_counter[neg_rank] += 1
				target_rank_pos_by_position[pos][pos_rank] += 1
				target_rank_neg_by_position[pos][neg_rank] += 1
				rank_delta_counter[rank_delta] += 1
				rank_delta_by_position[pos][rank_delta] += 1
				if is_accepted:
					accepted_rank_pos_counter[pos_rank] += 1
					accepted_rank_neg_counter[neg_rank] += 1
					accepted_rank_delta_counter[rank_delta] += 1
					if pos_rank == 1:
						accepted_rank_pos_top1 += 1
					if neg_rank == 1:
						accepted_rank_neg_top1 += 1
				else:
					rejected_rank_pos_counter[pos_rank] += 1
					rejected_rank_neg_counter[neg_rank] += 1
					rejected_rank_delta_counter[rank_delta] += 1
					rejected_pos_in_topk[pos] += 1
					if pos_rank == 1:
						rejected_rank_pos_top1 += 1
						rejected_pos_top1[pos] += 1
					if neg_rank == 1:
						rejected_rank_neg_top1 += 1

				total_rank_compared += 1
				rank_compare_by_position[pos]["total"] += 1
				if pos_rank < neg_rank:
					total_pos_better += 1
					rank_compare_by_position[pos]["pos_better"] += 1
					if is_accepted:
						accepted_pos_better += 1
					else:
						rejected_pos_better += 1
				elif pos_rank > neg_rank:
					total_neg_better += 1
					rank_compare_by_position[pos]["neg_better"] += 1
					if is_accepted:
						accepted_neg_better += 1
					else:
						rejected_neg_better += 1
				else:
					total_tie += 1
					rank_compare_by_position[pos]["tie"] += 1
					if is_accepted:
						accepted_tie += 1
					else:
						rejected_tie += 1
				if is_accepted:
					accepted_rank_compared += 1
				else:
					rejected_rank_compared += 1

	overall_top = _top_items(token_counter, top_n)

	per_position_top: Dict[str, List[Tuple[int, int, float]]] = {}
	for pos in sorted(token_counter_by_position.keys()):
		per_position_top[str(pos)] = _top_items(token_counter_by_position[pos], per_pos_top_n)

	rank_pos_hist = {str(k): int(v) for k, v in sorted(target_rank_pos_counter.items(), key=lambda x: x[0])}
	rank_neg_hist = {str(k): int(v) for k, v in sorted(target_rank_neg_counter.items(), key=lambda x: x[0])}
	accepted_rank_pos_hist = {str(k): int(v) for k, v in sorted(accepted_rank_pos_counter.items(), key=lambda x: x[0])}
	accepted_rank_neg_hist = {str(k): int(v) for k, v in sorted(accepted_rank_neg_counter.items(), key=lambda x: x[0])}
	rejected_rank_pos_hist = {str(k): int(v) for k, v in sorted(rejected_rank_pos_counter.items(), key=lambda x: x[0])}
	rejected_rank_neg_hist = {str(k): int(v) for k, v in sorted(rejected_rank_neg_counter.items(), key=lambda x: x[0])}
	topk_index_hist = {str(k): int(v) for k, v in sorted(target_topk_index_counter.items(), key=lambda x: x[0])}
	rank_delta_hist = {str(k): int(v) for k, v in sorted(rank_delta_counter.items(), key=lambda x: x[0])}
	accepted_rank_delta_hist = {str(k): int(v) for k, v in sorted(accepted_rank_delta_counter.items(), key=lambda x: x[0])}
	rejected_rank_delta_hist = {str(k): int(v) for k, v in sorted(rejected_rank_delta_counter.items(), key=lambda x: x[0])}

	def _mean_from_hist(counter: Counter) -> float | None:
		total = sum(counter.values())
		if total == 0:
			return None
		return float(sum(k * v for k, v in counter.items()) / total)

	rank_pos_mean = _mean_from_hist(target_rank_pos_counter)
	rank_neg_mean = _mean_from_hist(target_rank_neg_counter)
	accepted_rank_pos_mean = _mean_from_hist(accepted_rank_pos_counter)
	accepted_rank_neg_mean = _mean_from_hist(accepted_rank_neg_counter)
	rejected_rank_pos_mean = _mean_from_hist(rejected_rank_pos_counter)
	rejected_rank_neg_mean = _mean_from_hist(rejected_rank_neg_counter)
	topk_index_mean = _mean_from_hist(target_topk_index_counter)
	rank_delta_mean = _mean_from_hist(rank_delta_counter)
	accepted_rank_delta_mean = _mean_from_hist(accepted_rank_delta_counter)
	rejected_rank_delta_mean = _mean_from_hist(rejected_rank_delta_counter)

	def _rate(x: int, y: int) -> float:
		if y <= 0:
			return 0.0
		return float(x / y)

	return {
		"input_dir": str(input_dir),
		"num_shards": num_shards,
		"num_records": num_records,
		"num_records_missing_target_token_id": num_missing_target,
		"num_records_missing_acceptance_length": num_missing_acceptance,
		"num_records_missing_topk_or_logits": num_missing_topk_or_logits,
		"total_target_tokens": total_target_tokens,
		"unique_target_tokens": len(token_counter),
		"rank_stats": {
			"total_rankable_positions": total_rankable_positions,
			"target_in_topk": total_target_in_topk,
			"target_not_in_topk": total_target_not_in_topk,
			"target_in_topk_rate": (
				float(total_target_in_topk / total_rankable_positions)
				if total_rankable_positions > 0
				else 0.0
			),
			"target_topk_index_mean": topk_index_mean,
			"target_rank_positive_mean": rank_pos_mean,
			"target_rank_negative_mean": rank_neg_mean,
			"rank_delta_mean": rank_delta_mean,
			"rank_delta_definition": "delta = negative_rank - positive_rank; delta > 0 means positive rank is better",
			"rank_compared_count": total_rank_compared,
			"positive_better_count": total_pos_better,
			"negative_better_count": total_neg_better,
			"tie_count": total_tie,
			"positive_better_rate": _rate(total_pos_better, total_rank_compared),
			"negative_better_rate": _rate(total_neg_better, total_rank_compared),
			"tie_rate": _rate(total_tie, total_rank_compared),
			"accepted_part": {
				"rank_compared_count": accepted_rank_compared,
				"target_in_topk": accepted_target_in_topk,
				"target_not_in_topk": accepted_target_not_in_topk,
				"target_in_topk_rate": _rate(accepted_target_in_topk, accepted_rank_compared),
				"target_rank_positive_mean": accepted_rank_pos_mean,
				"target_rank_negative_mean": accepted_rank_neg_mean,
				"rank_delta_mean": accepted_rank_delta_mean,
				"target_rank_positive_top1_count": accepted_rank_pos_top1,
				"target_rank_negative_top1_count": accepted_rank_neg_top1,
				"target_rank_positive_top1_rate": _rate(accepted_rank_pos_top1, accepted_rank_compared),
				"target_rank_negative_top1_rate": _rate(accepted_rank_neg_top1, accepted_rank_compared),
				"positive_better_count": accepted_pos_better,
				"negative_better_count": accepted_neg_better,
				"tie_count": accepted_tie,
				"positive_better_rate": _rate(accepted_pos_better, accepted_rank_compared),
				"negative_better_rate": _rate(accepted_neg_better, accepted_rank_compared),
				"tie_rate": _rate(accepted_tie, accepted_rank_compared),
				"target_rank_positive_histogram": accepted_rank_pos_hist,
				"target_rank_negative_histogram": accepted_rank_neg_hist,
				"rank_delta_histogram": accepted_rank_delta_hist,
			},
			"rejected_part": {
				"rank_compared_count": rejected_rank_compared,
				"target_in_topk": rejected_target_in_topk,
				"target_not_in_topk": rejected_target_not_in_topk,
				"target_in_topk_rate": _rate(rejected_target_in_topk, rejected_rank_compared),
				"target_rank_positive_mean": rejected_rank_pos_mean,
				"target_rank_negative_mean": rejected_rank_neg_mean,
				"rank_delta_mean": rejected_rank_delta_mean,
				"target_rank_positive_top1_count": rejected_rank_pos_top1,
				"target_rank_negative_top1_count": rejected_rank_neg_top1,
				"target_rank_positive_top1_rate": _rate(rejected_rank_pos_top1, rejected_rank_compared),
				"target_rank_negative_top1_rate": _rate(rejected_rank_neg_top1, rejected_rank_compared),
				"positive_better_count": rejected_pos_better,
				"negative_better_count": rejected_neg_better,
				"tie_count": rejected_tie,
				"positive_better_rate": _rate(rejected_pos_better, rejected_rank_compared),
				"negative_better_rate": _rate(rejected_neg_better, rejected_rank_compared),
				"tie_rate": _rate(rejected_tie, rejected_rank_compared),
				"target_rank_positive_histogram": rejected_rank_pos_hist,
				"target_rank_negative_histogram": rejected_rank_neg_hist,
				"rank_delta_histogram": rejected_rank_delta_hist,
			},
			"target_topk_index_histogram": topk_index_hist,
			"target_rank_positive_histogram": rank_pos_hist,
			"target_rank_negative_histogram": rank_neg_hist,
			"rank_delta_histogram": rank_delta_hist,
			"target_rank_positive_top": [
				{"rank": r, "count": c, "ratio": p}
				for r, c, p in _top_items(target_rank_pos_counter, per_pos_top_n)
			],
			"target_rank_negative_top": [
				{"rank": r, "count": c, "ratio": p}
				for r, c, p in _top_items(target_rank_neg_counter, per_pos_top_n)
			],
			"target_topk_index_top": [
				{"topk_index": i, "count": c, "ratio": p}
				for i, c, p in _top_items(target_topk_index_counter, per_pos_top_n)
			],
			"rank_delta_top": [
				{"delta": d, "count": c, "ratio": p}
				for d, c, p in _top_items(rank_delta_counter, per_pos_top_n)
			],
		},
		"target_length_distribution": {
			str(k): int(v) for k, v in sorted(target_length_counter.items(), key=lambda x: x[0])
		},
		"acceptance_length_distribution": {
			str(k): int(v) for k, v in sorted(acceptance_counter.items(), key=lambda x: x[0])
		},
		"overall_top_tokens": [
			{"token_id": tok, "count": cnt, "ratio": ratio} for tok, cnt, ratio in overall_top
		],
		"per_position_top_tokens": {
			pos: [
				{"token_id": tok, "count": cnt, "ratio": ratio}
				for tok, cnt, ratio in rows
			]
			for pos, rows in per_position_top.items()
		},
		"per_position_rank_stats": {
			str(pos): {
				"target_topk_index_histogram": {
					str(k): int(v)
					for k, v in sorted(target_topk_index_by_position[pos].items(), key=lambda x: x[0])
				},
				"target_rank_positive_histogram": {
					str(k): int(v)
					for k, v in sorted(target_rank_pos_by_position[pos].items(), key=lambda x: x[0])
				},
				"target_rank_negative_histogram": {
					str(k): int(v)
					for k, v in sorted(target_rank_neg_by_position[pos].items(), key=lambda x: x[0])
				},
				"rank_delta_histogram": {
					str(k): int(v)
					for k, v in sorted(rank_delta_by_position[pos].items(), key=lambda x: x[0])
				},
				"comparison": {
					"count": int(rank_compare_by_position[pos]["total"]),
					"positive_better_count": int(rank_compare_by_position[pos]["pos_better"]),
					"negative_better_count": int(rank_compare_by_position[pos]["neg_better"]),
					"tie_count": int(rank_compare_by_position[pos]["tie"]),
					"positive_better_rate": _rate(
						rank_compare_by_position[pos]["pos_better"],
						rank_compare_by_position[pos]["total"],
					),
					"negative_better_rate": _rate(
						rank_compare_by_position[pos]["neg_better"],
						rank_compare_by_position[pos]["total"],
					),
					"tie_rate": _rate(
						rank_compare_by_position[pos]["tie"],
						rank_compare_by_position[pos]["total"],
					),
					"rank_delta_mean": _mean_from_hist(rank_delta_by_position[pos]),
				},
			}
			for pos in sorted(target_topk_index_by_position.keys())
		},
		"rejected_pos_top1_by_position": {
			str(pos): {
				"rejected_in_topk_count": int(rejected_pos_in_topk[pos]),
				"rejected_top1_count": int(rejected_pos_top1[pos]),
				"rejected_top1_rate": _rate(rejected_pos_top1[pos], rejected_pos_in_topk[pos]),
			}
			for pos in sorted(rejected_pos_in_topk.keys())
		},
	}


def _print_report(stats: Dict[str, Any], top_n: int, per_pos_top_n: int, show_positions: int) -> None:
	def _print_split_section(title: str, section: Dict[str, Any]) -> None:
		print(f"\n{title}:")
		print(f"  Rank compared positions        : {section['rank_compared_count']}")
		print(f"  Target in top-k                : {section['target_in_topk']}")
		print(f"  Target not in top-k            : {section['target_not_in_topk']}")
		print(f"  Target in top-k rate           : {section['target_in_topk_rate']:.6f}")
		print(f"  Mean positive rank (1=best)    : {section['target_rank_positive_mean']}")
		print(f"  Mean negative rank (1=best)    : {section['target_rank_negative_mean']}")
		print(f"  Mean rank delta (neg-pos)      : {section['rank_delta_mean']}")
		print(f"  Positive top-1 rate            : {section['target_rank_positive_top1_rate']:.6f}")
		print(f"  Negative top-1 rate            : {section['target_rank_negative_top1_rate']:.6f}")
		print(f"  Positive better rate           : {section['positive_better_rate']:.6f}")
		print(f"  Negative better rate           : {section['negative_better_rate']:.6f}")
		print(f"  Tie rate                       : {section['tie_rate']:.6f}")
		print(f"  Target rank positive histogram : {section['target_rank_positive_histogram']}")
		print(f"  Target rank negative histogram : {section['target_rank_negative_histogram']}")
		print(f"  Rank delta histogram           : {section['rank_delta_histogram']}")

	print("=" * 80)
	print("Target Token Analysis")
	print("=" * 80)
	print(f"Input dir                         : {stats['input_dir']}")
	print(f"Shards loaded                      : {stats['num_shards']}")
	print(f"Records loaded                     : {stats['num_records']}")
	print(
		f"Records missing target_token_id    : {stats['num_records_missing_target_token_id']}"
	)
	print(
		f"Records missing acceptance_length  : {stats['num_records_missing_acceptance_length']}"
	)
	print(
		f"Records missing topk/logits        : {stats['num_records_missing_topk_or_logits']}"
	)
	print(f"Total target-token observations    : {stats['total_target_tokens']}")
	print(f"Unique target tokens               : {stats['unique_target_tokens']}")

	rank_stats = stats["rank_stats"]
	print("\nRank stats (target token inside top-k candidate set):")
	print(f"  Rankable positions               : {rank_stats['total_rankable_positions']}")
	print(f"  Target in top-k                  : {rank_stats['target_in_topk']}")
	print(f"  Target not in top-k              : {rank_stats['target_not_in_topk']}")
	print(f"  Target in top-k rate             : {rank_stats['target_in_topk_rate']:.6f}")
	print(f"  Mean top-k index (0-based)       : {rank_stats['target_topk_index_mean']}")
	print(f"  Mean positive rank (1=best)      : {rank_stats['target_rank_positive_mean']}")
	print(f"  Mean negative rank (1=best)      : {rank_stats['target_rank_negative_mean']}")
	print(f"  Mean rank delta (neg-pos)        : {rank_stats['rank_delta_mean']}")
	print(f"  Compared positions               : {rank_stats['rank_compared_count']}")
	print(f"  Positive better count            : {rank_stats['positive_better_count']}")
	print(f"  Negative better count            : {rank_stats['negative_better_count']}")
	print(f"  Tie count                        : {rank_stats['tie_count']}")
	print(f"  Positive better rate             : {rank_stats['positive_better_rate']:.6f}")
	print(f"  Negative better rate             : {rank_stats['negative_better_rate']:.6f}")
	print(f"  Tie rate                         : {rank_stats['tie_rate']:.6f}")

	_print_split_section("Accepted part (pos < acceptance_length)", rank_stats["accepted_part"])
	_print_split_section("Rejected part (pos >= acceptance_length)", rank_stats["rejected_part"])

	print(f"\nTop {per_pos_top_n} target top-k index (0-based):")
	for row in rank_stats["target_topk_index_top"]:
		print(f"  {row['topk_index']:>8}  {row['count']:>10}  {row['ratio']:.6f}")

	print(f"\nTop {per_pos_top_n} target rank in positive logits:")
	for row in rank_stats["target_rank_positive_top"]:
		print(f"  {row['rank']:>8}  {row['count']:>10}  {row['ratio']:.6f}")

	print(f"\nTop {per_pos_top_n} target rank in negative logits:")
	for row in rank_stats["target_rank_negative_top"]:
		print(f"  {row['rank']:>8}  {row['count']:>10}  {row['ratio']:.6f}")

	print(f"\nTop {per_pos_top_n} rank delta (neg - pos):")
	for row in rank_stats["rank_delta_top"]:
		print(f"  {row['delta']:>8}  {row['count']:>10}  {row['ratio']:.6f}")

	rejected_pos_data = stats.get("rejected_pos_top1_by_position", {})
	if rejected_pos_data:
		print("\nRejected top-1 rate by position (only rejected tokens):")
		print(f"  {'Pos':>4}  {'Rejected':>10}  {'Top-1':>8}  {'Top-1%':>8}")
		for pos in sorted(rejected_pos_data.keys(), key=int):
			d = rejected_pos_data[pos]
			print(f"  {pos:>4}  {d['rejected_in_topk_count']:>10}  {d['rejected_top1_count']:>8}  {d['rejected_top1_rate']:>7.4f}")

	print("\nTarget length distribution (length -> blocks):")
	for length, count in stats["target_length_distribution"].items():
		print(f"  {length:>3} -> {count}")

	print("\nAcceptance length distribution (acceptance_length -> blocks):")
	for length, count in stats["acceptance_length_distribution"].items():
		print(f"  {length:>3} -> {count}")

	print(f"\nTop {top_n} target tokens overall (token_id, count, ratio):")
	for row in stats["overall_top_tokens"]:
		print(f"  {row['token_id']:>8}  {row['count']:>10}  {row['ratio']:.6f}")

	print(
		f"\nTop {per_pos_top_n} tokens by position (show first {show_positions} positions):"
	)
	for pos in range(show_positions):
		pos_key = str(pos)
		rows = stats["per_position_top_tokens"].get(pos_key, [])
		if not rows:
			continue
		print(f"  Position {pos}:")
		for row in rows:
			print(f"    {row['token_id']:>8}  {row['count']:>10}  {row['ratio']:.6f}")


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Analyze target token statistics from collected alpha record shards."
	)
	parser.add_argument(
		"--input-dir",
		type=str,
		default="alpha_model/collected_alpha_records_darft",
		help="Directory containing .pt shards (searched recursively).",
	)
	parser.add_argument(
		"--top-n",
		type=int,
		default=30,
		help="Top-N most frequent target tokens to report globally.",
	)
	parser.add_argument(
		"--per-pos-top-n",
		type=int,
		default=10,
		help="Top-N most frequent target tokens per position.",
	)
	parser.add_argument(
		"--show-positions",
		type=int,
		default=15,
		help="How many first positions to print in terminal report.",
	)
	parser.add_argument(
		"--max-shards",
		type=int,
		default=None,
		help="If set, only analyze the first N shards (for quick inspection).",
	)
	parser.add_argument(
		"--output-json",
		type=str,
		default=None,
		help="Optional path to save full analysis result as JSON.",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	input_dir = Path(args.input_dir)
	stats = analyze_target_tokens(
		input_dir=input_dir,
		top_n=args.top_n,
		per_pos_top_n=args.per_pos_top_n,
		max_shards=args.max_shards,
	)
	_print_report(
		stats,
		top_n=args.top_n,
		per_pos_top_n=args.per_pos_top_n,
		show_positions=max(0, args.show_positions),
	)

	if args.output_json:
		output_path = Path(args.output_json)
		output_path.parent.mkdir(parents=True, exist_ok=True)
		with output_path.open("w", encoding="utf-8") as f:
			json.dump(stats, f, ensure_ascii=False, indent=2)
		print(f"\nSaved JSON report to: {output_path}")


if __name__ == "__main__":
	main()
