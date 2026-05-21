"""
Phân tích đối chiếu phương pháp AR predict: So sánh token do AR rerun dự đoán
(target_token_id) với token thực tế được commit (committed_token_id) và với
các dự đoán khác (draft argmax, posterior argmax).

CHẠY NGOẠI TUYẾN: Script này chỉ phân tích dữ liệu JSONL đã thu thập.
KHÔNG cần GPU và KHÔNG chạy lại model.
"""

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_jsonl_records(path: Path) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def has_committed_fields(records: list[dict]) -> bool:
    return len(records) > 0 and "committed_token_id" in records[0]


def compute_stats(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "std": 0.0,
                "p10": 0.0, "p90": 0.0}
    n = len(values)
    sv = sorted(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": n,
        "mean": round(mean, 4),
        "median": round(sv[n // 2], 4),
        "min": round(min(values), 4),
        "max": round(max(values), 4),
        "std": round(var ** 0.5, 4),
        "p10": round(sv[max(0, int(n * 0.1))], 4),
        "p90": round(sv[min(n - 1, int(n * 0.9))], 4),
    }


def topk_rate(values: list[int], k: int) -> float:
    if not values:
        return 0.0
    return sum(1 for v in values if v <= k) / len(values) * 100


def print_sep(title: str = "") -> None:
    print()
    print("=" * 78)
    if title:
        print(f"  {title}")
        print("=" * 78)


def analyze(ar_records: list[dict], rej_records: list[dict], result_dir: Path) -> dict:
    has_committed = has_committed_fields(ar_records)
    n_ar = len(ar_records)
    n_rej = len(rej_records)
    print(f"AR target rank records: {n_ar}")
    print(f"Reject records:         {n_rej}")
    print(f"Has committed_token_id: {has_committed}")

    # ──────────────────────────────────────────────
    # 1. PHÂN LOẠI THEO TOKEN SOURCE
    # ──────────────────────────────────────────────
    print_sep("1. PHÂN LOẠI THEO TOKEN SOURCE")

    verify = [r for r in ar_records if r.get("token_source") == "verify"]
    ar_suffix = [r for r in ar_records if r.get("token_source") == "ar_suffix"]

    print(f"\n  Verify tokens:   {len(verify):>6} ({len(verify)/max(n_ar,1)*100:.2f}%)")
    print(f"  AR suffix tokens: {len(ar_suffix):>6} ({len(ar_suffix)/max(n_ar,1)*100:.2f}%)")

    def rank_summary(records, label):
        ranks = [r["positive_target_rank"] for r in records]
        stats = compute_stats(ranks)
        print(f"\n  --- {label} (n={len(records)}) ---")
        print(f"  Rank: mean={stats['mean']:.1f}, median={stats['median']:.0f}, "
              f"p10={stats['p10']:.0f}, p90={stats['p90']:.0f}")
        print(f"  Top-1:  {topk_rate(ranks, 1):.2f}%")
        print(f"  Top-5:  {topk_rate(ranks, 5):.2f}%")
        print(f"  Top-10: {topk_rate(ranks, 10):.2f}%")
        print(f"  Top-50: {topk_rate(ranks, 50):.2f}%")

    rank_summary(verify, "VERIFY tokens (target rank in draft logits)")
    rank_summary(ar_suffix, "AR SUFFIX tokens (target rank in draft logits)")

    # ──────────────────────────────────────────────
    # 2. PHÂN TÍCH THEO BLOCK POSITION
    # ──────────────────────────────────────────────
    print_sep("2. PHÂN TÍCH THEO BLOCK POSITION")

    pos_data = defaultdict(lambda: {"verify": [], "ar_suffix": []})
    for r in ar_records:
        bp = r["block_position"]
        src = r.get("token_source", "unknown")
        pos_data[bp][src].append(r["positive_target_rank"])

    print(f"\n  {'Pos':>4} | {'#Verify':>7} {'Top1_V':>7} {'Rank_V':>7} | "
          f"{'#AR':>7} {'Top1_AR':>8} {'Rank_AR':>7} | {'V/AR':>6}")
    print(f"  " + "-" * 72)
    for bp in sorted(pos_data.keys()):
        v = pos_data[bp]["verify"]
        a = pos_data[bp]["ar_suffix"]
        t1v = topk_rate(v, 1) if v else 0.0
        mv = compute_stats(v)["mean"] if v else 0.0
        t1a = topk_rate(a, 1) if a else 0.0
        ma = compute_stats(a)["mean"] if a else 0.0
        ratio = ma / max(mv, 0.001)
        print(f"  {bp:>4} | {len(v):>7} {t1v:>6.2f}% {mv:>7.1f} | "
              f"{len(a):>7} {t1a:>7.2f}% {ma:>7.1f} | {ratio:>5.2f}x")

    # ──────────────────────────────────────────────
    # 3. VERIFY RATE THEO BLOCK POSITION
    # ──────────────────────────────────────────────
    print_sep("3. VERIFY RATE THEO BLOCK POSITION")

    print(f"\n  {'Pos':>4} | {'Total':>6} {'Verify':>7} {'AR':>7} {'Verify%':>8} {'Rank_avg':>8} {'Rank_p50':>8}")
    print(f"  " + "-" * 55)
    for bp in sorted(pos_data.keys()):
        v = pos_data[bp]["verify"]
        a = pos_data[bp]["ar_suffix"]
        total = len(v) + len(a)
        all_ranks = v + a
        rs = compute_stats(all_ranks)
        print(f"  {bp:>4} | {total:>6} {len(v):>7} {len(a):>7} "
              f"{len(v)/max(total,1)*100:>7.2f}% {rs['mean']:>8.1f} {rs['median']:>8.0f}")

    # ──────────────────────────────────────────────
    # 4. PHÂN TÍCH SEED TOKEN (VỊ TRÍ ĐẦU TIÊN SAU ACCEPTANCE)
    # ──────────────────────────────────────────────
    print_sep("4. PHÂN TÍCH SEED TOKEN (VỊ TRÍ acceptance_length + 1)")

    seed_ranks = []
    for r in ar_records:
        if r["block_position"] == r["acceptance_length"] + 1:
            seed_ranks.append(r["positive_target_rank"])

    if seed_ranks:
        rs = compute_stats(seed_ranks)
        print(f"\n  Seed token positions: {len(seed_ranks)}")
        print(f"  Rank: mean={rs['mean']:.1f}, median={rs['median']:.0f}, "
              f"p10={rs['p10']:.0f}, p90={rs['p90']:.0f}")
        print(f"  Top-1:  {topk_rate(seed_ranks, 1):.2f}%")
        print(f"  Top-5:  {topk_rate(seed_ranks, 5):.2f}%")
        print(f"  Top-10: {topk_rate(seed_ranks, 10):.2f}%")
    else:
        print("\n  (Không có dữ liệu seed token)")

    # ──────────────────────────────────────────────
    # 5. PHÂN TÍCH AR PREDICTION VS COMMITTED TOKEN
    # ──────────────────────────────────────────────
    print_sep("5. AR PREDICTION vs COMMITTED TOKEN")

    if has_committed:
        same_count = sum(1 for r in ar_records if r["target_token_id"] == r["committed_token_id"])
        diff_count = n_ar - same_count
        print(f"\n  target_token == committed_token: {same_count:>6} / {n_ar} "
              f"({same_count/max(n_ar,1)*100:.2f}%)")
        print(f"  target_token != committed_token: {diff_count:>6} / {n_ar} "
              f"({diff_count/max(n_ar,1)*100:.2f}%)")

        # Per source
        for src_name, src_records in [("verify", verify), ("ar_suffix", ar_suffix)]:
            src_same = sum(1 for r in src_records if r["target_token_id"] == r["committed_token_id"])
            src_n = len(src_records)
            print(f"\n  --- {src_name.upper()} ---")
            print(f"  target == committed: {src_same}/{src_n} "
                  f"({src_same/max(src_n,1)*100:.2f}%)")

            # When they differ: what's the rank of committed vs target?
            diff = [r for r in src_records if r["target_token_id"] != r["committed_token_id"]]
            if diff:
                target_r = [r["positive_target_rank"] for r in diff]
                committed_r = [r["committed_rank"] for r in diff]
                t_rs = compute_stats(target_r)
                c_rs = compute_stats(committed_r)
                print(f"  When differ (n={len(diff)}):")
                print(f"    Target rank:    mean={t_rs['mean']:.1f}, median={t_rs['median']:.0f}")
                print(f"    Committed rank: mean={c_rs['mean']:.1f}, median={c_rs['median']:.0f}")

                # Which is better?
                target_better = sum(1 for r in diff
                                    if r["positive_target_rank"] <= r["committed_rank"])
                print(f"    Target rank <= Committed rank: "
                      f"{target_better}/{len(diff)} ({target_better/len(diff)*100:.2f}%)")
    else:
        print("\n  Dữ liệu cũ không có committed_token_id.")
        print("  Dùng reject records để phân tích thay thế.")

    # ──────────────────────────────────────────────
    # 6. PHÂN TÍCH VỊ TRÍ REJECT (qua reject records)
    # ──────────────────────────────────────────────
    print_sep("6. BA CHIỀU: AR ARGMAX vs DRAFT ARGMAX vs POSTERIOR ARGMAX")

    if n_rej > 0:
        # Tại reject positions, so sánh:
        # - positive_pred_id = draft model's argmax
        # - target_token_id  = AR rerun's argmax (từ ar_records tại vị trí tương ứng)
        # - posterior_pred_id = target model's posterior argmax

        # Build a lookup from ar_records by (sample_idx, turn_idx, decode_step, absolute_position)
        ar_lookup: dict[tuple[int, int, int, int], dict] = {}
        for r in ar_records:
            key = (r["sample_idx"], r["turn_idx"], r["decode_step"], r["absolute_position"])
            ar_lookup[key] = r

        matched_ar = 0
        three_way_match = 0  # target == positive == posterior
        target_match_posterior = 0  # target == posterior
        positive_match_posterior = 0  # positive == posterior

        for rej in rej_records:
            key = (rej["sample_idx"], rej["turn_idx"], rej["decode_step"], rej["absolute_position"])
            ar_rec = ar_lookup.get(key)
            if ar_rec is None:
                continue
            matched_ar += 1
            target_id = ar_rec["target_token_id"]
            pos_id = rej["positive_pred_id"]
            post_id = rej["posterior_pred_id"]

            if target_id == pos_id == post_id:
                three_way_match += 1
            if target_id == post_id:
                target_match_posterior += 1
            if pos_id == post_id:
                positive_match_posterior += 1

        print(f"\n  Reject records matched with AR records: {matched_ar}/{n_rej}")
        if matched_ar > 0:
            print(f"  Three-way agreement (AR==Draft==Posterior): "
                  f"{three_way_match}/{matched_ar} ({three_way_match/matched_ar*100:.2f}%)")
            print(f"  AR argmax == Posterior argmax: "
                  f"{target_match_posterior}/{matched_ar} ({target_match_posterior/matched_ar*100:.2f}%)")
            print(f"  Draft argmax == Posterior argmax: "
                  f"{positive_match_posterior}/{matched_ar} ({positive_match_posterior/matched_ar*100:.2f}%)")

        # At reject positions, what is the AR token's rank?
        rej_ar_ranks = []
        for rej in rej_records:
            key = (rej["sample_idx"], rej["turn_idx"], rej["decode_step"], rej["absolute_position"])
            ar_rec = ar_lookup.get(key)
            if ar_rec is not None:
                rej_ar_ranks.append(ar_rec["positive_target_rank"])

        if rej_ar_ranks:
            rs = compute_stats(rej_ar_ranks)
            print(f"\n  --- AR argmax rank in draft logits at reject positions ---")
            print(f"  Count: {len(rej_ar_ranks)}")
            print(f"  Rank: mean={rs['mean']:.1f}, median={rs['median']:.0f}, "
                  f"p10={rs['p10']:.0f}, p90={rs['p90']:.0f}")
            print(f"  Top-1:  {topk_rate(rej_ar_ranks, 1):.2f}%")
            print(f"  Top-5:  {topk_rate(rej_ar_ranks, 5):.2f}%")
    else:
        print("\n  (Không có reject records để phân tích)")

    # ──────────────────────────────────────────────
    # 7. COUNTERFACTUAL: NẾU DÙNG AR PREDICTION THAY VÌ DRAFT?
    # ──────────────────────────────────────────────
    print_sep("7. COUNTERFACTUAL: DRAFT vs AR PREDICTION")

    if has_committed:
        # So sánh rank của committed token vs target token
        # Nếu target rank < committed rank -> AR predict tốt hơn draft sample
        better_count = sum(1 for r in ar_records
                           if r["positive_target_rank"] < r["committed_rank"])
        worse_count = sum(1 for r in ar_records
                          if r["positive_target_rank"] > r["committed_rank"])
        equal_count = sum(1 for r in ar_records
                          if r["positive_target_rank"] == r["committed_rank"])
        print(f"\n  AR target rank <  Committed rank (AR better): {better_count} "
              f"({better_count/max(n_ar,1)*100:.2f}%)")
        print(f"  AR target rank == Committed rank (equal):     {equal_count} "
              f"({equal_count/max(n_ar,1)*100:.2f}%)")
        print(f"  AR target rank >  Committed rank (AR worse):  {worse_count} "
              f"({worse_count/max(n_ar,1)*100:.2f}%)")

        # Per source
        for src_name, src_records in [("verify", verify), ("ar_suffix", ar_suffix)]:
            src_n = len(src_records)
            src_better = sum(1 for r in src_records
                             if r["positive_target_rank"] < r["committed_rank"])
            src_worse = sum(1 for r in src_records
                            if r["positive_target_rank"] > r["committed_rank"])
            src_equal = src_n - src_better - src_worse
            print(f"\n  --- {src_name.upper()} (n={src_n}) ---")
            print(f"    AR better: {src_better} ({src_better/max(src_n,1)*100:.2f}%)")
            print(f"    Equal:     {src_equal} ({src_equal/max(src_n,1)*100:.2f}%)")
            print(f"    AR worse:  {src_worse} ({src_worse/max(src_n,1)*100:.2f}%)")
    else:
        print("\n  (Cần committed_token_id để so sánh)")

    # ──────────────────────────────────────────────
    # 8. PHÂN TÍCH ERROR COMPOUNDING
    # ──────────────────────────────────────────────
    print_sep("8. PHÂN TÍCH ERROR COMPOUNDING (AR SUFFIX STREAKS)")

    blocks: dict[tuple[int, int, int, int], list[dict]] = defaultdict(list)
    for r in ar_records:
        key = (r["sample_idx"], r["turn_idx"], r["decode_step"], r["block_start_position"])
        blocks[key].append(r)

    streak_blocks = []
    for key, brecs in blocks.items():
        ar_sorted = sorted(
            [r for r in brecs if r.get("token_source") == "ar_suffix"],
            key=lambda x: x["block_position"]
        )
        if len(ar_sorted) >= 2:
            streak_blocks.append(ar_sorted)

    print(f"\n  Blocks with >=2 consecutive AR suffix positions: {len(streak_blocks)}")

    if streak_blocks:
        monotonic_up = 0
        monotonic_down = 0
        fluctuating = 0
        for suf in streak_blocks:
            ranks = [r["positive_target_rank"] for r in suf]
            inc = all(ranks[i] <= ranks[i + 1] for i in range(len(ranks) - 1))
            dec = all(ranks[i] >= ranks[i + 1] for i in range(len(ranks) - 1))
            if inc:
                monotonic_up += 1
            elif dec:
                monotonic_down += 1
            else:
                fluctuating += 1

        total = len(streak_blocks)
        print(f"  Monotonically increasing (error compounding): "
              f"{monotonic_up} ({monotonic_up/total*100:.1f}%)")
        print(f"  Monotonically decreasing: "
              f"{monotonic_down} ({monotonic_down/total*100:.1f}%)")
        print(f"  Fluctuating: "
              f"{fluctuating} ({fluctuating/total*100:.1f}%)")

        # Per-position rank change
        rank_deltas = defaultdict(list)
        for suf in streak_blocks:
            for i in range(len(suf) - 1):
                delta = suf[i + 1]["positive_target_rank"] - suf[i]["positive_target_rank"]
                rank_deltas[f"{suf[i]['block_position']}->{suf[i+1]['block_position']}"].append(delta)

        print(f"\n  --- Rank delta between consecutive AR suffix positions ---")
        for transition in sorted(rank_deltas.keys()):
            deltas = rank_deltas[transition]
            mean_delta = sum(deltas) / len(deltas)
            neg_pct = sum(1 for d in deltas if d < 0) / len(deltas) * 100
            pos_pct = sum(1 for d in deltas if d > 0) / len(deltas) * 100
            print(f"  {transition}: mean_delta={mean_delta:+.1f}, "
                  f"improve={neg_pct:.0f}%, degrade={pos_pct:.0f}% (n={len(deltas)})")

    # ──────────────────────────────────────────────
    # 9. PHÂN TÍCH PHỐI HỢP: AR RANK vs ACCEPTANCE LENGTH
    # ──────────────────────────────────────────────
    print_sep("9. AR TARGET RANK vs ACCEPTANCE LENGTH")

    # Gom theo acceptance length
    acc_buckets = defaultdict(list)
    for r in ar_records:
        al = r["acceptance_length"]
        acc_buckets[al].append(r["positive_target_rank"])

    print(f"\n  {'AccLen':>7} | {'Count':>6} | {'Rank_mean':>9} {'Rank_med':>8} {'Top1%':>6}")
    print(f"  " + "-" * 46)
    for al in sorted(acc_buckets.keys()):
        ranks = acc_buckets[al]
        rs = compute_stats(ranks)
        print(f"  {al:>7} | {len(ranks):>6} | {rs['mean']:>9.1f} {rs['median']:>8.0f} "
              f"{topk_rate(ranks,1):>5.1f}%")

    # ──────────────────────────────────────────────
    # 10. PHÂN TÍCH PHỐI HỢP: AR RANK vs BLOCK START POSITION (xem xu hướng dài hạn)
    # ──────────────────────────────────────────────
    print_sep("10. XU HƯỚNG DÀI HẠN (THEO DECODE STEP)")

    decode_buckets = defaultdict(list)
    for r in ar_records:
        ds = r["decode_step"]
        decode_buckets[ds].append(r["positive_target_rank"])

    print(f"\n  {'Decode':>6} | {'Count':>6} | {'Rank_mean':>9} {'Top1%':>6} {'Verify%':>8}")
    print(f"  " + "-" * 42)
    for ds in sorted(decode_buckets.keys()):
        recs = decode_buckets[ds]
        ranks = recs
        verify_count = sum(1 for r in ar_records if r["decode_step"] == ds
                           and r.get("token_source") == "verify")
        rs = compute_stats(ranks)
        total = len(ranks)
        print(f"  {ds:>6} | {total:>6} | {rs['mean']:>9.1f} "
              f"{topk_rate(ranks,1):>5.1f}% "
              f"{verify_count/max(total,1)*100:>7.1f}%")

    # ──────────────────────────────────────────────
    # 11. LƯU KẾT QUẢ
    # ──────────────────────────────────────────────
    output = {
        "summary": {
            "ar_records": n_ar,
            "reject_records": n_rej,
            "has_committed_token_id": has_committed,
            "verify_count": len(verify),
            "ar_suffix_count": len(ar_suffix),
        },
        "verify_rank_stats": _rank_stats_dict(verify),
        "ar_suffix_rank_stats": _rank_stats_dict(ar_suffix),
        "seed_token_rank_stats": _rank_stats_dict(
            [r for r in ar_records if r["block_position"] == r["acceptance_length"] + 1]
        ),
        "block_position_profile": _build_position_profile(pos_data),
    }

    if has_committed:
        output["target_vs_committed"] = {
            "same": same_count,
            "diff": diff_count,
            "same_rate_pct": round(same_count / max(n_ar, 1) * 100, 2),
        }
        output["counterfactual_ar_vs_draft"] = {
            "ar_better": better_count,
            "equal": equal_count,
            "ar_worse": worse_count,
            "ar_better_rate_pct": round(better_count / max(n_ar, 1) * 100, 2),
        }

    json_path = result_dir / "ar_prediction_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n\n  Kết quả lưu tại: {json_path}")

    return output


def _rank_stats_dict(records: list[dict]) -> dict:
    ranks = [r["positive_target_rank"] for r in records]
    if not ranks:
        return {"count": 0, "mean": 0.0, "median": 0.0, "top1_rate_pct": 0.0,
                "top5_rate_pct": 0.0, "top10_rate_pct": 0.0}
    rs = compute_stats(ranks)
    return {
        "count": len(ranks),
        "mean": rs["mean"],
        "median": rs["median"],
        "p10": rs["p10"],
        "p90": rs["p90"],
        "top1_rate_pct": round(topk_rate(ranks, 1), 2),
        "top5_rate_pct": round(topk_rate(ranks, 5), 2),
        "top10_rate_pct": round(topk_rate(ranks, 10), 2),
    }


def _build_position_profile(pos_data: dict) -> list[dict]:
    profile = []
    for bp in sorted(pos_data.keys()):
        v = pos_data[bp]["verify"]
        a = pos_data[bp]["ar_suffix"]
        all_r = v + a
        rs = compute_stats(all_r)
        profile.append({
            "block_position": bp,
            "total_count": len(v) + len(a),
            "verify_count": len(v),
            "ar_suffix_count": len(a),
            "verify_rate_pct": round(len(v) / max(len(v) + len(a), 1) * 100, 2),
            "rank_mean": rs["mean"],
            "rank_median": rs["median"],
            "top1_rate_pct": round(topk_rate(all_r, 1), 2),
            "top5_rate_pct": round(topk_rate(all_r, 5), 2),
            "top10_rate_pct": round(topk_rate(all_r, 10), 2),
        })
    return profile


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phân tích đối chiếu kết quả phương pháp AR predict"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="analysis/results/tok_level_ana_168697611.gadi-pbs",
        help="Thư mục chứa kết quả phân tích (có ar_target_rank_records.jsonl, reject_records.jsonl)",
    )
    parser.add_argument(
        "--ar-file",
        type=str,
        default=None,
        help="Đường dẫn đến ar_target_rank_records.jsonl (ghi đè --result-dir)",
    )
    parser.add_argument(
        "--reject-file",
        type=str,
        default=None,
        help="Đường dẫn đến reject_records.jsonl (ghi đè --result-dir)",
    )
    args = parser.parse_args()

    result_dir = Path(args.result_dir)

    ar_path = Path(args.ar_file) if args.ar_file else result_dir / "ar_target_rank_records.jsonl"
    reject_path = Path(args.reject_file) if args.reject_file else result_dir / "reject_records.jsonl"

    if not ar_path.exists():
        # Fallback to CSV
        ar_csv = ar_path.with_suffix(".csv")
        if ar_csv.exists():
            print(f"JSONL không tồn tại, đọc CSV: {ar_csv}")
            ar_records = load_jsonl_records(ar_csv)  # This won't work, need CSV reader
            print("Lỗi: CSV chưa được hỗ trợ. Vui lòng cung cấp file JSONL.")
            sys.exit(1)
        else:
            print(f"Không tìm thấy: {ar_path}")
            sys.exit(1)

    if not reject_path.exists():
        reject_path = result_dir / "reject_records.jsonl"
        if not reject_path.exists():
            print(f"Warning: Không tìm thấy reject records tại {reject_path}")
            rej_records = []
        else:
            rej_records = load_jsonl_records(reject_path)
    else:
        rej_records = load_jsonl_records(reject_path)

    ar_records = load_jsonl_records(ar_path)

    if not ar_records:
        print("Không có dữ liệu AR records để phân tích.")
        sys.exit(1)

    analyze(ar_records, rej_records, result_dir)


if __name__ == "__main__":
    main()
