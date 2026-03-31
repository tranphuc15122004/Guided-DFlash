# VCD Prod Diagnostics Report

## Executive Summary
- Used 5000 step records for analysis (total collected: 6074).
- Mean acceptance length (positive/negative/final): 5.306 / 0.831 / 5.286.
- Fraction of steps with positive > negative: 78.70%.
- Fraction of steps with final <= positive: 97.32%.
- KL divergence (avg/min/max): 2.80589 / 0.00017 / 18.25000.

## Interpretation
Final VCD branch is not better than the positive branch for most steps, so contrastive mixing is not translating into longer accepted spans.

## Artifacts
- `prod_curve_mean.png`
- `accept_len_branch_hist.png`
- `accept_len_gap_vs_kl.png`
- `prod_summary.json`
- `prod_step_records.jsonl` (optional, based on CLI flag)
