# Draft Output Probability Shape

## Overall
- Number of draft-token records: **90870**
- Entropy logits source: **negative**
- Decoding speedup (bs=1 vs bs=block): **3.45x**
- Average acceptance length: **6.49**
- Analyzed argmax hit rate (context): **18.57%**
- Accepted-by-target rate (context): **35.51%**

## Core Distribution Stats
- Top-1 probability mean / median / p90: **0.4775 / 0.4136 / 0.9422**
- Normalized entropy mean / median / p90: **0.1728 / 0.1725 / 0.3122**
- Effective support mean / median / p90: **16.6545 / 7.8339 / 41.4602**

## Top-1 Probability Bands
| Band | Count | Rate |
|---|---:|---:|
| very_sharp_[0.9,1.0] | 11739 | 12.92% |
| sharp_[0.7,0.9) | 11103 | 12.22% |
| medium_[0.5,0.7) | 14228 | 15.66% |
| flat_[0.0,0.5) | 53800 | 59.21% |

## Normalized-Entropy Bands
| Band | Count | Rate |
|---|---:|---:|
| low_entropy_[0.0,0.4) | 90086 | 99.14% |
| mid_entropy_[0.4,0.7) | 784 | 0.86% |
| high_entropy_[0.7,1.0] | 0 | 0.00% |

## Position-wise Distribution (Within Block)
| block_pos | count | rate | top1 median | normalized-entropy median | effective-support median |
|---:|---:|---:|---:|---:|---:|
| 1 | 6058 | 6.67% | 0.5965 | 0.1131 | 3.854 |
| 2 | 6058 | 6.67% | 0.5951 | 0.1142 | 3.908 |
| 3 | 6058 | 6.67% | 0.5551 | 0.1271 | 4.558 |
| 4 | 6058 | 6.67% | 0.5139 | 0.1396 | 5.289 |
| 5 | 6058 | 6.67% | 0.4867 | 0.1487 | 5.898 |
| 6 | 6058 | 6.67% | 0.4591 | 0.1586 | 6.634 |
| 7 | 6058 | 6.67% | 0.4363 | 0.1642 | 7.096 |
| 8 | 6058 | 6.67% | 0.4110 | 0.1722 | 7.800 |
| 9 | 6058 | 6.67% | 0.3866 | 0.1801 | 8.578 |
| 10 | 6058 | 6.67% | 0.3689 | 0.1882 | 9.441 |
| 11 | 6058 | 6.67% | 0.3482 | 0.1972 | 10.514 |
| 12 | 6058 | 6.67% | 0.3213 | 0.2080 | 11.962 |
| 13 | 6058 | 6.67% | 0.3013 | 0.2190 | 13.638 |
| 14 | 6058 | 6.67% | 0.2801 | 0.2283 | 15.241 |
| 15 | 6058 | 6.67% | 0.2531 | 0.2390 | 17.308 |

## Mean Histogram of Full-Vocabulary Probabilities (Positions 1-15)
| block_pos | distributions | vocab_values | mass in first bin | mass in last bin |
|---:|---:|---:|---:|---:|
| 1 | 6058 | 920428288 | 99.9974% | 0.0001% |
| 2 | 6058 | 920428288 | 99.9973% | 0.0001% |
| 3 | 6058 | 920428288 | 99.9971% | 0.0001% |
| 4 | 6058 | 920428288 | 99.9969% | 0.0001% |
| 5 | 6058 | 920428288 | 99.9968% | 0.0001% |
| 6 | 6058 | 920428288 | 99.9966% | 0.0001% |
| 7 | 6058 | 920428288 | 99.9965% | 0.0001% |
| 8 | 6058 | 920428288 | 99.9963% | 0.0000% |
| 9 | 6058 | 920428288 | 99.9962% | 0.0000% |
| 10 | 6058 | 920428288 | 99.9961% | 0.0000% |
| 11 | 6058 | 920428288 | 99.9959% | 0.0000% |
| 12 | 6058 | 920428288 | 99.9957% | 0.0000% |
| 13 | 6058 | 920428288 | 99.9956% | 0.0000% |
| 14 | 6058 | 920428288 | 99.9954% | 0.0000% |
| 15 | 6058 | 920428288 | 99.9952% | 0.0000% |

Detailed per-bin values are saved in `draft_vocab_probability_hist_by_position.csv` and `draft_entropy_summary.json`.

## Interpreting Sharp vs Flat
- More sharp: higher top1 probability, lower normalized entropy, lower effective support.
- More flat: lower top1 probability, higher normalized entropy, higher effective support.
- Quick ratio (top1>=0.7 vs top1<0.5): **25.14% vs 59.21%**
