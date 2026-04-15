# Draft Output Probability Shape

## Overall
- Number of draft-token records: **89265**
- Decoding speedup (bs=1 vs bs=block): **3.64x**
- Average acceptance length: **6.51**
- Positive argmax hit rate (context): **57.85%**
- Accepted-by-target rate (context): **35.76%**

## Core Distribution Stats
- Top-1 probability mean / median / p90: **0.6510 / 0.6775 / 0.9997**
- Normalized entropy mean / median / p90: **0.1077 / 0.0850 / 0.2582**
- Effective support mean / median / p90: **7.9588 / 2.7559 / 21.7771**

## Top-1 Probability Bands
| Band | Count | Rate |
|---|---:|---:|
| very_sharp_[0.9,1.0] | 30703 | 34.40% |
| sharp_[0.7,0.9) | 12555 | 14.06% |
| medium_[0.5,0.7) | 13805 | 15.47% |
| flat_[0.0,0.5) | 32202 | 36.07% |

## Normalized-Entropy Bands
| Band | Count | Rate |
|---|---:|---:|
| low_entropy_[0.0,0.4) | 89221 | 99.95% |
| mid_entropy_[0.4,0.7) | 44 | 0.05% |
| high_entropy_[0.7,1.0] | 0 | 0.00% |

## Position-wise Distribution (Within Block)
| block_pos | count | rate | top1 median | normalized-entropy median | effective-support median |
|---:|---:|---:|---:|---:|---:|
| 1 | 5951 | 6.67% | 0.9857 | 0.0078 | 1.097 |
| 2 | 5951 | 6.67% | 0.9513 | 0.0214 | 1.291 |
| 3 | 5951 | 6.67% | 0.8965 | 0.0386 | 1.585 |
| 4 | 5951 | 6.67% | 0.8478 | 0.0518 | 1.855 |
| 5 | 5951 | 6.67% | 0.7826 | 0.0637 | 2.138 |
| 6 | 5951 | 6.67% | 0.7233 | 0.0753 | 2.454 |
| 7 | 5951 | 6.67% | 0.6970 | 0.0810 | 2.629 |
| 8 | 5951 | 6.67% | 0.6580 | 0.0907 | 2.950 |
| 9 | 5951 | 6.67% | 0.6089 | 0.1020 | 3.376 |
| 10 | 5951 | 6.67% | 0.5732 | 0.1132 | 3.859 |
| 11 | 5951 | 6.67% | 0.5384 | 0.1243 | 4.405 |
| 12 | 5951 | 6.67% | 0.5034 | 0.1350 | 5.004 |
| 13 | 5951 | 6.67% | 0.4677 | 0.1464 | 5.736 |
| 14 | 5951 | 6.67% | 0.4392 | 0.1575 | 6.546 |
| 15 | 5951 | 6.67% | 0.3907 | 0.1750 | 8.073 |

## Mean Histogram of Full-Vocabulary Probabilities (Positions 1-15)
| block_pos | distributions | vocab_values | mass in first bin | mass in last bin |
|---:|---:|---:|---:|---:|
| 1 | 5951 | 904171136 | 99.9988% | 0.0004% |
| 2 | 5951 | 904171136 | 99.9985% | 0.0003% |
| 3 | 5951 | 904171136 | 99.9983% | 0.0002% |
| 4 | 5951 | 904171136 | 99.9980% | 0.0002% |
| 5 | 5951 | 904171136 | 99.9978% | 0.0002% |
| 6 | 5951 | 904171136 | 99.9977% | 0.0002% |
| 7 | 5951 | 904171136 | 99.9975% | 0.0002% |
| 8 | 5951 | 904171136 | 99.9974% | 0.0002% |
| 9 | 5951 | 904171136 | 99.9972% | 0.0001% |
| 10 | 5951 | 904171136 | 99.9971% | 0.0001% |
| 11 | 5951 | 904171136 | 99.9969% | 0.0001% |
| 12 | 5951 | 904171136 | 99.9967% | 0.0001% |
| 13 | 5951 | 904171136 | 99.9965% | 0.0001% |
| 14 | 5951 | 904171136 | 99.9964% | 0.0001% |
| 15 | 5951 | 904171136 | 99.9961% | 0.0001% |

Detailed per-bin values are saved in `draft_vocab_probability_hist_by_position.csv` and `draft_entropy_summary.json`.

## Interpreting Sharp vs Flat
- More sharp: higher top1 probability, lower normalized entropy, lower effective support.
- More flat: lower top1 probability, higher normalized entropy, higher effective support.
- Quick ratio (top1>=0.7 vs top1<0.5): **48.46% vs 36.07%**
