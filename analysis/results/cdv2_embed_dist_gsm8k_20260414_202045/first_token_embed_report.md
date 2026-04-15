# CDv2 First-Token Embedding Distance Analysis

## Case Definitions
- `improved_push_to_top1`: positive miss, contrastive hits target top-1.
- `worsened_drop_from_top1`: positive hits target top-1, contrastive drops it.
- `worsened_not_push_to_top1`: both positive and contrastive do not put target at top-1.
- `kept_top1`: both positive and contrastive keep target at top-1.

## Overall
- Total token-level records: **89400**
- First-token same-id rate (negative token equals positive token): **0.0000%**
- Target in candidate-mask rate: **100.00%**
- Final accepted-prefix rate at position: **35.71%**
- Corr(L2, delta_target_logprob): **+0.0241**
- Corr(L2, delta_target_rank): **-0.0036**

## Focus Case Stats
| Case | Count | Rate | L2 mean | L2 median | L2 p90 | CosDist mean | Delta logprob mean | Delta rank mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved: target pushed to top-1 | 0 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Worsened: target dropped from top-1 | 0 | 0.00% | 0.0000 | 0.0000 | 0.0000 | 0.0000 | +0.0000 | +0.0000 |
| Worsened: target still not top-1 | 37742 | 42.22% | 1.5413 | 1.5483 | 1.7002 | 1.0321 | -8.5139 | -10.3653 |

## Requested 3-Case Embedding Distance Stats
| Case | Count | Rate | L2 mean | L2 median | L2 p90 | CosDist mean | CosDist median | CosDist p90 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Improved: target pushed to top-1 | 0 | 0.00% | n/a | n/a | n/a | n/a | n/a | n/a |
| Worsened: target dropped from top-1 | 0 | 0.00% | n/a | n/a | n/a | n/a | n/a | n/a |
| Worsened: target still not top-1 | 37742 | 42.22% | 1.5413 | 1.5483 | 1.7002 | 1.0321 | 1.0152 | 1.1528 |

## Not-Push Subtypes
| Subtype | Count | Rate within not-push |
|---|---:|---:|
| rank_unchanged_not_top1 | 25998 | 68.88% |
| rank_worse_not_top1 | 7859 | 20.82% |
| rank_improved_but_not_top1 | 3885 | 10.29% |

## Distance Bin Analysis
| Bin | L2 range | Count | Rate | Improve | Drop | Not-push | Keep-top1 | Mean d_logprob | Mean d_rank |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | [0.8800, 1.4405] | 14895 | 16.66% | 0.00% | 0.00% | 44.22% | 55.78% | -3.7551 | -4.2050 |
| 1 | [1.4405, 1.5055] | 14895 | 16.66% | 0.00% | 0.00% | 44.40% | 55.60% | -3.7605 | -3.4386 |
| 2 | [1.5055, 1.5541] | 14910 | 16.68% | 0.00% | 0.00% | 42.70% | 57.30% | -3.6877 | -2.6860 |
| 3 | [1.5541, 1.6048] | 14895 | 16.66% | 0.00% | 0.00% | 42.43% | 57.57% | -3.6286 | -5.8250 |
| 4 | [1.6048, 1.6716] | 14895 | 16.66% | 0.00% | 0.00% | 40.57% | 59.43% | -3.3591 | -4.1650 |
| 5 | [1.6716, 1.9773] | 14910 | 16.68% | 0.00% | 0.00% | 38.99% | 61.01% | -3.2119 | -5.9360 |

## Top Improved Cases
| sample | turn | step | abs_pos | block_pos | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |
|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|

## Top Worsened Cases: Drop From Top-1
| sample | turn | step | abs_pos | block_pos | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |
|---:|---:|---:|---:|---:|---|---:|---:|---|---|---|---:|---:|---:|---:|

## Top Worsened Cases: Not Push To Top-1
| sample | turn | step | abs_pos | block_pos | subtype | first_token -> negative | l2 | cos_dist | target | positive_pred | contrastive_pred | d_logprob | d_rank | target_in_mask | accepted |
|---:|---:|---:|---:|---:|---|---|---:|---:|---|---|---|---:|---:|---:|---:|
| 82 | 0 | 75 | 601 | 10 | rank_worse_not_top1 | } (92) -> 所以他 (113106) | 1.6749 | 1.0767 | ### (14374) | \n (198) | \n (198) | -16.1497 | -13634 | 1 | 0 |
| 115 | 0 | 66 | 520 | 9 | rank_worse_not_top1 | ** (334) -> formed (10155) | 1.7182 | 1.0538 | ### (14374) | \n (198) | \n (198) | -19.2892 | -11664 | 1 | 0 |
| 117 | 0 | 40 | 396 | 11 | rank_worse_not_top1 |  all (678) -> bike (55155) | 1.5613 | 1.0802 | ### (14374) | \n (198) | \n (198) | -19.3426 | -11632 | 1 | 0 |
| 20 | 0 | 33 | 361 | 10 | rank_worse_not_top1 | 3 (18) -> 잳 (151017) | 1.2304 | 1.4239 | shot (6340) |  up (705) |  up (705) | -24.2638 | -10698 | 1 | 0 |
| 115 | 0 | 66 | 522 | 11 | rank_worse_not_top1 | ** (334) -> formed (10155) | 1.7182 | 1.0538 | ### (14374) |  week (2003) |  week (2003) | -18.5752 | -8871 | 1 | 0 |
| 5 | 0 | 84 | 493 | 12 | rank_worse_not_top1 |  starts (8471) -> .LOGIN (89025) | 1.3938 | 0.9432 | To (1249) | \n (198) | \n (198) | -17.4583 | -8700 | 1 | 0 |
| 78 | 0 | 68 | 480 | 4 | rank_worse_not_top1 | 8 (23) ->  uniquely (41229) | 1.6230 | 1.0525 | ants (1783) |  the (279) |  the (279) | -19.0519 | -8093 | 1 | 0 |
| 115 | 0 | 65 | 521 | 13 | rank_worse_not_top1 | 2 (17) ->  безопасн (140599) | 1.4795 | 1.2752 | ### (14374) | \n (198) | \n (198) | -20.5200 | -7493 | 1 | 0 |
| 68 | 0 | 7 | 107 | 9 | rank_worse_not_top1 | 6 (21) ->  kèm (136944) | 1.5889 | 1.1254 | related (9721) |  ** (3070) |  ** (3070) | -19.3287 | -7321 | 1 | 0 |
| 117 | 0 | 40 | 397 | 12 | rank_worse_not_top1 |  all (678) -> bike (55155) | 1.5613 | 1.0802 | ### (14374) | \n (198) | \n (198) | -17.4770 | -7269 | 1 | 0 |
| 62 | 0 | 3 | 127 | 10 | rank_worse_not_top1 |  per (817) -> 力度 (102061) | 1.6588 | 1.0367 | gets (18691) | 1 (16) | 1 (16) | -23.8547 | -7247 | 1 | 0 |
| 115 | 0 | 65 | 522 | 14 | rank_worse_not_top1 | 2 (17) ->  безопасн (140599) | 1.4795 | 1.2752 | ### (14374) |  week (2003) |  week (2003) | -17.0250 | -7029 | 1 | 0 |
| 117 | 0 | 40 | 398 | 13 | rank_worse_not_top1 |  all (678) -> bike (55155) | 1.5613 | 1.0802 | ### (14374) | \n (198) | \n (198) | -16.6988 | -6662 | 1 | 0 |
| 87 | 0 | 17 | 214 | 4 | rank_worse_not_top1 | 3 (18) -> ﻩ (145355) | 1.5661 | 1.1762 | d (67) |   (220) |   (220) | -26.7030 | -6647 | 1 | 0 |
| 20 | 0 | 45 | 475 | 9 | rank_worse_not_top1 | } (92) ->  dalle (85244) | 1.4244 | 1.0498 | --- (4421) |  faster (10596) |  faster (10596) | -20.9507 | -6236 | 1 | 0 |
| 117 | 0 | 40 | 399 | 14 | rank_worse_not_top1 |  all (678) -> bike (55155) | 1.5613 | 1.0802 | ### (14374) | \n (198) | \n (198) | -16.1578 | -6042 | 1 | 0 |
| 5 | 0 | 84 | 494 | 13 | rank_worse_not_top1 |  starts (8471) -> .LOGIN (89025) | 1.3938 | 0.9432 | To (1249) | <\|im_end\|> (151645) | <\|im_end\|> (151645) | -16.9098 | -6000 | 1 | 0 |
| 115 | 0 | 66 | 521 | 10 | rank_worse_not_top1 | ** (334) -> formed (10155) | 1.7182 | 1.0538 | ### (14374) | \n (198) | \n (198) | -18.8745 | -5440 | 1 | 0 |
| 83 | 0 | 39 | 321 | 12 | rank_worse_not_top1 |  on (389) ->  Gauge (72060) | 1.5530 | 1.0949 | ### (14374) | \n (198) | \n (198) | -15.1684 | -5017 | 1 | 0 |
| 37 | 0 | 20 | 238 | 11 | rank_worse_not_top1 | \n (198) ->  Particularly (96385) | 1.6897 | 1.1908 | lot (9184) | 3 (18) | 3 (18) | -24.5191 | -4997 | 1 | 0 |
| 85 | 0 | 35 | 228 | 4 | rank_worse_not_top1 | . (13) -> 😞 (145618) | 1.5379 | 1.2600 | Ali (17662) | <\|im_end\|> (151645) | <\|im_end\|> (151645) | -19.7180 | -4961 | 1 | 0 |
| 44 | 0 | 3 | 84 | 13 | rank_worse_not_top1 | ---\n\n (44364) ->  העוב (139546) | 1.3187 | 0.9069 |  App (1845) | ira (8832) | ira (8832) | -19.6067 | -4859 | 1 | 0 |
| 55 | 0 | 82 | 614 | 9 | rank_worse_not_top1 | 6 (21) -> automation (67080) | 1.6487 | 1.1179 | 1 (16) | ### (14374) | ### (14374) | -24.5228 | -4823 | 1 | 0 |
| 29 | 0 | 29 | 373 | 8 | rank_worse_not_top1 | 2 (17) -> 攻 (99546) | 1.6459 | 1.0525 | bring (81377) | 2 (17) | 2 (17) | -16.2227 | -4747 | 1 | 0 |
| 21 | 0 | 28 | 271 | 11 | rank_worse_not_top1 |  got (2684) -> магазин (127006) | 1.4734 | 0.9790 | ### (14374) | \n (198) | \n (198) | -18.1503 | -4714 | 1 | 0 |
| 25 | 0 | 54 | 453 | 14 | rank_worse_not_top1 | 4 (19) -> RUN (47390) | 1.5830 | 1.1257 | To (1249) | 4 (19) | 4 (19) | -18.6726 | -4681 | 1 | 0 |
| 41 | 0 | 3 | 107 | 13 | rank_worse_not_top1 |  reported (4961) -> terr (68669) | 1.6297 | 0.9569 | Wait (14190) |  ** (3070) |  ** (3070) | -19.4842 | -4609 | 1 | 0 |
| 101 | 0 | 41 | 370 | 11 | rank_worse_not_top1 |  hike (34231) ->  filtering (29670) | 1.5260 | 0.9732 | led (832) |  the (279) |  the (279) | -19.8781 | -4296 | 1 | 0 |
| 21 | 0 | 28 | 274 | 14 | rank_worse_not_top1 |  got (2684) -> магазин (127006) | 1.4734 | 0.9790 | ### (14374) | <\|im_end\|> (151645) | <\|im_end\|> (151645) | -18.4240 | -4235 | 1 | 0 |
| 47 | 0 | 52 | 502 | 13 | rank_worse_not_top1 |  fill (5155) -> 栝 (120329) | 1.6316 | 0.9694 | ### (14374) | \n (198) | \n (198) | -19.2693 | -4038 | 1 | 0 |