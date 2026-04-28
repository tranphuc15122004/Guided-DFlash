# Reject Token Analysis: Positive vs CDv2

## Overall
- Reject events: **5855**
- Positive hit rate: **12.57%**
- CDv2 hit rate: **0.00%**
- Hit-rate gain (CDv2 - Positive): **-12.57%**
- CDv2 shadow fix rate @reject: **0.00%**
- Mean rescue-prob (Positive / CDv2-shadow): **0.2159 / 0.1305**
- Mean rescue-prob gain (CDv2 - Positive): **-0.0854**
- Rescue-prob improved-rate (>0): **9.09%**
- Prediction changed rate: **18.12%**

## Target LogProb Delta (CDv2 - Positive)
- Mean: -0.47508
- Median: 0.00000
- P10 / P90: -2.27492 / 0.00000
- Improved rate (>0): 5.43%

## Target Rank Delta (Positive - CDv2)
- Mean: -28.935
- Median: 0.000
- P10 / P90: -2.000 / 0.000
- Improved rate (>0): 2.97%

## Reject Taxonomy
| Taxonomy | Count | Rate |
|---|---:|---:|
| no_vcd_effect_positive_wrong | 4794 | 81.88% |
| vcd_regression_from_positive | 736 | 12.57% |
| vcd_shifted_but_still_wrong | 325 | 5.55% |

## Top Improved Reject Events
| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |
|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|
| 33 | 0 | 41 | 463 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 6.6854 | -2 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 96 | 0 | 33 | 338 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 6.5564 | 7 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 79 | 0 | 51 | 528 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | \n (198) | ast (559) | \n (198) | 6.2090 | 85 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `\n` sang `\n`. Ca positive va contrastive deu chua dua target token len top-1. |
| 40 | 0 | 16 | 189 | no_vcd_effect_positive_wrong |  afternoon (13354) | 1 (16) | 1 (16) | 1 (16) | 5.9135 | -69 | Reject vi draft de xuat `1` (16) khac posterior ` afternoon` (13354). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 114 | 0 | 56 | 549 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.8964 | -1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 108 | 0 | 33 | 262 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.8661 | 3 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 103 | 0 | 43 | 357 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.4972 | 15 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 108 | 0 | 24 | 168 | no_vcd_effect_positive_wrong | ---\n\n (44364) | Sub (3136) | Sub (3136) | Sub (3136) | 5.4955 | 4 | Reject vi draft de xuat `Sub` (3136) khac posterior `---\n\n` (44364). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 11 | 0 | 17 | 205 | no_vcd_effect_positive_wrong | ### (14374) | , (11) | , (11) | , (11) | 5.4725 | 4 | Reject vi draft de xuat `,` (11) khac posterior `###` (14374). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 40 | 0 | 21 | 228 | no_vcd_effect_positive_wrong |  thrown (14989) | }\n (532) | }\n (532) | }\n (532) | 5.4434 | 1 | Reject vi draft de xuat `}\n` (532) khac posterior ` thrown` (14989). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 84 | 0 | 53 | 457 | no_vcd_effect_positive_wrong | $$ (14085) |  \ (1124) |  \ (1124) |  \ (1124) | 5.2397 | 0 | Reject vi draft de xuat ` \` (1124) khac posterior `$$` (14085). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 8 | 0 | 54 | 394 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.1954 | 39 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 2 | 0 | 55 | 549 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.1544 | 6 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 105 | 0 | 45 | 455 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | . (13) | <|im_end|> (151645) | 4.9715 | -19 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Ca positive va contrastive deu chua dua target token len top-1. |
| 115 | 0 | 67 | 517 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | <|im_end|> (151645) | <|im_end|> (151645) | 4.9298 | -39 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 25 | 0 | 56 | 448 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.9202 | 1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 52 | 0 | 28 | 263 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.7310 | 1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 122 | 0 | 41 | 394 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.6666 | 8 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 75 | 0 | 24 | 228 | no_vcd_effect_positive_wrong | Ali (17662) | \n (198) | \n (198) | \n (198) | 4.4255 | 17 | Reject vi draft de xuat `\n` (198) khac posterior `Ali` (17662). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 36 | 0 | 41 | 283 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | \n (198) | <|im_end|> (151645) | 4.3460 | 5 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Ca positive va contrastive deu chua dua target token len top-1. |
| 15 | 0 | 39 | 324 | no_vcd_effect_positive_wrong | 0 (15) | ** (334) | ** (334) | ** (334) | 4.1844 | -27 | Reject vi draft de xuat `**` (334) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 39 | 0 | 114 | 604 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | \n (198) | <|im_end|> (151645) | \n (198) | 4.1408 | -81 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `\n` sang `\n`. Ca positive va contrastive deu chua dua target token len top-1. |
| 81 | 0 | 31 | 246 | no_vcd_effect_positive_wrong | Unc (63718) | \n (198) | \n (198) | \n (198) | 4.1377 | 16 | Reject vi draft de xuat `\n` (198) khac posterior `Unc` (63718). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 69 | 0 | 52 | 354 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.0948 | 1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 107 | 0 | 7 | 103 | no_vcd_effect_positive_wrong |   (220) |  is (374) |  is (374) |  is (374) | 4.0023 | 1 | Reject vi draft de xuat ` is` (374) khac posterior ` ` (220). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 86 | 0 | 36 | 216 | no_vcd_effect_positive_wrong | 0 (15) | 3 (18) | 3 (18) | 3 (18) | 3.9662 | 3 | Reject vi draft de xuat `3` (18) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 55 | 0 | 26 | 210 | no_vcd_effect_positive_wrong |  hours (4115) | week (10264) | week (10264) | week (10264) | 3.8670 | 6 | Reject vi draft de xuat `week` (10264) khac posterior ` hours` (4115). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 124 | 0 | 14 | 157 | no_vcd_effect_positive_wrong |  On (1913) |  He (1260) |  He (1260) |  He (1260) | 3.7664 | 0 | Reject vi draft de xuat ` He` (1260) khac posterior ` On` (1913). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 14 | 0 | 59 | 478 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | \n (198) | <|im_end|> (151645) | 3.5646 | -907 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Ca positive va contrastive deu chua dua target token len top-1. |
| 70 | 0 | 103 | 558 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 3.5247 | 4 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |

## Top Worsened Reject Events
| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |
|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|
| 118 | 0 | 16 | 204 | no_vcd_effect_positive_wrong | : (25) | atoes (20187) | atoes (20187) | atoes (20187) | -11.2200 | -1319 | Reject vi draft de xuat `atoes` (20187) khac posterior `:` (25). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 79 | 0 | 48 | 495 | vcd_regression_from_positive | 2 (17) | 6 (21) | 2 (17) | 6 (21) | -9.1812 | -2093 | Reject vi draft de xuat `6` (21) khac posterior `2` (17). Contrastive shadow sample da doi token tu `6` sang `6`. Contrastive lam mat token target ma positive von chon dung. |
| 25 | 0 | 26 | 234 | vcd_regression_from_positive | 4 (19) | 效 (59355) | 4 (19) | 效 (59355) | -8.1235 | -285 | Reject vi draft de xuat `效` (59355) khac posterior `4` (19). Contrastive shadow sample da doi token tu `效` sang `效`. Contrastive lam mat token target ma positive von chon dung. |
| 79 | 0 | 37 | 361 | vcd_regression_from_positive |  weighed (46612) |  was (572) |  weighed (46612) |  was (572) | -7.4620 | -82 | Reject vi draft de xuat ` was` (572) khac posterior ` weighed` (46612). Contrastive shadow sample da doi token tu ` was` sang ` was`. Contrastive lam mat token target ma positive von chon dung. |
| 115 | 0 | 59 | 454 | vcd_regression_from_positive | Total (7595) | E (36) | Total (7595) | E (36) | -7.2528 | -13 | Reject vi draft de xuat `E` (36) khac posterior `Total` (7595). Contrastive shadow sample da doi token tu `E` sang `E`. Contrastive lam mat token target ma positive von chon dung. |
| 21 | 0 | 30 | 256 | vcd_regression_from_positive | 5 (20) | 免责声明 (108218) | 5 (20) | 免责声明 (108218) | -7.1971 | -59 | Reject vi draft de xuat `免责声明` (108218) khac posterior `5` (20). Contrastive shadow sample da doi token tu `免责声明` sang `免责声明`. Contrastive lam mat token target ma positive von chon dung. |
| 104 | 0 | 5 | 110 | vcd_shifted_but_still_wrong |  Number (5624) |  Carmen (69958) |  She (2932) |  Carmen (69958) | -7.1295 | -74 | Reject vi draft de xuat ` Carmen` (69958) khac posterior ` Number` (5624). Contrastive shadow sample da doi token tu ` Carmen` sang ` Carmen`. Ca positive va contrastive deu chua dua target token len top-1. |
| 72 | 0 | 6 | 121 | vcd_regression_from_positive |  as (438) | half (37006) |  as (438) | half (37006) | -6.9465 | -24 | Reject vi draft de xuat `half` (37006) khac posterior ` as` (438). Contrastive shadow sample da doi token tu `half` sang `half`. Contrastive lam mat token target ma positive von chon dung. |
| 63 | 0 | 6 | 137 | vcd_regression_from_positive |   (220) | : (25) |   (220) | : (25) | -6.8468 | -54 | Reject vi draft de xuat `:` (25) khac posterior ` ` (220). Contrastive shadow sample da doi token tu `:` sang `:`. Contrastive lam mat token target ma positive von chon dung. |
| 85 | 0 | 16 | 124 | no_vcd_effect_positive_wrong | 0 (15) | }{ (15170) | }{ (15170) | }{ (15170) | -6.7644 | -2 | Reject vi draft de xuat `}{` (15170) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 71 | 0 | 21 | 208 | vcd_regression_from_positive | 9 (24) | 九十 (118572) | 9 (24) | 九十 (118572) | -6.7573 | -36 | Reject vi draft de xuat `九十` (118572) khac posterior `9` (24). Contrastive shadow sample da doi token tu `九十` sang `九十`. Contrastive lam mat token target ma positive von chon dung. |
| 60 | 0 | 15 | 161 | no_vcd_effect_positive_wrong | 5 (20) | 1 (16) | 1 (16) | 1 (16) | -6.5389 | -309 | Reject vi draft de xuat `1` (16) khac posterior `5` (20). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 82 | 0 | 30 | 250 | vcd_regression_from_positive |   (220) |  $$ (26107) |   (220) |  $$ (26107) | -6.3672 | -14 | Reject vi draft de xuat ` $$` (26107) khac posterior ` ` (220). Contrastive shadow sample da doi token tu ` $$` sang ` $$`. Contrastive lam mat token target ma positive von chon dung. |
| 118 | 0 | 18 | 235 | vcd_regression_from_positive | . (13) | : (25) | . (13) | : (25) | -6.3587 | -79 | Reject vi draft de xuat `:` (25) khac posterior `.` (13). Contrastive shadow sample da doi token tu `:` sang `:`. Contrastive lam mat token target ma positive von chon dung. |
| 27 | 0 | 18 | 188 | vcd_regression_from_positive | 6 (21) | 4 (19) | 6 (21) | 4 (19) | -6.3068 | -4 | Reject vi draft de xuat `4` (19) khac posterior `6` (21). Contrastive shadow sample da doi token tu `4` sang `4`. Contrastive lam mat token target ma positive von chon dung. |
| 96 | 0 | 20 | 238 | vcd_regression_from_positive | 5 (20) | 騰 (119046) | 5 (20) | 騰 (119046) | -6.2605 | -114 | Reject vi draft de xuat `騰` (119046) khac posterior `5` (20). Contrastive shadow sample da doi token tu `騰` sang `騰`. Contrastive lam mat token target ma positive von chon dung. |
| 6 | 0 | 35 | 269 | vcd_regression_from_positive | 1 (16) |  Perform (25001) | 1 (16) |  Perform (25001) | -6.2196 | -23 | Reject vi draft de xuat ` Perform` (25001) khac posterior `1` (16). Contrastive shadow sample da doi token tu ` Perform` sang ` Perform`. Contrastive lam mat token target ma positive von chon dung. |
| 90 | 0 | 21 | 228 | vcd_shifted_but_still_wrong |  Lo (6485) |  Grocery (95581) |  Remaining (89630) |  Grocery (95581) | -6.1616 | -108 | Reject vi draft de xuat ` Grocery` (95581) khac posterior ` Lo` (6485). Contrastive shadow sample da doi token tu ` Grocery` sang ` Grocery`. Ca positive va contrastive deu chua dua target token len top-1. |
| 110 | 0 | 38 | 293 | vcd_regression_from_positive | \n\n (271) | <|im_end|> (151645) | \n\n (271) | <|im_end|> (151645) | -6.0989 | -1 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `\n\n` (271). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Contrastive lam mat token target ma positive von chon dung. |
| 72 | 0 | 27 | 240 | vcd_regression_from_positive |  has (702) |  ** (3070) |  has (702) |  ** (3070) | -5.7457 | -10 | Reject vi draft de xuat ` **` (3070) khac posterior ` has` (702). Contrastive shadow sample da doi token tu ` **` sang ` **`. Contrastive lam mat token target ma positive von chon dung. |
| 91 | 0 | 32 | 223 | vcd_regression_from_positive |   (220) | bles (38763) |   (220) | bles (38763) | -5.7438 | -19 | Reject vi draft de xuat `bles` (38763) khac posterior ` ` (220). Contrastive shadow sample da doi token tu `bles` sang `bles`. Contrastive lam mat token target ma positive von chon dung. |
| 54 | 0 | 41 | 321 | vcd_shifted_but_still_wrong | 1 (16) | 宁 (99503) | 5 (20) | 宁 (99503) | -5.7139 | -148 | Reject vi draft de xuat `宁` (99503) khac posterior `1` (16). Contrastive shadow sample da doi token tu `宁` sang `宁`. Ca positive va contrastive deu chua dua target token len top-1. |
| 84 | 0 | 55 | 479 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) |  buys (49531) | \n (198) |  buys (49531) | -5.6411 | -1761 | Reject vi draft de xuat ` buys` (49531) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu ` buys` sang ` buys`. Ca positive va contrastive deu chua dua target token len top-1. |
| 79 | 0 | 42 | 422 | vcd_regression_from_positive |  pounds (16302) |  poo (82710) |  pounds (16302) |  poo (82710) | -5.5838 | -6 | Reject vi draft de xuat ` poo` (82710) khac posterior ` pounds` (16302). Contrastive shadow sample da doi token tu ` poo` sang ` poo`. Contrastive lam mat token target ma positive von chon dung. |
| 43 | 0 | 26 | 242 | no_vcd_effect_positive_wrong | ** (334) | :\n (510) | :\n (510) | :\n (510) | -5.5744 | -28 | Reject vi draft de xuat `:\n` (510) khac posterior `**` (334). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 63 | 0 | 37 | 300 | vcd_regression_from_positive |  roses (60641) |  lacks (36756) |  roses (60641) |  lacks (36756) | -5.5584 | -36 | Reject vi draft de xuat ` lacks` (36756) khac posterior ` roses` (60641). Contrastive shadow sample da doi token tu ` lacks` sang ` lacks`. Contrastive lam mat token target ma positive von chon dung. |
| 62 | 0 | 32 | 280 | vcd_shifted_but_still_wrong |  per (817) | /table (45326) |  of (315) | /table (45326) | -5.5314 | -37 | Reject vi draft de xuat `/table` (45326) khac posterior ` per` (817). Contrastive shadow sample da doi token tu `/table` sang `/table`. Ca positive va contrastive deu chua dua target token len top-1. |
| 18 | 0 | 26 | 241 | vcd_regression_from_positive |  and (323) | old (813) |  and (323) | old (813) | -5.5096 | -55 | Reject vi draft de xuat `old` (813) khac posterior ` and` (323). Contrastive shadow sample da doi token tu `old` sang `old`. Contrastive lam mat token target ma positive von chon dung. |
| 22 | 0 | 39 | 259 | vcd_regression_from_positive | **: (95518) |   \n (2303) | **: (95518) |   \n (2303) | -5.4914 | -11 | Reject vi draft de xuat `  \n` (2303) khac posterior `**:` (95518). Contrastive shadow sample da doi token tu `  \n` sang `  \n`. Contrastive lam mat token target ma positive von chon dung. |
| 67 | 0 | 46 | 396 | vcd_shifted_but_still_wrong |  he (566) |  Jama (40159) |  ** (3070) |  Jama (40159) | -5.4130 | -14 | Reject vi draft de xuat ` Jama` (40159) khac posterior ` he` (566). Contrastive shadow sample da doi token tu ` Jama` sang ` Jama`. Ca positive va contrastive deu chua dua target token len top-1. |

## Detailed Case Explanations
### Case 1: sample=33, turn=0, step=41, pos=463
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 6.6854 / -2
- Positive top-3: \n (198, p=0.9657); <|im_end|> (151645, p=0.0330); . (13, p=0.0010)
- CDv2 top-3: \n (198, p=0.2862); <|im_end|> (151645, p=0.0351); . (13, p=0.0171)
- Posterior top-3: <|endoftext|> (151643, p=0.9924); </think> (151668, p=0.0041); <|im_end|> (151645, p=0.0028)

### Case 2: sample=96, turn=0, step=33, pos=338
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 6.5564 / 7
- Positive top-3: \n (198, p=0.9941); <|im_end|> (151645, p=0.0046); . (13, p=0.0007)
- CDv2 top-3: \n (198, p=0.5866); 0 (15, p=0.0063); <|im_end|> (151645, p=0.0046)
- Posterior top-3: <|endoftext|> (151643, p=0.9971); </think> (151668, p=0.0015); <|im_end|> (151645, p=0.0010)

### Case 3: sample=79, turn=0, step=51, pos=528
- Taxonomy: `vcd_shifted_but_still_wrong` - Contrastive co doi huong du doan nhung chua dua target len top-1.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `ast` (559), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `\n` sang `\n`. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 6.2090 / 85
- Positive top-3: ast (559, p=0.5161); M (44, p=0.4019); \n (198, p=0.0374)
- CDv2 top-3: \n (198, p=0.2570); \n\n (271, p=0.1140); <|im_end|> (151645, p=0.1006)
- Posterior top-3: <|endoftext|> (151643, p=0.9942); <|im_end|> (151645, p=0.0028); </think> (151668, p=0.0022)

### Case 4: sample=40, turn=0, step=16, pos=189
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `1` (16); posterior token: ` afternoon` (13354)
- Positive pred: `1` (16), CDv2 pred: `1` (16)
- Reason: Reject vi draft de xuat `1` (16) khac posterior ` afternoon` (13354). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 5.9135 / -69
- Positive top-3: 1 (16, p=0.5720); 2 (17, p=0.2384);  ** (3070, p=0.0877)
- CDv2 top-3: 1 (16, p=0.1703); 2 (17, p=0.0269);  end (835, p=0.0245)
- Posterior top-3:  afternoon (13354, p=0.9996);  day (1899, p=0.0003);  fight (4367, p=0.0001)

### Case 5: sample=114, turn=0, step=56, pos=549
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 5.8964 / -1
- Positive top-3: \n (198, p=0.9789); <|im_end|> (151645, p=0.0203); ** (334, p=0.0004)
- CDv2 top-3: \n (198, p=0.2441);  Step (14822, p=0.0172); <|im_end|> (151645, p=0.0147)
- Posterior top-3: <|endoftext|> (151643, p=0.9871); <|im_end|> (151645, p=0.0075); </think> (151668, p=0.0040)

### Case 6: sample=118, turn=0, step=16, pos=204
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `atoes` (20187); posterior token: `:` (25)
- Positive pred: `atoes` (20187), CDv2 pred: `atoes` (20187)
- Reason: Reject vi draft de xuat `atoes` (20187) khac posterior `:` (25). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -11.2200 / -1319
- Positive top-3: atoes (20187, p=0.7610); : (25, p=0.2180);  potatoes (34167, p=0.0058)
- CDv2 top-3: atoes (20187, p=0.9459);  potatoes (34167, p=0.0148); ates (973, p=0.0042)
- Posterior top-3: : (25, p=1.0000); :\n (510, p=0.0000); ： (5122, p=0.0000)

### Case 7: sample=79, turn=0, step=48, pos=495
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: `6` (21); posterior token: `2` (17)
- Positive pred: `2` (17), CDv2 pred: `6` (21)
- Reason: Reject vi draft de xuat `6` (21) khac posterior `2` (17). Contrastive shadow sample da doi token tu `6` sang `6`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -9.1812 / -2093
- Positive top-3: 2 (17, p=0.6405); 1 (16, p=0.3026); 3 (18, p=0.0206)
- CDv2 top-3: 9 (24, p=0.0259); 6 (21, p=0.0259);  Graham (25124, p=0.0139)
- Posterior top-3: 2 (17, p=1.0000); 0 (15, p=0.0000); 3 (18, p=0.0000)

### Case 8: sample=25, turn=0, step=26, pos=234
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: `效` (59355); posterior token: `4` (19)
- Positive pred: `4` (19), CDv2 pred: `效` (59355)
- Reason: Reject vi draft de xuat `效` (59355) khac posterior `4` (19). Contrastive shadow sample da doi token tu `效` sang `效`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -8.1235 / -285
- Positive top-3: 4 (19, p=0.9999); 5 (20, p=0.0000); 0 (15, p=0.0000)
- CDv2 top-3: 效 (59355, p=0.0075); signup (28725, p=0.0040);  Bek (70219, p=0.0035)
- Posterior top-3: 4 (19, p=1.0000); 3 (18, p=0.0000); 5 (20, p=0.0000)

### Case 9: sample=79, turn=0, step=37, pos=361
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: ` was` (572); posterior token: ` weighed` (46612)
- Positive pred: ` weighed` (46612), CDv2 pred: ` was` (572)
- Reason: Reject vi draft de xuat ` was` (572) khac posterior ` weighed` (46612). Contrastive shadow sample da doi token tu ` was` sang ` was`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -7.4620 / -82
- Positive top-3:  weighed (46612, p=0.8428);  was (572, p=0.1464);  had (1030, p=0.0034)
- CDv2 top-3:  was (572, p=0.6330);  had (1030, p=0.0571);  would (1035, p=0.0335)
- Posterior top-3:  weighed (46612, p=1.0000);  weighted (36824, p=0.0000);  was (572, p=0.0000)

### Case 10: sample=115, turn=0, step=59, pos=454
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: `E` (36); posterior token: `Total` (7595)
- Positive pred: `Total` (7595), CDv2 pred: `E` (36)
- Reason: Reject vi draft de xuat `E` (36) khac posterior `Total` (7595). Contrastive shadow sample da doi token tu `E` sang `E`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -7.2528 / -13
- Positive top-3: Total (7595, p=0.6634); E (36, p=0.3134); Amount (10093, p=0.0045)
- CDv2 top-3: E (36, p=0.9532); En (1702, p=0.0026); Y (56, p=0.0009)
- Posterior top-3: Total (7595, p=0.9356); E (36, p=0.0598); Weekly (80516, p=0.0043)
