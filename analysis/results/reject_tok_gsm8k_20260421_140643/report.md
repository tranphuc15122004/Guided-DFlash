# Reject Token Analysis: Positive vs CDv2

## Overall
- Reject events: **5739**
- Positive hit rate: **9.91%**
- CDv2 hit rate: **0.00%**
- Hit-rate gain (CDv2 - Positive): **-9.91%**
- CDv2 shadow fix rate @reject: **0.00%**
- Mean rescue-prob (Positive / CDv2-shadow): **0.2018 / 0.1347**
- Mean rescue-prob gain (CDv2 - Positive): **-0.0671**
- Rescue-prob improved-rate (>0): **9.32%**
- Prediction changed rate: **14.76%**

## Target LogProb Delta (CDv2 - Positive)
- Mean: -0.38926
- Median: 0.00000
- P10 / P90: -2.03528 / 0.00000
- Improved rate (>0): 4.29%

## Target Rank Delta (Positive - CDv2)
- Mean: -30.698
- Median: 0.000
- P10 / P90: -1.000 / 0.000
- Improved rate (>0): 2.35%

## Reject Taxonomy
| Taxonomy | Count | Rate |
|---|---:|---:|
| no_vcd_effect_positive_wrong | 4892 | 85.24% |
| vcd_regression_from_positive | 569 | 9.91% |
| vcd_shifted_but_still_wrong | 278 | 4.84% |

## Top Improved Reject Events
| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |
|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|
| 33 | 0 | 40 | 463 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 7.4564 | 8 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 96 | 0 | 33 | 338 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 6.5257 | 8 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 79 | 0 | 51 | 528 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | \n (198) | ast (559) | \n (198) | 6.2090 | 85 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `\n` sang `\n`. Ca positive va contrastive deu chua dua target token len top-1. |
| 108 | 0 | 33 | 262 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.8661 | 3 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 108 | 0 | 24 | 168 | no_vcd_effect_positive_wrong | ---\n\n (44364) | Sub (3136) | Sub (3136) | Sub (3136) | 5.4955 | 4 | Reject vi draft de xuat `Sub` (3136) khac posterior `---\n\n` (44364). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 8 | 0 | 53 | 394 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.4951 | 37 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 52 | 0 | 28 | 263 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.4808 | 1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 114 | 0 | 55 | 549 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 5.0296 | -19 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 24 | 0 | 37 | 373 | no_vcd_effect_positive_wrong |  by (553) | ---\n\n (44364) | ---\n\n (44364) | ---\n\n (44364) | 5.0209 | -9880 | Reject vi draft de xuat `---\n\n` (44364) khac posterior ` by` (553). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 105 | 0 | 45 | 455 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | . (13) | <|im_end|> (151645) | 4.9715 | -19 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Ca positive va contrastive deu chua dua target token len top-1. |
| 73 | 0 | 28 | 239 | no_vcd_effect_positive_wrong | Human (33975) |  ** (3070) |  ** (3070) |  ** (3070) | 4.7234 | -26830 | Reject vi draft de xuat ` **` (3070) khac posterior `Human` (33975). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 122 | 0 | 41 | 394 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.6736 | 9 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 36 | 0 | 41 | 283 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | \n (198) | <|im_end|> (151645) | 4.3460 | 5 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Ca positive va contrastive deu chua dua target token len top-1. |
| 75 | 0 | 24 | 228 | no_vcd_effect_positive_wrong | Ali (17662) | \n (198) | \n (198) | \n (198) | 4.2284 | 14 | Reject vi draft de xuat `\n` (198) khac posterior `Ali` (17662). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 32 | 0 | 33 | 363 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | <|im_end|> (151645) | <|im_end|> (151645) | 4.1943 | 13 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 120 | 0 | 42 | 358 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.1474 | 1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 29 | 0 | 13 | 228 | vcd_shifted_but_still_wrong | ult (494) |  an (458) |  \ (1124) |  an (458) | 4.1340 | -1121 | Reject vi draft de xuat ` an` (458) khac posterior `ult` (494). Contrastive shadow sample da doi token tu ` an` sang ` an`. Ca positive va contrastive deu chua dua target token len top-1. |
| 109 | 0 | 40 | 348 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 4.0832 | 0 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 107 | 0 | 7 | 103 | no_vcd_effect_positive_wrong |   (220) |  is (374) |  is (374) |  is (374) | 4.0023 | 1 | Reject vi draft de xuat ` is` (374) khac posterior ` ` (220). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 90 | 0 | 39 | 303 | vcd_shifted_but_still_wrong | Final (19357) | Answer (16141) |  Answer (21806) | Answer (16141) | 3.9797 | 1 | Reject vi draft de xuat `Answer` (16141) khac posterior `Final` (19357). Contrastive shadow sample da doi token tu `Answer` sang `Answer`. Ca positive va contrastive deu chua dua target token len top-1. |
| 15 | 0 | 37 | 324 | no_vcd_effect_positive_wrong | 0 (15) | ** (334) | ** (334) | ** (334) | 3.9732 | -56 | Reject vi draft de xuat `**` (334) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 86 | 0 | 36 | 216 | no_vcd_effect_positive_wrong | 0 (15) | 3 (18) | 3 (18) | 3 (18) | 3.9253 | 3 | Reject vi draft de xuat `3` (18) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 55 | 0 | 26 | 210 | no_vcd_effect_positive_wrong |  hours (4115) | week (10264) | week (10264) | week (10264) | 3.8670 | 6 | Reject vi draft de xuat `week` (10264) khac posterior ` hours` (4115). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 127 | 0 | 25 | 246 | no_vcd_effect_positive_wrong |  Subtract (93210) |  Deposit (48471) |  Deposit (48471) |  Deposit (48471) | 3.8243 | 34 | Reject vi draft de xuat ` Deposit` (48471) khac posterior ` Subtract` (93210). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 58 | 0 | 72 | 578 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) |  will (686) |  will (686) |  will (686) | 3.7657 | 102 | Reject vi draft de xuat ` will` (686) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 69 | 0 | 52 | 354 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 3.7432 | -1 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 126 | 0 | 83 | 510 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) |  are (525) |  left (2115) |  are (525) | 3.6220 | -1165 | Reject vi draft de xuat ` are` (525) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu ` are` sang ` are`. Ca positive va contrastive deu chua dua target token len top-1. |
| 40 | 0 | 25 | 259 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | <|im_end|> (151645) | <|im_end|> (151645) | <|im_end|> (151645) | 3.5751 | 0 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 37 | 0 | 30 | 291 | no_vcd_effect_positive_wrong | 0 (15) | }\n (532) | }\n (532) | }\n (532) | 3.5552 | 0 | Reject vi draft de xuat `}\n` (532) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 61 | 0 | 55 | 431 | no_vcd_effect_positive_wrong | <|endoftext|> (151643) | \n (198) | \n (198) | \n (198) | 3.5468 | 3 | Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |

## Top Worsened Reject Events
| sample | turn | step | abs_pos | taxonomy | target | sampled_draft | positive_pred | contrastive_pred | d_logprob | d_rank | why_reject |
|---:|---:|---:|---:|---|---|---|---|---|---:|---:|---|
| 79 | 0 | 48 | 495 | vcd_regression_from_positive | 2 (17) | 6 (21) | 2 (17) | 6 (21) | -9.1812 | -2093 | Reject vi draft de xuat `6` (21) khac posterior `2` (17). Contrastive shadow sample da doi token tu `6` sang `6`. Contrastive lam mat token target ma positive von chon dung. |
| 79 | 0 | 37 | 361 | vcd_regression_from_positive |  weighed (46612) |  was (572) |  weighed (46612) |  was (572) | -7.4620 | -82 | Reject vi draft de xuat ` was` (572) khac posterior ` weighed` (46612). Contrastive shadow sample da doi token tu ` was` sang ` was`. Contrastive lam mat token target ma positive von chon dung. |
| 104 | 0 | 5 | 110 | vcd_shifted_but_still_wrong |  Number (5624) |  Carmen (69958) |  She (2932) |  Carmen (69958) | -7.1295 | -74 | Reject vi draft de xuat ` Carmen` (69958) khac posterior ` Number` (5624). Contrastive shadow sample da doi token tu ` Carmen` sang ` Carmen`. Ca positive va contrastive deu chua dua target token len top-1. |
| 63 | 0 | 6 | 137 | vcd_regression_from_positive |   (220) | : (25) |   (220) | : (25) | -6.8468 | -54 | Reject vi draft de xuat `:` (25) khac posterior ` ` (220). Contrastive shadow sample da doi token tu `:` sang `:`. Contrastive lam mat token target ma positive von chon dung. |
| 64 | 0 | 0 | 109 | vcd_shifted_but_still_wrong |  the (279) | On (1925) |  ** (3070) | On (1925) | -6.8241 | -771 | Reject vi draft de xuat `On` (1925) khac posterior ` the` (279). Contrastive shadow sample da doi token tu `On` sang `On`. Ca positive va contrastive deu chua dua target token len top-1. |
| 60 | 0 | 15 | 161 | no_vcd_effect_positive_wrong | 5 (20) | 1 (16) | 1 (16) | 1 (16) | -6.2739 | -228 | Reject vi draft de xuat `1` (16) khac posterior `5` (20). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 115 | 0 | 59 | 454 | vcd_regression_from_positive | Total (7595) | E (36) | Total (7595) | E (36) | -6.0966 | -2 | Reject vi draft de xuat `E` (36) khac posterior `Total` (7595). Contrastive shadow sample da doi token tu `E` sang `E`. Contrastive lam mat token target ma positive von chon dung. |
| 110 | 0 | 38 | 293 | vcd_regression_from_positive | \n\n (271) | <|im_end|> (151645) | \n\n (271) | <|im_end|> (151645) | -6.0485 | -1 | Reject vi draft de xuat `<|im_end|>` (151645) khac posterior `\n\n` (271). Contrastive shadow sample da doi token tu `<|im_end|>` sang `<|im_end|>`. Contrastive lam mat token target ma positive von chon dung. |
| 27 | 0 | 16 | 188 | vcd_regression_from_positive | 6 (21) | erie (26105) | 6 (21) | erie (26105) | -6.0181 | -4 | Reject vi draft de xuat `erie` (26105) khac posterior `6` (21). Contrastive shadow sample da doi token tu `erie` sang `erie`. Contrastive lam mat token target ma positive von chon dung. |
| 88 | 0 | 8 | 137 | no_vcd_effect_positive_wrong | Total (7595) | arnings (14857) | arnings (14857) | arnings (14857) | -5.7898 | -29 | Reject vi draft de xuat `arnings` (14857) khac posterior `Total` (7595). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 64 | 0 | 43 | 367 | no_vcd_effect_positive_wrong |  \ (1124) | 0 (15) | 0 (15) | 0 (15) | -5.7769 | -1 | Reject vi draft de xuat `0` (15) khac posterior ` \` (1124). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 54 | 0 | 41 | 321 | vcd_shifted_but_still_wrong | 1 (16) | 宁 (99503) | 5 (20) | 宁 (99503) | -5.7139 | -148 | Reject vi draft de xuat `宁` (99503) khac posterior `1` (16). Contrastive shadow sample da doi token tu `宁` sang `宁`. Ca positive va contrastive deu chua dua target token len top-1. |
| 42 | 0 | 35 | 364 | vcd_regression_from_positive | 0 (15) | , (11) | 0 (15) | , (11) | -5.6851 | -4 | Reject vi draft de xuat `,` (11) khac posterior `0` (15). Contrastive shadow sample da doi token tu `,` sang `,`. Contrastive lam mat token target ma positive von chon dung. |
| 8 | 0 | 5 | 106 | no_vcd_effect_positive_wrong | ** (334) | .\n\n (382) | .\n\n (382) | .\n\n (382) | -5.6708 | -18 | Reject vi draft de xuat `.\n\n` (382) khac posterior `**` (334). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 100 | 0 | 26 | 252 | vcd_shifted_but_still_wrong |  cost (2783) |  additional (5107) |  total (2790) |  additional (5107) | -5.6487 | -38 | Reject vi draft de xuat ` additional` (5107) khac posterior ` cost` (2783). Contrastive shadow sample da doi token tu ` additional` sang ` additional`. Ca positive va contrastive deu chua dua target token len top-1. |
| 79 | 0 | 42 | 422 | vcd_regression_from_positive |  pounds (16302) |  poo (82710) |  pounds (16302) |  poo (82710) | -5.5838 | -6 | Reject vi draft de xuat ` poo` (82710) khac posterior ` pounds` (16302). Contrastive shadow sample da doi token tu ` poo` sang ` poo`. Contrastive lam mat token target ma positive von chon dung. |
| 103 | 0 | 27 | 240 | no_vcd_effect_positive_wrong | 0 (15) | , (11) | , (11) | , (11) | -5.5835 | -161 | Reject vi draft de xuat `,` (11) khac posterior `0` (15). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 63 | 0 | 37 | 300 | vcd_regression_from_positive |  roses (60641) |  lacks (36756) |  roses (60641) |  lacks (36756) | -5.5584 | -36 | Reject vi draft de xuat ` lacks` (36756) khac posterior ` roses` (60641). Contrastive shadow sample da doi token tu ` lacks` sang ` lacks`. Contrastive lam mat token target ma positive von chon dung. |
| 94 | 0 | 27 | 306 | vcd_regression_from_positive | 4 (19) | 2 (17) | 4 (19) | 2 (17) | -5.5291 | -1 | Reject vi draft de xuat `2` (17) khac posterior `4` (19). Contrastive shadow sample da doi token tu `2` sang `2`. Contrastive lam mat token target ma positive von chon dung. |
| 18 | 0 | 26 | 241 | vcd_regression_from_positive |  and (323) | old (813) |  and (323) | old (813) | -5.5096 | -55 | Reject vi draft de xuat `old` (813) khac posterior ` and` (323). Contrastive shadow sample da doi token tu `old` sang `old`. Contrastive lam mat token target ma positive von chon dung. |
| 78 | 0 | 35 | 300 | vcd_shifted_but_still_wrong |  the (279) |  spouses (65212) |  we (582) |  spouses (65212) | -5.4120 | -244 | Reject vi draft de xuat ` spouses` (65212) khac posterior ` the` (279). Contrastive shadow sample da doi token tu ` spouses` sang ` spouses`. Ca positive va contrastive deu chua dua target token len top-1. |
| 29 | 0 | 17 | 287 | vcd_regression_from_positive | / (14) | /ad (44460) | / (14) | /ad (44460) | -5.4049 | -110 | Reject vi draft de xuat `/ad` (44460) khac posterior `/` (14). Contrastive shadow sample da doi token tu `/ad` sang `/ad`. Contrastive lam mat token target ma positive von chon dung. |
| 62 | 0 | 31 | 270 | no_vcd_effect_positive_wrong |  pounds (16302) | { (90) | { (90) | { (90) | -5.3662 | -7 | Reject vi draft de xuat `{` (90) khac posterior ` pounds` (16302). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1. |
| 124 | 0 | 53 | 376 | vcd_regression_from_positive | 0 (15) | } (92) | 0 (15) | } (92) | -5.3449 | -29 | Reject vi draft de xuat `}` (92) khac posterior `0` (15). Contrastive shadow sample da doi token tu `}` sang `}`. Contrastive lam mat token target ma positive von chon dung. |
| 46 | 0 | 43 | 359 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) |  inches (14924) | \n (198) |  inches (14924) | -5.2024 | -10971 | Reject vi draft de xuat ` inches` (14924) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu ` inches` sang ` inches`. Ca positive va contrastive deu chua dua target token len top-1. |
| 113 | 0 | 19 | 230 | vcd_regression_from_positive | 6 (21) | 驮 (119309) | 6 (21) | 驮 (119309) | -5.1455 | -1 | Reject vi draft de xuat `驮` (119309) khac posterior `6` (21). Contrastive shadow sample da doi token tu `驮` sang `驮`. Contrastive lam mat token target ma positive von chon dung. |
| 118 | 0 | 21 | 251 | vcd_shifted_but_still_wrong |  carrot (74194) |  hex (12371) |  price (3349) |  hex (12371) | -5.0679 | -12377 | Reject vi draft de xuat ` hex` (12371) khac posterior ` carrot` (74194). Contrastive shadow sample da doi token tu ` hex` sang ` hex`. Ca positive va contrastive deu chua dua target token len top-1. |
| 3 | 0 | 14 | 233 | vcd_shifted_but_still_wrong |  \ (1124) | 个月内 (112579) |  hour (6460) | 个月内 (112579) | -5.0455 | -53 | Reject vi draft de xuat `个月内` (112579) khac posterior ` \` (1124). Contrastive shadow sample da doi token tu `个月内` sang `个月内`. Ca positive va contrastive deu chua dua target token len top-1. |
| 23 | 0 | 30 | 266 | vcd_shifted_but_still_wrong | <|endoftext|> (151643) |  adds (11367) | \n (198) |  adds (11367) | -5.0257 | -5665 | Reject vi draft de xuat ` adds` (11367) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample da doi token tu ` adds` sang ` adds`. Ca positive va contrastive deu chua dua target token len top-1. |
| 29 | 0 | 27 | 339 | vcd_regression_from_positive | children (5864) | kids (73896) | children (5864) | kids (73896) | -5.0134 | -1 | Reject vi draft de xuat `kids` (73896) khac posterior `children` (5864). Contrastive shadow sample da doi token tu `kids` sang `kids`. Contrastive lam mat token target ma positive von chon dung. |

## Detailed Case Explanations
### Case 1: sample=33, turn=0, step=40, pos=463
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 7.4564 / 8
- Positive top-3: \n (198, p=0.9837); <|im_end|> (151645, p=0.0159); \n\n (271, p=0.0002)
- CDv2 top-3: \n (198, p=0.2211); <|im_end|> (151645, p=0.0704); 8 (23, p=0.0065)
- Posterior top-3: <|endoftext|> (151643, p=0.9884); </think> (151668, p=0.0059); <|im_end|> (151645, p=0.0046)

### Case 2: sample=96, turn=0, step=33, pos=338
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 6.5257 / 8
- Positive top-3: \n (198, p=0.9941); <|im_end|> (151645, p=0.0046); . (13, p=0.0007)
- CDv2 top-3: \n (198, p=0.5996); 0 (15, p=0.0065); <|im_end|> (151645, p=0.0044)
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

### Case 4: sample=108, turn=0, step=33, pos=262
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `\n` (198); posterior token: `<|endoftext|>` (151643)
- Positive pred: `\n` (198), CDv2 pred: `\n` (198)
- Reason: Reject vi draft de xuat `\n` (198) khac posterior `<|endoftext|>` (151643). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 5.8661 / 3
- Positive top-3: \n (198, p=0.9854); <|im_end|> (151645, p=0.0141); \n\n (271, p=0.0002)
- CDv2 top-3: \n (198, p=0.6654); <|endoftext|> (151643, p=0.0429); <|im_end|> (151645, p=0.0295)
- Posterior top-3: <|endoftext|> (151643, p=0.9987); <|im_end|> (151645, p=0.0006); </think> (151668, p=0.0006)

### Case 5: sample=108, turn=0, step=24, pos=168
- Taxonomy: `no_vcd_effect_positive_wrong` - Contrastive khong doi huong du doan va positive da sai target tu dau.
- Proposed draft token: `Sub` (3136); posterior token: `---\n\n` (44364)
- Positive pred: `Sub` (3136), CDv2 pred: `Sub` (3136)
- Reason: Reject vi draft de xuat `Sub` (3136) khac posterior `---\n\n` (44364). Contrastive shadow sample khong doi token so voi original o vi tri reject. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: 5.4955 / 4
- Positive top-3: Sub (3136, p=0.7539); $$ (14085, p=0.1156); S (50, p=0.1020)
- CDv2 top-3: Sub (3136, p=0.2514); ---\n\n (44364, p=0.2150); Since (12549, p=0.1898)
- Posterior top-3: ---\n\n (44364, p=0.8485); Sub (3136, p=0.1474); $$ (14085, p=0.0035)

### Case 6: sample=79, turn=0, step=48, pos=495
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: `6` (21); posterior token: `2` (17)
- Positive pred: `2` (17), CDv2 pred: `6` (21)
- Reason: Reject vi draft de xuat `6` (21) khac posterior `2` (17). Contrastive shadow sample da doi token tu `6` sang `6`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -9.1812 / -2093
- Positive top-3: 2 (17, p=0.6405); 1 (16, p=0.3026); 3 (18, p=0.0206)
- CDv2 top-3: 9 (24, p=0.0259); 6 (21, p=0.0259);  Graham (25124, p=0.0139)
- Posterior top-3: 2 (17, p=1.0000); 0 (15, p=0.0000); 3 (18, p=0.0000)

### Case 7: sample=79, turn=0, step=37, pos=361
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: ` was` (572); posterior token: ` weighed` (46612)
- Positive pred: ` weighed` (46612), CDv2 pred: ` was` (572)
- Reason: Reject vi draft de xuat ` was` (572) khac posterior ` weighed` (46612). Contrastive shadow sample da doi token tu ` was` sang ` was`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -7.4620 / -82
- Positive top-3:  weighed (46612, p=0.8428);  was (572, p=0.1464);  had (1030, p=0.0034)
- CDv2 top-3:  was (572, p=0.6330);  had (1030, p=0.0571);  would (1035, p=0.0335)
- Posterior top-3:  weighed (46612, p=1.0000);  weighted (36824, p=0.0000);  was (572, p=0.0000)

### Case 8: sample=104, turn=0, step=5, pos=110
- Taxonomy: `vcd_shifted_but_still_wrong` - Contrastive co doi huong du doan nhung chua dua target len top-1.
- Proposed draft token: ` Carmen` (69958); posterior token: ` Number` (5624)
- Positive pred: ` She` (2932), CDv2 pred: ` Carmen` (69958)
- Reason: Reject vi draft de xuat ` Carmen` (69958) khac posterior ` Number` (5624). Contrastive shadow sample da doi token tu ` Carmen` sang ` Carmen`. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -7.1295 / -74
- Positive top-3:  She (2932, p=0.4590);  Carmen (69958, p=0.2457);  Number (5624, p=0.1024)
- CDv2 top-3:  Carmen (69958, p=0.9231);  Carm (34452, p=0.0210);  Carla (93770, p=0.0047)
- Posterior top-3:  Number (5624, p=0.9421);  Carmen (69958, p=0.0532);  Total (10657, p=0.0030)

### Case 9: sample=63, turn=0, step=6, pos=137
- Taxonomy: `vcd_regression_from_positive` - Positive dung target nhung contrastive day target xuong, dan den reject.
- Proposed draft token: `:` (25); posterior token: ` ` (220)
- Positive pred: ` ` (220), CDv2 pred: `:` (25)
- Reason: Reject vi draft de xuat `:` (25) khac posterior ` ` (220). Contrastive shadow sample da doi token tu `:` sang `:`. Contrastive lam mat token target ma positive von chon dung.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -6.8468 / -54
- Positive top-3:   (220, p=0.9631); : (25, p=0.0137); Step (8304, p=0.0051)
- CDv2 top-3: : (25, p=0.0318);  of (315, p=0.0073); Step (8304, p=0.0047)
- Posterior top-3:   (220, p=1.0000);  � (93920, p=0.0000);   (4102, p=0.0000)

### Case 10: sample=64, turn=0, step=0, pos=109
- Taxonomy: `vcd_shifted_but_still_wrong` - Contrastive co doi huong du doan nhung chua dua target len top-1.
- Proposed draft token: `On` (1925); posterior token: ` the` (279)
- Positive pred: ` **` (3070), CDv2 pred: `On` (1925)
- Reason: Reject vi draft de xuat `On` (1925) khac posterior ` the` (279). Contrastive shadow sample da doi token tu `On` sang `On`. Ca positive va contrastive deu chua dua target token len top-1.
- Candidate mask: target_in=1, sampled_in=1
- Delta target logprob/rank: -6.8241 / -771
- Positive top-3:  ** (3070, p=0.3549); On (1925, p=0.1152);  day (1899, p=0.1017)
- CDv2 top-3: On (1925, p=0.3096); Day (10159, p=0.1374); Custom (10268, p=0.0610)
- Posterior top-3:  the (279, p=0.9993);  ** (3070, p=0.0007);  day (1899, p=0.0000)
