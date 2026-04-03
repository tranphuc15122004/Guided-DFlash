# Phan tich insight cho run `reject_tok_gsm8k_20260402_014507` (CDv4)

## 1. Pham vi va cach doc dung run nay

Nguon dung de phan tich:

- `analysis/results/reject_tok_gsm8k_20260402_014507/summary.json`
- `analysis/results/reject_tok_gsm8k_20260402_014507/reject_records.jsonl`
- `analysis/results/reject_tok_gsm8k_20260402_014507/report.md`
- File tham chieu cach viet: `analysis/results/reject_tok_gsm8k_20260326_105221/vcd_reject_token_insights.md`

Meta cua run:

- Dataset: `gsm8k`
- `block_size = 16`
- `vcd_alpha = 0.1`
- `vcd_beta = 0.1`
- `negative_context_dropout = 0.3`
- `negative_context_noise_std = 0.0`
- `evaluation_mode = positive_rollout_with_contrastive_shadow`
- `compare_name = CDv4`

Pham vi event-level trong folder nay la `128` sample, `sample_idx` chay tu `0` den `127`. Tong cong co `5491` reject event, trung binh `42.90` reject moi sample, median `40`, max `187`, min `19`.

Can nhan manh: day khong phai do chat luong rollout CDv4 end-to-end. Day la phan tich tai vi tri reject tren quy dao positive rollout, sau do tinh them mot nhanh contrastive shadow de hoi:

1. CDv4 co doi top-1 tai vi tri reject khong?
2. Neu co doi, no co dua token ve dung posterior token khong?
3. Neu khong sua duoc, do target bi candidate mask loai hay do contrastive score qua yeu?

Noi ngan gon: folder nay do **kha nang can thiep o first-error token**, khong do toan bo rollout khi de CDv4 dieu khien thuc su.

## 2. Headline ket qua

| Chi so | Gia tri | Dien giai ngan |
|---|---:|---|
| Reject events | 5491 | So vi tri reject duoc quan sat |
| `positive_hit_rate` | 1.18% | O mot so it event, positive step logits van dat target top-1 |
| `vcd_hit_rate` | 0.00% | CDv4 khong co exact fix nao o reject token |
| `pred_changed_rate` | 2.29% | CDv4 rat hiem khi thuc su doi top-1 |
| `target_in_candidate_mask_rate` | 67.66% | Khoang 2/3 reject con reachable ve mat mask |
| `no_vcd_effect_positive_wrong` | 65.78% | Phan lon reject la CDv4 gan nhu khong tac dong |
| `target_filtered_by_candidate_mask` | 32.34% | Gan 1/3 reject bi chan ngay tu candidate mask |
| `vcd_regression_from_positive` | 1.18% | Co mot nhom nho ma CDv4 lam xau di so voi positive |
| `vcd_shifted_but_still_wrong` | 0.69% | Co doi huong, nhung van khong sua duoc reject |

Headline quan trong nhat cua run nay la:

- CDv4 gan nhu **khong can thiep** vao reject token.
- Khi can thiep, no **khong tao ra exact fix nao**.
- So it lan can thiep hien co con bao gom ca **regression** tu nhung event ma positive logits dang dung.

## 3. Phan ra cac bottleneck chinh

### 3.1 Failure mode lon nhat la "khong doi duoc top-1"

Trong `5491` reject:

- `5365` event, tuong duong `97.71%`, CDv4 khong doi prediction
- Chi `126` event, tuong duong `2.29%`, la CDv4 co doi prediction

Ngay ca trong nhom target van con nam trong candidate mask (`3715` event), ty le doi prediction cung chi la `2.77%`. Nghia la van de chinh khong chi la hard mask; ngay ca tren reachable set, contrastive signal cua CDv4 cung qua yeu de lat top-1 cua positive.

Dieu nay rat khac voi run tham chieu `reject_tok_gsm8k_20260326_105221`, noi VCD shadow truoc do co:

- `pred_changed_rate = 26.13%`
- `vcd_hit_rate = 13.16%`

Trong khi run CDv4 hien tai chi con:

- `pred_changed_rate = 2.29%`
- `vcd_hit_rate = 0.00%`

Neu chi nhin tren behavioral level, CDv4 trong run nay giong mot **perturbation rat nhe** hon la mot local reranker co kha nang sua reject.

### 3.2 Candidate mask van la hard ceiling rat lon

`1776` event, tuong duong `32.34%`, roi vao taxonomy `target_filtered_by_candidate_mask`. O nhung event nay, target bi loai khoi candidate set ngay tu dau, nen CDv4 khong con quyen chon target du contrastive score co muon day len cung khong duoc.

Mot vai dau hieu ro:

- `target_in_candidate_mask_rate = 67.66%`
- Trong nhom `target_in_mask = 0`, `vcd_hit_rate = 0%`
- `delta_target_logprob.mean` cua toan run thanh `-Infinity` vi cac event bi mask gan target logit ve gia tri cuc tieu cua dtype

Do sau cua target trong positive ranking lien quan rat manh den kha nang song sot qua mask:

| Positive target rank | Ty le target con trong mask |
|---|---:|
| 1 | 100.00% |
| 2 | 87.39% |
| 3 | 70.12% |
| 4 | 54.61% |
| 5 | 38.16% |
| 6 | 33.74% |
| 7 | 22.32% |
| 8 | 13.04% |
| 9 | 9.84% |
| 10 | 10.77% |
| >10 | 1.98% |

Nghia la khi target khong con nam sat frontier cua positive, hard beta mask gan nhu dong cua hoan toan.

Case dai dien:

- `sample=0`, `decode_step=14`, `pos=141`
- target posterior la ` how`
- positive/CDv4 deu chon ` Samantha`
- `positive_target_rank = 19`
- `target_in_candidate_mask = 0`
- posterior lai uu tien ` how` rat manh: `p(target) = 0.9225`, trong khi `p(sampled) = 0.0316`

Day la nhom reject "unrecoverable by design" duoi cau hinh mask hien tai.

### 3.3 Tren nhung lan CDv4 co can thiep, mot nua la regression

Trong `126` event ma `pred_changed = 1`, taxonomy tach ra nhu sau:

- `65` event (`51.6%`) la `vcd_regression_from_positive`
- `38` event (`30.2%`) la `vcd_shifted_but_still_wrong`
- `23` event (`18.3%`) van la `target_filtered_by_candidate_mask`, tuc CDv4 co doi token nhung van khong the toi target

Nhom regression dac biet dang chu y:

- Tat ca `65` event regression deu co `positive_hit = 1`
- Tat ca deu co `positive_target_rank = 1`
- Nhung CDv4 lai day target xuong, nen `vcd_hit = 0`

Noi cach khac: nhung lan CDv4 thuc su "pha" o run nay phan lon la pha tren cac event ma positive step logits dang xep target dung top-1.

Case dai dien:

- `sample=11`, `decode_step=4`, `pos=127`
- target la ` the`
- positive top-1 la ` the` voi `p = 0.3077`
- CDv4 doi top-1 sang ` how` voi `p = 0.3513`, day ` the` xuong rank-2
- posterior lai rat ro rang uu tien ` the` (`p = 0.9960`)

Diem can luu y la `delta_target_logprob` o case nay van duong `+0.1285`. Tuc la chi nhin vao target logprob se de danh gia sai: CDv4 co the tang score cua target, nhung van tang score cua token sai nhieu hon va lam top-1 bi lat.

### 3.4 Co mot nhom "co ich nhung chua du" rat nho

`38` event `vcd_shifted_but_still_wrong` cho thay CDv4 khong hoan toan vo dung. O nhom nay:

- `pred_changed = 1`
- `positive_hit = 0`
- `vcd_hit = 0`
- mean `delta_target_logprob = +0.2003`
- `78.95%` event co `delta_target_logprob > 0`

Tuc la tren nhom nho nay, CDv4 thuong day target len gan hon, nhung van khong du de len top-1.

Case dai dien:

- `sample=71`, `decode_step=11`, `pos=141`
- positive top-1 la ` squirt`
- CDv4 doi top-1 sang ` bottles`
- target posterior la ` guns`
- xac suat cua ` guns` duoc nang tu `0.0497` len `0.1542`
- nhung ` bottles` van top-1, nen reject khong duoc sua

Nhom nay cho thay CDv4 van co mot it local signal, nhung cuong do hien tai qua nhe de bien thanh exact correction.

## 4. Cach doc dung `delta_target_logprob`

`summary.json` ghi:

```json
"delta_target_logprob": {
  "mean": -Infinity,
  "median": 0.0,
  "p10": -3.3895313892515355e+38,
  "p90": 0.1075512170791626
}
```

So `-Infinity` nay khong co nghia la moi reject deu bi CDv4 lam te di. No chu yeu do:

1. target bi loai khoi `candidate_mask`
2. logit cua target bi gan thanh gia tri cuc tieu cua dtype
3. khi tong hop bang `float32`, mean bi overflow thanh `-Infinity`

Neu chi nhin tren reachable set (`target_in_mask = 1`), buc tranh dung hon la:

- mean `delta_target_logprob = +0.0332`
- improved-rate `delta_target_logprob > 0` la `21.21%`
- nhung `vcd_hit_rate` van la `0%`
- `pred_changed_rate` tren reachable set van chi la `2.77%`

Insight o day la:

- CDv4 co luc tang score cua target
- nhung thuong **khong tang du manh** de vuot qua token dang dung o top-1
- vi the neu chi doc logprob ma khong doc exact-hit va changed-rate, rat de over-estimate tac dung cua CDv4

Case `sample=110`, `decode_step=13`, `pos=173` la vi du dien hinh:

- target posterior la token khoang trang ` `
- positive va CDv4 deu giu top-1 la `:`
- nhung `p(target)` duoc nang tu `0.0972` len `0.2866`
- posterior lai gan nhu chac chan chon target (`p = 0.99995`)

Day la mot near-miss rat ro: CDv4 day dung huong, nhung khong du luc de lat token sai.

## 5. Ket luan tong hop

Run `reject_tok_gsm8k_20260402_014507` cho thay CDv4 trong cau hinh hien tai (`alpha=0.1`, `beta=0.1`) dang gap ba gioi han lon:

1. Candidate mask van chan mat khoang `1/3` reject ngay tu dau.
2. Tren phan reachable con lai, CDv4 rat it khi doi duoc top-1.
3. Trong so it lan doi top-1, mot nua lai la regression tu cac case ma positive step logits dang dung.

Ket qua thuc nghiem quan trong nhat cua folder nay la:

- CDv4 hien chua hoat dong nhu mot co che rescue reject hieu qua.
- Tac dung quan sat duoc chu yeu la nhich target logprob o mot so case, chua chuyen thanh exact fix.
- So voi run VCD tham chieu trong folder `20260326_105221`, do manh can thiep cua CDv4 hien tai giam rat ro.

## 6. Huong debug / thi nghiem tiep theo

Tu ket qua tren, nhung huong uu tien de kiem tra tiep la:

1. Tang do manh contrastive.
   - `pred_changed_rate = 2.29%` qua thap, nen can xem `alpha=0.1` co qua nhe khong.

2. Giam do chat cua hard mask.
   - Vi `32.34%` reject bi filter ngay tu dau, co the thu `beta` nho hon hoac soft-mask thay vi hard-mask.

3. Kiem tra rieng nhom regression.
   - `65` event regression deu co `positive_target_rank = 1`; day la dau hieu rat cu the rang CDv4 dang pha frontier o mot nhom case le ra co the giu nguyen.

4. Theo doi them metric "top-1 margin sau contrastive".
   - Run nay co nhieu case tang target logprob nhung khong doi top-1; margin giua target va token dang top-1 se giup thay ro tai sao.

5. Tach danh gia thanh hai che do.
   - `reachable set`: target con trong mask
   - `unreachable set`: target bi filter
   Vi gom chung hai nhom nay khien mot so mean metric, dac biet `delta_target_logprob`, de gay hieu nham.
