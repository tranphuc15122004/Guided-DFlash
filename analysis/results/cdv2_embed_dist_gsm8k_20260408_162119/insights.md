# Phân tích insight cho run `cdv2_embed_dist_gsm8k_20260408_162119`

## 1. Phạm vi và cách đọc kết quả

Nguồn dùng để phân tích:

- `analysis/results/cdv2_embed_dist_gsm8k_20260408_162119/first_token_embed_summary.json`
- `analysis/results/cdv2_embed_dist_gsm8k_20260408_162119/first_token_case_records.csv`
- `analysis/results/cdv2_embed_dist_gsm8k_20260408_162119/first_token_embed_report.md`
- `analysis/CDv2_embed_distance_ana.py`

Mục tiêu của folder này là đo:

1. khoảng cách embedding giữa `first_token` của positive sample và `first_token` đã bị thay ngẫu nhiên ở negative sample,
2. mức thay đổi của target token dưới cơ chế contrastive,
3. và liệu việc thay `first_token` của negative sample có giúp contrastive đẩy target đúng lên top-1 hay không.

Cần nhấn mạnh: đây là phân tích **token-level shadow** trên quỹ đạo positive rollout, không phải benchmark end-to-end khi để CDv2 tự điều khiển toàn bộ generation.

## 2. Headline kết quả

| Chỉ số | Giá trị | Ý nghĩa |
|---|---:|---|
| Tổng record | `90,570` | số token-level event được đo |
| `improved_push_to_top1` | `0` | không có exact rescue nào |
| `worsened_drop_from_top1` | `0` | không có case positive đúng nhưng contrastive làm rơi khỏi top-1 |
| `final_pred_id == positive_pred_id` | `90,570 / 90,570 = 100%` | argmax cuối không bao giờ đổi |
| `negative_pred_id != positive_pred_id` | `66,161 / 90,570 = 73.05%` | negative branch bị perturb khá mạnh |
| `target_in_candidate_mask = 1` | `73,085 / 90,570 = 80.69%` | target còn reachable trong khoảng 4/5 case |
| reachable wrong cases | `21,140` | positive sai nhưng target vẫn còn trong candidate mask |
| true rank-lift trên reachable wrong | `551 / 21,140 = 2.61%` | số case có cải thiện rank thật sự |
| rank-worse trên reachable wrong | `1,173 / 21,140 = 5.55%` | số case bị làm xấu đi |
| score-only nudge trên reachable wrong | `18,364 / 21,140 = 86.87%` | target logprob tăng nhưng rank không đổi |
| reachable wrong có `candidate_size <= 6` | `12,096 / 21,140 = 57.22%` | hơn nửa case bị “đóng băng” về mặt cơ chế |

Kết luận ngắn nhất của run này là:

- Thay `first_token` của negative sample **có perturb negative branch**.
- Nhưng perturb đó **không bao giờ đi xuyên qua combine stage để đổi final argmax**.
- Các tín hiệu “rank improved but not top1” trong report gốc phần lớn là **artefact do candidate mask**, không phải rescue thật.

## 3. Insight 1: Perturbation dừng ở negative branch, không đi tới final decision

Đây là phát hiện quan trọng nhất của toàn bộ folder:

- `negative_pred_id` khác `positive_pred_id` ở `73.05%` record.
- KL giữa positive và negative branch là khá rõ: mean toàn cục khoảng `2.77`.
- Nhưng `final_pred_id` lại trùng `positive_pred_id` ở `100%` record.

Nói cách khác:

- việc thay `first_token` của negative sample **không phải là quá yếu để negative branch không cảm nhận được**,
- mà là **tín hiệu đó bị triệt tiêu hoặc bị chặn ở stage contrastive combine + candidate filtering + keep-positive**,
- nên cuối cùng top-1 không hề thay đổi.

Đây là pattern thất bại mang tính cơ chế, không phải noise ngẫu nhiên.

## 4. Insight 2: Khoảng cách embedding của first token là tín hiệu rất yếu

Các chỉ số chính:

- `corr(L2, delta_target_logprob) = +0.0184`
- `corr(L2, delta_target_rank) = +0.0009`
- `corr(cos_dist, delta_target_logprob) = +0.0372`
- `corr(cos_dist, delta_target_rank) = +0.0021`

So sánh `kept_top1` với `worsened_not_push_to_top1`:

- chênh lệch mean L2 chỉ khoảng `+0.0097`
- Cohen's d của L2 chỉ khoảng `0.071`
- Cohen's d của cosine distance chỉ khoảng `0.135`

Điều này cho thấy:

- raw embedding distance của `first_token` gần như **không có sức phân biệt thực dụng** cho success/failure,
- nếu chỉ nhìn histogram, boxplot hoặc chia bin theo L2 thì rất dễ over-interpret,
- L2 lớn hơn một chút có xu hướng đi cùng `keep_top1` cao hơn, nhưng effect rất nhỏ và nhiều khả năng là **confounded bởi độ dễ của context**, không phải do khoảng cách embedding thật sự điều khiển outcome.

Một cách nói gọn hơn:

- `first-token embedding distance` hiện tại là một **proxy yếu** cho “độ mạnh của negative sample”.

## 5. Insight 3: Có một artefact thống kê lớn trong `delta_target_rank`

Đây là điểm dễ đọc sai nhất trong report gốc.

### 5.1. Phần lớn `rank_improved_but_not_top1` không phải improvement thật

Trong report hiện tại:

- `rank_improved_but_not_top1 = 12,145`

Nhưng khi tách theo `target_in_candidate_mask`:

| Tập con | Count | Improve | Unchanged | Worse |
|---|---:|---:|---:|---:|
| positive wrong + target reachable | `21,140` | `551` | `19,416` | `1,173` |
| positive wrong + target unreachable | `17,485` | `11,594` | `1,196` | `4,695` |

Tức là:

- `11,594 / 12,145 = 95.46%` số case được gắn nhãn `rank_improved_but_not_top1` đến từ **target đã bị loại khỏi candidate mask**.

Đây không phải rescue thật. Đây là artefact của cách đo rank sau khi target bị gán logit cực tiểu.

### 5.2. Vì sao artefact này xuất hiện

Khi `target_in_candidate_mask = 0`:

- `final_target_logprob` bị đẩy về giá trị cực tiểu của dtype,
- nhiều token ngoài mask cùng bị kéo về mức này,
- hàm tính rank chỉ đếm số token có logit lớn hơn target,
- nên target có thể nhận một rank “đẹp hơn” một cách giả tạo dù thực chất đã bị loại khỏi search space.

Dấu hiệu rõ nhất:

- trong `17,485` unreachable wrong cases, `final_target_rank = 7` xuất hiện tới `12,673` lần,
- và với nhiều case, `final_target_rank = candidate_size + 1`.

Đây là plateau do chỉ còn một số ít candidate sống sót, không phải do target thật sự được contrastive kéo gần lên top.

### 5.3. Hệ quả thực hành

Vì vậy:

- không nên dùng global mean của `delta_target_rank`,
- không nên dùng trực tiếp subtype `rank_improved_but_not_top1`,
- và không nên đọc `not_push` như thể tất cả improvement trong đó đều có nghĩa.

Muốn đọc đúng, bắt buộc phải tách:

1. `target_in_candidate_mask = 0`
2. `target_in_candidate_mask = 1`

## 6. Insight 4: “Thành công” thật sự chỉ là score-only nudge, chưa thành ranking change

Nếu chỉ nhìn reachable wrong cases (`positive_hit = 0`, `target_in_candidate_mask = 1`):

- `18,585 / 21,140 = 87.91%` có `delta_target_logprob > 0`
- nhưng trong số đó:
  - chỉ `221` case có rank improve kèm logprob tăng,
  - `18,364` case còn lại chỉ là **target logprob tăng nhưng rank không đổi**

Tức là:

- contrastive thường đẩy score của target lên đúng hướng,
- nhưng gần như luôn **không đủ mạnh để vượt đối thủ đứng trên**.

Thống kê robust trên reachable wrong cases:

- `delta_target_logprob`: median `+0.153`, p75 `+0.322`, p90 `+0.556`
- `delta_target_rank`: median `0`, p75 `0`, p90 `0`

Diễn giải:

- tác động chủ yếu đang nằm ở mức **reweight nhẹ xác suất**,
- chưa chuyển hóa thành **đảo thứ hạng**,
- nên nếu mục tiêu là rescue contrastive thật sự, chỉ tăng target logprob là chưa đủ.

## 7. Insight 5: Pattern thành công và thất bại phụ thuộc mạnh vào `positive_target_rank`

Nếu chỉ xét reachable wrong cases:

| Positive target rank | Count | `delta_logprob > 0` | Rank improve | Rank worse | `candidate_size <= 6` |
|---|---:|---:|---:|---:|---:|
| `2` | `9,854` | `99.99%` | `0.00%` | `0.01%` | `79.05%` |
| `3` | `3,866` | `99.79%` | `0.00%` | `0.21%` | `61.82%` |
| `4-5` | `3,510` | `96.84%` | `0.46%` | `3.16%` | `39.89%` |
| `6-10` | `2,490` | `30.64%` | `14.78%` | `31.33%` | `4.58%` |
| `11+` | `710` | `0.28%` | `23.52%` | `38.45%` | `0.00%` |

Pattern đọc ra từ bảng này:

### 7.1. Khi target đã gần top (`rank = 2-3`)

- contrastive gần như luôn tăng target logprob,
- nhưng gần như **không bao giờ đổi được rank**.

Đây là pattern “đúng hướng nhưng không đủ lực”.

### 7.2. Khi target ở xa hơn (`rank >= 6`)

- hệ thống bắt đầu có khả năng làm target dịch chuyển rank,
- nhưng movement này **thường mang tính hỗn loạn**:
  - rank-worse còn nhiều hơn rank-improve,
  - tỉ lệ `delta_logprob > 0` giảm rất mạnh.

Đây là pattern “có room để di chuyển, nhưng negative perturbation không đủ target-aware nên di chuyển chủ yếu là nhiễu”.

Nói gọn:

- target gần top: có tín hiệu tốt nhưng bị chặn,
- target xa top: có tự do di chuyển nhưng di chuyển sai hướng nhiều hơn đúng hướng.

## 8. Insight 6: `n_keep = 6` đang tạo một vùng “đóng băng” cơ chế

Trong `analysis/CDv2_embed_distance_ana.py`, phần combine có đoạn:

```python
n_keep = min(6, final_draft_logits.size(-1))
top_indices = torch.topk(final_draft_logits, k=n_keep, dim=-1).indices
positive_top_values = positive_draft_logits.gather(dim=-1, index=top_indices)
final_draft_logits.scatter_(dim=-1, index=top_indices, src=positive_top_values)
```

Hệ quả trực tiếp:

- nếu `candidate_size <= 6`, toàn bộ candidate đang sống có thể bị reset về positive scores,
- nên contrastive gần như không còn cơ hội đổi thứ hạng trong candidate set.

Đúng như dữ liệu cho thấy, trong reachable wrong cases:

- `12,096 / 21,140 = 57.22%` có `candidate_size <= 6`
- và trên nhóm `candidate_size = 2..6`, rank improve = `0%`, rank worse = `0%`

Đây là một kết quả rất mạnh:

- hơn nửa reachable error hiện tại là **frozen-by-design**,
- nên dù thay `first_token` negative sample có tạo ra tín hiệu nào đi nữa, nó cũng không thể hiện ra ở final rank.

Ngay cả khi `candidate_size > 6`, movement mới xuất hiện, nhưng vẫn yếu:

- `candidate_size = 7`: improve `1.97%`, worse `2.68%`
- `candidate_size = 10`: improve `5.85%`, worse `10.99%`
- `candidate_size = 11`: improve `7.34%`, worse `13.42%`

Tức là:

- khi cơ chế không còn bị đóng băng hoàn toàn, movement xuất hiện,
- nhưng vẫn thiên về regression hơn là rescue.

## 9. Insight 7: KL divergence hữu ích hơn L2, nhưng vẫn không đủ để tạo correction

So với raw L2, KL giữa positive và negative branch có tín hiệu tốt hơn:

- `corr(KL, keep_top1) = +0.241`
- `corr(KL, accepted_by_final) = +0.312`
- `corr(KL, delta_logprob_finite) = +0.102`
- `corr(KL, L2) = +0.062`

Điều này cho thấy:

- mức divergence thật ở logit-space mới là thứ gần với “tác động thực” hơn,
- còn raw first-token embedding distance chỉ là một proxy khá xa.

Tuy nhiên, ngay cả khi KL lớn:

- negative branch hầu như luôn đổi argmax,
- nhưng final argmax vẫn không đổi.

Vì vậy, KL lớn hơn chỉ cho biết perturbation đã vào negative branch, chứ chưa nói gì về khả năng rescue ở final branch.

## 10. Insight 8: Identity của random negative token ảnh hưởng ít hơn độ khó của context

Phân loại thô token type cho thấy:

- `negative_first_token_text` thuộc nhóm `ascii_alpha` chiếm `58.84%`
- `non_ascii` chiếm `36.97%`

Nhưng keep-rate theo negative token category của các nhóm lớn khá sát nhau:

- `ascii_alpha`: `56.88%`
- `non_ascii`: `58.10%`
- `ascii_symbol`: `58.37%`

Trong khi đó, keep-rate theo **positive first token category** chênh rõ hơn:

- `ascii_alpha`: `52.92%`
- `ascii_symbol`: `61.82%`
- `ascii_digit`: `70.67%`
- `whitespace`: `62.47%`

Tức là:

- kiểu token gốc của context phản ánh độ dễ/khó của event mạnh hơn,
- còn việc negative first token rơi vào loại nào chỉ tạo chênh lệch nhỏ.

Nói ngắn gọn:

- outcome hiện tại bị chi phối nhiều hơn bởi **cấu trúc context và độ gần của target với frontier**,
- chứ không bị chi phối mạnh bởi việc random token thay vào là “weird token” hay “common token”.

## 11. Pattern thành công và không thành công

### 11.1. Pattern “thành công” theo nghĩa yếu

Nếu nới định nghĩa success từ “push to top-1” sang “đẩy target đi đúng hướng”, thì pattern có ích nhất hiện tại là:

- `target_in_candidate_mask = 1`
- `positive_target_rank = 2` hoặc `3`
- `delta_target_logprob > 0`
- final rank giữ nguyên

Đây là pattern:

- contrastive có tín hiệu đúng,
- nhưng đang bị combine stage chặn nên không biến thành rescue thật.

### 11.2. Pattern thất bại loại 1: target bị mask loại khỏi candidate set

Đây là failure mode lớn nhất về mặt thống kê:

- `17,485` case wrong nằm ở nhóm này
- mọi “improvement rank” trong nhóm này gần như không có giá trị cơ chế

### 11.3. Pattern thất bại loại 2: target còn reachable nhưng candidate set quá nhỏ

Đây là failure mode mang tính thiết kế:

- `candidate_size <= 6`
- toàn bộ candidate dễ bị reset về positive scores
- contrastive không có không gian để thay ranking

### 11.4. Pattern thất bại loại 3: target đủ xa để có movement, nhưng movement chủ yếu sai hướng

Khi `positive_target_rank >= 6`:

- movement bắt đầu xuất hiện,
- nhưng `rank_worse > rank_improve`,
- cho thấy perturbation hiện tại không phải “hard negative có hướng”, mà giống một nguồn nhiễu không ổn định hơn.

## 12. Kết luận cơ chế

Kết quả của folder này cho thấy:

1. Thay `first_token` của negative sample **không vô dụng hoàn toàn**, vì nó làm negative branch đổi mạnh.
2. Nhưng tín hiệu đó **không đi qua được lớp combine hiện tại** để đổi final argmax.
3. `first_token` embedding distance **không phải là control knob tốt** để dự đoán rescue.
4. Phần lớn “improvement” trong report gốc thực chất là **artefact của hard mask**.
5. Trên tập reachable thật sự, hiệu ứng quan sát được chủ yếu là **score-only nudge**, chưa phải **ranking correction**.

Vì vậy, insight chính không phải là:

- “L2 lớn thì tốt hơn”,

mà là:

- “cơ chế hiện tại đang chặn hầu hết tác động của negative perturbation trước khi nó có thể trở thành rescue”.

## 13. Khuyến nghị thí nghiệm tiếp theo

### 13.1. Đổi cách report

Bắt buộc báo cáo tách riêng:

1. `target_in_candidate_mask = 0`
2. `target_in_candidate_mask = 1`

Và nên thêm 4 metric chính:

- `final_argmax_change_rate`
- `negative_argmax_change_rate`
- `reachable_rank_lift_rate`
- `score_only_nudge_rate`

### 13.2. Sweep `n_keep`

Nên chạy ablation với:

- `n_keep = 0`
- `n_keep = 1`
- `n_keep = 2`
- `n_keep = 6`

để đo xem exact rescue có xuất hiện không khi bỏ bớt vùng frozen.

### 13.3. Không dùng random token thuần túy làm negative first token

Với dữ liệu hiện tại, random replacement trên toàn vocab:

- tạo L2 variation nhưng ít giá trị giải thích,
- tạo negative divergence nhưng không đủ target-aware.

Hợp lý hơn là thử:

- nearest-neighbor hard negatives,
- confusion-aware negatives quanh top competitor,
- hoặc negatives được ràng buộc theo cùng kiểu token / cùng cụm embedding.

### 13.4. Dùng KL hoặc argmax-change của negative branch làm biến điều khiển chính

Nếu muốn nghiên cứu “độ mạnh của perturbation”, nên ưu tiên:

- `step_kl_positive_vs_negative`
- `negative_pred_id != positive_pred_id`

hơn là chỉ nhìn `first_token_embedding_l2`.

## 14. Ghi chú implementation cần kiểm tra thêm

`analysis/CDv2_embed_distance_ana.py` đang dùng logic keep-positive trên **top vocab entries** của `final_draft_logits`.

Trong khi đó `scheme/CD_v2.py` hiện có đoạn:

```python
final_draft_logits[:, :n_keep, :] = positive_draft_logits[:, :n_keep, :]
```

Đoạn này tác động theo **sequence dimension**, không giống logic trong file analysis.

Nếu đây là code path thật dùng trong runtime, thì:

- analysis hiện tại không hoàn toàn isomorphic với runtime,
- và nên kiểm tra lại xem đây có phải bug hay chủ ý hay không trước khi kết luận về deploy-time behavior.
