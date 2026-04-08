# Phân tích insight cho run `reject_tok_gsm8k_20260402_014507` (CDv4)

## 1. Phạm vi và cách đọc của run này

Nguồn dùng để phân tích:

- `analysis/results/reject_tok_gsm8k_20260402_014507/summary.json`
- `analysis/results/reject_tok_gsm8k_20260402_014507/reject_records.jsonl`
- `analysis/results/reject_tok_gsm8k_20260402_014507/report.md`
- File tham chiếu cách viết: `analysis/results/reject_tok_gsm8k_20260326_105221/vcd_reject_token_insights.md`

Meta của run:

- Dataset: `gsm8k`
- `block_size = 16`
- `vcd_alpha = 0.1`
- `vcd_beta = 0.1`
- `negative_context_dropout = 0.3`
- `negative_context_noise_std = 0.0`
- `evaluation_mode = positive_rollout_with_contrastive_shadow`
- `compare_name = CDv4`

Phạm vi event-level trong folder này là `128` sample, `sample_idx` chạy từ `0` đến `127`. Tổng cộng có `5491` reject event; trung bình `42.90` reject mỗi sample, median `40`, max `187`, min `19`.

Cần nhấn mạnh: đây không phải đánh giá chất lượng rollout CDv4 end-to-end. Đây là phân tích tại vị trí reject trên quỹ đạo positive rollout, sau đó tính thêm một nhánh contrastive shadow để trả lời:

1. CDv4 có đổi top-1 tại vị trí reject không?
2. Nếu có đổi, nó có đưa token về đúng posterior target không?
3. Nếu không sửa được, là do target bị candidate mask loại hay do contrastive score quá yếu?

Nói ngắn gọn: folder này đo **khả năng can thiệp ở first-error token**, không đo toàn bộ rollout khi để CDv4 điều khiển thực sự.

## 2. Headline kết quả

| Chỉ số | Giá trị | Diễn giải ngắn |
|---|---:|---|
| Reject events | 5491 | số vị trí bị reject |
| `positive_hit_rate` | 1.18% | Ở một số ít event, positive step logits vẫn đạt target top-1 |
| `vcd_hit_rate` | 0.00% | CDv4 không có exact fix nào ở reject token. |
| `pred_changed_rate` | 2.29% | CDv4 rất hiếm khi thực hiện đổi top-1 |
| `target_in_candidate_mask_rate` | 67.66% | Khoảng 2/3 reject còn reachable về mặt mask |
| `no_vcd_effect_positive_wrong` | 65.78% | Phần lớn reject CDv4 gần như không có tác động |
| `target_filtered_by_candidate_mask` | 32.34% | Gần 1/3 reject bị chặn ngay từ candidate mask |
| `vcd_regression_from_positive` | 1.18% | Có một nhóm nhỏ mà CDv4 làm xấu đi so với positive |
| `vcd_shifted_but_still_wrong` | 0.69% | Có đổi hướng nhưng không sửa được reject. |

Headline quan trọng nhất của run này là:

- CDv4 gần như **không can thiệp** vào reject token.
- Khi can thiệp, nó **không tạo ra exact fix nào**.
- Số ít lần can thiệp hiện có còn bao gồm cả **regression** từ những event mà positive logits đang đúng.

## 3. Phân ra các bottleneck chính

### 3.1 Failure mode lớn nhất là "không đổi được top-1"

Trong `5491` reject:

- `5365` event, tương đương `97.71%`, CDv4 không đổi prediction.
- Chỉ `126` event, tương đương `2.29%`, là CDv4 có đổi prediction.

Ngay cả trong nhóm target vẫn còn nằm trong candidate mask (`3715` event), tỉ lệ đổi prediction cũng chỉ là `2.77%`. Nghĩa là vấn đề chính không chỉ là hard mask; ngay cả trên reachable set, tín hiệu contrastive của CDv4 cũng quá yếu để lật top-1 của positive.

Điều này rất khác với run tham chiếu `reject_tok_gsm8k_20260326_105221`, nơi VCD shadow trước đó có:

- `pred_changed_rate = 26.13%`
- `vcd_hit_rate = 13.16%`

Trong khi run CDv4 hiện tại chỉ còn:

- `pred_changed_rate = 2.29%`
- `vcd_hit_rate = 0.00%`

Nếu chỉ nhìn ở behavioral level, CDv4 trong run này giống một **perturbation rất nhẹ** hơn là một local reranker có khả năng sửa reject.

### 3.2 Candidate mask vẫn là hard ceiling rất lớn

`1776` event, tương đương `32.34%`, rơi vào taxonomy `target_filtered_by_candidate_mask`. Ở những event này, target bị loại khỏi candidate set ngay từ đầu, nên CDv4 không còn quyền chọn target dù contrastive score có muốn đẩy lên cũng không được.

Một vài dấu hiệu rõ:

- `target_in_candidate_mask_rate = 67.66%`.
- Trong nhóm `target_in_mask = 0`, `vcd_hit_rate = 0%`.
- `delta_target_logprob.mean` của toàn run thành `-Infinity` vì các event bị mask khiến logit target về giá trị cực tiểu của dtype.

Do vị trí của target trong positive ranking liên quan rất mạnh đến khả năng sống sót qua mask:

| Positive target rank | Tỷ lệ target còn trong mask |
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

Nghĩa là khi target không còn nằm sát frontier của positive, hard beta mask gần như đóng cửa hoàn toàn.

Case đại diện:

- `sample=0`, `decode_step=14`, `pos=141`.
- target posterior là ` how`.
- positive/CDv4 đều chọn ` Samantha`.
- `positive_target_rank = 19`.
- `target_in_candidate_mask = 0`.
- posterior lại ưu tiên ` how` rất mạnh: `p(target) = 0.9225`, trong khi `p(sampled) = 0.0316`.

Đây là nhóm reject "unrecoverable by design" dưới cấu hình mask hiện tại.

### 3.3 Trên những lần CDv4 có can thiệp, một nửa là regression

Trong `126` event mà `pred_changed = 1`, taxonomy tách ra như sau:

- `65` event (`51.6%`) là `vcd_regression_from_positive`.
- `38` event (`30.2%`) là `vcd_shifted_but_still_wrong`.
- `23` event (`18.3%`) vẫn là `target_filtered_by_candidate_mask`, tức CDv4 có đổi token nhưng vẫn không thể tới target.

Nhóm regression đặc biệt đáng chú ý:

- Tất cả `65` event regression đều có `positive_hit = 1`.
- Tất cả đều có `positive_target_rank = 1`.
- Nhưng CDv4 lại đẩy target xuống, nên `vcd_hit = 0`.

Nói cách khác: những lần CDv4 thực sự "phá" ở run này phần lớn là phá trên các event mà positive step logits đang xếp target đúng top-1.

Case đại diện:

- `sample=11`, `decode_step=4`, `pos=127`.
- target là ` the`.
- positive top-1 là ` the` với `p = 0.3077`.
- CDv4 đổi top-1 sang ` how` với `p = 0.3513`, đẩy ` the` xuống rank-2.
- posterior lại rất rõ ràng ưu tiên ` the` (`p = 0.9960`).

Điểm cần lưu ý là `delta_target_logprob` ở case này vẫn dương `+0.1285`. Tức là chỉ nhìn vào target logprob sẽ dễ đánh giá sai: CDv4 có thể tăng score của target, nhưng vẫn tăng score của token sai nhiều hơn và làm top-1 bị lật.

### 3.4 Có một nhóm "có ích nhưng chưa đủ" rất nhỏ

`38` event `vcd_shifted_but_still_wrong` cho thấy CDv4 không hoàn toàn vô dụng. Ở nhóm này:

- `pred_changed = 1`.
- `positive_hit = 0`.
- `vcd_hit = 0`.
- mean `delta_target_logprob = +0.2003`.
- `78.95%` event có `delta_target_logprob > 0`.

Tức là trên nhóm nhỏ này, CDv4 thường đẩy target lên gần hơn, nhưng vẫn chưa đủ để lên top-1.

Case đại diện:

- `sample=71`, `decode_step=11`, `pos=141`.
- positive top-1 là ` squirt`.
- CDv4 đổi top-1 sang ` bottles`.
- target posterior là ` guns`.
- xác suất của ` guns` được nâng từ `0.0497` lên `0.1542`.
- nhưng ` bottles` vẫn top-1, nên reject không được sửa.

Nhóm này cho thấy CDv4 vẫn có một ít tín hiệu cục bộ, nhưng cường độ hiện tại quá nhẹ để biến thành exact correction.

## 4. Cách đọc `delta_target_logprob`

`summary.json` ghi:

```json
"delta_target_logprob": {
  "mean": -Infinity,
  "median": 0.0,
  "p10": -3.3895313892515355e+38,
  "p90": 0.1075512170791626
}
```

Số `-Infinity` này không có nghĩa là mọi reject đều bị CDv4 làm tệ đi. Nó chủ yếu do:

1. target bị loại khỏi `candidate_mask`.
2. logit của target bị gán thành giá trị cực tiểu của dtype.
3. khi tổng hợp bằng `float32`, mean bị overflow thành `-Infinity`.

Nếu chỉ nhìn trên reachable set (`target_in_mask = 1`), bức tranh đúng hơn là:

- mean `delta_target_logprob = +0.0332`.
- improved-rate `delta_target_logprob > 0` là `21.21%`.
- nhưng `vcd_hit_rate` vẫn là `0%`.
- `pred_changed_rate` trên reachable set vẫn chỉ là `2.77%`.

Insight ở đây là:

- CDv4 có lúc tăng score của target.
- nhưng thường **không tăng đủ mạnh** để vượt qua token đang đứng ở top-1.
- vì thế nếu chỉ đọc logprob mà không đọc exact-hit và changed-rate, rất dễ over-estimate tác dụng của CDv4.

Case `sample=110`, `decode_step=13`, `pos=173` là ví dụ điển hình:

- target posterior là token khoảng trắng ` `.
- positive và CDv4 đều giữ top-1 là `:`.
- nhưng `p(target)` được nâng từ `0.0972` lên `0.2866`.
- posterior lại gần như chắc chắn chọn target (`p = 0.99995`).

Đây là một near-miss rõ ràng: CDv4 đẩy đúng hướng, nhưng không đủ lực để lật token sai.

## 5. Kết luận tổng hợp

Run `reject_tok_gsm8k_20260402_014507` cho thấy CDv4 trong cấu hình hiện tại (`alpha=0.1`, `beta=0.1`) đang gặp ba giới hạn lớn:

1. Candidate mask vẫn chặn mất khoảng `1/3` reject ngay từ đầu.
2. Trên phần reachable còn lại, CDv4 rất ít khi đổi được top-1.
3. Trong số ít lần đổi top-1, một nửa lại là regression từ các case mà positive step logits đang đúng.

Kết quả thực nghiệm quan trọng nhất của folder này là:

- CDv4 hiện chưa hoạt động như một cơ chế rescue reject hiệu quả.
- Tác dụng quan sát được chủ yếu là nhích target logprob ở một số case, chưa chuyển thành exact fix.
- So với run VCD tham chiếu trong folder `20260326_105221`, độ mạnh can thiệp của CDv4 hiện tại giảm rất rõ.

## 6. Hướng debug / thí nghiệm tiếp theo

Từ kết quả trên, những hướng ưu tiên để kiểm tra tiếp là:

1. Tăng độ mạnh contrastive.
   - `pred_changed_rate = 2.29%` quá thấp, nên cần xem `alpha=0.1` có quá nhẹ không.

2. Giảm độ chặt của hard mask.
   - Vì `32.34%` reject bị filter ngay từ đầu, có thể thử `beta` nhỏ hơn hoặc soft-mask thay vì hard-mask.

3. Kiểm tra riêng nhóm regression.
   - `65` event regression đều có `positive_target_rank = 1`; đây là dấu hiệu rất cụ thể rằng CDv4 đang phá frontier ở một nhóm case lẽ ra có thể giữ nguyên.

4. Theo dõi thêm metric "top-1 margin sau contrastive".
   - Run này có nhiều case tăng target logprob nhưng không đổi top-1; margin giữa target và token đang top-1 sẽ giúp thấy rõ lý do.

5. Tách đánh giá thành hai chế độ.
   - `reachable set`: target còn trong mask.
   - `unreachable set`: target bị filter.
   Việc gom chung hai nhóm này khiến một số mean metric, đặc biệt `delta_target_logprob`, dễ gây hiểu nhầm.

---

Nếu bạn muốn, tôi có thể tiếp tục: (a) rà soát ngôn ngữ cho trôi chảy hơn nữa; hoặc (b) tạo phiên bản tiếng Anh song song.
