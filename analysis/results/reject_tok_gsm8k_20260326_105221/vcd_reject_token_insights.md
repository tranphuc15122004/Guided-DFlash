# Phân tích `check_reject_tok.py` và insight về tác động của VCD lên drafting ở mức token

## 1. Phạm vi và cách đọc đúng run này

Nguồn dùng để phân tích:

- Code: `Analysis/check_reject_tok.py`
- Kết quả gốc: `Analysis/results/reject_tok_gsm8k_20260326_105221/summary.json`
- Event-level data: `Analysis/results/reject_tok_gsm8k_20260326_105221/reject_records.jsonl`
- Report auto-generated: `Analysis/results/reject_tok_gsm8k_20260326_105221/report.md`

Các meta đã ghi lại trong `summary.json`:

- Dataset: `gsm8k`
- `block_size = 16`
- `vcd_alpha = 0.5`
- `vcd_beta = 0.1`
- `negative_context_dropout = 0.3`
- `negative_context_noise_std = 0.0`
- `evaluation_mode = positive_rollout_with_vcd_shadow`
- `vcd_shadow_num_samples = 1`

Phạm vi dữ liệu thực tế trong folder này là `128` sample, với `sample_idx` chạy liên tiếp từ `0` đến `127`, không phải toàn bộ GSM8K test. Tổng cộng có `5496` reject event, tức trung bình khoảng `42.94` reject mỗi sample, median `39.5`, max `189`.

Điểm quan trọng nhất để đọc đúng run này là: đây **không phải** rollout VCD thật. Ở `Analysis/check_reject_tok.py:738-749`, script luôn giữ quỹ đạo generate bằng `positive_sampled_tokens`, còn VCD chỉ được tính ở chế độ shadow để hỏi:

1. Ở đúng vị trí reject của positive draft, VCD có đổi token không?
2. Nếu đổi, VCD có kéo token về đúng posterior token không?
3. Nếu không kéo được, nguyên nhân đến từ candidate mask hay từ contrastive score chưa đủ mạnh?

Nói ngắn gọn: run này đo **khả năng cứu reject tại first-error token**, chứ chưa đo toàn bộ chất lượng rollout khi thực sự để VCD điều khiển cả block.

## 2. Script đang đo cái gì ở mức token

Pipeline cốt lõi của script là:

1. Dựng `positive_draft_logits` và `negative_draft_logits` bằng cách thay context target hidden bằng một phiên bản negative có dropout/noise, đồng thời randomize token đầu của block negative (`Analysis/check_reject_tok.py:110-138`).
2. Tạo `candidate_mask` từ positive distribution: chỉ giữ các token có xác suất lớn hơn hoặc bằng `beta * max_prob`, đồng thời ép top-1 của positive luôn được giữ lại (`Analysis/check_reject_tok.py:141-147`).
3. Tính VCD logits theo công thức imported từ `model/utils.py:38-45`:
   `final = (1 + alpha) * positive - alpha * negative`
4. Filter VCD logits bằng candidate mask (`Analysis/check_reject_tok.py:150-151`, `738-744`).
5. Positive draft vẫn là quỹ đạo thật; VCD chỉ sample shadow (`Analysis/check_reject_tok.py:745-749`).
6. Reject event được ghi khi positive draft token đầu tiên trong block không khớp posterior token của target model ở bước verify (`Analysis/check_reject_tok.py:762-769`).

Vì vậy, reject record ở đây mô tả đúng một tình huống rất cụ thể:

- Positive draft đã chọn sai token so với posterior.
- Ta xem tại chính token đó, VCD có giúp chọn lại token tốt hơn không.

Đây là thiết kế tốt để tách riêng cơ chế sửa lỗi local của VCD khỏi hiệu ứng dây chuyền của rollout dài.

## 3. Kết quả headline, nhưng đã hiệu chỉnh cách diễn giải

### 3.1 Các con số có ý nghĩa trực tiếp

| Chỉ số | Giá trị | Diễn giải đúng |
|---|---:|---|
| Reject events | 5496 | Số first-reject token được quan sát trên 128 sample |
| `target_in_candidate_mask_rate` | 67.56% | Khoảng 2/3 reject còn “reachable” đối với VCD |
| `pred_changed_rate` | 26.13% | VCD chỉ thật sự can thiệp top-1 ở khoảng 1/4 reject |
| `vcd_hit_rate` | 13.16% | Tỉ lệ VCD shadow đổi đúng về posterior token trên toàn bộ reject |
| `vcd_hit_rate | target_in_mask=1` | 19.47% | Nếu target còn nằm trong mask, xác suất cứu reject tăng lên gần 1/5 |
| `vcd_hit_rate | pred_changed=1` | 50.35% | Mỗi khi VCD thật sự đổi token, khoảng một nửa số lần đổi là đổi đúng |

### 3.2 Decomposition theo bottleneck

Trong 5496 reject:

- `1783` event, tương đương `32.44%`, rơi vào `target_filtered_by_candidate_mask`
- `2537` event, tương đương `46.16%`, là `no_vcd_effect_positive_wrong`
- `453` event, tương đương `8.24%`, là `vcd_shifted_but_still_wrong`
- `723` event, tương đương `13.16%`, là `vcd_shadow_can_fix_reject`

Điều này cho thấy VCD đang vấp hai tầng bottleneck rất rõ:

1. **Gate bottleneck**: `32.44%` reject là bất khả cứu với cấu hình `beta=0.1` hiện tại vì target bị loại khỏi candidate mask.
2. **Rerank bottleneck**: Trong `3713` reject mà target vẫn còn trong mask, VCD vẫn:
   - sửa đúng `19.47%`
   - đổi token nhưng vẫn sai `12.20%`
   - không đổi được top-1 và vẫn sai `68.33%`

Nói cách khác, candidate mask là trần cứng đầu tiên; còn sau khi đã qua trần đó, bài toán chính là VCD chưa đủ lực để lật được top-1 của positive ở đa số case.

## 4. Insight sâu về cơ chế tác động của VCD

### 4.1 VCD ở run này là một cơ chế rerank cục bộ, không phải cơ chế “retrieval” token từ sâu trong tail

Đây là insight quan trọng nhất.

Tỉ lệ target lọt vào candidate mask giảm rất nhanh khi target đi xuống trong positive ranking:

| Positive target rank | Tỉ lệ target còn trong mask | Tỉ lệ VCD sửa đúng |
|---|---:|---:|
| 1 | 100.00% | 42.22% |
| 2 | 87.04% | 19.57% |
| 3 | 70.81% | 7.50% |
| 4 | 54.81% | 4.69% |
| 5 | 37.61% | 1.71% |
| 6-10 | 13.19% đến 33.54% | gần 0% đến 1.86% |
| >10 | 1.99% | 0.00% |

Ngoài ra:

- `79.39%` toàn bộ fix của VCD đến từ các case mà target đang ở `positive_target_rank = 2`
- `7.88%` fix đến từ các case target đã ở `rank = 1` nhưng thua ở bước tie-breaking
- Chỉ `0.69%` fix đến từ case `positive_target_rank > 5`
- Trong các case target còn trong mask, target đã nằm trong `positive top-5` tới `96.55%`
- Trong các case VCD sửa thành công, target nằm trong `positive top-5` tới `99.17%`

Kết luận:

- VCD mạnh nhất khi target đã ở rất gần frontier của positive draft, đặc biệt là rank-2.
- VCD gần như không “kéo” được target từ vùng rank sâu.
- Về bản chất token-level, VCD đang hoạt động như một **local reranker / local disambiguator**, chứ chưa phải một cơ chế phục hồi token sâu ngoài positive frontier.

Đây là đúng với trực giác của công thức contrastive: nó sửa những near-miss local tốt hơn nhiều so với việc truy hồi một token mà positive gần như không cân nhắc.

### 4.2 Candidate mask đang là hard ceiling lớn nhất của VCD

`candidate_mask` được xây từ positive distribution, với ngưỡng `prob >= beta * max_prob` (`Analysis/check_reject_tok.py:141-147`). Sau đó mọi token ngoài mask bị gán logit cực tiểu (`Analysis/check_reject_tok.py:150-151`), nên target bị lọc là coi như VCD không còn quyền chọn target nữa.

Hệ quả trong dữ liệu:

- `32.44%` reject là **unrecoverable by design** dưới `beta = 0.1`
- Ở các event bị filter, `vcd_hit_rate = 0%`
- Mean positive target probability của nhóm bị filter chỉ khoảng `0.0202`, median `0.0124`, tức target vốn đã ở dưới ngưỡng cạnh tranh của frontier positive

Ví dụ đại diện:

- `sample=14`, `decode_step=32`, `pos=290`
- target posterior là `third`, positive draft chọn `child`
- `positive_target_rank = 5`, target không nằm trong candidate mask
- posterior lại ưu tiên `third` hơn `child` rất mạnh, với `target_minus_sampled_posterior_logprob = 56.25`
- VCD có đổi token từ `child` sang `street`, nhưng vẫn không thể chọn `third` vì target đã bị mask loại

Ý nghĩa nghiên cứu:

- Nếu mục tiêu là cải thiện drafting ở mức token, thì tuning `beta` hoặc thay hard mask bằng soft mask sẽ là hướng đòn bẩy rất lớn.
- Hiện tại, gần 1/3 reject bị “chặn cửa” trước khi contrastive score kịp phát huy.

### 4.3 Một phần lợi ích thật của VCD là phá tie đúng hướng, và metric rank hiện tại đang undercount lợi ích này

Script dùng hàm rank:

`rank(token) = (#logits > token_logit) + 1` tại `Analysis/check_reject_tok.py:161-163`

Vì dùng dấu `>` thay vì `>=`, nên các token đồng logit top-1 đều có rank bằng 1. Trong dữ liệu có:

- `135` reject mà `positive_target_rank = 1` nhưng `positive_hit = 0`
- Ở cả `135` case này, `positive_target_logprob == positive_sampled_logprob` chính xác
- Điều đó nghĩa là target bị thua không phải do điểm thấp hơn, mà do **tie-breaking của argmax/sample**

Trong 135 case này, VCD sửa đúng được `57` case, tương đương `42.22%`.

Ví dụ:

- `sample=0`, `decode_step=7`, `pos=115`
- positive chọn `are`, target là `need`
- `positive_target_logprob = positive_sampled_logprob = -0.9131`
- rank của target không đổi (`delta_target_rank = 0`), nhưng VCD vẫn chuyển top-1 sang `need` và cứu reject

Điểm rất quan trọng:

- Có đúng `57` fix của VCD mà `delta_target_rank = 0`
- Tức `7.88%` toàn bộ fix của VCD là **invisible nếu chỉ nhìn rank improvement**

Vì vậy, khi đánh giá VCD ở mức token, không nên chỉ dựa vào `delta_target_rank`; cần xem thêm exact-hit và các case tie.

### 4.4 Khi VCD thực sự can thiệp, nó thường đẩy draft về gần posterior hơn

Trong `1436` event mà VCD đổi prediction (`pred_changed = 1`):

- `50.35%` trở thành exact fix
- `72.35%` có `vcd_shadow_sampled_rank_in_posterior` tốt hơn token positive ban đầu
- median posterior rank của token shadow giảm từ `4` xuống `1`

Điều này rất đáng chú ý: không phải cứ VCD đổi token là đổi ngẫu nhiên. Phần lớn các lần đổi token thật sự đang di chuyển draft theo hướng posterior-compatible hơn.

Ví dụ partial improvement nhưng chưa đủ:

- `sample=126`, `decode_step=113`, `pos=643`
- target là `Unknown`
- positive chọn khoảng trắng ` `
- VCD chuyển sang `?`
- target được kéo từ `positive_target_rank = 10` lên `vcd_target_rank = 4`
- posterior gap với token VCD tốt hơn hẳn token positive cũ (`4.125` xuống `1.875`)
- nhưng vì chưa lên top-1 nên reject vẫn xảy ra

Điều này gợi ý một insight tinh hơn:

- VCD không chỉ có hai trạng thái “sửa đúng” hoặc “vô dụng”
- Có một vùng trung gian khá thực: VCD đã làm draft “ít sai hơn”, nhưng chưa đủ để vượt ngưỡng acceptance ở first reject token

### 4.5 Phần lớn fix đến từ near-miss rank-2, tức VCD đang chỉnh đúng chỗ mô hình draft “gần đúng nhưng chưa đủ chắc”

Trong toàn bộ `723` fix:

- `574` fix đến từ `positive_target_rank = 2`
- `57` fix đến từ tie `positive_target_rank = 1`
- `64` fix đến từ `rank = 3`
- chỉ `28` fix còn lại đến từ `rank >= 4`

Case đại diện cho “rank-2 rescue”:

- `sample=70`, `decode_step=19`, `pos=175`
- positive chọn `two`
- posterior token là `told`
- target đang ở rank-2 dưới positive
- VCD kéo target lên rank-1, với `delta_target_logprob = +2.266`
- posterior cũng ưu tiên `told` hơn `two` rất mạnh (`target_minus_sampled_posterior_logprob = 19.125`)

Ý nghĩa:

- VCD phát huy tốt nhất ở các token mà positive draft đã chứa đủ signal để “biết gần đúng”, nhưng vẫn bị nhiễu local đánh bại.
- Đây chính là kiểu lỗi mà contrastive drafting nên sửa: những lỗi cục bộ ở frontier, không phải lỗi coverage toàn bộ không gian token.

## 5. Câu chuyện logprob: report auto-generated đúng số, nhưng dễ bị hiểu sai

`report.md` ghi `delta_target_logprob` mean là `-inf`. Số này **không sai**, nhưng rất dễ bị diễn giải sai.

Lý do:

- Khi target bị loại khỏi candidate mask, `apply_vcd_candidate_filter` gán logit target về `torch.finfo(dtype).min`
- Sau softmax/log-softmax, logprob target của VCD trở thành một giá trị âm cực lớn
- Vì vậy mean toàn cục bị thống trị bởi các event “target bị filter”, chứ không phản ánh chất lượng rerank trên các event còn reachable

Nếu đọc đúng theo điều kiện:

- Trên toàn bộ nhóm `target_in_mask = 1`, mean `delta_target_logprob = -0.240`, median `-0.051`
- Trên nhóm `pred_changed = 1`, mean `delta_target_logprob = +0.360`, median `+0.523`
- Trên nhóm fix thật (`vcd_hit = 1`), mean `delta_target_logprob = +0.834`, median `+0.730`
- Mean target probability trong nhóm fix tăng từ `0.289` lên `0.630`

Diễn giải đúng là:

- Nếu tính cả event unreachable, logprob delta toàn cục sẽ trông rất xấu vì hard mask.
- Nếu chỉ nhìn những lần VCD thực sự can thiệp, VCD thường tăng xác suất của target.
- Nhưng số case no-op và wrong-shift vẫn còn nhiều, nên trung bình có điều kiện trên toàn bộ reachable set chưa dương.

Nói cách khác: VCD hiện tại là một cơ chế có **high payoff on wins**, nhưng coverage của vùng win vẫn chưa đủ rộng.

## 6. Các metric trong report hiện tại không nên over-interpret

Một số metric trong `report.md` là hệ quả cấu trúc của setup, không nên đọc như tín hiệu độc lập:

1. `positive_hit_rate = 0%`
   - Đây gần như là hệ quả tất yếu vì reject event được định nghĩa trên quỹ đạo positive rollout.
   - Ở `Analysis/check_reject_tok.py:745` positive token được lấy bằng `sample(positive_draft_logits)`; với `model/utils.py:28-35`, khi không truyền temperature thì đây là greedy argmax.
   - Do reject chỉ được ghi khi positive token khác posterior token, nên `positive_hit` về bản chất gần như luôn bằng 0.

2. `target_is_posterior_argmax_rate = 100%`
   - Đây cho thấy posterior trong run này về thực chất cũng đang greedy.
   - Vì vậy metric này không cung cấp thêm insight ngoài việc xác nhận setup gần với `temperature = 0`.

3. `sampled_in_candidate_mask_rate = 100%`
   - Đây là hệ quả trực tiếp của `candidate_mask.scatter_(top_token_indices, True)` tại `Analysis/check_reject_tok.py:145-146`.
   - Positive sampled token là top-1 của positive, nên luôn ở trong mask.

4. `vcd_shadow_fix_rate` trùng hoàn toàn `vcd_hit_rate`
   - Trong run này, `vcd_pred_id == vcd_shadow_sampled_id` cho mọi record, vì VCD shadow token cũng được lấy greedy ở `Analysis/check_reject_tok.py:746`.
   - Do đó hai metric này không độc lập.

5. `positive_rescue_prob_est` và `vcd_shadow_rescue_prob_est`
   - Với `vcd_shadow_num_samples = 1`, hai metric này chỉ nhận giá trị `0` hoặc `1`.
   - Chúng là một Bernoulli Monte Carlo một mẫu, không phải estimate xác suất ổn định.
   - Có thể giữ như một tín hiệu phụ, nhưng không nên dùng làm nền tảng lập luận chính.

## 7. Kết luận thực chất về tác động của VCD lên drafting ở mức token

Từ run này, có thể rút ra kết luận đúng đắn và khá rõ:

1. VCD **có cải thiện drafting ở mức token**, nhưng improvement tập trung vào first-reject token nơi target đã ở rất gần positive frontier.
2. Tác dụng chính của VCD là **rerank local**:
   - phá tie đúng hướng
   - kéo rank-2 lên rank-1
   - đôi khi kéo token sai về gần posterior hơn dù chưa thành exact hit
3. VCD **không phải** công cụ phục hồi token ở vùng rank sâu dưới cấu hình hiện tại; candidate mask `beta=0.1` chặn mất khoảng `32.44%` reject ngay từ đầu.
4. Khi VCD thật sự can thiệp, chất lượng đổi token khá tốt:
   - khoảng một nửa là exact fix
   - hơn 72% đưa token shadow về gần posterior hơn
5. Failure lớn nhất hiện tại không phải là VCD luôn làm sai, mà là:
   - nhiều case target không còn reachable do mask
   - trong các case reachable, contrastive score chưa đủ mạnh để lật top-1 positive ở đa số vị trí

Nếu mục tiêu nghiên cứu là “VCD cải thiện quá trình drafting ở mức token như thế nào”, thì mô tả chính xác nhất là:

> VCD đang hoạt động như một cơ chế chỉnh sửa cục bộ ở frontier của positive draft. Nó hữu ích nhất cho các reject kiểu near-miss, đặc biệt là tie/rank-2, nhưng bị giới hạn mạnh bởi candidate mask và chưa đủ mạnh để cứu các token nằm ngoài positive frontier.

## 8. Các thí nghiệm tiếp theo nên làm để kết luận mạnh hơn

1. Chạy lại với **true VCD rollout**, không chỉ shadow, để đo xem các fix local này có chuyển hóa thành acceptance length / speedup / end-task quality hay không.
2. Sweep `beta` (`0.05`, `0.1`, `0.2`) để tách rõ trade-off giữa reachability và noise.
3. Tăng `vcd_shadow_num_samples` lên ít nhất `64` hoặc log luôn exact target probability để thay thế Bernoulli estimate hiện tại.
4. Log thêm `target_prob / max_prob` hoặc kích thước candidate set để thấy chính xác target thua vì bị filter hay vì contrastive score.
5. Tách riêng analysis cho:
   - tie cases
   - rank-2 near-miss
   - deep-rank cases
   nhằm tối ưu VCD đúng vào vùng nó đang có leverage cao nhất.
