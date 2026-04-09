# Phân tích insight cho run `draft_entropy_gsm8k_20260409_080744`

## 1. Phạm vi và cách đọc đúng của folder này

Nguồn dùng để phân tích:

- `analysis/results/draft_entropy_gsm8k_20260409_080744/draft_entropy_summary.json`
- `analysis/results/draft_entropy_gsm8k_20260409_080744/draft_entropy_records.csv`
- `analysis/results/draft_entropy_gsm8k_20260409_080744/draft_entropy_report.md`
- `analysis/CDv2_check_model_entropy.py`

Folder này đo entropy và confidence của **draft model positive branch** tại từng draft token, rồi so sánh với:

1. `positive_argmax_hit`: local token correctness, tức top-1 của draft có trùng target posterior không.
2. `accepted_by_target`: token đó có nằm trong accepted prefix của target không.

Điểm cực quan trọng là:

- `accepted_by_target` là **prefix-level signal**,
- còn `positive_argmax_hit` là **local token-level signal**.

Vì thế:

- `rejected token` không đồng nghĩa với `draft dự đoán sai ở token đó`,
- một phần lớn reject là do **một lỗi sớm hơn trong cùng block** khiến các token đúng phía sau cũng không được accept.

Nếu không tách hai khái niệm này, rất dễ đọc sai insight về entropy.

## 2. Headline kết quả

| Chỉ số | Giá trị | Ý nghĩa |
|---|---:|---|
| Tổng token record | `90,525` | số draft-token event được đo |
| Local hit rate (`positive_argmax_hit`) | `57.42%` | độ đúng cục bộ của draft top-1 |
| Accepted-by-target rate | `35.47%` | độ accept theo prefix thực sự |
| Rejected rate | `64.53%` | phần lớn token không nằm trong accepted prefix |
| Overall ECE | `0.0705` | calibration tổng thể khá ổn |
| Rejected-only ECE | `0.1598` | calibration trên rejected subset xấu hơn rõ |
| Mean entropy accepted subset | `0.336` nats | accepted token rất peaked |
| Mean entropy rejected subset | `1.871` nats | rejected token bất định hơn nhiều |
| Mean entropy local-hit subset | `0.706` nats | token đúng cục bộ vẫn khá confident |
| Mean entropy local-wrong subset | `2.164` nats | token sai cục bộ có entropy cao rõ rệt |

Headline ngắn nhất của run này là:

- entropy và top-1 probability dự đoán **local correctness** rất tốt,
- nhưng `accepted_by_target` còn bị chi phối mạnh bởi **vị trí trong block và prefix truncation**,
- nên các phân tích trên rejected tokens phải tách thành `rejected_wrong` và `rejected_hit`.

## 3. Insight 1: Rejected token không đồng nghĩa với local failure

Ma trận đếm quan trọng nhất của cả folder:

| `accepted_by_target` | `positive_argmax_hit` | Count | Rate |
|---|---:|---:|---:|
| 0 | 0 | `38,453` | `42.48%` |
| 0 | 1 | `19,966` | `22.06%` |
| 1 | 0 | `97` | `0.11%` |
| 1 | 1 | `32,009` | `35.36%` |

Điều này có nghĩa là:

- trong `58,419` rejected tokens, có tới `19,966` token vẫn local đúng,
- tức `34.18%` rejected subset là **rejected nhưng local-hit**.

Một cách đọc khác:

- với các token mà draft top-1 đúng cục bộ, chỉ `61.59%` được accept thật sự,
- phần còn lại chủ yếu bị mất do earlier mismatch trong block.

Vì vậy, nếu chỉ nhìn “entropy của rejected token” rồi kết luận draft bất định hoặc sai, kết luận đó sẽ bị trộn giữa:

1. **true local failure**: draft top-1 sai,
2. **prefix spillover**: token đúng nhưng nằm sau vị trí mismatch đầu tiên.

Đây là insight cốt lõi nhất để đọc đúng folder này.

## 4. Insight 2: Có ba chế độ token khác nhau, không phải chỉ hai

Nếu tách token thành ba nhóm có nghĩa hơn:

| Nhóm | Count | Mean entropy | Median entropy | Mean top1 prob | Median effective support | Mean block pos |
|---|---:|---:|---:|---:|---:|---:|
| `accepted_hit` | `32,009` | `0.333` | `0.0567` | `0.906` | `1.06` | `5.26` |
| `rejected_hit` | `19,966` | `1.303` | `1.102` | `0.643` | `3.01` | `9.50` |
| `rejected_wrong` | `38,453` | `2.166` | `2.144` | `0.424` | `8.53` | `9.51` |

Ta có thể đọc ra:

### 4.1. `accepted_hit`

- rất low entropy,
- top-1 probability rất cao,
- effective support gần 1,
- xuất hiện sớm hơn trong block.

Đây là vùng “draft rất chắc và prefix chưa gãy”.

### 4.2. `rejected_hit`

- entropy cao hơn accepted_hit khá nhiều,
- nhưng vẫn thấp hơn rejected_wrong rõ rệt,
- top-1 probability ở mức trung gian.

Đây là vùng “draft vẫn đúng cục bộ, nhưng không còn nằm trong accepted prefix”.

### 4.3. `rejected_wrong`

- entropy cao nhất,
- support rộng nhất,
- confidence thấp nhất.

Đây mới là vùng “true uncertainty / true local failure”.

Insight quan trọng ở đây là:

- rejected subset không phải một cụm đồng nhất,
- mà là hỗn hợp của một cụm “đúng nhưng bị spillover” và một cụm “sai thật”.

## 5. Insight 3: Entropy và confidence dự đoán local correctness rất mạnh

Theo quintile của `entropy_nats`:

| Entropy quintile | Hit rate | Accepted rate | Rejected-hit | Rejected-wrong |
|---|---:|---:|---:|---:|
| thấp nhất | `98.87%` | `89.19%` | `9.68%` | `1.13%` |
| Q2 | `83.21%` | `55.17%` | `28.04%` | `16.79%` |
| Q3 | `54.77%` | `23.79%` | `31.34%` | `44.87%` |
| Q4 | `32.70%` | `7.64%` | `25.21%` | `67.15%` |
| cao nhất | `17.53%` | `1.54%` | `16.02%` | `82.44%` |

Theo quintile của `top1_prob`:

| Top1 prob quintile | Hit rate | Accepted rate | Rejected-hit | Rejected-wrong |
|---|---:|---:|---:|---:|
| thấp nhất | `17.08%` | `1.32%` | `15.84%` | `82.84%` |
| Q2 | `33.33%` | `8.79%` | `24.98%` | `66.24%` |
| Q3 | `54.84%` | `23.91%` | `30.95%` | `45.14%` |
| Q4 | `82.97%` | `54.19%` | `28.77%` | `17.03%` |
| cao nhất | `98.86%` | `89.12%` | `9.74%` | `1.14%` |

Diễn giải:

- entropy thấp hoặc top-1 prob cao gần như đồng nghĩa với local correctness,
- khi entropy tăng, rejected_wrong tăng rất nhanh,
- effective support cho ra cùng thứ tự vì `exp(H)` là biến đổi đơn điệu của entropy, nhưng dễ hiểu hơn về mặt trực giác.

Nếu muốn có một chỉ số “draft có đang chắc chắn ở token này không”, thì:

- `top1_prob`,
- `top1_margin`,
- hoặc `effective_support`

đều hữu ích hơn việc nhìn raw normalized entropy band hiện tại.

## 6. Insight 4: Normalized entropy band hiện tại gần như không còn giá trị phân tách

Report gốc đang dùng các band:

- `low_entropy_[0.0,0.4)`
- `mid_entropy_[0.4,0.7)`
- `high_entropy_[0.7,1.0]`

Nhưng dữ liệu thực tế là:

- max normalized entropy chỉ `0.4466`
- `99.84%` toàn bộ record nằm trong band `[0.0, 0.4)`
- chỉ `0.16%` nằm ở `>= 0.4`
- không có record nào ở `>= 0.7`

Nói cách khác:

- banding hiện tại gần như collapse thành một band duy nhất,
- nên phần `Entropy Bands` trong report hiện tại ít giá trị thực nghiệm.

Nguyên nhân không phải vì model “siêu chắc ở mọi nơi”, mà vì:

- normalized entropy đang chia cho `log(vocab_size)`,
- trong khi phân phối thực tế của draft rất concentrated,
- nên giá trị normalized bị nén mạnh về vùng thấp.

Khuyến nghị:

- dùng **quantile bins** thay vì absolute bands,
- hoặc chuyển sang `entropy_nats` / `effective_support` để đọc trực quan hơn.

## 7. Insight 5: Block position là confounder cực mạnh

`block_position` chỉ chạy từ `1` đến `15`, vì draft đang dự đoán `block_size - 1` token.

Từ đầu đến cuối block:

| Block pos | Hit rate | Accepted rate | Rejected-hit rate | Mean entropy | Mean top1 prob |
|---|---:|---:|---:|---:|---:|
| 1 | `88.70%` | `88.68%` | `0.17%` | `0.389` | `0.878` |
| 5 | `64.18%` | `44.74%` | `19.60%` | `1.086` | `0.702` |
| 10 | `49.33%` | `20.80%` | `28.62%` | `1.536` | `0.588` |
| 15 | `36.32%` | `8.96%` | `27.36%` | `2.095` | `0.459` |

Insight ở đây:

- càng về sau trong block, local hit giảm đều,
- accepted rate còn giảm mạnh hơn local hit,
- rejected-hit tăng rất mạnh từ gần 0 lên khoảng 27-29%.

Điều này phản ánh hai cơ chế cùng lúc:

1. token sau trong block intrinsically khó hơn,
2. prefix acceptance làm những token đúng phía sau bị reject nếu block đã gãy từ trước.

Vì vậy:

- mọi phân tích entropy trên rejected subset nếu không control theo `block_position` sẽ bị confound rất mạnh.

## 8. Insight 6: Entropy không chỉ là proxy của block position

Dù `block_position` là confounder mạnh, entropy/confidence vẫn có giá trị riêng bên trong từng vị trí block.

Ví dụ ở `block_position = 12`:

- theo tertile của `entropy_nats`, hit rate đi từ `79.87%` xuống `37.12%` rồi `16.47%`
- theo tertile của `top1_prob`, hit rate đi từ `17.22%` lên `36.11%` rồi `78.27%`

Ở `block_position = 15`:

- top1 tertile thấp nhất: hit `16.01%`, accepted `0.10%`
- top1 tertile cao nhất: hit `64.42%`, accepted `24.81%`

Điều này cho thấy:

- entropy/confidence không chỉ phản ánh “đang ở token muộn hơn trong block”,
- mà vẫn chứa tín hiệu local uncertainty thật sự ngay cả khi giữ `block_position` cố định.

## 9. Insight 7: Cách đọc đúng nhóm “high-confidence rejected”

Nhìn bề ngoài, các token bị reject nhưng có `top1_prob` rất cao có thể trông như một failure calibration nghiêm trọng.

Nhưng khi tách kỹ:

| Nhóm trong rejected subset | Count | Rate trong rejected | Local-hit rate | Mean block pos | Mean entropy |
|---|---:|---:|---:|---:|---:|
| `top1_prob >= 0.9` | `6,898` | `11.81%` | `74.60%` | `8.90` | `0.190` |
| `0.7 <= top1_prob < 0.9` | `8,067` | `13.81%` | `49.68%` | `8.85` | `0.812` |
| `0.5 <= top1_prob < 0.7` | `11,273` | `19.30%` | `36.49%` | `8.84` | `1.374` |
| `top1_prob < 0.5` | `32,181` | `55.09%` | `20.82%` | `10.04` | `2.671` |

Điều này cho thấy:

- high-confidence rejected không chủ yếu là “model rất tự tin nhưng sai”,
- mà phần lớn là “model đúng cục bộ nhưng không còn trong accepted prefix”.

Vì vậy:

- nếu chỉ nhìn bảng `Most-Confident Rejected Tokens`, rất dễ overstate vấn đề overconfidence,
- trong khi thực tế đây là mixture giữa prefix truncation và local error.

Nói gọn:

- `confident + rejected` không đồng nghĩa với `confident + wrong`.

## 10. Insight 8: Calibration tổng thể ổn, nhưng rejected subset được calibrate kém hơn rõ rệt

Chỉ số calibration:

- overall `ECE = 0.0705`
- rejected-only `ECE = 0.1598`

Trong overall calibration:

- gap lớn nhất xuất hiện ở bin confidence `0.7-0.8`, khoảng `13.39` điểm phần trăm.

Trong rejected-only calibration:

- gap lớn nhất lên tới khoảng `30.90` điểm phần trăm ở vùng confidence `0.8-0.9`.

Tuy nhiên cần đọc rất cẩn thận:

- rejected subset là một tập đã bị condition theo prefix acceptance,
- nên calibration kém hơn ở đây không chỉ do local uncertainty,
- mà còn do prefix truncation làm nhãn `accepted=0` khác với local correctness.

Nếu mục tiêu là đánh giá draft model như một local predictor, thì:

- nên calibrate theo `positive_argmax_hit`,
- không nên chỉ calibrate theo `accepted_by_target`.

## 11. Insight 9: Một vài hard trajectory đang thống trị entropy tail

Outlier concentration:

- sample `126` có `2,248` rejected tokens, cao nhất toàn bộ run
- sample `126` cũng chiếm `303 / 500` token entropy cao nhất

Ngoài ra các sample như:

- `70`
- `28`
- `67`
- `5`
- `97`

cũng xuất hiện nhiều trong reject/high-entropy tail.

Điều này cho thấy:

- tail của entropy không phân bố đều trên dataset,
- mà bị chi phối mạnh bởi một số trajectory rất khó.

Hệ quả:

- nếu muốn debug chi tiết, inspect thủ công `sample_idx = 126` sẽ có giá trị hơn là chỉ nhìn aggregate histograms.

## 12. Pattern thành công và thất bại

### 12.1. Pattern “thành công”

Một token draft có xác suất cao là local đúng và được accept khi:

- nằm sớm trong block,
- entropy thấp / effective support gần 1,
- top1 probability rất cao.

Ví dụ ở `block_position = 1`:

- `top1_prob >= 0.9` phủ `67.72%` token đầu block,
- và đạt `98.48%` hit/accepted.

### 12.2. Pattern “đúng cục bộ nhưng không survive prefix”

Pattern này xuất hiện nhiều ở:

- block position trung bình đến muộn,
- confidence vẫn trung bình hoặc cao,
- entropy không quá lớn.

Đây chính là nhóm `rejected_hit`.

### 12.3. Pattern “true local failure”

Pattern thất bại thật sự là:

- entropy cao,
- effective support lớn,
- top1 probability thấp,
- thường rơi vào nửa sau của block.

Đây là nhóm `rejected_wrong`.

## 13. Kết luận cơ chế

Folder này cho thấy ba kết luận chính:

1. Draft entropy là predictor mạnh cho **local correctness**.
2. Accepted-by-target không phải local metric thuần, vì bị chi phối mạnh bởi **prefix truncation**.
3. Các band normalized entropy hiện tại không đủ tốt để tách chế độ hoạt động của draft model.

Insight quan trọng nhất không phải là:

- “rejected token có entropy cao”,

mà là:

- “rejected token đang trộn hai chế độ hoàn toàn khác nhau: local wrong và local right but prefix-rejected”.

Nếu không tách hai chế độ này, ta sẽ đánh giá sai cả entropy lẫn calibration của draft model.

## 14. Khuyến nghị cho các run phân tích tiếp theo

### 14.1. Báo cáo tách riêng ba nhóm

Nên report riêng:

- `accepted_hit`
- `rejected_hit`
- `rejected_wrong`

thay vì chỉ `accepted` và `rejected`.

### 14.2. Thay entropy bands bằng quantile bands

Banding hiện tại theo normalized entropy không còn hiệu quả. Nên đổi sang:

- quintile / decile của `entropy_nats`,
- hoặc band theo `effective_support`.

### 14.3. Khi nghiên cứu routing, nên tách token đầu block khỏi phần còn lại

Với `block_position = 1`, threshold theo confidence sạch hơn rất nhiều:

- `top1_prob >= 0.9` phủ `67.72%`
- hit/accepted đều `98.48%`

Đây là vị trí rất phù hợp cho routing hoặc early trust.

### 14.4. Nếu mục tiêu là debug failure thật, ưu tiên `rejected_wrong`

Các token `rejected_hit` không phải local error thật, nên:

- không nên trộn chúng vào tập hard failure,
- và không nên dùng chúng làm evidence chính cho “draft overconfident”.

### 14.5. Ưu tiên inspect thủ công các sample tail

Đặc biệt:

- `sample_idx = 126`
- `sample_idx = 67`
- `sample_idx = 28`

vì các sample này đang đóng góp disproportionately vào reject/high-entropy tail.
