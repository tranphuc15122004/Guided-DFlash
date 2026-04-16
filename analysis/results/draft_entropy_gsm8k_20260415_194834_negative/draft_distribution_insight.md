# Insight: Phân Phối Của Draft Model Ở Nhánh Negative Trong Contrastive

## Câu hỏi trung tâm
Phân phối của draft model khi phân tích trên `negative sample` trong cơ chế contrastive có đặc điểm gì, và nó khác gì về mặt "độ nhọn" so với nhánh positive?

## Kết luận ngắn
Nhánh `negative` vẫn là một phân phối **head-heavy**, nhưng đã bị **làm phẳng và mở rộng rõ rệt** so với nhánh `positive`.

Nói chính xác hơn:

- Nó **không uniform** và cũng không thực sự "bè ra toàn vocab".
- Mass vẫn tập trung khá mạnh vào `top-5` và `top-10`.
- Tuy nhiên, mass ở `top-1` và `top-2` giảm rõ, `tail` ngoài `top-10` tăng lên, và `effective support` tăng mạnh.

Với góc nhìn contrastive, đây là một kết quả hợp lý:

> Negative sample không phá hủy cấu trúc phân phối của draft model, mà chủ yếu làm giảm độ tự tin và làm mở rộng phân phối, khiến nhánh negative trở thành một phiên bản "kém sắc, kém tập trung, kém gần target hơn" so với nhánh positive.

## Bằng chứng định lượng quan trọng
Tổng số record: **90,870**

Meta quan trọng:

- `entropy_logit_source = negative`
- `analyzed_argmax_hit_rate = 18.57%`
- `accepted_by_target_rate = 35.51%`

## 1. Nhánh negative có tập trung vào top-rank không?
Có, nhưng yếu hơn nhiều so với nhánh positive.

### Head-mass của nhánh negative
| Thành phần | Mean | Median | Diễn giải |
|---|---:|---:|---|
| Rank 1 | 0.4775 | 0.4136 | Top-1 vẫn giữ mass lớn nhất, nhưng median chỉ ~41.4% |
| Rank 2 | 0.1459 | 0.1326 | Rank 2 chiếm vai trò lớn hơn so với nhánh positive |
| Rank 1-2 cộng dồn | 0.6234 | 0.6214 | Median: 2 rank đầu giữ ~62.1% mass |
| Rank 3-5 | 0.1573 | 0.1653 | Một lượng mass đáng kể bị đẩy xuống rank 3-5 |
| Rank 6-10 | 0.0854 | 0.0774 | Top-10 vẫn giữ một phần lớn xác suất |
| Top-5 mass | 0.7807 | 0.8392 | Median: top-5 giữ ~83.9% mass |
| Top-10 mass | 0.8661 | 0.9250 | Median: top-10 giữ ~92.5% mass |
| Tail sau top-10 | 0.1339 | 0.0750 | Median: còn ~7.5% nằm ngoài top-10 |

Đọc bảng này theo câu hỏi "negative có còn tập trung không?":

- Câu trả lời là **có**: ở median case, top-10 vẫn giữ ~92.5% mass.
- Nhưng nó đã **kém nhọn hơn rõ rệt**: top-1 chỉ còn ~41.4%, top-2 chỉ giữ ~62.1%, và tail ngoài top-10 đã tăng lên ~7.5%.

## 2. Bao nhiêu trường hợp thật sự nhọn?
Phân loại theo `top1_prob`:

- `top1 >= 0.9`: **12.92%**
- `top1 >= 0.7`: **25.14%**
- `top1 < 0.5`: **59.21%**

Ý nghĩa:

- Chỉ khoảng **1/8 dataset** còn rất nhọn ở `top-1`.
- Chỉ khoảng **1/4 dataset** có `top1 >= 0.7`.
- Gần **60%** trường hợp đã rơi vào vùng `top1 < 0.5`.

Đây là dấu hiệu rõ ràng cho thấy negative sample đã làm suy giảm độ sắc ở phần đầu phân phối.

## 3. Mức độ tập trung top-k còn mạnh đến đâu?
Thêm một vài ngưỡng để đọc nhanh:

- `top2_cum >= 0.8`: **33.04%**
- `top2_cum >= 0.9`: **23.19%**
- `top5 >= 0.9`: **40.57%**
- `top5 >= 0.95`: **30.58%**
- `top10 >= 0.95`: **42.99%**
- `top10 >= 0.99`: **23.58%**
- `tail_after_top10 <= 0.05`: **42.99%**
- `tail_after_top10 <= 0.01`: **23.58%**
- `top1_margin >= 0.5`: **27.85%**

Điều này cho thấy:

- Nhánh negative vẫn còn tính `head-heavy`, nhưng chỉ ở mức vừa phải.
- Rất nhiều mass đã được đẩy từ `top-1/top-2` xuống `top-3..10` và cả `tail`.
- Contrastive vì thế có thể khai thác một sự khác biệt đáng kể giữa positive và negative, nhưng khác biệt này là kiểu "phẳng hơn" chứ không phải "vỡ cấu trúc".

## 4. So sánh trực tiếp với nhánh positive
Đây là phần quan trọng nhất nếu mục tiêu là hiểu vai trò của negative sample trong contrastive.

### So sánh median: negative vs positive
| Metric | Negative | Positive | Delta (neg - pos) |
|---|---:|---:|---:|
| `top1_prob` | 0.4136 | 0.6775 | -0.2639 |
| `top1 + top2` | 0.6214 | 0.8855 | -0.2641 |
| `top5_mass` | 0.8392 | 0.9739 | -0.1347 |
| `top10_mass` | 0.9250 | 0.9901 | -0.0651 |
| `tail_after_top10` | 0.0750 | 0.0099 | +0.0651 |
| `effective_support` | 7.8339 | 2.7559 | +5.0780 |
| `top1_margin` | 0.1995 | 0.5050 | -0.3055 |
| `target_prob` | 0.0251 | 0.4806 | -0.4556 |

Thông điệp từ bảng này:

- Negative sample không chỉ làm giảm `top1_prob`, mà còn làm giảm mạnh **toàn bộ head concentration**.
- `effective_support` median tăng từ **2.76** lên **7.83**: phân phối rộng hơn rất rõ.
- `top1_margin` median giảm từ **0.5050** xuống **0.1995**: rank-1 không còn vượt trội rank-2 như ở nhánh positive.
- `target_prob` median rơi từ **0.4806** xuống **0.0251**: nhánh negative trở nên rất kém gần target token.

Nếu cần một câu mô tả ngắn:

> Negative sample khiến draft distribution chuyển từ trạng thái "sắc và ưu tiên mạnh cho target/head token" sang trạng thái "vẫn còn có đầu, nhưng rộng hơn, cạnh tranh hơn, và target bị đẩy ra khỏi đầu phân phối".

## 5. Position-wise: càng về sau trong block, nhánh negative càng phẳng
### Tổng hợp theo 3 vùng
| Segment | top1 median | top5 median | top10 median | tail10 median | effective support median | flat rate |
|---|---:|---:|---:|---:|---:|---:|
| Early (1-5) | 0.5497 | 0.9207 | 0.9641 | 0.0359 | 4.5682 | 44.53% |
| Mid (6-10) | 0.4127 | 0.8395 | 0.9256 | 0.0744 | 7.8814 | 59.56% |
| Late (11-15) | 0.2984 | 0.7196 | 0.8572 | 0.1428 | 13.6057 | 73.53% |

### Ví dụ đầu block và cuối block
| Block pos | top1 median | top5 median | top10 median | tail10 median | effective support median | flat rate |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.5965 | 0.9394 | 0.9725 | 0.0275 | 3.8537 | 39.06% |
| 15 | 0.2531 | 0.6624 | 0.8178 | 0.1822 | 17.3083 | 79.47% |

Điều này cho thấy:

- Ở đầu block, negative branch vẫn tương đối có cấu trúc.
- Càng về sau, phân phối negative mở rộng nhanh:
  - `top1` giảm mạnh
  - `tail` tăng nhanh
  - `effective support` tăng rất rõ

So với nhánh positive, xu hướng này cũng tồn tại, nhưng nhánh negative bị "trừng phạt" mạnh hơn nhiều.

## 6. Entropy thấp nhưng không nên hiểu nhầm là rất nhọn
`normalized_entropy` của nhánh negative vẫn thấp:

- mean: **0.1728**
- median: **0.1725**
- `low_entropy_[0.0, 0.4)`: **99.14%**

Nhưng điều này **không có nghĩa** là nhánh negative vẫn rất nhọn ở top-1.

Lý do:

- Vocab của model rất lớn, nên chỉ cần phân phối không trải đều trên toàn vocab thì entropy chuẩn hóa đã có thể thấp.
- Các chỉ số phản ánh đúng câu hỏi "mass có thực sự dồn vào rank cao nhất không" vẫn là:
  - `top1_prob`
  - `top1 + top2`
  - `top5_mass`
  - `top10_mass`
  - `tail_after_top10`
  - `effective_support`
  - `top1_margin`

Và các chỉ số này đều cho thấy nhánh negative đã **phẳng hơn rõ rệt**.

## 7. Vai trò của negative sample trong contrastive
Nếu nhìn từ góc độ cơ chế:

- Nhánh positive cung cấp một phân phối tập trung hơn, target-aligned hơn.
- Nhánh negative tạo ra một phân phối vẫn có đầu, nhưng:
  - kém sắc hơn
  - kém tự tin hơn
  - kém gần target hơn
  - mass bị đẩy xuống rank 3-10 và tail nhiều hơn

Điều này rất hợp với mục tiêu contrastive:

> Negative sample không cần phải tạo ra một phân phối ngẫu nhiên. Nó chỉ cần tạo ra một phân phối có cấu trúc nhưng "lệch hướng" và "giảm độ tập trung" so với nhánh positive, để phép trừ contrastive có thể hạ điểm các token được negative branch ủng hộ.

## Insight chốt để đưa vào note/paper
Nếu cần một đoạn ngắn gọn:

> Trên `draft_entropy_gsm8k_20260415_194834_negative`, phân phối draft model ở nhánh negative vẫn mang tính head-heavy, nhưng mở rộng hơn đáng kể so với nhánh positive. Ở median case, `top1_prob` giảm từ `0.6775` xuống `0.4136`, `top1+top2` giảm từ `0.8855` xuống `0.6214`, `tail_after_top10` tăng từ `0.0099` lên `0.0750`, và `effective_support` tăng từ `2.76` lên `7.83`. Như vậy, negative sample không phá vỡ cấu trúc phân phối, mà chủ yếu làm giảm độ sắc, làm tăng cạnh tranh trong nhóm top-rank, và đẩy target ra xa phần đầu phân phối. Đây chính là kiểu khác biệt mà cơ chế contrastive có thể khai thác hiệu quả.

## File nguồn
- `draft_entropy_summary.json`
- `draft_entropy_report.md`
- `draft_entropy_records.csv`
