# Insight: Phân Phối Của Draft Model Trên GSM8K

## Câu Hỏi Trung Tâm
Draft model có thực sự tập trung (nhọn) vào một vài điểm rank cao nhất hay không?

## Kết Luận Ngắn
Có, nhưng cần nói chính xác hơn:

- Phân phối của draft model là **head-heavy rõ rệt**: xác suất chủ yếu nằm trong **top-5**, và rất nhiều trường hợp đã gần như đóng lại trong **top-10**.
- Tuy nhiên, nếu câu hỏi là có "**rất nhọn ở top-1**" không, thì câu trả lời là **có nhưng không đồng đều**. Chỉ một phần dữ liệu rất nhọn ở rank-1; một phần khác vẫn tập trung ở top-rank nhưng không độc đáo vào top-1.

Nói cách khác:

- **Có sự tập trung mạnh vào các rank cao nhất**.
- **Không phải lúc nào cũng là kiểu one-hot ở rank-1**.

## Bằng Chứng Định Lượng Quan Trọng
Tổng số record: **89.265**

### 1. Mức Độ Tập Trung Theo Head-Mass
Phân bố xác suất theo các nhóm rank:

| Thành Phần | Mean | Median | Diễn Giải |
|---|---:|---:|---|
| Rank 1 | 0.6510 | 0.6775 | Median cho thấy token đúng top-1 thường đã án gần 68% mass |
| Rank 2 | 0.1233 | 0.1086 | Rank 2 vẫn giữ một phần đáng kể |
| Rank 1-2 công dồn | 0.7743 | 0.8855 | Median: 2 rank đầu đã giữ gần 88.6% mass |
| Rank 3-5 | 0.1082 | 0.0737 | Phân bố sung cho top-5, nhưng nhỏ hơn top-1 rất nhiều |
| Rank 6-10 | 0.0509 | 0.0143 | Sau top-5, mass giảm rõ |
| Top-5 mass | 0.8825 | 0.9739 | Median: top-5 giữ ~97.4% tổng xác suất |
| Top-10 mass | 0.9334 | 0.9901 | Median: top-10 giữ ~99.0% tổng xác suất |
| Tail sau top-10 | 0.0666 | 0.0099 | Median: chỉ còn ~1.0% nằm ngoài top-10 |

Đọc bảng này theo góc nhìn "độ nhọn":

- Ở **median case**, phân phối rất có dấu: rank-1 đã ~67.7%, top-2 đã ~88.5%, top-5 đã ~97.4%.
- Điều này là dấu hiệu rất mạnh rằng phân phối **không trải đều**, mà **có xu hướng đon vào vài rank đầu**.

### 2. Bao Nhiêu Trường Hợp Thực Sự Rất Nhọn?
Phân loại theo `top1_prob`:

- `top1 >= 0.9`: **34,40%**
- `top1 >= 0.7`: **48,46%**
- `top1 < 0.5`: **36,07%**

Ý nghĩa:

- Khoảng **1/3 dataset** là trường hợp rất nhọn ở top-1.
- Gần **một nửa dataset** có top-1 tương đối mạnh (`>= 0.7`).
- Nhưng cũng có **36,1% trường hợp** mà top-1 chưa đủ mạnh (`< 0.5`).

Vì vậy, nếu phát biểu "draft model lúc nào cũng nhọn ở top-1" thì **không đúng**.
Phát biểu đúng hơn là: **draft model thường tập trung vào top-rank, nhưng độ nhọn ở rank-1 thay đổi khá mạnh giữa các vị trí/bối cảnh**.

### 3. Top-Rank Có Thực Sự Ăn Phần Lớn Xác Suất Không?
Thêm một vài chỉ số ngưỡng:

- `top2_cum >= 0.8`: **58,24%**
- `top2_cum >= 0.9`: **48,42%**
- `top5 >= 0.9`: **65,71%**
- `top5 >= 0.95`: **56,85%**
- `top10 >= 0.99`: **50,09%**
- `tail_after_top10 <= 0.01`: **50,09%**
- `top1_margin >= 0.5`: **50,28%**

Điều này cũng có nghĩa:

- Trong hơn **một nửa** số record, chỉ **2 rank đầu** đã nắng gần hết mass.
- Trong gần **2/3** số record, **top-5** đã giữ ít nhất 90% mass.
- Trong **một nửa** số record, phần ngoài **top-10** còn tối đa 1%.

Nếu mục tiêu là kiểm tra "mass có thực sự đon vào một vài rank đầu không", thì đây là bằng chứng rất mạnh rằng **có**.

## Phân Phối Không Đồng Đều Theo Block Position
Độ nhọn giảm dần khi đi sâu vào block.

### Tổng Hợp Theo 3 Vùng
| Segment (Khu Vực) | top1 median | top10 median | flat rate (`top1 < 0.5`) | effective support median |
|---|---:|---:|---:|---:|
| Early (1-5) | 0.9187 | 0.9988 | 17,94% | 1.4553 |
| Mid (6-10) | 0.6519 | 0.9874 | 37,13% | 2.9895 |
| Late (11-15) | 0.4680 | 0.9558 | 53,16% | 5.8119 |

### Ví Dụ Cực Trị Đầu Và Cuối Block
| block pos | top1 median | top5 median | top10 median | flat rate | effective support median |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.9857 | 0.9996 | 0.9998 | 6,84% | 1.0972 |
| 15 | 0.3907 | 0.8313 | 0.9292 | 60,78% | 8.0726 |

Diễn Giải:

- Ở đầu block, draft model **rất nhọn**, gần như khoá vào vài token đầu.
- Càng về sau, phân phối **mở rộng ra**. Tuy vậy, ngay cả ở cuối block, mass vẫn ưu tiên head khá mạnh: median top-10 vẫn ~92,9%.

## Điểm Quan Trọng Khi Diễn Giải Entropy
`normalized_entropy` rất thấp (median ~0.085), nhưng không nên chỉ dựa vào chỉ số này để kết luận "top-1 rất nhọn".

Lý Do:

- Vocab của model rất lớn, nên chỉ cần mass không trải đều trên toàn vocab là entropy chuẩn hoá đã thấp.
- Chỉ số phản ánh đúng câu hỏi "có đon vào rank cao nhất không" hơn là:
  - `top1_prob`
  - `top1 + top2`
  - `top5_mass`
  - `top10_mass`
  - `tail_after_top10`
  - `top1_margin`

## Insight Chốt
Nếu chỉ cần một kết luận để đưa vào paper/note:

> Draft model trên tập GSM8K có phân phối xác suất mang tính head-heavy rất rõ: ở median case, top-2 đã giữ ~88,5% mass, top-5 giữ ~97,4%, và phần ngoài top-10 chỉ còn ~1,0%. Tuy nhiên, độ nhọn ở rank-1 không đồng đều; chỉ ~34,4% trường hợp có `top1 >= 0.9`, trong khi ~36,1% trường hợp có `top1 < 0.5`. Vì vậy, mô tả chính xác nhất là draft model **tập trung mạnh vào một vài rank cao nhất**, nhưng **không phải lúc nào cũng cực nhọn tại duy nhất rank-1**, đặc biệt ở các vị trí về sau trong block.

## Nguồn Số Liệu
- File summary gốc: `draft_entropy_summary.json`
- File record chi tiết: `draft_entropy_records.csv`
- File report gốc: `draft_entropy_report.md`
