# Insight: Xác Suất Của Target Trong Phân Phối Sample Và Tác Động Của CD

## Cách Đọc
Trong note này, "xác suất của target" được hiểu là:

- `p(target) = exp(target_logprob)`

Tức là xác suất mà mỗi nhánh (`positive`, `negative`, `cd_raw`, `cd_final`) gán cho đúng token target tại từng vị trí.

## Kết Luận Ngắn
CD trong run `cdv2_target_rank_gsm8k_20260416_235623` có tác động, nhưng tác động này chủ yếu là **hiệu chỉnh xác suất yếu** chứ chưa phải **reranking mạnh**.

- `target` luôn nằm trong candidate set: `target_in_candidate_mask_rate = 100%`
- Vì vậy, nút thắt không nằm ở recall của candidate mask
- Nút thắt nằm ở chỗ CD có đủ lực để nâng **thứ hạng tương đối** của target trước các đối thủ hay không

Nhìn tổng thể:

- `negative` gán xác suất cho target thấp hơn `positive` rất rõ
- `cd_raw` thường tăng nhẹ logprob của target
- nhưng `cd_final` chỉ hiếm khi đẩy target lên top-1, và số case làm mất top-1 còn nhiều hơn số case cứu top-1

## 1. Target Có Thực Sự "Ở Trong Phân Phối" Không?
Có, và còn ở rất rõ.

### Cohort `reject_only` (`n = 5,523`)
| Variant | Mean `p(target)` | Median `p(target)` | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| positive | 0.1573 | 0.1270 | 3.86% | 83.27% |
| negative | 0.0771 | 0.0071 | 8.73% | 37.41% |
| cd_raw | 0.1584 | 0.1281 | 3.98% | 83.05% |
| cd_final | 0.1560 | 0.1269 | 2.15% | 83.22% |

### Cohort `reject_to_tail` (`n = 58,617`)
| Variant | Mean `p(target)` | Median `p(target)` | Top-1 | Top-5 |
|---|---:|---:|---:|---:|
| positive | 0.2810 | 0.1352 | 35.67% | 71.81% |
| negative | 0.1007 | 0.0210 | 13.94% | 43.72% |
| cd_raw | 0.2736 | 0.1288 | 34.88% | 71.24% |
| cd_final | 0.2752 | 0.1296 | 34.85% | 71.37% |

Insight chính:

- `positive` luôn gán xác suất cho target cao hơn `negative` rất nhiều
- median của `negative` đặc biệt thấp: chỉ `0.0071` ở `reject_only` và `0.0210` ở `reject_to_tail`
- nghĩa là negative branch đúng là đang xem target là token "ít phù hợp" hơn khá mạnh

Tức là CD **có tín hiệu để khai thác**. Vấn đề là tín hiệu đó chưa chuyển hóa thành cải thiện top-1 đủ mạnh.

## 2. Vị Trí Của Target Trong Phân Phối `positive` Và `negative`
Phần trên mới cho thấy độ lớn của xác suất. Phần này trả lời câu hỏi trực tiếp hơn:

- trong toàn bộ phân phối token, target thường đứng ở hạng nào dưới `positive`?
- và khi sang `negative`, thứ hạng đó dịch chuyển ra sao?

### Thống kê rank tổng quát

#### Cohort `reject_only`
| Variant | Mean rank | Median rank | P90 rank | Max rank |
|---|---:|---:|---:|---:|
| positive | 11.78 | 2 | 9.0 | 21,400 |
| negative | 225.67 | 9 | 142.4 | 143,709 |

#### Cohort `reject_to_tail`
| Variant | Mean rank | Median rank | P90 rank | Max rank |
|---|---:|---:|---:|---:|
| positive | 76.12 | 2 | 77.0 | 97,404 |
| negative | 139.29 | 7 | 93.0 | 143,709 |

Ý nghĩa:

- ở cả hai cohort, median rank của `positive` chỉ là `2`, tức là target gần như luôn nằm sát đỉnh phân phối
- `negative` đẩy target xuống rõ rệt: median rank tăng lên `9` trong `reject_only` và `7` trong `reject_to_tail`
- mean rank của `negative` cũng lớn hơn rất mạnh, đặc biệt ở `reject_only` (`225.67` so với `11.78`)

### Phân phối rank theo bucket

#### Cohort `reject_only`
| Variant | Top-1 | Rank 2-5 | Rank 6-10 | Rank 11-50 | Rank 51-100 | Rank >100 |
|---|---:|---:|---:|---:|---:|---:|
| positive | 3.86% | 79.41% | 8.67% | 6.39% | 0.78% | 0.89% |
| negative | 8.73% | 28.68% | 15.50% | 28.28% | 6.52% | 12.29% |

#### Cohort `reject_to_tail`
| Variant | Top-1 | Rank 2-5 | Rank 6-10 | Rank 11-50 | Rank 51-100 | Rank >100 |
|---|---:|---:|---:|---:|---:|---:|
| positive | 35.67% | 36.14% | 9.70% | 12.68% | 2.29% | 3.52% |
| negative | 13.94% | 29.77% | 16.00% | 26.30% | 5.55% | 8.44% |

Insight quan trọng:

- `positive` tập trung rất mạnh ở vùng rank thấp:
  - `reject_only`: có tới `83.27%` target nằm trong top-5
  - `reject_to_tail`: có `71.81%` nằm trong top-5
- `negative` làm target trượt khỏi vùng đỉnh:
  - `reject_only`: chỉ còn `37.41%` trong top-5
  - `reject_to_tail`: chỉ còn `43.72%` trong top-5
- đặc biệt, xác suất target rơi xuống ngoài top-100 tăng mạnh dưới `negative`:
  - `0.89% -> 12.29%` ở `reject_only`
  - `3.52% -> 8.44%` ở `reject_to_tail`

### So sánh trực tiếp `positive` với `negative`
| Cohort | `positive` rank tốt hơn `negative` | Bằng nhau | `negative` tốt hơn `positive` |
|---|---:|---:|---:|
| reject_only | 74.60% | 10.94% | 14.47% |
| reject_to_tail | 67.61% | 15.45% | 16.93% |

Điều này củng cố một ý rất rõ:

- trong đa số trường hợp, `positive` đặt target ở vị trí tốt hơn `negative`
- nhưng không phải tuyệt đối
- vẫn có khoảng `14-17%` case mà `negative` lại xếp target cao hơn `positive`

Nói cách khác, `negative` không chỉ đơn giản là một nhánh luôn "ghét" target; nó tạo ra một phân phối khác, trong đó phần lớn case target bị kéo xuống, nhưng vẫn có một tập con nhỏ nơi target lại được ưu tiên hơn.

### Theo vị trí trong block
Nhìn theo `block_position`, target vẫn được `positive` ưu tiên hơn `negative` gần như xuyên suốt cả block.

#### Cohort `reject_to_tail`
| Block pos | Median rank positive | Median rank negative | Top-5 positive | Top-5 negative |
|---|---:|---:|---:|---:|
| 1 | 2 | 16 | 89.91% | 30.55% |
| 3 | 2 | 9 | 80.47% | 39.20% |
| 6 | 2 | 7 | 73.38% | 44.78% |
| 7 | 2 | 6 | 73.35% | 46.00% |
| 10 | 2 | 7 | 70.21% | 43.42% |
| 15 | 3 | 7 | 65.47% | 43.82% |

#### Cohort `reject_only`
| Block pos | Median rank positive | Median rank negative | Top-5 positive | Top-5 negative |
|---|---:|---:|---:|---:|
| 1 | 2 | 16 | 89.91% | 30.55% |
| 3 | 2 | 10 | 83.99% | 34.74% |
| 6 | 2 | 9 | 78.61% | 38.70% |
| 7 | 2 | 6 | 81.90% | 46.67% |
| 10 | 3 | 10 | 75.24% | 37.38% |
| 15 | 2 | 7 | 85.19% | 44.44% |

Đọc các bảng này theo trực giác:

- từ đầu đến cuối block, `positive` thường giữ target quanh rank `2-3`
- `negative` thường đặt target ở khoảng rank `6-16`
- chênh lệch này tồn tại khá ổn định theo vị trí, nên tín hiệu phân biệt giữa hai nhánh là có thật và không chỉ đến từ một vài vị trí lẻ

## 3. CD Có Thực Sự Nâng Xác Suất Của Target Không?
Có, nhưng chủ yếu ở mức nhẹ.

### So với `positive`, tỷ lệ CD làm tăng xác suất target
| Cohort | `cd_raw > positive` | `cd_final > positive` |
|---|---:|---:|
| reject_only | 56.69% | 17.42% |
| reject_to_tail | 45.42% | 33.60% |

### So với `positive`, tỷ lệ CD cải thiện rank của target
| Cohort | `cd_raw` cải thiện rank | `cd_final` cải thiện rank |
|---|---:|---:|
| reject_only | 11.62% | 2.70% |
| reject_to_tail | 10.63% | 7.99% |

Điều này cho thấy:

- CD khá thường xuyên làm `logprob` của target tăng lên
- nhưng phần lớn mức tăng đó **không đủ lớn** để biến thành cải thiện rank
- nói cách khác, CD đang "nhích xác suất" của target nhiều hơn là "đẩy target thắng hẳn"

## 4. CD Có Cứu Được Top-1 Không?
Hầu như không, và nhìn tổng thể còn hơi bất lợi.

### Chuyển trạng thái top-1 từ `positive` sang `cd_final`
| Cohort | Sai -> Đúng | Đúng -> Sai | Net |
|---|---:|---:|---:|
| reject_only | 2 | 96 | -94 |
| reject_to_tail | 404 | 884 | -480 |

### Khi `positive` đang sai top-1
| Cohort | Số case `positive` sai top-1 | CD đẩy lên top-1 |
|---|---:|---:|
| reject_only | 5,310 | 0.04% |
| reject_to_tail | 37,708 | 1.07% |

Đây là insight quan trọng nhất:

- CD gần như không có khả năng "sửa sai" khi `positive` đã không chọn target ở top-1
- số case CD làm target tụt khỏi top-1 lại lớn hơn số case CD cứu được

Vì vậy, trong run này, CD chưa phải là một **corrective reranker** đủ mạnh.

## 5. Hiệu Ứng Theo Vị Trí Trong Block
`final_override_keep = 6`, nên ở các vị trí đầu block, `cd_final` gần như không tạo khác biệt.

- Ở `block_position = 1..6`, mean `delta_logprob_cd_final_vs_positive = 0`
- Từ `block_position >= 7`, CD mới bắt đầu tác động rõ hơn

Ví dụ trong `reject_to_tail`:

- vị trí 7: mean logprob gain của target là `+0.0492`, nhưng top-1 lại giảm từ `39.92%` xuống `38.21%`
- vị trí 10: mean logprob gain là `+0.0560`, nhưng top-1 giảm từ `37.39%` xuống `36.45%`
- vị trí 15: mean logprob gain là `+0.0503`, nhưng top-1 giảm từ `31.94%` xuống `30.80%`

Trong `reject_only` hiện tượng này còn rõ hơn:

- vị trí 7: target được tăng xác suất trung bình khoảng `1.11x`, nhưng top-1 giảm từ `7.30%` xuống `0.32%`
- vị trí 15: target được tăng xác suất trung bình khoảng `1.17x`, nhưng top-1 giảm từ `9.26%` xuống `0.93%`

Ý nghĩa:

- CD có thể tăng xác suất tuyệt đối của target
- nhưng nó đồng thời còn nâng hoặc giữ các đối thủ mạnh hơn, nên **thứ hạng tương đối** của target không được cải thiện tương xứng

## 6. Insight Cơ Chế: Khi Negative "Ghét" Target Hơn, CD Có Giúp Không?
Có, nhưng chỉ ở mức vừa phải.

Nếu chia `reject_to_tail` theo 4 nhóm của khoảng cách:

- `gap = positive_target_logprob - negative_target_logprob`

thì ở quartile cao nhất:

- mean gap: `5.69`
- xác suất CD làm tăng logprob target: `43.85%`
- mean `delta_logprob_cd_final_vs_positive`: `+0.0864`
- tỷ lệ top-1 được cứu: `1.94%`
- tỷ lệ top-1 bị làm hỏng: `0.87%`

Trong khi ở quartile thấp nhất:

- mean gap: `-1.16`
- xác suất CD làm tăng logprob target: `32.18%`
- mean `delta_logprob_cd_final_vs_positive`: `+0.0363`
- tỷ lệ top-1 được cứu: `0.00%`
- tỷ lệ top-1 bị làm hỏng: `1.67%`

Diễn giải:

- khi `positive` đánh giá target tốt hơn `negative` càng rõ, CD càng có xu hướng giúp target
- nhưng tương quan vẫn yếu: khoảng `0.08` với `delta_logprob` và `0.08` với cải thiện rank
- tức là đây là tín hiệu hữu ích, nhưng chưa đủ mạnh để quyết định winner

## Insight Chốt
Nếu cần một kết luận ngắn để đưa vào note hoặc paper:

> Trong run `cdv2_target_rank_gsm8k_20260416_235623`, target luôn hiện diện trong candidate distribution, nên thất bại của CD không đến từ việc target bị loại khỏi tập ứng viên mà đến từ việc CD không cải thiện đủ thứ hạng tương đối của target trước các đối thủ. Negative branch thật sự cung cấp tín hiệu phân biệt vì nó gán xác suất cho target thấp hơn positive khá mạnh, và CD cũng thường tăng nhẹ logprob của target. Tuy nhiên, tác động này chủ yếu dừng ở mức calibration yếu: `cd_final` chỉ cứu được `0.04%` case sai top-1 trong `reject_only` và `1.07%` trong `reject_to_tail`, trong khi số case làm mất top-1 vẫn lớn hơn số case cứu được. Nói ngắn gọn, CD ở cấu hình hiện tại đang là một weak probabilistic regularizer hơn là một strong corrective reranker.

## Nguồn Số Liệu

- `target_rank_report.md`
- `target_rank_summary.json`
- `target_rank_records.csv`
