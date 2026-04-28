# Insight: Khoảng Cách Embedding Của First-Token Giữa 2 Mẫu Trong Contrastive

## Câu Hỏi Trung Tâm
Khoảng cách embedding của `first token` giữa 2 mẫu trong contrastive

- positive sample (mẫu dương)
- negative sample (mẫu âm)

có thực sự dự báo được việc contrastive sẽ đẩy target lên top-1 hay không?

## Kết Luận Ngắn
Không theo nghĩa "flip winner" (lật kết quả).

- Trong toàn bộ **89.400 token records**, không có bất kỳ case nào mà contrastive:
  - sửa được từ sai thành đúng top-1
  - hoặc làm rối từ đúng top-1 thành sai
- Vì vậy, `first-token embed distance` không đóng vai trò như một tín hiệu đủ mạnh để đảo winner của positive branch.
- Tuy nhiên, distance lớn hơn vẫn có liên hệ nhẹ với việc block ổn định hơn: giữ top-1 tốt hơn một chút, accepted nhiều hơn một chút, và mức phạt lên target logprob nhẹ hơn một chút.

Nói ngắn gọn:

- Đây là một **weak calibration signal** (tín hiệu hiệu chỉnh yếu)
- Chứ không phải một **strong corrective signal** (tín hiệu sửa chữa mạnh)

## Bằng Chứng Định Lượng Quan Trọng
Tổng số record: **89.400**

### 1. Confusion matrix cho thấy contrastive không tạo ra flip (lật kết quả)

- `positive_wrong_final_wrong`: **37.742** (dương sai, kết quả cuối sai)
- `positive_wrong_final_correct`: **0** (dương sai, kết quả cuối đúng)
- `positive_correct_final_wrong`: **0** (dương đúng, kết quả cuối sai)
- `positive_correct_final_correct`: **51.658** (dương đúng, kết quả cuối đúng)

Ý nghĩa:

- Nếu positive branch đã sai top-1, contrastive không cứu được.
- Nếu positive branch đã đúng top-1, contrastive cũng không làm rối top-1.

Do đó, nếu mục tiêu là tìm một chỉ số "first-token distance có giúp contrastive đổi kết quả hay không", thì câu trả lời ở run này là: **không**.

### 2. Distance gần như không tách được 2 nhóm đúng và sai
So sánh 2 nhóm thực sự xuất hiện:

| Trường Hợp | Số Lượng | L2 trung bình | L2 trung vị | CosDist trung bình | CosDist trung vị |
|---|---:|---:|---:|---:|---:|
| `kept_top1` (giữ top-1) | 51.658 | 1.5518 | 1.5577 | 1.0439 | 1.0296 |
| `worsened_not_push_to_top1` (xấu hơn, không đẩy lên top-1) | 37.742 | 1.5413 | 1.5483 | 1.0321 | 1.0152 |

Chênh lệch rất nhỏ:

- `delta L2 trung bình`: **+0.0105**
- `delta CosDist trung bình`: **+0.0118**
- Cohen's d:
  - `L2`: **0.0766**
  - `CosDist`: **0.1286**

Điều này nói rằng:

- nhóm giữ được top-1 có distance lớn hơn một chút
- nhưng overlap giữa 2 phân phối rất lớn
- nên distance này **không đủ sắc** để tách rõ "tốt" và "xấu"

### 3. Tương quan toàn cục rất yếu

- Corr(`L2`, `delta_target_logprob`) = **+0.0241**
- Corr(`L2`, `delta_target_rank`) = **-0.0036**
- Tương quan cấp độ block:
  - Corr(`L2`, `final_acc`) = **+0.0725**
  - Corr(`L2`, `accepted_rate`) = **+0.0516**
  - Corr(`CosDist`, `final_acc`) = **+0.1216**

Thông điệp chính:

- distance có xu hướng "đúng hướng", nhưng rất yếu
- không có dấu hiệu ngưỡng rõ ràng kiểu "distance vượt mốc X thì contrastive bắt đầu hữu ích mạnh"

## Distance Lớn Hơn Có Giúp Gì Không?
Có, nhưng chỉ ở mức nhẹ.

### 1. Theo sextile của L2
So sánh nhóm L2 thấp nhất và cao nhất:

| Nhóm L2 | Số Lượng | Tỷ Lệ Giữ Top-1 | Tỷ Lệ Accepted | Trung Bình Delta Logprob |
|---|---:|---:|---:|---:|
| Low sextile (thấp nhất) | 14.910 | 55,79% | 33,78% | -3,7546 |
| High sextile (cao nhất) | 14.910 | 61,01% | 38,42% | -3,2119 |

Ý nghĩa:

- Khi first-token distance lớn hơn, contrastive có xu hướng:
  - giữ target top-1 nhiều hơn khoảng **5,2 điểm**
  - accepted nhiều hơn khoảng **4,6 điểm**
  - làm target bị phạt logprob ít hơn

Nhưng cần nhấn mạnh:

- đây là **cải thiện nhẹ**
- không hề biến thành những case "sai thành đúng"

### 2. Theo block-level robustness (độ bền vững cấp độ block)
Vì `first_token_embedding_l2` là hằng số trong suốt 15 vị trí của cùng một block, nên nhìn theo block hợp lý hơn nhìn token-level.

Theo 6 nhóm L2 cấp độ block:

| Nhóm L2 | Trung Bình final_acc | Trung Bình accepted_rate | Trung Bình first_fail_pos |
|---|---:|---:|---:|
| Thấp nhất | 0,5579 | 0,3378 | 5,07 |
| Cao nhất | 0,6101 | 0,3842 | 5,76 |

Đọc bảng này theo ngôn ngữ có chuẩn:

- distance lớn hơn không sửa winner (kết quả thắng)
- nhưng nó có vẻ giúp block "sống lâu" hơn một chút trước khi bắt đầu sai

## Nút Thắt Thực Tế Không Nằm Ở Candidate Mask
Đây là một insight quan trọng:

- `target_in_candidate_mask_rate` = **100%**

Nghĩa là:

- target token lúc nào cũng nằm trong candidate set (tập ứng viên)
- contrastive thất bại không phải vì target bị loại khỏi mask
- mà vì score sau contrastive **không đủ để đẩy target lên top-1**

Nói cách khác, bài toán ở đây là **rescoring strength** (sức mạnh của việc tính lại điểm), không phải **candidate recall** (khả năng ghi nhận ứng viên).

## Trong nhóm not-push, phần lớn là "không đủ lực đẩy"
Trong 37.742 case `worsened_not_push_to_top1`:

| Kiểu | Số Lượng | Tỷ Lệ |
|---|---:|---:|
| `rank_unchanged_not_top1` (hạng không đổi, không top-1) | 25.998 | 68,88% |
| `rank_worse_not_top1` (hạng tệ hơn, không top-1) | 7.859 | 20,82% |
| `rank_improved_but_not_top1` (hạng cải thiện nhưng vẫn không top-1) | 3.885 | 10,29% |

Ý nghĩa:

- Gần **69%** trường hợp, contrastive thực chất không đổi được hạng của target.
- Khoảng **10%** trường hợp, hạng có cải thiện nhưng vẫn không đủ lên top-1.
- Chỉ khoảng **21%** trường hợp là hạng tệ đi thực sự.

Thêm một dấu hiệu hay:

- Trong nhóm not-push, `delta_target_logprob` có:
  - mean = **-8,5139**
  - median = **+0,0094**

Median dương nhưng mean âm rất mạnh cho thấy:

- đa số case thay đổi rất nhỏ, gần như không đổi
- nhưng tồn tại một số outlier rất xấu kéo mean xuống mạnh

Tức là contrastive không phải lúc nào cũng "phá", mà thường là **không đủ lực để cứu**, và thỉnh thoảng mới có một số cú sốc âm lớn.

## Hiệu Ứng Vị Trí Trong Block Mạnh Hơn Hiệu Ứng Distance
Nếu nhìn token-level, một yếu tố mạnh hơn L2 là `block_position` (vị trí trong block).

Từ đầu đến cuối block:

- `not_push_rate` (tỷ lệ không đẩy): **11,29% → 63,39%**
- `keep_top1_rate` (tỷ lệ giữ top-1): **88,71% → 36,61%**
- `accepted_rate` (tỷ lệ chấp nhận): **88,71% → 9,06%**

Trong khi đó, `L2 mean` theo `block_position` gần như giữ nguyên vì đây là cùng một first-token distance lặp lại trong block.

Điều này rất quan trọng cho cách diễn giải:

- token-level result bị chi phối mạnh bởi độ khó tăng dần theo vị trí trong block
- vì vậy, để hiểu đúng vai trò của first-token distance, nên ưu tiên góc nhìn **block-level** (cấp độ block)

## Một Insight Cơ Chế Rất Đáng Chú Ý

- `first_token_same_id_rate` = **0%**

Tức là first token của negative sample **luôn khác** first token của positive sample.

Điều này gợi ý rằng contrastive trong run này đang so sánh:

- một positive trajectory (quỹ đạo dương)
- với một negative trajectory có điểm neo mở đầu đã khác hẳn

Thay vì một perturbation nhỏ quanh cùng một token đầu.

Dưới cấu hình hiện tại:

- `cd_alpha = 0.1`
- `negative_context_dropout = 0.3`
- `negative_hidden_mode = mask_zero`

có vẻ như negative branch tạo ra một hướng phân biệt đủ rõ để "reweight" nhẹ, nhưng chưa đủ mạnh để lật top-1 của positive branch.

## Insight Chốt
Nếu chỉ cần một kết luận để đưa vào note/paper:

> Trong run `cdv2_embed_dist_gsm8k_20260414_202045`, khoảng cách embedding của first token giữa positive sample và negative sample không đủ bảo được khả năng contrastive flip (lật) kết quả top-1: không có bất kỳ case nào được sửa từ sai thành đúng, và cũng không có case nào bị làm rối từ đúng thành sai. Tuy nhiên, distance lớn hơn vẫn liên hệ nhẹ với độ bền của block: keep-top1 rate tăng từ 55,8% lên 61,0% và accepted rate tăng từ 33,8% lên 38,4% từ nhóm L2 thấp nhất sang cao nhất. Vì `target` luôn nằm trong candidate mask, nút thắt chính không nằm ở recall mà nằm ở sức mạnh rescoring (tính lại điểm); contrastive ở đây đóng vai trò regularizer/calibration yếu, chứ chưa trở thành corrective reranker đủ mạnh để đẩy target lên top-1.

## Nguồn Số Liệu

- `first_token_embed_summary.json`
- `first_token_case_records.csv`
- `first_token_embed_report.md`
