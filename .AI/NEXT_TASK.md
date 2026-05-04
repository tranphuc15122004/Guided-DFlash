# Kế hoạch thực hiện tiếp theo

## Tổng quan

Hiện tại chúng ta đã có:

- ✅ Định nghĩa bài toán MDP hoàn chỉnh.
- ✅ Script thu thập dữ liệu offline (`data_collecting.py`).
- ✅ Mô hình actor (`ContextualBanditAlpha`) và critic (`Critic`) — `alpha_model/model/alpha_model.py`.
- ✅ Các hàm tính reward vector hóa — `alpha_model/train/alpha_simulate.py`.
- ✅ Offline dataset loader (HDF5) — `alpha_model/train/offline_dataset_h5.py`.
- ✅ Script huấn luyện A2C bandit — `alpha_model/train/train_AC.py`.
- ✅ Package structure với `__init__.py`.

Cần triển khai các bước còn lại để tích hợp policy và đánh giá.

## Các task chi tiết

### ✅ Task T2: Xây dựng OfflineBanditDataset — ĐÃ HOÀN THÀNH

- **File:** `alpha_model/train/offline_dataset_h5.py`
- Lớp `HDF5BanditDataset` đọc dữ liệu từ file HDF5.
- Hàm `collate_bandit_batch` để gộp batch.
- `baseline_acc_len` dtype float32 để khớp với reward computation.

### ✅ Task T3: Huấn luyện A2C bandit — ĐÃ HOÀN THÀNH

- **File:** `alpha_model/train/train_AC.py`
- `GaussianPolicy` kế thừa `ContextualBanditAlpha` + `log_std` học được.
- Training loop hỗ trợ full batch (không per-sample).
- Có gradient clipping, entropy coef configurable.
- Đã smoke test thành công trên dữ liệu giả lập.

### Task T4: Tích hợp policy vào Dflash thực tế

### 🔲 Task T4: Tích hợp policy vào Dflash thực tế

**Mục tiêu**: Thay thế α cố định trong `dflash_generate` bằng α động từ actor đã huấn luyện.

**Công việc**:

1. Load actor checkpoint từ `checkpoints/actor_epochN.pt`.
2. Tại mỗi block trong quá trình decode:
   - Xây dựng state từ pos_logits, neg_logits (top-K), vị trí, alpha_prev.
   - Gọi `actor.forward()` → `mean, log_std` → sample/clamp → `alpha_per_token` (S, 3).
   - Dùng `alpha_per_token` trong contrastive decoding (gather alpha theo bucket của từng token).
   - Cập nhật `alpha_prev` cho block tiếp theo.
3. Chú ý:
   - Đồng bộ device (GPU) và dtype (float32).
   - Khi inference có thể chỉ dùng `mean` (không sample) hoặc sample với noise nhỏ.
   - Range α ∈ [-2, 2] như thiết kế.
   - Cần xác định bucket cho từng token dựa trên rank trong positive logits.

**Output dự kiến**:

- Script `dflash_with_alpha_agent.py` (hoặc sửa `data_collecting.py` thêm mode `--use-alpha-agent`).

**Kiểm tra**:

- So sánh acceptance length trung bình với baseline α cố định 0.1 và 1.0.
- Kiểm tra overhead của actor forward đến tốc độ decode.

---

### 🔲 Task T5: Đánh giá và tinh chỉnh

**Mục tiêu**: Đánh giá toàn diện và cải thiện nếu cần.

**Công việc**:

1. Chạy trên tập test (ví dụ 200 mẫu), ghi lại:
   - Acceptance length distribution.
   - Time per output token.
   - Speedup so với target model only.
2. Phân tích kết quả:
   - Nếu chưa tốt, thử điều chỉnh reward weights (w1, w2, w3), entropy_coef, max_grad_norm.
   - Thử thay đổi số bucket, topK, hidden_dim, learning rate.
   - Thử huấn luyện MDP thay vì bandit (cần next_state).
   - Thử thêm PPO clipped surrogate thay vì A2C thuần.
3. So sánh với các baseline:
   - α=0 (chỉ positive).
   - α=0.1 (α nhỏ).
   - α=1.0.
   - α=2.0.
   - α cố định tối ưu (tìm bằng grid search).
4. Nếu thành công, lưu lại final model và ghi chú.

---

## Thứ tự ưu tiên

1. ✅ **T2** (offline dataset) — đã hoàn thành (`offline_dataset_h5.py`).
2. ✅ **T3** (huấn luyện A2C bandit) — đã hoàn thành (`train_AC.py`).
3. 🔲 **T4** (tích hợp policy vào Dflash) — ưu tiên cao nhất hiện tại.
4. 🔲 **T5** (đánh giá và tinh chỉnh) — sau khi có model tích hợp.

## Ghi chú

- Dữ liệu hiện dùng định dạng HDF5 (merge từ các chunk `.pt`).
- `num_workers=0` trong DataLoader do h5py không fork-safe.
- Hàm `compute_reward_components_batch` đã hỗ trợ full batch (B, S, K).
- Có thể cần cân nhắc chuyển sang PPO clipped nếu A2C không ổn định.
- GPU driver cần cập nhật nếu muốn chạy trên CUDA (hiện tại `cpu` fallback).
