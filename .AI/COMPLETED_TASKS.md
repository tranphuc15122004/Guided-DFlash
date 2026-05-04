# Các công việc đã hoàn thành

## 1. Định nghĩa bài toán và thiết kế MDP

- Xác định state, action, reward, transition, episode cho bài toán điều chỉnh α động trong Dflash.
- Thống nhất các tham số: block_size=16 (S=15), top_k=32, số bucket=3 (rank 0-9, 10-19, ≥20).
- Quyết định dùng greedy sampling, normalize vị trí, reward gồm r1 (thay đổi rank), r2 (thưởng top-1), r3 (thay đổi acceptance length).
- Lưu ý: r1 và r2 được tính trên toàn bộ token (không mask theo accept length), chỉ r3 dùng baseline từ dữ liệu.

## 2. Script thu thập dữ liệu

**File:** `data_collecting.py`

- Chạy Dflash với α cố định (`--cd-alpha`) để sinh responses cho một dataset.
- Tại mỗi block, tính positive logits từ draft model và negative logits bằng cách:
  - Thay đổi ngẫu nhiên token đầu block.
  - Dropout target hidden (chế độ `mask_zero`).
- Trích xuất top‑K logits (K=32) và các thông tin:
  - `draft_topk_token_ids`, `draft_topk_logits`, `neg_logits_on_draft_topk_ids`
  - `block_position` (normalized), `absolute_position` (normalized)
  - `target_token_id` (từ target model)
  - `acceptance_length` (baseline)
  - `alpha_prev` (α từ block trước, khởi tạo zero)
  - `alpha_applied` (α cố định dùng trong collect)
- Lưu mỗi block thành một record (dictionary) trong buffer, định kỳ flush thành file chunk `.pt`.
- Hỗ trợ distributed (MPI) để tăng tốc thu thập.
- Có thể cấu hình số lượng mẫu, max_new_tokens, temperature (thường 0).

**Data collected**: Các thư mục `alpha_model/collected_alpha_records/rank_*/chunk_*.pt` → sau đó merge thành file HDF5 để training.

## 3. Mô hình RL (Actor & Critic)

**File:** `alpha_model/model/alpha_model.py`

### 3.1. StateFeatureExtractor

- Trích xuất các đặc trưng từ pos_logits, neg_logits (shape B×S×K):
  - logits, log_softmax, softmax
  - entropy, top-1 margin
  - top‑k mass cho k = 5, 10, 20
  - diff logits (pos - neg)
  - KL divergence
  - block position, absolute position
  - alpha_prev (nếu có)
- Ghép tất cả thành vector feature (input_dim được tính tự động).

### 3.2. Rotary Embedding & Transformer

- `RotaryEmbedding` hỗ trợ RoPE.
- `TransformerRoPEBlock`: attention causal, RoPE, FFN.
- `TransformerContext`: stack 2 block, hidden_dim=128, 4 heads.

### 3.3. ContextualBanditAlpha (Actor)

- FeatureExtractor → StateEncoder (MLP) → TransformerContext → Linear(128 → 3)
- Output: `max_alpha * tanh(logits)` (α ∈ [-2,2]).
- Forward nhận pos_logits, neg_logits, block_pos, abs_pos, alpha_prev (optional).

### 3.4. Critic

- Dùng FeatureExtractor và StateEncoder giống actor, nhưng value head (MLP → 1), không dùng transformer.
- Output: giá trị V(s) cho mỗi token (S,).

### 3.5. Kiểm tra shape

- Code kiểm thử nhanh trong `if __name__ == "__main__"` xác nhận output shapes.

## 4. Hàm simulate và tính reward vector hóa

**File:** `alpha_model/train/alpha_simulate.py`

### 4.1. Các hàm phụ trợ

- `compute_bucket_thresholds(topk, num_buckets)`: chia topK đều, trả về thresholds.
- `get_rank_in_topk_batch`: tìm rank của target token trong topk_token_ids (hỗ trợ batch B).
- `get_bucket_from_rank_batch`: xác định bucket từ rank và thresholds.
- `apply_contrastive_batch`: `pos - alpha * neg`.
- `greedy_sample_batch`: argmax.

### 4.2. `simulate_acceptance_length_batch`

- Nhận pos_logits, neg_logits, topk_token_ids, target_token_ids, alpha_per_token (đều hỗ trợ batch B).
- Tính rank → bucket → alphas → contrastive logits → pred_tokens → correct.
- Dùng `cumprod` để tìm acceptance length và correct_mask.
- Trả về (acc_len, correct_mask) — mỗi shape (B,) và (B, S).

### 4.3. `compute_reward_components_batch`

- Hỗ trợ full batch (B, S, K).
- Tính ranks_before, buckets, alphas, contrastive logits/probs.
- Tính ranks_after dựa trên target prob (đếm prob > target_prob).
- Tính delta_rank, weights (exp(-i/γ)).
- **r1**, **r2** tính trên toàn bộ token (không mask).
- Tính acceptance length contrastive (dùng cumprod).
- delta = acc_len_contrastive - baseline_acc_len.
- **r3** = max(delta,0) - λ * max(-delta,0).
- Trả về (r1, r2, r3) — mỗi shape (B,).

### 4.4. `total_reward`

- Kết hợp với w1=0.1, w2=0.1, w3=1.0.
- Có alias `total_reward_batch` và backward-compatible alias `total_reward`.

### 4.5. `_ensure_batched_inputs`

- Tự động thêm batch dim cho single sample (2D → 3D).

### 4.6. Kiểm thử

- Chạy với dữ liệu giả lập (S=15, K=32) trên CPU/GPU để đảm bảo không lỗi và output hợp lý.

## 5. Kết quả kiểm thử ban đầu

- Các hàm vector hóa chạy đúng shape và cho kết quả số.
- So sánh `simulate_acceptance_length_vectorized` với baseline (alpha_applied) cho thấy tương thích.
- **Lưu ý**: Hiện tại chỉ kiểm thử trên dữ liệu giả lập, chưa chạy trên dữ liệu thật do chưa có dataset loader.

## 6. Task T2: OfflineBanditDataset (HDF5)

**File:** `alpha_model/train/offline_dataset_h5.py`

### 6.1. HDF5BanditDataset

- Đọc dữ liệu từ file HDF5 (đã được merge từ các chunk `.pt`).
- Mỗi record bao gồm 8 field bắt buộc (S=15, K=32):
  - `draft_topk_logits` (S, K), `neg_logits_on_draft_topk_ids` (S, K)
  - `draft_topk_token_ids` (S, K), `target_token_id` (S,)
  - `block_position` (S,), `absolute_position` (S,)
  - `alpha_prev` (S, 3), `acceptance_length` (scalar)
- Tự động validate required fields khi khởi tạo.
- Trả về dictionary với tất cả tensor đã ép kiểu đúng (float32 / long).

### 6.2. collate_bandit_batch

- Gộp batch từ list các dictionary → dictionary với tensor đã stack dim=0.
- `baseline_acc_len` được ép kiểu `float32` (để khớp với reward computation).

### 6.3. Lưu ý

- `num_workers=0` trong DataLoader vì h5py không fork-safe.

## 7. Task T3: Huấn luyện A2C bandit

**File:** `alpha_model/train/train_AC.py`

### 7.1. GaussianPolicy (Actor mở rộng)

- Kế thừa `ContextualBanditAlpha`, thêm `log_std` là `nn.Parameter` học được cho 3 buckets.
- Output: `(mean, log_std)` — cả hai shape (B, S, 3).
- Dùng `rsample()` (reparameterization trick) để gradient lan truyền qua action.
- Action được clamp về `[-max_alpha, max_alpha]`.

### 7.2. Training loop (full batch)

- **Không lặp per-sample** — actor, critic, reward đều chạy full batch.
- Pipeline mỗi batch:
  1. Actor forward → mean, log_std → sample action.
  2. `compute_reward_components_batch` → r1, r2, r3 → total_reward (B,).
  3. Critic forward → value (B, S) → mean over S → (B,).
  4. advantage = reward - value.detach().
  5. Actor loss = -(log_prob *advantage).mean() - entropy_coef* entropy.
  6. Critic loss = MSE(value, reward).
  7. Gradient clipping (`clip_grad_norm_` với max_norm=1.0) cho cả actor & critic.
  8. Cập nhật optimizer.

### 7.3. Cấu hình (argparse)

| Tham số | Mặc định | Ý nghĩa |
|---------|----------|---------|
| `--batch-size` | 64 | Số record mỗi batch |
| `--epochs` | 10 | Số epoch huấn luyện |
| `--lr` | 3e-4 | Learning rate Adam |
| `--top-k` | 32 | Kích thước top-K |
| `--hidden-dim` | 128 | Hidden dim của transformer/MLP |
| `--num-buckets` | 3 | Số bucket cho α |
| `--max-alpha` | 2.0 | Giới hạn α ∈ [-2, 2] |
| `--gamma-decay` | 7.0 | Hệ số decay vị trí cho reward |
| `--lambda-r3` | 3.0 | Penalty cho acceptance ngắn hơn |
| `--w1, --w2, --w3` | 0.1, 0.1, 1.0 | Trọng số reward |
| `--entropy-coef` | 0.01 | Hệ số entropy bonus |
| `--max-grad-norm` | 1.0 | Gradient clipping |
| `--save-interval` | 1 | Lưu checkpoint mỗi N epoch |

### 7.4. Cấu trúc package

Đã tạo `__init__.py` cho:

- `alpha_model/model/__init__.py`
- `alpha_model/train/__init__.py`

Cho phép import từ package:

```python
from alpha_model.model.alpha_model import ContextualBanditAlpha, Critic
from alpha_model.train.alpha_simulate import compute_reward_components_vectorized, total_reward
from alpha_model.train.offline_dataset_h5 import HDF5BanditDataset, collate_bandit_batch
```

### 7.5. Cách chạy

```bash
conda activate ai
cd /home/admin_wsl/projects/dflash
python3 -m alpha_model.train.train_AC --data-path alpha_model/alpha_dataset.h5 --epochs 10
```

### 7.6. Kiểm thử

- Đã chạy smoke test toàn bộ pipeline trên dữ liệu giả lập (256 records).
- Không lỗi shape, không lỗi dtype.
- Loss giảm qua các epoch.

## 8. Tài liệu (đã viết)

- `PROBLEM_FOMULATION.md`: định nghĩa bài toán MDP đầy đủ.
- `COMPLETED_TASKS.md`: tài liệu này.
- `NEXT_TASK.md`: kế hoạch các task tiếp theo.
