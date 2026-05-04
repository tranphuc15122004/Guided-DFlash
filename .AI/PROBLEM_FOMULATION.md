# Bài toán MDP cho điều chỉnh động α trong Dflash

## 1. Bối cảnh

- **Mô hình**: Dflash (diffusion speculative decoding), block size = 16 tokens.
- Token đầu mỗi block là token đã được target model xác thực từ block trước.
- Cần diffusion 15 token còn lại.
- **Contrastive decoding**:
logit_contrastive = logit_positive - α * logit_negative

- **Vấn đề**: Hệ số α cố định không hiệu quả → cần một RL agent điều chỉnh α động theo từng token dựa trên trạng thái của drafter.

## 2. Định nghĩa MDP

M = (S, A, P, R, γ)

### 2.1 State (S)

Tại mỗi block, state được xây dựng từ các thông tin sau:

- **Positive sample** (từ Dflash model, 32 top token theo logit):
- log(logit)
- log(softmax(logit))
- softmax(logit)
- entropy, top‑1 margin, top‑k mass (k=5,10,20)
- **Negative sample** (tương tự, nhưng được tạo bằng cách thay đổi token đầu block và dropout target hidden):
- log(logit)
- log(softmax(logit))
- softmax(logit)
- entropy, top‑1 margin, top‑k mass
- **Cross features**:
- hiệu (pos - neg)
- KL divergence
- **Vị trí**:
- block position (normalized)
- absolute token position (normalized)
- **Hệ số α từ block trước**: α_prev (shape S×3)

Kích thước: mỗi feature có batch=1, seq_len=S = block_size-1, K=32. Tất cả được ghép thành vector.

### 2.2 Action (A)

Mỗi token (vị trí i trong block) được gán 3 hệ số α tương ứng với 3 nhóm rank buckets:

- **Bucket 0**: top 0..9 (rank 0-9)
- **Bucket 1**: top 10..19 (rank 10-19)
- **Bucket 2**: rank >= 20

Do đó, action có kích thước `(S, 3)`. Mỗi α ∈ [0, 2] (có thể mở rộng sang [-2,2] nếu cần).  
Cách xác định bucket: dựa trên rank của target token trong **positive logits** (trước contrastive).

### 2.3 Transition (P)

1. Agent nhận state từ drafter.
2. Agent sinh action (α_per_token).
3. Tính logit_contrastive = pos - α * neg (theo bucket cho từng token).
4. Target model verify:

- Token được chọn = argmax(logit_contrastive) (greedy).
- So sánh với target token (từ target model).
- Nếu đúng → accept, tiếp tục; nếu sai → dừng.

5. Nhận reward (xem §2.4).
2. Chuyển sang block tiếp theo (trạng thái mới được xây dựng từ block đó, với α_prev = action hiện tại).

### 2.4 Reward (R)

Reward tổng hợp cho mỗi block:
R = w1 *r1 + w2* r2 + w3 * r3

với w1=0.1, w2=0.1, w3=1.0.

**r1**: thay đổi rank của target token (độ cải thiện thứ hạng)

r1 = Σ_{i=0}^{S-1} (rank_before[i] - rank_after[i]) * exp(-i/γ)

γ = 7.0

**r2**: thưởng thêm nếu target token lên top‑1

r2 = Σ_{i=0}^{S-1} 2 *exp(-i/γ)* 𝟙(rank_after[i] == 0)

**r3**: dựa trên sự thay đổi của acceptance length so với baseline

Δ = acc_len_contrastive - baseline_acc_len
r3 = max(Δ, 0) - λ * max(-Δ, 0)

với λ = 3.0.

**Ghi chú**:

- `baseline_acc_len` là acceptance length ghi nhận được khi thu thập dữ liệu (ứng với α cố định, coi như baseline).
- `rank_before` và `rank_after` được tính trên top‑K (K=32). Nếu target token không nằm trong top‑K, rank = K.
- Các reward r1, r2 được tính trên **toàn bộ S token** (không mask theo accept length). Chỉ r3 dùng acceptance length thực tế.

### 2.5 Episode

Một episode là toàn bộ quá trình sinh câu trả lời cho một prompt. Dữ liệu được thu thập offline dưới dạng các record (mỗi record là một block) theo đúng trình tự `(sample_id, turn_index, block_index)`.

## 3. Dữ liệu thu thập (offline dataset)

Mỗi record (đã được squeeze batch) bao gồm:

| Field | Shape | Dtype | Mô tả |
|-------|-------|-------|-------|
| sample_id | int | | ID của mẫu |
| turn_index | int | | Thứ tự turn trong hội thoại |
| block_index | int | | Chỉ số block trong quá trình decode |
| draft_topk_token_ids | (S, K) | int32 | Top‑K token IDs từ positive draft |
| draft_topk_logits | (S, K) | float32 | Logits tương ứng trên các token đó |
| neg_logits_on_draft_topk_ids | (S, K) | float32 | Logits negative trên cùng tập token IDs |
| block_position | (S,) | float32 | Vị trí trong block (normalized 0..1) |
| absolute_position | (S,) | float32 | Vị trí tuyệt đối (normalized 0..1) |
| target_token_id | (S,) | int32 | Token do target model sinh ra |
| acceptance_length | int | | Baseline acceptance length (với α cố định) |
| alpha_prev | (S, 3) | float32 | α của block trước (zero cho block đầu) |
| alpha_applied | (S, 3) | float32 | α thực tế đã dùng khi collect (thường là hằng số) |

## 4. Mô hình RL

### 4.1 Actor: ContextualBanditAlpha

- **File**: `alpha_model/model/alpha_model.py`.
- **Input**: pos_logits, neg_logits (B×S×K), block_pos, abs_pos (B×S), alpha_prev (B×S×3)
- **Kiến trúc**:
  - `StateFeatureExtractor`: trích xuất các đặc trưng thống kê (entropy, margin, top‑k mass, KL div, diff, vị trí, alpha_prev)
  - `StateEncoder`: MLP 2 lớp (128 → 128)
  - `TransformerContext`: 2 layer RoPE‑Transformer, causal, hidden_dim=128, 4 heads
  - `alpha_head`: Linear(128 → 3) → `max_alpha * tanh` → α ∈ [-2,2]
- **Output**: (B, S, 3) action
- **Gaussian extension**: `GaussianPolicy` (trong `train_AC.py`) thêm `log_std` học được → output `(mean, log_std)`.

### 4.2 Critic

- **File**: `alpha_model/model/alpha_model.py`.
- Input tương tự actor, không dùng transformer (chỉ MLP), output giá trị V(s) cho mỗi token (B, S).

## 5. Huấn luyện

### 5.1 Dataset

- **Định dạng**: HDF5 (`alpha_dataset.h5`), merge từ các chunk `.pt`.
- **Dataset loader**: `alpha_model/train/offline_dataset_h5.py` — lớp `HDF5BanditDataset`.
- Mỗi record = 1 block, gồm 8 field (pos_logits, neg_logits, topk_ids, target_ids, positions, alpha_prev, baseline).
- Collate function: `collate_bandit_batch` — stack batch dim.

### 5.2 Thuật toán huấn luyện

- **A2C bandit** (Advantage Actor-Critic, không PPO clipped).
- **Actor**: `GaussianPolicy` (kế thừa `ContextualBanditAlpha`) — output `(mean, log_std)`.
- **Action sampling**: reparameterization trick (`rsample()`), clamp về `[-max_alpha, max_alpha]`.
- **Loss**:
  - Actor: `-(log_prob * advantage).mean() - entropy_coef * entropy`.
  - Critic: `MSE(value, reward)`.
- **Gradient clipping**: `clip_grad_norm_` với max_norm=1.0.
- **Optimizer**: Adam (lr=3e-4).

### 5.3 Script huấn luyện

- **File**: `alpha_model/train/train_AC.py`.
- Hỗ trợ full batch (không per-sample loop).
- Cấu hình qua argparse (batch_size, epochs, lr, entropy_coef, max_grad_norm, ...).
- Lưu checkpoint sau mỗi epoch.
- Cách chạy: `python3 -m alpha_model.train.train_AC --data-path alpha_model/alpha_dataset.h5 --epochs 10`.

### 5.4 Hàm tính reward (đã triển khai)

- **File**: `alpha_model/train/alpha_simulate.py`.
- `compute_reward_components_batch`: hỗ trợ full batch (B, S, K), tính r1, r2, r3.
- `simulate_acceptance_length_batch`: mô phỏng acceptance length với α bất kỳ (có thể dùng để validate).
- `compute_bucket_thresholds`: tính thresholds cho bucket từ topK và num_buckets.
- Các alias backward-compatible: `compute_reward_components_vectorized`, `simulate_acceptance_length_vectorized`, `total_reward`.

## 6. Các quyết định thiết kế quan trọng

- **Top‑K**: 32 (có thể mở rộng sau).
- **Greedy sampling** (temperature=0).
- **Normalize vị trí**: có sẵn trong dữ liệu.
- **Không cần baseline acceptance length riêng**: dùng acceptance_length có trong record làm baseline.
- **r1, r2 tính trên toàn bộ token** (không mask accept length). r3 dùng độ dài thực tế.
- **Huấn luyện A2C bandit** (không PPO clipped), có thể mở rộng MDP sau.
- **Gaussian policy** với `log_std` học được (reparameterization trick).
- **Gradient clipping** để ổn định huấn luyện.
- **Dataset HDF5** thay vì `.pt` chunks để tiện đọc/ghi và memory mapping.

## 7. Các công việc đã thực hiện

- [x] Định nghĩa bài toán MDP (tài liệu này).
- [x] Xây dựng script thu thập dữ liệu (`alpha_model/data_collecting.py`).
- [x] Thiết kế mô hình RL: Actor (`ContextualBanditAlpha`), Critic — `alpha_model/model/alpha_model.py`.
- [x] Triển khai hàm simulate và reward vector hóa — `alpha_model/train/alpha_simulate.py`.
- [x] Task T2: Dataset loader HDF5 — `alpha_model/train/offline_dataset_h5.py`.
- [x] Task T3: Script huấn luyện A2C bandit — `alpha_model/train/train_AC.py`.
- [x] Cấu trúc package với `__init__.py` (`alpha_model/model/`, `alpha_model/train/`).
- [x] Smoke test toàn bộ pipeline trên dữ liệu giả lập — không lỗi.

## 8. Kế hoạch tiếp theo (tóm tắt)

1. ✅ **Task T2**: OfflineBanditDataset (HDF5) — đã hoàn thành.
2. ✅ **Task T3**: Huấn luyện A2C bandit với actor và critic — đã hoàn thành.
3. 🔲 **Task T4**: Tích hợp policy đã train vào Dflash thực tế.
4. 🔲 **Task T5**: Đánh giá và tinh chỉnh.
