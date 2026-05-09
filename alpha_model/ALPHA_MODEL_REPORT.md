# Báo cáo: Alpha Model — Reinforcement Learning cho Dynamic Alpha trong Contrastive Decoding

## 1. Tổng quan

**Mục tiêu:** Trong cơ chế Contrastive Decoding (CD) của DFlash, hệ số alpha ($\text{logits} = \text{pos} - \alpha \cdot \text{neg}$) được cố định cho toàn bộ quá trình decode. Điều này không tối ưu vì mức độ "khác biệt" giữa positive và negative draft thay đổi theo từng token, từng block. Mục tiêu là train một mô hình RL nhẹ (`ContextualBanditAlpha`) để **điều chỉnh alpha động** cho từng token trong mỗi decode block.

**Ý tưởng:** Sử dụng Actor-Critic (A2C) với:
- **Actor** (`ContextualBanditAlpha` / `GaussianPolicy`): sinh 3 hệ số alpha cho mỗi vị trí trong block (mỗi alpha áp dụng cho một bucket rank).
- **Critic**: ước lượng value function $V(s)$ để tính advantage.

**Kết quả hiện tại:** Mô hình được train 70 epochs trên ~4 triệu records (gsm8k + magicoder). Training bị gián đoạn do walltime (đạt epoch 70/100). Metrics cho thấy model đang học chậm — reward cải thiện nhẹ từ -17.28 → -16.54, r1 (rank improvement) từ -13.31 → -6.37, r2 (top-1 bonus) từ 6.19 → 6.69.

---

## 2. Bài toán MDP (Markov Decision Process)

Được mô tả chi tiết trong [`alpha_problem_formulation_note.md`](alpha_problem_formulation_note.md).

### 2.1 State $S$

Tại mỗi decode block, state bao gồm các features từ positive và negative draft logits (top-32 tokens):

| Nhóm | Features | Kích thước |
|------|----------|-----------|
| **Positive side** | `logits` (K), `log_softmax` (K), `softmax` (K), entropy (1), top-1 margin (1), top-k masses (3) | $3K + 2 + \|ks\|$ |
| **Negative side** | (tương tự positive) | $3K + 2 + \|ks\|$ |
| **Cross features** | `pos_logits - neg_logits` (K), KL divergence (1) | $K + 1$ |
| **Positional** | block position (0..14, normalized), absolute position (0..max_len, normalized) | 2 |
| **History** | alpha từ block trước $\alpha_{t-1}$ | 3 |

Tổng input dimension: $2 \times (3K + 2 + \|ks\|) + (K + 1) + 2 + 3 = 2 \times 101 + 33 + 5 = 240$ (với K=32, |ks|=3).

### 2.2 Action $A$

Mỗi action là một bộ 3 hệ số alpha, mỗi alpha áp dụng cho một **bucket rank**:
- **Bucket 0**: rank 0–10 (top tokens có xác suất cao nhất)
- **Bucket 1**: rank 11–20
- **Bucket 2**: rank 21+ (bao gồm cả token không nằm trong top-K)

$$
a_t \in \mathbb{R}^{(B-1) \times 3}, \quad a_t \in [-\alpha_{\max}, \alpha_{\max}],\; \alpha_{\max}=2.0
$$

Việc dùng 3 buckets thay vì alpha riêng cho từng token làm giảm độ phức tạp và giúp học dễ hơn.

### 2.3 Reward $R$

Reward gồm 3 thành phần:

1. **$r_1$ — Rank improvement**: thưởng khi rank của target token được cải thiện sau contrastive:
   $$r_1 = \sum_{i=1}^{B-1} \Delta_{\text{target rank}} \cdot \exp\left(-\frac{i-1}{\gamma}\right)$$

2. **$r_2$ — Top-1 bonus**: thưởng thêm nếu target được đẩy lên top-1:
   $$r_2 = \sum_{i=1}^{B-1} 2 \cdot \exp\left(-\frac{i-1}{\gamma}\right) \cdot \mathbb{1}[\text{rank}_\text{after} = 0]$$

3. **$r_3$ — Acceptance length**: ưu tiên không làm giảm acceptance length trước khi cải thiện nó:
   $$r_3 = \max(\Delta \text{acc\_len}, 0) - \lambda \cdot \max(-\Delta \text{acc\_len}, 0)$$

**Reward tổng hợp:**
$$R = w_1 r_1 + w_2 r_2 + w_3 r_3$$

Với $w_1=0.1, w_2=0.1, w_3=1.0, \lambda=3.0, \gamma=7$.

### 2.4 Transition $P$

Sau khi draft model sinh positive và negative logits cho block hiện tại:
1. Alpha model đề xuất 3 hệ số alpha (action)
2. Contrastive decoding áp dụng alpha để sinh draft tokens
3. Target model verify, trả về acceptance length và token thật tại vị trí reject

### 2.5 Episode

Một episode = toàn bộ quá trình decode một câu trả lời (multi-turn). Dữ liệu được collect offline → train ngẫu nhiên (i.i.d.) để tận dụng toàn bộ dữ liệu.

---

## 3. Kiến trúc Model

### 3.1 State Feature Extractor (`StateFeatureExtractor`)

- **Input:** `pos_logits`, `neg_logits` (B, S, K), `block_pos`, `abs_pos`, `alpha_prev` (optional)
- **Xử lý:**
  - Tính softmax, log_softmax cho cả 2 phía
  - Tính entropy, top-1 margin, top-k mass (k=5,10,20)
  - Cross features: diff_logits, KL divergence
  - Ghép với positional features và alpha_prev
- **Output:** feature vector kích thước 240

### 3.2 State Encoder (`StateEncoder`)

```python
nn.Sequential(
    Linear(240 → 128), GELU,
    Linear(128 → 128), LayerNorm(128)
)
```

### 3.3 Transformer với RoPE (`TransformerContext`)

- 2 layers Transformer với RoPE positional encoding và causal mask
- 4 attention heads, hidden dim = 128
- Mục đích: mô hình hóa phụ thuộc giữa các token trong block (sequential dependency)

### 3.4 Actor Head

```python
Linear(128 → 3)  # 3 buckets
alphas = max_alpha * tanh(logits)  # ∈ [-max_alpha, max_alpha]
```

### 3.5 Critic

- Cùng feature extractor và encoder với actor
- Value head: `Linear(128 → 128) → ReLU → Linear(128 → 1)` → scalar per position

### 3.6 GaussianPolicy (training)

Kế thừa `ContextualBanditAlpha`, thêm `log_std` (3 learnable parameters) để tạo phân phối Gaussian cho reparameterization trick:
```
mean = super().forward(...)
log_std = self.log_std  # (3,) broadcast to (B, S, 3)
return mean, log_std
```

### 3.7 Số tham số

| Module | Parameters |
|--------|-----------|
| `StateFeatureExtractor` | 0 (stateless, chỉ compute graph) |
| `StateEncoder` | ~49,408 |
| `TransformerContext` (2 layers × RoPEBlock) | ~446,976 |
| `AlphaHead` | 387 |
| **Actor total** | **~496,771** |
| **Critic total** | **~497,025** |
| **GaussianPolicy extra** | 3 (`log_std`) |

→ Tổng ~0.5M parameters — rất nhẹ so với target/draft model.

---

## 4. Pipeline Thu thập Dữ liệu

### 4.1 Data Collecting (`data_collecting.py`)

Chạy inference với DFlash để collect per-block records:

```bash
python -m alpha_model.data_collecting \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --collector-output-dir alpha_model/collected_alpha_records/my_run/
```

**Mỗi record lưu:**
- `sample_id`, `turn_index`, `block_index` — identity
- `draft_topk_token_ids` (15, K) — int32, top-K token IDs từ positive draft
- `draft_topk_logits` (15, K) — float32, logits tương ứng
- `neg_logits_on_draft_topk_ids` (15, K) — float32, negative logits lấy trên cùng token IDs
- `block_position` (15,) — normalized (0..1)
- `absolute_position` (15,) — normalized (0..1)
- `target_token_id` (15,) — int32, token thật từ target model
- `acceptance_length` () — int, số token được accept
- `alpha_prev` (15, 3) — alpha từ block trước

**Chunking:** Cứ 2048 records → flush thành 1 file `.pt`.

### 4.2 Merge to HDF5 (`merge_pt_shards_to_hdf5.py`)

```bash
python alpha_model/merge_pt_shards_to_hdf5.py \
  --input-dir alpha_model/collected_alpha_records/ \
  --output-file alpha_model/alpha_dataset.h5 \
  --compression gzip
```

Scan toàn bộ shards để infer shape/dtype, sau đó ghi vào HDF5 dataset với chunking tối ưu.

### 4.3 Dataset hiện tại

| Dataset | Số records | Tỉ lệ |
|---------|-----------|-------|
| gsm8k | 7,473 samples × ? blocks | ~ |
| magicoder | ~20,000 samples × ? blocks | ~ |
| **Total** | **4,071,799 records** | ~23GB RAM |

---

## 5. Training Pipeline

### 5.1 Dataset Loading (`HDF5BanditDataset`)

- Load toàn bộ HDF5 vào RAM (~23GB) ngay khi khởi tạo — chunking HDF5 không tối ưu cho random access.
- Fields: `pos_logits`, `neg_logits`, `topk_token_ids`, `target_token_ids`, `block_pos`, `abs_pos`, `alpha_prev`, `baseline_acc_len`

### 5.2 Training Loop (`train_AC.py`)

**Hyperparameters:**

| Parameter | Value | Note |
|-----------|-------|------|
| Batch size | 1024 | |
| Epochs | 100 (đạt 70) | Bị gián đoạn bởi walltime |
| Learning rate | 1e-4 | Adam optimizer |
| Top-K | 32 | |
| Hidden dim | 128 | |
| Num buckets | 3 | |
| Max alpha | 2.0 | |
| Entropy coef | 0.01 | |
| Max grad norm | 1.0 | Gradient clipping |
| Save interval | 1 | Save mỗi epoch |

**Training algorithm** (Advantage Actor-Critic):

1. **Actor forward:** `mean, log_std = actor(state)` → sample action từ Normal(mean, std) via reparameterization
2. **Reward computation:** `compute_reward_components_vectorized(pos, neg, topk_ids, target, action, baseline, ...)` vectorized trên toàn batch
3. **Critic forward:** `value = critic(state)` → advantage = reward - value
4. **Actor loss:** $-\log \pi(a|s) \cdot A + \beta \cdot H(\pi)$ (entropy bonus)
5. **Critic loss:** MSE(value, reward)
6. **Update:** gradient clipping @ 1.0

### 5.3 Reward Simulation (`alpha_simulate.py`)

- `simulate_acceptance_length_batch()`: mô phỏng quá trình verification trên offline data
- `compute_reward_components_batch()`: tính r1, r2, r3 vectorized
- `total_reward_batch()`: tổng hợp R = w1·r1 + w2·r2 + w3·r3

### 5.4 Kết quả Training

```
Epoch    reward     r1         r2        r3         entropy    baseline_acc
─────────────────────────────────────────────────────────────────────────────
    1   -17.28    -13.31      6.19     -16.57      42.93        ?
   10   -16.72     -8.27      6.53     -16.57      43.81        ?
   20   -16.62     -7.07      6.63     -16.57      44.68        ?
   30   -16.60     -6.92      6.64     -16.57      45.27        ?
   40   -16.57     -6.64      6.66     -16.57      45.77        ?
   50   -16.56     -6.57      6.66     -16.57      46.12        ?
   60   -16.56     -6.60      6.66     -16.57      46.39        ?
   70   -16.54     -6.37      6.69     -16.57      45.79        ?
```

**Nhận xét:**
- **r3 (acceptance length)** hầu như không đổi (~ -16.57) — model chưa học cách cải thiện acceptance length đáng kể
- **r1 (rank improvement)** cải thiện rõ: từ -13.31 → -6.37 (giảm độ xấu của rank)
- **r2 (top-1 bonus)** cải thiện nhẹ: 6.19 → 6.69
- **Entropy** tăng từ 42.93 → 45.79 (policy vẫn giữ exploration)
- **Critic loss** giảm chậm: 77.1 → 53.1 (value estimation còn nhiều noise)
- Training dừng ở epoch 70/100 do walltime (6h), chưa hội tụ hoàn toàn

---

## 6. Inference Integration

### 6.1 CD_alpha_model scheme (`scheme/CD_alpha_model.py`)

Tích hợp alpha model vào DFlash generation pipeline:
1. Tính positive & negative draft logits
2. Nếu có alpha model → forward qua `ContextualBanditAlpha` để lấy 3 alphas
3. `_build_full_alpha_from_buckets()`: mở rộng 3 alpha buckets → full vocabulary alpha mapping
4. `apply_cd_logits_dynamic()`: $\log\text{softmax}(\text{pos}) - \alpha \cdot \log\text{softmax}(\text{neg})$
5. Candidate filtering + top-6 clamp
6. Sample và verify với target model

**Checkpoint loading:**
- Load `actor_epoch70.pt` (state dict của `GaussianPolicy`)
- Tạo `ContextualBanditAlpha` (base class) → load với `strict=False` (bỏ qua `log_std`)

### 6.2 So sánh: Fixed Alpha vs Dynamic Alpha

| Aspect | Fixed Alpha (baseline) | Dynamic Alpha (model) |
|--------|----------------------|----------------------|
| Alpha values | Constant (e.g. 0.6) | Per-position, per-bucket |
| Computation | Trivial | ~0.5M param forward |
| Adaptivity | None | Adapts to logit distribution |
| Current status | Working | Training ongoing |

---

## 7. Tổng kết và Hướng phát triển

### Đã làm được
✅ Xây dựng MDP formulation hoàn chỉnh cho bài toán dynamic alpha  
✅ Implement actor-critic architecture với Transformer + RoPE  
✅ Pipeline thu thập dữ liệu offline (4M records)  
✅ Training loop với reward simulation vectorized  
✅ Tích hợp inference vào DFlash pipeline  

### Cần cải thiện
❌ **r3 (acceptance length) không cải thiện** — reward function cần điều chỉnh (tăng w3, giảm penalty λ)  
❌ **Critic loss cao** — value estimation chưa tốt, có thể cần thêm features hoặc tăng model capacity  
❌ **Training chưa hội tụ** — cần thêm epochs (~200-300)  
❌ **Walltime limit** — 6h không đủ cho 100 epochs với 4M records; cần tối ưu data loading hoặc tăng batch size  

### Hướng phát triển
1. **Tuning reward weights:** thử $w_3=2.0$ hoặc $\lambda=1.0$ để khuyến khích tăng acceptance length
2. **Curriculum learning:** train trên các block dễ trước, khó sau
3. **Online RL:** thay vì offline dataset, collect + train online để tránh distribution shift
4. **Multi-GPU training:** dùng `torchrun` để tận dụng nhiều GPU
5. **Cải thiện state representation:** thêm context features (layer depth, attention patterns)
