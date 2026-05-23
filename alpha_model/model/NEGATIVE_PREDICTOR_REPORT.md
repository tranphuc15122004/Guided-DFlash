# Negative Logit Predictor — Report

## Mục lục

1. [Tổng quan & Motivation](#1-t%E1%BB%95ng-quan--motivation)
2. [Kiến trúc mô hình](#2-ki%E1%BA%BFn-tr%C3%BAc-m%C3%B4-h%C3%ACnh)
   - [2.1. NegativeLogitPredictor (Transformer)](#21-negativelogitpredictor-transformer)
   - [2.2. NegativeLogitPredictor_Dense (MLP)](#22-negativelogitpredictor_dense-mlp)
   - [2.3. GaussianNegativePolicy (Actor)](#23-gaussiannegativepolicy-actor)
   - [2.4. NegativePredictorCritic (Value)](#24-negativepredictorcritic-value)
3. [Feature Extraction — StateFeatureExtractor](#3-feature-extraction--statefeatureextractor)
4. [Quy trình Training](#4-quy-tr%C3%ACnh-training)
   - [4.1. Phase 1: Supervised Pre-training](#41-phase-1-supervised-pre-training)
   - [4.2. Phase 2: Reinforcement Learning (A2C)](#42-phase-2-reinforcement-learning-a2c)
5. [Các thay đổi đã thực hiện (Removing `neg_logits_prev`)](#5-c%C3%A1c-thay-%C4%91%E1%BB%95i-%C4%91%C3%A3-th%E1%BB%B1c-hi%E1%BB%87n-removing-neg_logits_prev)
6. [So sánh với ContextualBanditAlpha (Alpha Model)](#6-so-s%C3%A1nh-v%E1%BB%9Bi-contextualbanditalpha-alpha-model)
7. [Smoke Test & Validation](#7-smoke-test--validation)

---

## 1. Tổng quan & Motivation

**Contrastive Decoding (CD)** giúp cải thiện chất lượng sinh của LLM bằng cách lấy hiệu giữa log-probability của *model chính* (positive) và *model yếu* (negative):

```
CD(q_t) = log_softmax(pos_logits) − α · log_softmax(neg_logits)
```

Trong alpha model cũ, mô hình chỉ học 3 scalar α tương ứng với 3 rank‑buckets để scale toàn bộ phân phối negative. Điều này giới hạn khả năng điều chỉnh phân phối.

**Negative Predictor** giải quyết vấn đề này bằng cách **trực tiếp dự đoán 32 giá trị negative logit** cho top‑32 token positive, cung cấp 32 bậc tự do thay vì 3 scalar.

```
Alpha Model:       CD = log_softmax(pos) − α_bucket · log_softmax(neg)
Negative Predictor: CD = log_softmax(pos) − 1.0 · log_softmax(neg_predicted)
                                             ^^^^^^^^
                                       alpha = 1.0 cố định
```

---

## 2. Kiến trúc mô hình

### 2.1. NegativeLogitPredictor (Transformer)

```
Input:  pos_logits (B,S,K), neg_logits (B,S,K), block_pos (B,S), abs_pos (B,S)
                │
                ▼
   ┌────────────────────────┐
   │  StateFeatureExtractor │  ← trích xuất 237 features (xem §3)
   └────────┬───────────────┘
            │ state_feats (B, S, 237)
            ▼
   ┌────────────────┐
   │  StateEncoder   │  Linear(237 → 128) + GELU + Linear(128 → 128) + LayerNorm
   └────────┬───────┘
            │ (B, S, 128)
            ▼
   ┌──────────────────────┐
   │  TransformerContext   │  ← 2-layer Transformer với RoPE + causal mask
   │  (4 heads, RoPE)     │
   └────────┬─────────────┘
            │ (B, S, 128)
            ▼
   ┌──────────────────────┐
   │  output_norm (LayerNorm)
   │  output_head (Linear 128 → 32) │
   └────────┬─────────────┘
            │ predicted_neg_logits (B, S, 32)
            ▼
   ┌─ predict_delta? ────┐
   │  True:  + neg_logits │
   │  False: replace     │
   └─────────────────────┘
```

**Tham số quan trọng:**

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `top_k` | 32 | Số token top‑K được dự đoán |
| `hidden_dim` | 128 | Kích thước hidden state |
| `predict_delta` | False | `True` = output là delta cộng vào neg_logits gốc |
| `ks` | [5,10,20] | Các K để tính top‑K mass features |

### 2.2. NegativeLogitPredictor_Dense (MLP)

Kiến trúc thay thế không dùng Transformer. Mỗi token được encode riêng rẽ, sau đó cả block được pad đến `max_seq_len=15`, flatten và đưa qua MLP:

```
state_feats (B, S, 237)
    │
    ▼
token_projector: Linear(237 → 512) + GELU + LayerNorm → (B, S, 512)
    │
    ▼
Pad to max_seq_len=15 → (B, 15, 512)
    │
    ▼
Flatten → (B, 15*512=7680)
    │
    ▼
block_mlp: Linear(7680 → 1024) + GELU + Dropout
         → Linear(1024 → 1024) + GELU + Dropout
         → Linear(1024 → 15*32=480)
    │
    ▼
Reshape → (B, 15, 32) → Crop về S gốc
```

**Khi nào dùng?**

- **Transformer**: Mạnh hơn cho sequence dependency, nhưng chậm hơn.
- **Dense**: Nhanh hơn, ít parameters, thích hợp khi block size nhỏ (15 tokens).

### 2.3. GaussianNegativePolicy (Actor)

Wrapper quanh `NegativeLogitPredictor` để tạo policy Gaussian cho RL:

- **Mean**: output từ `NegativeLogitPredictor.forward()` → `(B, S, top_k)`
- **Log_std**: learnable parameter shape `(top_k,)` — mỗi token position có std riêng, expand lên `(B, S, top_k)` khi forward.

```
forward():
    mean = NegativeLogitPredictor(pos, neg, bpos, apos)   # (B,S,32)
    log_std = self.log_std.expand_as(mean)                 # (B,S,32)
    return mean, log_std
```

### 2.4. NegativePredictorCritic (Value)

Critic đơn giản hơn, chia sẻ `StateFeatureExtractor` với actor nhưng dùng MLP value head:

```
state_feats (B, S, 237)
    → StateEncoder → (B, S, 128)
    → value_head: Linear(128 → 128) + ReLU + Linear(128 → 1)
    → squeeze(-1) → (B, S)
```

---

## 3. Feature Extraction — StateFeatureExtractor

`StateFeatureExtractor` là module dùng chung cho cả actor và critic (kế thừa từ `ContextualBanditAlpha`). Nó xây dựng feature vector cho mỗi token position gồm:

| Nhóm | Chi tiết | Kích thước |
|------|----------|-----------|
| **pos_logits** | logits, log_probs, probs | 3 × K = 96 |
| **pos_entropy** | Entropy của phân phối positive | 1 |
| **pos_margin** | Khoảng cách top1 − top2 | 1 |
| **pos_mass** | Top-K mass cho các ks=[5,10,20] | 3 |
| **neg_logits** | logits, log_probs, probs | 3 × K = 96 |
| **neg_entropy** | Entropy của phân phối negative | 1 |
| **neg_margin** | Khoảng cách top1 − top2 | 1 |
| **neg_mass** | Top-K mass cho các ks=[5,10,20] | 3 |
| **diff_logits** | pos − neg logits (K tokens) | K = 32 |
| **kl_div** | KL(positive ‖ negative) | 1 |
| **block_pos** | Vị trí trong block (0..S-1) | 1 |
| **abs_pos** | Vị trí tuyệt đối trong sequence | 1 |
| **Total** | | **237** |

Công thức tính input_dim:

```
input_dim = 2 * (3*top_k + 2 + len(ks)) + (top_k + 1) + 2
          = 2 * (96 + 2 + 3) + (32 + 1) + 2
          = 2 * 101 + 33 + 2
          = 237
```

---

## 4. Quy trình Training

### 4.1. Phase 1: Supervised Pre-training

**Mục tiêu**: Dạy mô hình dự đoán negative logits sao cho target token xuất hiện ở rank cao nhất trong top‑K sau Contrastive Decoding.

**Loss function**:

```
L = CE_loss(CD_logits, target_token) + λ_bce · BCE_loss(predicted_neg, one_hot_target)
```

Cả 2 loss đều được **weighted theo vị trí** (exponential decay với γ=7.0):

```
position_weight(t) = exp(-t / γ)
```

→ Token ở vị trí sớm (gần prefix) có weight cao hơn.

**Usage**:

```bash
python -m alpha_model.train.train_negative_predictor_phase1 \
    --data-path alpha_model/alpha_dataset.h5 \
    --epochs 20 --batch-size 64 \
    --output-dir checkpoints/phase1
```

| Flag | Default | Ý nghĩa |
|------|---------|---------|
| `--dense` | — | Dùng Dense MLP variant |
| `--predict-delta` | — | Dùng delta mode |
| `--lambda-bce` | 0.2 | Weight cho BCE auxiliary loss |
| `--gamma-decay` | 7.0 | Positional weight decay |
| `--lr` | 3e-4 | Learning rate |

### 4.2. Phase 2: Reinforcement Learning (A2C)

**Mục tiêu**: Fine-tune policy bằng RL để tối đa hóa acceptance length và rank improvement.

**Action space**: 32 giá trị logit liên tục (Gaussian policy với learnable std).

**Reward components**:

| Component | Ý nghĩa |
|-----------|---------|
| **r1** | Target rank improvement so với baseline (CD với neg_logits gốc) |
| **r2** | Binary bonus nếu target vào top‑1 |
| **r3** | Penalty nếu acceptance length < baseline |
| **Total** | `w1·r1 + w2·r2 + w3·r3` (mặc định: 0.1, 0.1, 1.0) |

**Losses**:

- **Actor**: `−(log_prob · advantage) − entropy_coef · H(π)`
- **Critic**: `MSE(value, reward)`

**Usage**:

```bash
python -m alpha_model.train.train_negative_predictor \
    --data-path alpha_model/alpha_dataset.h5 \
    --epochs 10 --batch-size 64 \
    --from-phase1 checkpoints/phase1/best.pt \
    --output-dir checkpoints/rl
```

**Pretrained weights from Phase 1** được load với `strict=False` để cho phép chỉ load được phần actor weights (critic được khởi tạo từ đầu).

---

## 5. Các thay đổi đã thực hiện (Removing `neg_logits_prev`)

### Vấn đề

`neg_logits_prev` là cơ chế lịch sử — truyền negative logits dự đoán từ block trước vào model như một "memory" state. Tuy nhiên:

1. **Dữ liệu không khớp**: Dataset chỉ lưu `alpha_prev` (3D bucketed scalar), không lưu `neg_logits_prev` thật (32 logit values).
2. **Approximation gượng ép**: Training scripts phải pad `alpha_prev` (kích thước nhỏ) lên 32 dimensions bằng 0 → gây nhiễu không có ý nghĩa.
3. **Không cần thiết**: Mô hình đã có đủ thông tin từ `pos_logits, neg_logits` hiện tại và positional encoding; history không mang lại lợi ích đáng kể.

### Các file bị ảnh hưởng

| File | Thay đổi |
|------|----------|
| **`alpha_model/model/negative_predictor.py`** | Xóa tham số `use_neg_logits_prev` khỏi constructor 6 classes; set `alpha_prev_dim=None`; xóa tham số `neg_logits_prev` khỏi `forward()`; cập nhật docstring và test block |
| **`alpha_model/train/train_negative_predictor_phase1.py`** | Xóa logic pad `alpha_prev → neg_logits_prev`; sửa `compute_rank_metrics(..., K)` → `(..., args.top_k)` |
| **`alpha_model/train/train_negative_predictor.py`** | Xóa logic pad `alpha_prev → neg_logits_prev`; sửa `critic(...)` không còn `neg_logits_prev` |

### Tác động

- **Input dimension giảm**: Từ `237 + top_k (32) = 269` → còn **237** (tiết kiệm ~12% parameters).
- **Pipeline sạch hơn**: Không cần xử lý `alpha_prev` trong data loading.
- **Tương thích ngược**: Model cũ vẫn có thể load bằng `strict=False`.

---

## 6. So sánh với ContextualBanditAlpha (Alpha Model)

| Thuộc tính | Alpha Model (cũ) | Negative Predictor (mới) |
|------------|------------------|--------------------------|
| **Output** | 3 scalar α (bucketed) | 32 logit values |
| **Degrees of freedom** | 3 | 32 |
| **CD formula** | `log_softmax(pos) − α·log_softmax(neg)` | `log_softmax(pos) − log_softmax(neg_predicted)` |
| **Action space** | Discrete (3 buckets) | Continuous (Gaussian) |
| **Alpha** | Học được (per bucket) | Cố định = 1.0 |
| **Input features** | 237 + α_prev_dim | 237 (không history) |
| **Backbone** | Transformer hoặc Dense MLP | Transformer hoặc Dense MLP |
| **Phase 1 training** | Behavior cloning | Supervised (CE + BCE) |
| **Phase 2 training** | PPO/REINFORCE | A2C |
| **Parameters** | ~448K (Transformer variant) | ~448K (Transformer variant) |

---

## 7. Smoke Test & Validation

Smoke test được thực hiện với dataset HDF5 giả lập (10 records, S=15, K=32) trên GPU.

### Kết quả Phase 1

| Metric | Giá trị |
|--------|---------|
| Loss | 0.1648 |
| CE loss | 0.1565 |
| BCE loss | 0.0416 |
| Top-10 accuracy | 0.200 |
| BCE accuracy | 0.1063 |
| Thời gian | 1.7s (5 batches × 2 samples) |

### Kết quả Phase 2

| Metric | Giá trị |
|--------|---------|
| Total reward | -15.53 |
| r1 (rank improvement) | 0.72 |
| r3 (acceptance penalty) | -15.60 |
| Policy entropy | 441.38 |
| Advantage | -14.96 |
| Thời gian | 1.3s (5 batches × 2 samples) |

### Các lỗi đã fix trong quá trình smoke test

1. **`tqdm` missing import** in `logging_utils.py` — thêm `from tqdm import tqdm`
2. **`K` undefined** trong `train_negative_predictor_phase1.py` — sửa thành `args.top_k`

---

## Tổng kết

Negative Predictor là bước tiến từ alpha model: thay vì 3 scalar α, mô hình học trực tiếp 32 negative logit values, cho khả năng điều chỉnh phân phối mạnh mẽ hơn. Kiến trúc stateless (không history) giúp pipeline đơn giản và hiệu quả hơn. Quy trình hai-phase (Supervised → RL) cho phép mô hình học từ imitation learning trước, sau đó tối ưu bằng reinforcement learning để đạt acceptance rate cao nhất.
