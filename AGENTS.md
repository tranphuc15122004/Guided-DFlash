# Guided-DFlash Agent Guide

## Project Overview

**Guided-DFlash** extends [DFlash](https://github.com/z-lab/dflash) — a block diffusion speculative decoding framework — with **learned contrastive decoding** via two approaches:

1. **Alpha Model** (legacy): Learns 3 scalar alpha coefficients (one per rank bucket) to scale the negative distribution in `log_softmax(pos) − α · log_softmax(neg)`.
2. **Negative Predictor** (new): Directly predicts 32 negative logit values for the top‑32 positive tokens (32 degrees of freedom), with a fixed α = 1.0, replacing `α · log_softmax(neg)` with `log_softmax(neg_predicted)`.

Both approaches use reinforcement learning (A2C / GRPO) to optimize acceptance rate.

Core repo: `https://github.com/tranphuc15122004/Guided-DFlash`
Upstream paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

---

## Quick Start

```bash
# Environment
conda create -n dflash python=3.11
conda activate dflash
pip install uv
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation   # optional
```

---

## Architecture

```
model/                          — Core DFlash draft model + CD/VCD logit functions
  dflash.py                     — Qwen3DFlashDraftModel (modified attention with target context)
  utils.py                      — apply_cd_logits(), apply_vcd_logits(), sample(), load_and_process_dataset()

scheme/                         — 10 inference/decoding variants
  CD_v1.py .. CD_v4.py          — Contrastive Decoding variants
  VCD_v1.py .. VCD_v3.py        — Vectorized CD variants
  CD_alpha_model.py             — CD with learned 3-bucket alpha (ContextualBanditAlpha)
  CD_negative_model.py          — CD with learned 32 logit values (NegativeLogitPredictor)
  mask_50_redraft.py            — Mask 50% + redraft variant
  run_metadata.py               — Run configuration logging

alpha_model/                    — Learning pipeline (alpha + negative predictor)
  __init__.py                   — Public API exports (both model families)
  data_collecting.py            — Per-block record collection from generation
  merge_pt_shards_to_hdf5.py    — Merge .pt shards into single HDF5 file
  fix_target_token.py           — Target token correction utility
  auto_submit_job_data_collect.py — Automated PBS job submission for data collection
  utils.py                      — Shared dataset utilities
  model/
    alpha_model.py              — ContextualBanditAlpha, Critic, StateFeatureExtractor, TransformerContext
    negative_predictor.py       — NegativeLogitPredictor, GaussianNegativePolicy, NegativePredictorCritic
    NEGATIVE_PREDICTOR_REPORT.md — Full architecture report (Vietnamese)
  train/
    train_AC.py                 — A2C training for Alpha Model (3-bucket)
    train_GRPO.py               — GRPO training for Alpha Model
    train_negative_predictor_phase1.py — Phase 1 supervised training for Negative Predictor
    train_negative_predictor.py — Phase 2 RL (A2C) training for Negative Predictor
    alpha_simulate.py           — Loss functions, reward computation, acceptance simulation
    offline_dataset_h5.py       — HDF5 dataset loader (full-RAM loading)
    metrics.py                  — Comprehensive metrics (rank, distribution, action, acceptance)
    logging_utils.py            — TrainingLogger class (CSV, checkpoint, CLI)

alpha_adjusting/                — Offline alpha sweep analysis
  alpha_analysis.py             — Grid search over alpha values on collected logits
  data_colecting.py             — RejectTokenCollector (legacy data collection)

analysis/                       — Benchmarking, evaluation, and visualization
  tok_level_ana.py              — Token-level acceptance analysis
  token_level_core.py           — Core analysis utilities
  token_level_reporting.py      — Reporting for token-level analysis
  CD_vs_Dflash.py               — Compare CD variants vs vanilla DFlash
  VCD_vs_Dflash.py              — Compare VCD vs DFlash
  alpha_bucket_analysis.py      — Bucket statistics from training logs
  alpha_bucket_deep_analysis.py — Deep bucket analysis (target-vs-predicted)
  CDv2_check_model_entropy.py   — Entropy analysis for CDv2
  CDv2_embed_distance_ana.py    — Embedding distance analysis
  CDv2_target_survey.py         — Target token survey
  inspect_target_hidden.py      — Target hidden state inspection
  target_tok.py / target_tok_2.py — Target token analysis
  build_token_neighbor_table.py — Build token neighbor table for enhanced negative sampling
  visualize.py                  — Visualization tools

script/                         — PBS job submission scripts
  run_h200_alpha_collect.sh     — H200 data collection launcher
  run_a100_alpha_collect.sh     — A100 data collection launcher
  cd_alpha_model.pbs            — PBS script for alpha model evaluation
  train_alpha_h200.pbs          — PBS script for alpha model training
  tok_level_ana_h100.pbs        — PBS token-level analysis
  target_tok_h100.pbs           — PBS target token analysis
  fix_target_token.pbs          — PBS target token fix
  h200_alpha_ana.pbs            — PBS alpha analysis
  merge_data.pbs                — PBS data merge
  debug_*_flash.pbs             — PBS debug scripts (A100/H100/V100)
  cache_alpha_datasets.py       — Cache datasets locally
  parse_*_training_log*.py      — Training log parsers

test/                           — Smoke / unit tests
  test_infer.py                 — Inference smoke test
  test_dataest.py               — Dataset loading test
  test_alpha_dataset_loading.py — HDF5 alpha dataset loading test

checkpoints/                    — Trained model weights
  actor_best.pt                 — Best actor checkpoint
  critic_best.pt                — Best critic checkpoint
  training_history.csv          — CSV training history

distributed.py                  — Multi-GPU init, gather, barrier utilities
```

---

## Running Benchmarks

```bash
# Transformers backend (single GPU)
bash run_benchmark.sh

# SGLang backend (multi-concurrency)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python benchmark_sglang.py --target-model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16

# Specific scheme benchmark (edit benchmark.py to set SCHEME)
python benchmark.py --target-model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 --dataset gsm8k

# Analysis (run as module from repo root)
python -m analysis.tok_level_ana
python -m analysis.CD_vs_Dflash
python -m analysis.VCD_vs_Dflash
```

---

## Inference Schemes

All 10 decoding variants live in `scheme/`. Each is runnable standalone:

```bash
python -m scheme.CD_v2 --target-model Qwen/Qwen3-4B --draft-model z-lab/Qwen3-4B-DFlash-b16 --dataset gsm8k
```

### CD Family (Contrastive Decoding)

| Scheme | Negative Source | Key Feature |
|--------|----------------|-------------|
| `CD_v1` | Random token replacement (first token) | Basic CD |
| `CD_v2` | Random token replacement + candidate filtering | CD with top‑K candidate filter |
| `CD_v2_soften` | Softened candidate masking | Softer probability filtering |
| `CD_v3` | Random replacement | Alternative CD formulation |
| `CD_v4` | Random replacement | Alternative CD formulation |

### VCD Family (Vectorized Contrastive Decoding)

| Scheme | Negative Source | Key Feature |
|--------|----------------|-------------|
| `VCD_v1` | Remask + redraft first token | Vectorized, one draft forward pass |
| `VCD_v2` | Remask + redraft | VCD variant |
| `VCD_v3` | Remask + redraft | VCD variant |

### Learned Model Schemes

| Scheme | Model | Output | Action Space |
|--------|-------|--------|-------------|
| `CD_alpha_model` | `ContextualBanditAlpha` | 3 scalar α (bucketed) | Discrete (3 buckets) |
| `CD_negative_model` | `NegativeLogitPredictor` | 32 logit values | Continuous (Gaussian) |

### Mask Variant

| Scheme | Description |
|--------|-------------|
| `mask_50_redraft` | Mask 50% of tokens + redraft from draft model |

---

## Alpha Model Training (Legacy, 3-bucket)

### 1. Data Collection

```bash
python -m alpha_model.data_collecting \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --collector-output-dir alpha_model/collected_alpha_records/my_run/
```

This stores per-block records as `.pt` shards.

### 2. Merge into HDF5

```bash
python alpha_model/merge_pt_shards_to_hdf5.py \
  --input-dir alpha_model/collected_alpha_records \
  --output-file alpha_model/alpha_dataset.h5 \
  --compression gzip
```

Stored fields:

- `draft_topk_logits` — positive logits (B, S, K)
- `neg_logits_on_draft_topk_ids` — negative logits on top-K (B, S, K)
- `draft_topk_token_ids` — token IDs for top-K (B, S, K)
- `target_token_id` — ground truth tokens (B, S)
- `block_position` — position in block (B, S)
- `absolute_position` — absolute position (B, S)
- `alpha_prev` — previous alpha values (B, S, 3)
- `acceptance_length` — baseline acceptance length (B,)

### 3. Train Alpha Model (A2C)

```bash
python -m alpha_model.train.train_AC \
  --h5-path alpha_model/alpha_dataset.h5 \
  --epochs 10 --batch-size 32 \
  --output-path checkpoints/alpha_policy.pt
```

### 4. Train Alpha Model (GRPO)

```bash
python -m alpha_model.train.train_GRPO \
  --data-path alpha_model/alpha_dataset.h5 \
  --epochs 10 --batch-size 32 \
  --output-dir checkpoints/grpo
```

---

## Negative Predictor Training (New, 32-logit)

The Negative Predictor is a two-phase training pipeline:

### Phase 1: Supervised Pre-training

**Goal**: Teach the model to predict negative logits such that the target token ranks #1 in top‑K after CD.

**Loss**: `CE_loss(CD_logits, target) + λ_bce · BCE_loss(predicted_neg, one_hot_target)`

Both losses are **positionally weighted** (exponential decay with γ=7.0) and have an **early-boost** mechanism (default: first 6 tokens × 2.0 weight).

```bash
python -m alpha_model.train.train_negative_predictor_phase1 \
  --data-path alpha_model/alpha_dataset.h5 \
  --epochs 20 --batch-size 64 \
  --lr 3e-4 --lr-scheduler cosine --lr-min 1e-6 \
  --early-boost-n 6 --early-boost-weight 2.0 \
  --output-dir checkpoints/phase1
```

**PBS submission (Gadi HPC):**

```bash
qsub script/train_negative_predictor_phase1.pbs
```

Override config via env vars:

```bash
qsub -v EPOCHS=50,LR=1e-4,LAMBDA_BCE=0.3,DENSE=1 script/train_negative_predictor_phase1.pbs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--dense` | — | Use Dense MLP variant (faster, no transformer) |
| `--predict-delta` | — | Predict delta added to original neg_logits |
| `--lambda-bce` | 0.2 | BCE auxiliary loss weight |
| `--gamma-decay` | 7.0 | Positional weight decay (higher = flatter) |
| `--early-boost-n` | 6 | Number of earliest positions to boost |
| `--early-boost-weight` | 2.0 | Extra weight multiplier for boosted positions |
| `--lr-scheduler` | `cosine` | `cosine`, `linear`, `plateau`, or `none` |
| `--resume` | — | Resume from checkpoint |

### Phase 2: Reinforcement Learning (A2C)

**Goal**: Fine-tune the policy via RL to maximize acceptance length and rank improvement.

**Action space**: 32 continuous logit values (Gaussian policy with learnable log_std).

**Reward components**:

| Component | Formula | Weight | Description |
|-----------|---------|--------|-------------|
| **r1** | `Σ Δrank · weight` | w1=0.1 | Target rank improvement vs baseline |
| **r2** | `Σ (rank==0) · 2 · weight` | w2=0.1 | Top-1 bonus |
| **r3** | `max(Δ,0) − λ·max(−Δ,0)` | w3=1.0 | Acceptance length improvement (asymmetric penalty) |
| **Total** | `w1·r1 + w2·r2 + w3·r3` | | |

```bash
python -m alpha_model.train.train_negative_predictor \
  --data-path alpha_model/alpha_dataset.h5 \
  --epochs 10 --batch-size 64 \
  --lr 3e-4 --lr-scheduler cosine --lr-min 1e-6 \
  --from-phase1 checkpoints/phase1/best.pt \
  --w1 0.1 --w2 0.1 --w3 1.0 \
  --entropy-coef 0.01 \
  --output-dir checkpoints/rl
```

**PBS submission (Gadi HPC):**

```bash
# Initialize from Phase 1 checkpoint
qsub -v FROM_PHASE1=checkpoints/phase1/best.pt script/train_negative_predictor_rl.pbs

# Resume interrupted training
qsub -v RESUME=checkpoints/rl/checkpoints/latest.pt script/train_negative_predictor_rl.pbs

# Custom config
qsub -v EPOCHS=30,LR=1e-4,W1=0.2,W2=0.2,W3=0.5 script/train_negative_predictor_rl.pbs
```

| Flag | Default | Description |
|------|---------|-------------|
| `--from-phase1` | — | Initialize actor from Phase 1 checkpoint |
| `--resume` | — | Resume full training (overrides `--from-phase1`) |
| `--entropy-coef` | 0.01 | Entropy bonus coefficient |
| `--lambda-r3` | 3.0 | Asymmetric penalty multiplier for shorter acceptance |
| `--predict-delta` | — | Delta mode (output adds to original neg_logits) |

### Resume & LR Scheduling

Both training scripts support:

- **`--resume <checkpoint.pt>`**: Restores actor/critic weights, optimizer state, scheduler state, and best-metric tracking. Epoch loop resumes from `ckpt['epoch'] + 1`.
- **`--lr-scheduler`**: Four options:
  - `cosine` (default): `CosineAnnealingLR` from `lr` → `lr_min` over total epochs
  - `linear`: Linear decay from `lr` → `lr_min`
  - `plateau`: `ReduceLROnPlateau` (factor=0.5, patience=3)
  - `none`: Fixed LR

---

## Model Architecture Details

### NegativeLogitPredictor (Transformer)

```
Input: pos_logits(B,S,32), neg_logits(B,S,32), block_pos(B,S), abs_pos(B,S)
  │
  ▼
StateFeatureExtractor → 237-dim feature vector per token
  │
  ▼
StateEncoder: Linear(237→128) + GELU + Linear(128→128) + LayerNorm
  │
  ▼
TransformerContext: 2-layer Transformer with RoPE + causal mask (4 heads)
  │
  ▼
output_norm (LayerNorm) → output_head (Linear 128→32)
  │
  ▼
predicted_neg_logits (B, S, 32)
```

| Variant | Architecture | Params | Speed |
|---------|-------------|--------|-------|
| `NegativeLogitPredictor` | Transformer (RoPE, 4 heads, 2 layers) | ~448K | Slower, context-aware |
| `NegativeLogitPredictor_Dense` | MLP (token proj → flatten → block MLP) | ~4.5M | Faster, stateless per token |

### NegativePredictorCritic

Shares `StateFeatureExtractor` + `StateEncoder` with actor, uses MLP value head:

```
state_feats → StateEncoder → Linear(128→128) + ReLU + Linear(128→1) → squeeze → (B,S)
```

### ContextualBanditAlpha (Alpha Model)

Same feature extraction backbone, but outputs 3 alpha values (one per rank bucket):

```
... → alpha_head (Linear 128→3) → tanh → max_alpha * tanh(logits) → (B,S,3)
```

### Feature Vector (237 dimensions)

| Group | Components | Dims |
|-------|-----------|------|
| **pos_logits** | logits, log_probs, probs | 3×32 = 96 |
| **pos_entropy** | `-Σ p·log p` | 1 |
| **pos_margin** | top1 − top2 probability | 1 |
| **pos_mass** | top-K mass for ks=[5,10,20] | 3 |
| **neg_logits** | logits, log_probs, probs | 3×32 = 96 |
| **neg_entropy** | `-Σ p·log p` | 1 |
| **neg_margin** | top1 − top2 probability | 1 |
| **neg_mass** | top-K mass for ks=[5,10,20] | 3 |
| **diff_logits** | pos − neg (element-wise) | 32 |
| **kl_div** | KL(positive ‖ negative) | 1 |
| **block_pos** | Position in block (0..S-1) | 1 |
| **abs_pos** | Absolute position | 1 |
| **Total** | | **237** |

---

## Offline Alpha Analysis

```bash
# Sweep alpha values on collected logits
python -m alpha_adjusting.alpha_analysis \
  --collect-dir alpha_model/collected_alpha_records/my_run/ \
  --alpha-min 0.0 --alpha-max 2.0 --alpha-steps 21 \
  --output-dir alpha_adjusting/results/
```

This tests a range of alpha values (e.g., 0.0 to 2.0 in 0.1 steps) on collected logits and reports per-sample and aggregate acceptance length, rank improvement, and optimal alpha distributions.

---

## Analysis Tools

```bash
# Token-level acceptance analysis
python -m analysis.tok_level_ana

# Compare CD vs vanilla DFlash
python -m analysis.CD_vs_Dflash

# Compare VCD vs DFlash
python -m analysis.VCD_vs_Dflash

# Bucket analysis from training logs
python analysis/alpha_bucket_analysis.py

# Deep bucket analysis (target vs predicted)
python analysis/alpha_bucket_deep_analysis.py

# Build token neighbor table for enhanced negative sampling
python analysis/build_token_neighbor_table.py

# CDv2 specific analyses
python -m analysis.CDv2_check_model_entropy
python -m analysis.CDv2_embed_distance_ana
python -m analysis.CDv2_target_survey
```

---

## Gadi HPC Cluster (PBS)

| Queue | GPU | Notes |
|-------|-----|-------|
| `gpuhopper` | H100/H200 | Default for experiments |
| `dgxa100` | A100 | Needs `ncpus>=16` |
| `gpuvolta` | V100 | torch 2.5.1+cu118, fp16 fallback |

**PBS Header** (always include):

```bash
#PBS -P jp09
#PBS -l storage=scratch/jp09+gdata/hn98
```

**CRITICAL**: The venv at `/scratch/jp09/dd9648/venvs/dflash_h100` is a symlink to `/g/data/hn98/mini3/bin/python3`. Without `+gdata/hn98` in the storage directive, compute nodes fall back to system Python 3.6.8 with no packages.

**Avoid `module load pytorch`** — it corrupts PYTHONPATH. Use the venv directly.

### Available PBS Scripts

| Script | Purpose |
|--------|---------|
| `script/train_alpha_h200.pbs` | Train alpha model on H200 |
| `script/train_negative_predictor_phase1.pbs` | Phase 1 supervised training for Negative Predictor |
| `script/train_negative_predictor_rl.pbs` | Phase 2 RL (A2C) training for Negative Predictor |
| `script/cd_alpha_model.pbs` | Evaluate alpha model scheme |
| `script/tok_level_ana_h100.pbs` | Token-level analysis on H100 |
| `script/target_tok_h100.pbs` | Target token analysis |
| `script/fix_target_token.pbs` | Fix target tokens in dataset |
| `script/h200_alpha_ana.pbs` | Alpha analysis on H200 |
| `script/merge_data.pbs` | Merge collected shards |
| `script/debug_a100_flash.pbs` | Debug on A100 with flash-attn |
| `script/debug_h100_flash.pbs` | Debug on H100 with flash-attn |
| `script/debug_v100_flash.pbs` | Debug on V100 with flash-attn |

---

## Dataset & Cache

- HF datasets are pre-cached at `/scratch/jp09/dd9648/.cache/huggingface`
- **Always set `OFFLINE_MODE=1`** after pre-caching
- When going online, set `HF_DATASETS_OFFLINE_MODE=0` explicitly (not auto)
- Supported datasets: gsm8k, metamath, magicoder, Math-Instruct, math500, aime24, aime25, alpaca
- The HDF5 dataset loader (`offline_dataset_h5.py`) loads all data into RAM at init (~23GB for full dataset). This is necessary because HDF5 chunking is inefficient for random access.
- **Automated collection**: Use `alpha_model/auto_submit_job_data_collect.py` to sequentially submit data collection jobs for multiple datasets.

---

## Key Conventions

- **Block size = 16**: Each pass generates 15 tokens (16 − 1 prefix). All sequence dimensions use S = 15.
- **Top-K = 32**: Only top-32 logits are stored during collection to save space. Models operate on K = 32.
- **Alpha buckets**: 3 values per position (rank buckets 0–10, 11–20, 21+), defined by thresholds `[10, 20]`.
- **Seeding**: Use `set_global_seed(seed, deterministic=True)` for reproducibility.
- **Device**: CUDA default, multi-GPU via `torchrun` with `RANK` / `LOCAL_RANK` env vars (see `distributed.py`).
- **Analysis scripts** must be run as modules (`python -m analysis.X`) or with `PYTHONPATH` set to repo root.
- **Training scripts** must be run as modules: `python -m alpha_model.train.train_negative_predictor_phase1`.
- **DFlash draft models** are on HF: `z-lab/Qwen3-*-DFlash-b16`, `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`, etc.
- **CD formula**: Always use `torch.log_softmax(..., dim=-1)`, never `torch.log(...)`.
- **Negative Predictor CD**: `log_softmax(pos) − log_softmax(neg_predicted)` with α = 1.0 fixed.

---

## Common Pitfalls

1. **CD logits formula**: Use `torch.log_softmax(..., dim=-1)` NOT `torch.log(...)` when computing contrastive logits in `model/utils.py`.
2. **HDF5 merge fields**: `absolute_position` and `block_position` are `(15,)` float32 arrays, not int32 scalars.
3. **HF offline mode**: `load_dataset` throws `OfflineModeIsEnabled` unless `HF_DATASETS_OFFLINE_MODE` is explicitly `0` during online use.
4. **PBS job limits**: If `qstat` shows "Server job limit reached for project jp09", the queue won't accept new jobs until running jobs finish — no resource change helps.
5. **Progress bar output**: Carriage-return progress bars in logs can be decoded via `tr '\r' '\n' < log`.
6. **HDF5 chunking**: The HDF5 files have poor chunking for random access (chunks = (63622, 1, 1)). Always use the full-RAM loader (`HDF5BanditDataset`) which loads everything into memory at init.
7. **`--resume` vs `--from-phase1`**: In Phase 2, `--resume` restores full training state (actor, critic, optimizers, scheduler). `--from-phase1` only loads actor weights (with `strict=False`). If both are specified, `--resume` takes precedence.
8. **Dense model max_seq_len**: `NegativeLogitPredictor_Dense` has a fixed `max_seq_len=15`. If your dataset has different sequence length, adjust the constructor or crop accordingly.
9. **LambdaLR resume**: When resuming a `LambdaLR` scheduler, `last_epoch` is correctly restored from checkpoint — no manual adjustment needed.
10. **Model loading with strict=False**: Phase 1 checkpoints only contain `actor_state_dict`. Phase 2 loads with `strict=False` to ignore missing critic keys.

---

## Key Dependencies

- `torch`, `transformers`, `accelerate` — core ML stack
- `sglang` — serving/inference (installed from GH PR #16818)
- `h5py` — HDF5 data format for alpha datasets
- `matplotlib` — analysis visualization
- `flash-attn` — optional speedup (v2.7.4 confirmed working)
- `loguru` — structured logging
- `tqdm` — progress bars
- `rich` — pretty console output

---

## Key Files

| File | Purpose |
|------|---------|
| `AGENTS.md` | This guide |
| `DFlash_README.md` | Upstream DFlash docs (install, model support, quick start) |
| `model/dflash.py` | `Qwen3DFlashDraftModel` — core draft architecture (modified attention) |
| `model/utils.py` | `apply_cd_logits()`, `apply_vcd_logits()`, `sample()`, `load_and_process_dataset()` |
| `scheme/` | 10 decoding variants (CD_v1–v4, VCD_v1–v3, CD_alpha_model, CD_negative_model, mask_50_redraft) |
| `alpha_model/__init__.py` | Public API — exports all model and training classes |
| `alpha_model/model/alpha_model.py` | `ContextualBanditAlpha`, `Critic`, `StateFeatureExtractor`, `TransformerContext` |
| `alpha_model/model/negative_predictor.py` | `NegativeLogitPredictor`, `GaussianNegativePolicy`, `NegativePredictorCritic` (+ Dense variants) |
| `alpha_model/train/train_AC.py` | A2C training loop for Alpha Model (3-bucket) |
| `alpha_model/train/train_GRPO.py` | GRPO training loop for Alpha Model |
| `alpha_model/train/train_negative_predictor_phase1.py` | Phase 1 supervised training for Negative Predictor |
| `alpha_model/train/train_negative_predictor.py` | Phase 2 RL (A2C) training for Negative Predictor |
| `alpha_model/train/alpha_simulate.py` | Reward computation, loss functions, acceptance simulation |
| `alpha_model/train/offline_dataset_h5.py` | `HDF5BanditDataset` — full-RAM HDF5 loader |
| `alpha_model/train/metrics.py` | Comprehensive metrics suite (rank, distribution, action, acceptance) |
| `alpha_model/train/logging_utils.py` | `TrainingLogger` — checkpointing, CSV, logging |
| `alpha_model/data_collecting.py` | Per-block record collector (generates .pt shards) |
| `alpha_model/merge_pt_shards_to_hdf5.py` | Merge .pt shards → HDF5 |
| `alpha_model/model/NEGATIVE_PREDICTOR_REPORT.md` | Full architecture report (Vietnamese) |
| `distributed.py` | `init()`, `rank()`, `gather()` for multi-GPU |
| `analysis/tok_level_ana.py` | Token-level acceptance analysis |
| `analysis/CD_vs_Dflash.py` | CD variants vs vanilla DFlash benchmark |
| `analysis/alpha_bucket_analysis.py` | Bucket statistics from training logs |
| `alpha_adjusting/alpha_analysis.py` | Offline alpha value sweep |
| `script/run_h200_alpha_collect.sh` | H200 data collection bash launcher |

---

## Codex Working Memory

Use this section as the cross-session working summary for future Codex work.

- The research goal is to improve DFlash speculative decoding acceptance by changing how the negative draft distribution is used during contrastive decoding.
- There are two learned control surfaces:
  - **Alpha model**: learns per-position bucket alphas, typically 3 buckets over top-32 ranks. Runtime path is `scheme/CD_alpha_model.py`; model code is `alpha_model/model/alpha_model.py`; RL simulation is in `alpha_model/train/alpha_simulate.py`.
  - **Negative predictor**: learns direct top-32 negative logits with fixed alpha=1.0. Model code is `alpha_model/model/negative_predictor.py`; supervised phase is `alpha_model/train/train_negative_predictor_phase1.py`; RL phase is `alpha_model/train/train_negative_predictor.py`.
- The canonical CD formula in this repo is `log_softmax(pos) - alpha * log_softmax(neg)`. For the negative predictor it is `log_softmax(pos_topk) - log_softmax(predicted_neg_topk)`. Avoid raw `torch.log(softmax)` and avoid logit-space subtraction unless working on VCD.
- Data collection stores positive top-k logits, negative logits gathered on the same positive top-k token IDs, top-k token IDs, target token IDs, normalized block/absolute positions, previous alphas, and baseline acceptance length. Training reads this through `HDF5BanditDataset`, which intentionally loads the whole HDF5 into RAM.
- Runtime inference builds positive/negative draft logits by pairing normal and corrupted draft inputs in one draft forward pass. Negative context is usually random first-token replacement plus hidden-state masking (`mask_zero`) or hidden-state shuffling (`shuffle_tokens`).
- Candidate filtering is a major behavior boundary: CD schemes usually mask to positive-model candidates via beta threshold or top-64 probability mask. If a target is outside top-32/top-k or filtered out by the candidate mask, learned alpha/logit shaping cannot recover it.
- `scheme/CD_alpha_model.py` is the more current learned-runtime path: it supports checkpoint config inference, dense vs transformer alpha models, deterministic seeding, SDPA fallback, distributed gathering, and full-vocab alpha expansion from top-k buckets.
- `scheme/CD_negative_model.py` should be treated carefully before benchmarking: it appears less aligned with the current `NegativeLogitPredictor.forward(pos, neg, block_pos, abs_pos)` API, because its generation path passes an extra `neg_logits_prev` argument and includes an older target verification/cache path. Verify and modernize it against `CD_alpha_model.py` before relying on results.
- For HPC/PBS, preserve `#PBS -P jp09` and `#PBS -l storage=scratch/jp09+gdata/hn98`; use the venv directly and avoid `module load pytorch`.
- The user often works on `scheme/CD_alpha_model.py`, `scheme/CD_negative_model.py`, `alpha_model/train/alpha_simulate.py`, and PBS launch scripts. Do not overwrite local changes; check `git status --short` before edits.
