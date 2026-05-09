# Guided-DFlash Agent Guide

## Project Overview

**Guided-DFlash** extends [DFlash](https://github.com/z-lab/dflash) — a block diffusion speculative decoding framework — with dynamic alpha optimization via reinforcement learning. The goal is to learn per-token alpha coefficients that improve the contrastive decoding acceptance rate, accelerating LLM inference.

Core repo: `https://github.com/tranphuc15122004/Guided-DFlash`
Upstream paper: [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)

## Quick Start

```bash
# Environment
conda create -n dflash python=3.11
conda activate dflash
pip install uv
uv pip install -r requirements.txt
uv pip install flash-attn --no-build-isolation   # optional
```

## Architecture

```
model/          — Core DFlash draft model + CD/VCD logit functions
scheme/         — 8 contrastive decoding variants (CD_v1..v4, VCD_v1..v3, mask_50_redraft)
alpha_model/    — Alpha learning pipeline: data collection, model, training
alpha_adjusting/— Offline alpha sweep analysis
analysis/       — Benchmarking, evaluation, and visualization tools
script/         — PBS job submission scripts for Gadi HPC
test/           — Smoke tests for inference, datasets, HDF5 loading
```

## Running Benchmarks

```bash
# Transformers backend (single GPU)
bash run_benchmark.sh

# SGLang backend (multi-concurrency)
export SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1
python benchmark_sglang.py --target-model Qwen/Qwen3-8B --draft-model z-lab/Qwen3-8B-DFlash-b16

# Analysis (run as module from repo root)
python -m analysis.tok_level_ana
python -m analysis.CD_vs_Dflash
```

## Alpha Data Collection & Training

```bash
# 1. Collect per-block records from generation
python -m alpha_model.data_collecting \
  --model-name-or-path Qwen/Qwen3-4B \
  --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
  --dataset gsm8k \
  --collector-output-dir alpha_model/collected_alpha_records/my_run/

# 2. Merge .pt shards into HDF5
python alpha_model/merge_pt_shards_to_hdf5.py

# 3. Train alpha policy (actor-critic RL)
python -m alpha_model.train.train_AC \
  --h5-path alpha_model/alpha_dataset.h5 \
  --epochs 10 --batch-size 32 \
  --output-path checkpoints/alpha_policy.pt
```

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

## Dataset & Cache

- HF datasets are pre-cached at `/scratch/jp09/dd9648/.cache/huggingface`
- **Always set `OFFLINE_MODE=1`** after pre-caching
- When going online, set `HF_DATASETS_OFFLINE_MODE=0` explicitly (not auto)
- Supported datasets: gsm8k, metamath, magicoder, Math-Instruct

## Key Conventions

- **Block size = 16**: Each pass generates 15 tokens (16 − 1 prefix)
- **Top-K = 32**: Only top-32 logits are stored during collection to save space
- **Alpha buckets**: 3 values per position (rank buckets 0–10, 11–20, 21+)
- **Seeding**: Use `set_global_seed(seed, deterministic=True)` for reproducibility
- **Device**: CUDA default, multi-GPU via `torchrun` with `RANK` env vars
- **Analysis scripts** must be run as modules (`python -m analysis.X`) or with `PYTHONPATH` set to repo root
- **DFlash draft models** are on HF: `z-lab/Qwen3-*-DFlash-b16`, `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat`, etc.

## Common Pitfalls

1. **CD logits formula**: Use `torch.log_softmax(..., dim=-1)` NOT `torch.log(...)` when computing contrastive logits in `model/utils.py`.
2. **HDF5 merge fields**: `absolute_position` and `block_position` are `(15,)` float32 arrays, not int32 scalars.
3. **HF offline mode**: `load_dataset` throws `OfflineModeIsEnabled` unless `HF_DATASETS_OFFLINE_MODE` is explicitly `0` during online use.
4. **PBS job limits**: If `qstat` shows "Server job limit reached for project jp09", the queue won't accept new jobs until running jobs finish — no resource change helps.
5. **Progress bar output**: Carriage-return progress bars in logs can be decoded via `tr '\r' '\n' < log`.

## Key Dependencies

- `torch`, `transformers`, `accelerate` — core ML stack
- `sglang` — serving/inference (installed from GH PR #16818)
- `h5py` — HDF5 data format for alpha datasets
- `matplotlib` — analysis visualization
- `flash-attn` — optional speedup (v2.7.4 confirmed working)

## Key Files

| File | Purpose |
|------|---------|
| `DFlash_README.md` | Upstream DFlash docs (install, model support, quick start) |
| `model/dflash.py` | `Qwen3DFlashDraftModel` — core draft architecture |
| `model/utils.py` | `apply_cd_logits()`, `apply_vcd_logits()`, sampling helpers |
| `scheme/` | 8 decoding variants (CD_v1–v4, VCD_v1–v3, mask_50_redraft) |
| `alpha_model/model/alpha_model.py` | `ContextualBanditAlpha` — learned alpha policy |
| `alpha_model/train/train_AC.py` | Actor-critic training loop |
| `distributed.py` | `init()`, `rank()`, `gather()` utilities for multi-GPU |
