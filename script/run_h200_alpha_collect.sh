#!/usr/bin/env bash
set -euo pipefail

# Submit a small/large alpha data-collection job to H200 queue on Gadi.
# Usage:
#   bash script/run_h200_alpha_collect.sh
#   DATASET=gsm8k WALLTIME=01:00:00 bash script/run_h200_alpha_collect.sh

# Prefer PBS working directory when submitted via qsub, otherwise resolve from script path.
if [[ -n "${PBS_O_WORKDIR:-}" && -d "${PBS_O_WORKDIR}" ]]; then
  ROOT_DIR="${PBS_O_WORKDIR}"
else
  ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
cd "$ROOT_DIR"
mkdir -p logs

PROJECT="${PROJECT:-hn98}"
QUEUE="${QUEUE:-gpuhopper}"
NGPUS="${NGPUS:-1}"
NCPUS="${NCPUS:-12}"
MEM="${MEM:-48GB}"
WALLTIME="${WALLTIME:-00:30:00}"
STORAGE="${STORAGE:-scratch/jp09+gdata/hn98}"

VENV_PATH="${VENV_PATH:-/scratch/jp09/dd9648/venvs/dflash_h100}"
HF_HOME="${HF_HOME:-/scratch/jp09/dd9648/.cache/huggingface}"
OFFLINE_MODE="${OFFLINE_MODE:-1}"

MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-4B}"
DRAFT_NAME_OR_PATH="${DRAFT_NAME_OR_PATH:-z-lab/Qwen3-4B-DFlash-b16}"
DATASET="${DATASET:-metamath}"
BLOCK_SIZE="${BLOCK_SIZE:-16}"
COLLECTOR_TOPK="${COLLECTOR_TOPK:-32}"
COLLECTOR_CHUNK_SIZE="${COLLECTOR_CHUNK_SIZE:-6144}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-2048}"
SEED="${SEED:-0}"
START_INDEX="${START_INDEX:-0}"
NUM_INSTANCES="${NUM_INSTANCES:-200}"

RUN_TAG="${RUN_TAG:-h200_collect_${DATASET}_${START_INDEX}_${NUM_INSTANCES}}"
COLLECTOR_OUTPUT_DIR="${COLLECTOR_OUTPUT_DIR:-$ROOT_DIR/alpha_model/collected_alpha_records/$RUN_TAG}"
mkdir -p "$COLLECTOR_OUTPUT_DIR"

echo "Submitting H200 alpha collection job with settings:"
echo "  dataset=$DATASET max_new_tokens=$MAX_NEW_TOKENS"
echo "  output_dir=$COLLECTOR_OUTPUT_DIR"

JOB_ID="$({
qsub <<EOF
#!/bin/bash
#PBS -P ${PROJECT}
#PBS -q ${QUEUE}
#PBS -l ngpus=${NGPUS}
#PBS -l ncpus=${NCPUS}
#PBS -l mem=${MEM}
#PBS -l walltime=${WALLTIME}
#PBS -l storage=${STORAGE}
#PBS -j oe

set -euo pipefail
cd "${ROOT_DIR}"
mkdir -p logs

LOG_FILE="logs/h200_alpha_collect_\${PBS_JOBID}.log"
exec > >(tee -a "\$LOG_FILE") 2>&1

echo "[INFO] host=\$(hostname) jobid=\${PBS_JOBID}"
source "${VENV_PATH}/bin/activate"
unset PYTHONPATH
export HF_HOME="${HF_HOME}"
if [[ "${OFFLINE_MODE}" == "1" ]]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
  export HF_DATASETS_OFFLINE=1
  export HF_DATASETS_OFFLINE_MODE=1
  echo "[INFO] Offline mode enabled (HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1, HF_DATASETS_OFFLINE=1, HF_DATASETS_OFFLINE_MODE=1)"
else
  # Online by default so uncached datasets (e.g. math_instruct/metamath/magicoder) can be fetched.
  unset HF_HUB_OFFLINE
  unset TRANSFORMERS_OFFLINE
  unset HF_DATASETS_OFFLINE
  export HF_DATASETS_OFFLINE_MODE=0
  echo "[INFO] Offline mode disabled (Hub access enabled; HF_DATASETS_OFFLINE_MODE=0)"
fi
export CUDA_LAUNCH_BLOCKING=0
export PYTHONFAULTHANDLER=1
mkdir -p "\$HF_HOME"
echo "[INFO] HF_DATASETS_OFFLINE_MODE=${HF_DATASETS_OFFLINE_MODE:-unset} HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-unset} HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-unset}"

python - <<'PY'
import torch
print('torch', torch.__version__, 'cuda', torch.version.cuda, 'is_cuda_available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device_count', torch.cuda.device_count(), 'device_name', torch.cuda.get_device_name(0))
PY

mkdir -p "${COLLECTOR_OUTPUT_DIR}"

python -m alpha_model.data_collecting \
  --model-name-or-path "${MODEL_NAME_OR_PATH}" \
  --draft-name-or-path "${DRAFT_NAME_OR_PATH}" \
  --dataset "${DATASET}" \
  --block-size "${BLOCK_SIZE}" \
  --collect-dataset \
  --collector-output-dir "${COLLECTOR_OUTPUT_DIR}" \
  --collector-topk "${COLLECTOR_TOPK}" \
  --collector-chunk-size "${COLLECTOR_CHUNK_SIZE}" \
  --max-new-tokens "${MAX_NEW_TOKENS}" \
  --start-index "${START_INDEX}" \
  --num-instances "${NUM_INSTANCES}" \
  --seed "${SEED}" \

echo "[INFO] Collection completed. output_dir=${COLLECTOR_OUTPUT_DIR}"
EOF
} | tr -d '[:space:]')"

echo "Submitted job: $JOB_ID"
echo "Monitor: qstat -u ${USER}"
echo "Tail log: tail -f logs/h200_alpha_collect_${JOB_ID}.log"
