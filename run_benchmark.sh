NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29600}
EXTRA_BENCHMARK_ARGS=${EXTRA_BENCHMARK_ARGS:-}

mkdir -p logs

if [[ -n "${TASKS_OVERRIDE:-}" ]]; then
  TASKS=("${TASKS_OVERRIDE}")
else
  TASKS=(
    "gsm8k:128"
  )
fi

for task in "${TASKS[@]}"; do
  IFS=':' read -r DATASET_NAME MAX_SAMPLES <<< "$task"

  echo "========================================================"
  echo "Running Benchmark: $DATASET_NAME with $MAX_SAMPLES samples"
  echo "========================================================"

  torchrun \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_port=${MASTER_PORT} \
    -m alpha_adjusting.data_colecting \
    --dataset "$DATASET_NAME" \
    --max-samples "$MAX_SAMPLES" \
    --model-name-or-path Qwen/Qwen3-4B \
    --draft-name-or-path z-lab/Qwen3-4B-DFlash-b16 \
    --max-new-tokens 2048 \
    --temperature 0.0 \
    ${EXTRA_BENCHMARK_ARGS} \
    2>&1 | tee "logs/${DATASET_NAME}.log"

done