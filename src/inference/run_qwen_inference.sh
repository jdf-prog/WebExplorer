#!/usr/bin/env bash

set -euo pipefail

INFERENCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${INFERENCE_DIR}/../../.." && pwd)"

export DATASET="${DATASET:-browsecomp_validation_100}"
export QWEN_MODEL_NAME="${QWEN_MODEL_NAME:-models/Qwen3.6-35B-A3B}"
export QWEN_VLLM_PORT="${QWEN_VLLM_PORT:-8010}"
export QWEN_API_KEY="${QWEN_API_KEY:-EMPTY}"
export QWEN_API_BASE="${QWEN_API_BASE:-http://127.0.0.1:${QWEN_VLLM_PORT}/v1}"
export QWEN_OUTPUT_PATH="${QWEN_OUTPUT_PATH:-${REPO_ROOT}/outputs/qwen_function_call}"
export TEMPERATURE="${TEMPERATURE:-1.0}"
export TOP_P="${TOP_P:-0.95}"
export QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-${TEMPERATURE}}"
export QWEN_TOP_P="${QWEN_TOP_P:-${TOP_P}}"
export QWEN_TOP_K="${QWEN_TOP_K:-20}"
export QWEN_MIN_P="${QWEN_MIN_P:-0.0}"
export QWEN_PRESENCE_PENALTY="${QWEN_PRESENCE_PENALTY:-1.5}"
export QWEN_REPETITION_PENALTY="${QWEN_REPETITION_PENALTY:-1.0}"
export MAX_WORKERS="${MAX_WORKERS:-8}"
export ROLLOUT_COUNT="${ROLLOUT_COUNT:-1}"
export JUDGE_ENGINE="${JUDGE_ENGINE:-openai}"

mkdir -p "${QWEN_OUTPUT_PATH}"

cd "${INFERENCE_DIR}"

python -u run_multi_qwen.py \
  --dataset "${DATASET}" \
  --output "${QWEN_OUTPUT_PATH}" \
  --max_workers "${MAX_WORKERS}" \
  --model "${QWEN_MODEL_NAME}" \
  --temperature "${TEMPERATURE}" \
  --top_p "${TOP_P}" \
  --total_splits "${WORLD_SIZE:-1}" \
  --worker_split "$((${RANK:-0} + 1))" \
  --roll_out_count "${ROLLOUT_COUNT}" \
  --auto_judge \
  --judge_engine "${JUDGE_ENGINE}"
