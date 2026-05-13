#!/usr/bin/env bash
set -euo pipefail

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export TOKENIZERS_PARALLELISM=false
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-600}"
export VLLM_ENGINE_ITERATION_TIMEOUT_S="${VLLM_ENGINE_ITERATION_TIMEOUT_S:-600}"
export VLLM_ENGINE_TIMEOUT_S="${VLLM_ENGINE_TIMEOUT_S:-600}"

# vLLM core config
export MODEL_PATH="${MODEL_PATH:-/models/ms-mirrors/Qwen3.5-27B-FP8/release.20260226}"
export SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-5-27b-vllm-tq}"
export TP_SIZE="${TP_SIZE:-2}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
export MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
export ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}"
export ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}"
export TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
export DTYPE="${DTYPE:-bfloat16}"

# TurboQuant knobs
export TQ_ENABLE="${TQ_ENABLE:-1}"
export TQ_KEY_BITS="${TQ_KEY_BITS:-3}"
export TQ_VALUE_BITS="${TQ_VALUE_BITS:-4}"
export TQ_BUFFER_SIZE="${TQ_BUFFER_SIZE:-64}"

# Optional speculative config for vLLM (leave empty to disable)
export SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}"

PORT="${PORT:-8321}"
HOST="${HOST:-0.0.0.0}"
PATCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/turboquant_vllm_patch"
export PYTHONPATH="${PATCH_DIR}:${PYTHONPATH:-}"

prefix_cache_flag="--no-enable-prefix-caching"
if [[ "${ENABLE_PREFIX_CACHING}" == "1" ]]; then
  prefix_cache_flag="--enable-prefix-caching"
fi

chunked_prefill_flag="--no-enable-chunked-prefill"
if [[ "${ENABLE_CHUNKED_PREFILL}" == "1" ]]; then
  chunked_prefill_flag="--enable-chunked-prefill"
fi

trust_remote_code_flag=""
if [[ "${TRUST_REMOTE_CODE}" == "1" ]]; then
  trust_remote_code_flag="--trust-remote-code"
fi

echo "[start] model=${MODEL_PATH}"
echo "[start] served_name=${SERVED_MODEL_NAME}"
echo "[start] cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "[start] turboquant key_bits=${TQ_KEY_BITS}, value_bits=${TQ_VALUE_BITS}, buffer=${TQ_BUFFER_SIZE}"
echo "[start] turboquant patch=${PATCH_DIR}"
echo "[start] speculative=${SPECULATIVE_CONFIG:-<disabled>}"
echo "[start] listen=${HOST}:${PORT}"

cmd=(
  vllm serve "${MODEL_PATH}"
  -tp "${TP_SIZE}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-model-len "${MAX_MODEL_LEN}"
  "${prefix_cache_flag}"
  "${chunked_prefill_flag}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --dtype "${DTYPE}"
  --host "${HOST}"
  --port "${PORT}"
)

if [[ -n "${trust_remote_code_flag}" ]]; then
  cmd+=("${trust_remote_code_flag}")
fi

if [[ -n "${SPECULATIVE_CONFIG}" ]]; then
  cmd+=(--speculative-config "${SPECULATIVE_CONFIG}")
fi

echo "[start] command:"
printf ' %q' "${cmd[@]}"
echo

exec "${cmd[@]}"
