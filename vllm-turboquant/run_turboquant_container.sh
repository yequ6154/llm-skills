#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.20.0__turboquant}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3-5-27b-vllm-tq}"
GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
HOST_PORT="${HOST_PORT:-8321}"
START_SCRIPT_PATH="${START_SCRIPT_PATH:-${SCRIPT_DIR}/start_turboquant_openai.sh}"

MODEL_MOUNT="${MODEL_MOUNT:-/mnt/seaweedfs/data/buckets/cybertron/llmops-shared-pv/cached_models:/models}"
WORK_MOUNT="${WORK_MOUNT:-/home/lei.xiong:/home/lei.xiong}"

if [[ ! -f "${START_SCRIPT_PATH}" ]]; then
  echo "[run] start script not found: ${START_SCRIPT_PATH}" >&2
  exit 1
fi

docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true

echo "[run] image=${IMAGE_NAME}"
echo "[run] container=${CONTAINER_NAME}"
echo "[run] gpus=${GPU_DEVICES}"
echo "[run] port=${HOST_PORT}"
echo "[run] start_script=${START_SCRIPT_PATH}"

run_container() {
  docker run -d --rm \
    --name "${CONTAINER_NAME}" \
    "$@" \
    --network host \
    --ipc host \
    --entrypoint /bin/bash \
    -e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}" \
    -e TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}" \
    -e PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    -e PORT="${HOST_PORT}" \
    -e HOST="${HOST:-0.0.0.0}" \
    -e MODEL_PATH="${MODEL_PATH:-/models/ms-mirrors/Qwen3.5-27B-FP8/release.20260226}" \
    -e SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen3-5-27b-vllm-tq}" \
    -e TP_SIZE="${TP_SIZE:-2}" \
    -e GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}" \
    -e MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}" \
    -e MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}" \
    -e ENABLE_PREFIX_CACHING="${ENABLE_PREFIX_CACHING:-0}" \
    -e ENABLE_CHUNKED_PREFILL="${ENABLE_CHUNKED_PREFILL:-1}" \
    -e TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}" \
    -e DTYPE="${DTYPE:-bfloat16}" \
    -e TQ_ENABLE="${TQ_ENABLE:-1}" \
    -e TQ_KEY_BITS="${TQ_KEY_BITS:-3}" \
    -e TQ_VALUE_BITS="${TQ_VALUE_BITS:-4}" \
    -e TQ_BUFFER_SIZE="${TQ_BUFFER_SIZE:-128}" \
    -e SPECULATIVE_CONFIG="${SPECULATIVE_CONFIG:-}" \
    -v "${MODEL_MOUNT}" \
    -v "${WORK_MOUNT}" \
    "${IMAGE_NAME}" -lc "exec \"${START_SCRIPT_PATH}\""
}

run_out=""
if run_out="$(run_container --gpus "device=${GPU_DEVICES}" 2>&1)"; then
  echo "${run_out}"
else
  echo "${run_out}" >&2
  if [[ "${run_out}" == *"cannot set both Count and DeviceIDs on device request."* ]]; then
    echo "[run] detected Docker GPU request conflict, retry with --runtime=nvidia"
    docker rm -f "${CONTAINER_NAME}" >/dev/null 2>&1 || true
    run_out="$(run_container \
      --runtime nvidia \
      -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICES}" \
      -e NVIDIA_DRIVER_CAPABILITIES="${NVIDIA_DRIVER_CAPABILITIES:-compute,utility}" \
      2>&1)"
    echo "${run_out}"
  else
    exit 1
  fi
fi

echo "[run] started. logs:"
echo "docker logs -f ${CONTAINER_NAME}"
