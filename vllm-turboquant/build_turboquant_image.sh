#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_NAME="${IMAGE_NAME:-registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.20.0__turboquant}"
DOCKERFILE_PATH="${DOCKERFILE_PATH:-${SCRIPT_DIR}/Dockerfile.vllm-turboquant}"
BASE_IMAGE="${BASE_IMAGE:-registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.20.0}"
TURBOQUANT_GIT_URL="${TURBOQUANT_GIT_URL:-https://github.com/0xSero/turboquant.git}"
TURBOQUANT_GIT_REF="${TURBOQUANT_GIT_REF:-main}"

echo "[build] image=${IMAGE_NAME}"
echo "[build] dockerfile=${DOCKERFILE_PATH}"
echo "[build] base_image=${BASE_IMAGE}"
echo "[build] turboquant=${TURBOQUANT_GIT_URL}@${TURBOQUANT_GIT_REF}"

docker build \
  -t "${IMAGE_NAME}" \
  -f "${DOCKERFILE_PATH}" \
  --build-arg BASE_IMAGE="${BASE_IMAGE}" \
  --build-arg TURBOQUANT_GIT_URL="${TURBOQUANT_GIT_URL}" \
  --build-arg TURBOQUANT_GIT_REF="${TURBOQUANT_GIT_REF}" \
  "${SCRIPT_DIR}"

echo "[build] done"
