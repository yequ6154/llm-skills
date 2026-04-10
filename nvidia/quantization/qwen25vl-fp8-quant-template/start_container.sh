#!/bin/bash

# ============================================
# 启动量化容器
# 根据实际情况修改以下变量
# ============================================

CONTAINER_NAME="quant-qwen25vl"
GPUS='"device=0,1,2,3,4,5,6,7"'
MODEL_DIR="/mnt/seaweedfs/data/buckets/cybertron/llmops-shared-pv/cached_models/vision-llm"
IMAGE="registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.10.2"

docker run -dit --rm --init \
  --name "${CONTAINER_NAME}" \
  --gpus="${GPUS}" \
  --net=host \
  --ipc=host \
  -v /home/lei.xiong:/home/lei.xiong \
  -v "${MODEL_DIR}":/models \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "tail -f /dev/null"

echo "Container ${CONTAINER_NAME} started."
echo "Next step: bash setup_and_patch.sh ${CONTAINER_NAME}"
