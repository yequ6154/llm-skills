#!/bin/bash

# ============================================
# 在宿主机上执行：安装依赖 + patch transformers
# 用法: bash setup_and_patch.sh <容器名>
# ============================================

set -e

CONTAINER=${1:?"Usage: bash setup_and_patch.sh <container_name>"}

echo "=== Step 1: 安装 llmcompressor + compressed-tensors (--no-deps, 避免污染系统环境) ==="
docker exec -u xd "${CONTAINER}" pip3 install \
  llmcompressor==0.10.0.1 \
  compressed-tensors==0.14.0.1 \
  --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple

echo ""
echo "=== Step 2: Patch transformers 源码 (以 root 执行) ==="
docker exec -u root "${CONTAINER}" python3 /home/lei.xiong/project/quant/qwen25vl-fp8-quant-template/patch_transformers.py

echo ""
echo "=== Done! 现在可以进入容器运行量化: ==="
echo "  docker exec -it -u xd ${CONTAINER} bash"
echo "  python3 /home/lei.xiong/project/quant/qwen25vl-fp8-quant-template/quant_fp8.py"
