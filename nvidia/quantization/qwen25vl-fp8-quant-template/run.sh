#!/bin/bash

# ============================================================================
# Qwen2.5-VL FP8 一键量化脚本
#
# 用法:
#   bash run.sh <模型路径> <量化输出路径> [GPU编号] [用户名]
#
# 示例:
#   # 7B 模型, 单卡, 以当前用户身份运行
#   bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8 4
#
#   # 大模型, 2卡, 指定用户
#   bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8 4,5 xd
#
#   # 不指定 GPU 和用户, 默认 GPU 0、当前用户
#   bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8
# ============================================================================

set -e

MODEL_ID=${1:?"用法: bash run.sh <模型路径> <量化输出路径> [GPU编号] [用户名]"}
SAVE_DIR=${2:?"用法: bash run.sh <模型路径> <量化输出路径> [GPU编号] [用户名]"}
GPU_IDS=${3:-"0"}
RUN_USER=${4:-"$(whoami)"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_NAME="quant-fp8-$(date +%s)"
IMAGE="registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.10.2"

MODEL_REAL=$(realpath "${MODEL_ID}" 2>/dev/null || echo "${MODEL_ID}")
SAVE_REAL=$(realpath "$(dirname "${SAVE_DIR}")" 2>/dev/null || echo "$(dirname "${SAVE_DIR}")")

echo "============================================"
echo "  Qwen2.5-VL FP8 一键量化"
echo "============================================"
echo "  模型路径:   ${MODEL_ID}"
echo "  输出路径:   ${SAVE_DIR}"
echo "  GPU:        ${GPU_IDS}"
echo "  运行用户:   ${RUN_USER}"
echo "  容器名:     ${CONTAINER_NAME}"
echo "============================================"
echo ""

# ---- Step 1: 启动容器 ----
echo "[1/4] 启动容器..."
docker run -dit --rm --init \
  --name "${CONTAINER_NAME}" \
  --gpus=all \
  --net=host \
  --ipc=host \
  -v /home:/home \
  -v "$(dirname "${MODEL_REAL}")":"$(dirname "${MODEL_REAL}")" \
  -v "$(dirname "${SAVE_REAL}")":"$(dirname "${SAVE_REAL}")" \
  --entrypoint /bin/bash \
  "${IMAGE}" \
  -c "tail -f /dev/null" > /dev/null

cleanup() {
  echo ""
  echo "[cleanup] 停止并删除容器 ${CONTAINER_NAME}..."
  docker rm -f "${CONTAINER_NAME}" > /dev/null 2>&1 || true
}
trap cleanup EXIT

# ---- Step 2: 安装依赖（root） ----
echo "[2/4] 安装 llmcompressor + compressed-tensors..."
docker exec "${CONTAINER_NAME}" pip3 install \
  llmcompressor==0.10.0.1 \
  compressed-tensors==0.14.0.1 \
  loguru==0.7.2 \
  datasets==3.4.0 \
  pandas==2.2.3 \
  pytz==2024.1 \
  pyarrow==19.0.1 \
  multiprocess==0.70.15 \
  xxhash==3.5.0 \
  --no-deps \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  -q

# ---- Step 2.1: 创建用户 xd（root）----
docker exec "${CONTAINER_NAME}" groupadd -g 1013 xd 
docker exec "${CONTAINER_NAME}" useradd -u 1012 -g 1013 -m -s /bin/bash xd

# ---- Step 3: Patch transformers（root） ----
echo "[3/4] Patch transformers 源码..."
docker exec "${CONTAINER_NAME}" python3 "${SCRIPT_DIR}/patch_transformers.py"

# ---- Step 4: 运行量化（指定用户） ----
echo "[4/5] 以用户 ${RUN_USER} 开始量化..."
echo ""
docker exec -u "${RUN_USER}" "${CONTAINER_NAME}" python3 -c "
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '${GPU_IDS}'

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

MODEL_ID = '${MODEL_ID}'
SAVE_DIR = '${SAVE_DIR}'

print(f'Loading model from {MODEL_ID} ...')
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, trust_remote_code=False, torch_dtype='auto', device_map='auto'
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)

recipe = QuantizationModifier(
    targets='Linear',
    scheme='FP8_DYNAMIC',
    ignore=[
        're:lm_head.*', 're:model.lm_head.*', 're:.*lm_head',
        're:visual.*', 're:model.visual.*', 're:.*visual',
    ],
)

oneshot(model=model, recipe=recipe)

model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print(f'\nDone! Quantized model saved to: {SAVE_DIR}')
"

# ---- Step 5: 清理 config.json 中的不兼容字段 ----
echo "[5/5] 清理 config.json 中的 scale_dtype / zp_dtype 字段..."
docker exec -u "${RUN_USER}" "${CONTAINER_NAME}" python3 -c "
import json, os

config_path = os.path.join('${SAVE_DIR}', 'config.json')
with open(config_path) as f:
    config = json.load(f)

removed = []
def clean_fields(obj):
    if isinstance(obj, dict):
        for key in ('scale_dtype', 'zp_dtype'):
            if key in obj:
                del obj[key]
                removed.append(key)
        for v in obj.values():
            clean_fields(v)
    elif isinstance(obj, list):
        for v in obj:
            clean_fields(v)

clean_fields(config)

with open(config_path, 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

if removed:
    print(f'Removed fields: {removed}')
else:
    print('No scale_dtype/zp_dtype fields found, config is clean.')
"

echo ""
echo "============================================"
echo "  量化完成!"
echo "  输出路径: ${SAVE_DIR}"
echo "============================================"
