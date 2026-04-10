"""
Qwen2.5-VL FP8 Dynamic 量化脚本

前置条件:
  1. 使用 vllm-openai:v0.10.2 镜像启动容器
  2. 已执行 setup_and_patch.sh 安装依赖并 patch transformers
  3. 使用 device_map='cuda:0' 加载模型（必须单卡，不能用 'auto'）
"""

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

import os

# ============================================
# 根据实际情况修改以下三个变量
# ============================================
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
MODEL_ID = "/models/BR-VL-NPL-Classification-CreditCard-V2.0-20260409/release-v2"
SAVE_DIR = "/models/BR-VL-NPL-Classification-CreditCard-V2.0-20260409-FP8-Dynamic"

# 加载模型（必须用 cuda:0，不能用 auto，否则会触发 offload bug）
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID, trust_remote_code=False, torch_dtype="auto", device_map="cuda:0"
)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)

# FP8 动态量化配置：只量化 Linear 层，跳过 visual encoder 和 lm_head
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=[
        "re:lm_head.*",
        "re:model.lm_head.*",
        "re:.*lm_head",
        "re:visual.*",
        "re:model.visual.*",
        "re:.*visual",
    ],
)

# 执行量化
oneshot(model=model, recipe=recipe)

# 保存量化后的模型
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)

print(f"\nDone! Quantized model saved to: {SAVE_DIR}")
