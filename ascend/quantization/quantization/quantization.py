from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, Qwen2_5_VLForConditionalGeneration
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

MODEL_ID = "/data2/huggingface/bairong-inc/BR-Table-7B-V1.0-20250923/"
#MODEL_ID = "/root/.cache/huggingface/bairong-inc/BR-VL-FinDoc-V1.1/"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(MODEL_ID, trust_remote_code=False, torch_dtype='auto', device_map='auto')
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=False)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8_DYNAMIC",
    ignore=["re:lm_head.*", "re:model.lm_head.*", "re:.*lm_head", "re:visual.*", "re:model.visual.*", "re:.*visual"],
)

# Apply quantization and save to disk in compressed-tensors format.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = "/data2/huggingface/bairong-inc/BR-Table-7B-V1.0-20250923-FP8/"
#SAVE_DIR = "/root/.cache/huggingface/bairong-inc/BR-VL-FinDoc-V1.1-FP8-Dynamic/"
model.save_pretrained(SAVE_DIR, save_compressed=True)
processor.save_pretrained(SAVE_DIR)