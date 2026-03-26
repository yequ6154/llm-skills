from modelscope import AutoModel, AutoTokenizer
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier

import os
import gc

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

gc.collect()

MODEL_ID = "/model/Qianfan-OCR"
model = AutoModel.from_pretrained(
    MODEL_ID,
    torch_dtype='auto',
    trust_remote_code=True,
    device_map="auto",
    local_files_only=True
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True, local_files_only=True, fix_mistral_regex=True)

recipe = QuantizationModifier(
    targets="Linear",
    scheme="W8A8",
    ignore=["re:lm_head.*", 
    "re:model.lm_head.*",
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*", 
    "re:.*visual", 
    "re:.vision_model.*",
    "re:language_model.model.embed_tokens.*",
    "re:mlp1.*",],
)

# Apply quantization and save to disk in compressed-tensors format.
oneshot(model=model, recipe=recipe)

# Save to disk in compressed-tensors format.
SAVE_DIR = "/model/Qianfan-OCR-W8A8"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)