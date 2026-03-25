import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import torch.nn as nn
import torch
import torch.nn.functional as F
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForTokenClassification, AutoConfig,  AutoTokenizer, AutoProcessor,AutoModel, AutoModelForSequenceClassification, Qwen2_5_VLForConditionalGeneration
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLCausalLMOutputWithPast, Qwen2_5_VLModel


class Qwen2_5_VLForConditionalClassfication(Qwen2_5_VLForConditionalGeneration):

  def __init__(self, config):
    super().__init__(config)
    self.model = Qwen2_5_VLModel(config)
    self.lm_head = nn.Identity()
    self.score = nn.Linear(config.text_config.hidden_size, config.num_labels, bias=False)

    def forward(self,
                input_ids: torch.LongTensor | None = None,
                attention_mask: torch.Tensor | None = None,
                position_ids: torch.LongTensor | None = None,
                past_key_values=None,
                inputs_embeds: torch.FloatTensor | None = None,
                labels: torch.LongTensor | None = None,
                use_cache: bool | None = None,
                output_attentions: bool | None = None,
                output_hidden_states: bool | None = None,
                pixel_values: torch.Tensor | None = None,
                pixel_values_videos: torch.FloatTensor | None = None,
                image_grid_thw: torch.LongTensor | None = None,
                video_grid_thw: torch.LongTensor | None = None,
                rope_deltas: torch.LongTensor | None = None,
                cache_position: torch.LongTensor | None = None,
                second_per_grid_ts: torch.Tensor | None = None,
                logits_to_keep: int | torch.Tensor = 0,
                **kwargs,
               ) -> tuple | Qwen2_5_VLCausalLMOutputWithPast:
      output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
      output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
      )

      outputs = self.model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        pixel_values_videos=pixel_values_videos,
        image_grid_thw=image_grid_thw,
        video_grid_thw=video_grid_thw,
        second_per_grid_ts=second_per_grid_ts,
        position_ids=position_ids,
        attention_mask=attention_mask,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=True,
        cache_position=cache_position,
        **kwargs,
      )
      hidden_states = outputs[0]
      print(hidden_states.shape)
      # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
      slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
      #hidden_states = self.lm_head(hidden_states[:, slice_indices, :])
      score = self.score(hidden_states[:, slice_indices, :])

      loss = None

      return Qwen2_5_VLCausalLMOutputWithPast(
        loss=loss,
        logits=score,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        rope_deltas=outputs.rope_deltas,
      )

model_dir = "/data/llm_application/MODELS/资产魔方分类头/br-vision-asset-classification-creditcard-v1.0-20260113"

model = Qwen2_5_VLForConditionalClassfication.from_pretrained(model_dir, torch_dtype="auto", device_map="auto")

prompt = "<image>你是银行贷后材料分类专家。给出此图片分类。"

messages = [
  {"role": "user", "content": [
    {"type": "image", "image": "/data/llm_application/DATASETS/资产魔方/标注数据/中银/数据增强_0107/train/img_000000_original.png"},
    {"type": "text", "text": prompt}
  ]}
]

processor = AutoProcessor.from_pretrained(model_dir)

text = processor.apply_chat_template(
  messages, tokenize=False, add_generation_prompt=True
)

image_inputs, video_inputs = process_vision_info(messages)
tttt = text[75:-33]
inputs = processor(
  text=[tttt],
  images=image_inputs,
  videos=video_inputs,
  padding=True,
  return_tensors="pt",
)
inputs = inputs.to("cuda")

model.eval()
with torch.no_grad():
  res = model(**inputs)

def _get_seq_cls_logprobs(pred: int, logprobs: torch.Tensor, top_logprobs: int):
  idxs = logprobs.argsort(descending=True, dim=-1)[:top_logprobs].tolist()
  logprobs = logprobs.tolist()
  return {
    'content': [{
      'index': pred,
      'logprobs': [logprobs[p] for p in pred] if isinstance(pred, (list, tuple)) else logprobs[pred],
      'top_logprobs': [{
        'index': idx,
        'logprob': logprobs[idx]
      } for idx in idxs]
    }]
  }

top_logprobs = 20
batch_size = 1
logits = res.logits


pooled_logits = logits[torch.arange(batch_size, device=logits.device), -1]
preds = [(logprob >= 0.5).nonzero(as_tuple=True)[0].tolist() for logprob in torch.sigmoid(pooled_logits)]
logprobs = F.logsigmoid(pooled_logits)

logprobs = [_get_seq_cls_logprobs(pred, logprobs[i], top_logprobs) for i, pred in enumerate(preds)]
