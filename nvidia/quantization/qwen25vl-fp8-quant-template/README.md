# Qwen2.5-VL FP8 Dynamic 量化方案

## 问题背景

使用 `llmcompressor` 对 Qwen2.5-VL 系列模型做 FP8 动态量化时，`model.save_pretrained(save_compressed=True)` 会报错：

```
KeyError: 'visual.patch_embed.proj.weight'
```

**根因**：`transformers >= 4.52` 在 `save_pretrained` 中对 offloaded modules 的处理逻辑有 bug。
当模型通过 `device_map` 加载时，transformers 会认为模型有 offloaded 模块，进入一个特殊的保存分支。
该分支构建 `module_map` 时，与 Qwen2.5-VL 的 visual encoder 模块命名不兼容，导致 KeyError。
即使所有参数都在同一张 GPU 上（无 offload），该分支仍会被触发。

**解决方案**：在容器内 patch transformers 源码，跳过有 bug 的 offloaded modules 代码块。

---

## 环境要求

| 组件 | 版本 |
|------|------|
| Docker 镜像 | `vllm-openai:v0.10.2`（transformers 4.56.1 + torch 2.8.0+cu128） |
| llmcompressor | 0.10.0.1（`--no-deps` 安装，避免污染系统环境） |
| compressed-tensors | 0.14.0.1（`--no-deps` 安装） |
| GPU | 单卡显存 >= 20GB（7B 模型 bf16 约 15-17GB） |

> 其他 vLLM 镜像版本也适用，只要 transformers >= 4.52 都需要打 patch。

---

## 快速使用（推荐）

一行命令完成量化，自动启动容器、安装依赖、patch、量化、清理：

```bash
bash run.sh <模型路径> <量化输出路径> [GPU编号] [用户名]
```

| 参数 | 必填 | 默认值 | 说明 |
|------|------|--------|------|
| 模型路径 | 是 | - | 原始模型目录 |
| 量化输出路径 | 是 | - | 量化后模型保存目录 |
| GPU编号 | 否 | `0` | 使用的 GPU，多卡用逗号分隔 |
| 用户名 | 否 | 当前用户 | 容器内运行量化的用户，决定输出文件的归属 |

示例：

```bash
# 7B 模型, 单卡, 以当前用户身份运行
bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8 4

# 大模型, 2卡, 指定用户 xd
bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8 4,5 xd

# 不指定 GPU 和用户, 默认 GPU 0、当前用户
bash run.sh /models/BR-VL-xxx/release-v2 /models/BR-VL-xxx-FP8

# 不同机器上用不同用户
bash run.sh /models/xxx/release-v2 /models/xxx-FP8 0,1 lei.xiong
```

脚本内部流程：

1. 启动临时容器
2. 以 root 安装依赖 + patch transformers（修改系统文件必须 root）
3. 以**指定用户**运行量化（输出文件归该用户所有）
4. 清理 config.json 中的 `scale_dtype` / `zp_dtype` 字段（兼容 vLLM 0.11.0 等版本）
5. 结束后自动清理容器

> 必要文件：`run.sh` + `patch_transformers.py`（两个文件必须在同一目录下）

---

## 分步操作（可选）

如果需要进入容器调试或定制，可以按以下步骤手动执行：

### 1. 启动容器

```bash
bash start_container.sh
```

根据实际情况修改 `start_container.sh` 中的容器名、GPU 编号和模型挂载路径。

### 2. 以 root 身份安装依赖 + 打 patch

```bash
# 在宿主机上执行（一行命令搞定）
bash setup_and_patch.sh <容器名>
```

例如：

```bash
bash setup_and_patch.sh test-lei-16
```

这个脚本会：
1. 以 root 安装 `llmcompressor` 和 `compressed-tensors`（`--no-deps`）
2. 以 root patch transformers 源码

### 3. 进入容器运行量化

```bash
docker exec -it -u <用户名> <容器名> bash
python3 quant_fp8.py
```

### 4. 量化完成后清理（可选）

如果这个容器还要用来跑 vLLM 推理服务，需要卸载用户级包，恢复系统环境：

```bash
pip3 uninstall llmcompressor compressed-tensors -y
```

---

## 文件说明

```
qwen25vl-fp8-quant-template/
├── README.md               # 本文档
├── run.sh                   # 一键量化脚本（推荐使用）
├── patch_transformers.py    # patch 脚本（被 run.sh 和 setup_and_patch.sh 调用）
├── start_container.sh       # [分步用] 启动容器
├── setup_and_patch.sh       # [分步用] 安装依赖 + patch transformers
└── quant_fp8.py             # [分步用] 量化脚本（在容器内执行）
```

---

## 已知问题及处理

### config.json 中的 scale_dtype / zp_dtype 字段

量化时使用的 `compressed-tensors 0.14.0.1` 会在 `config.json` 的量化配置中写入 `scale_dtype` 和 `zp_dtype` 字段。
但部分 vLLM 版本（如 0.11.0）自带的 `compressed-tensors` 较旧，其 Pydantic model 不认识这两个字段，部署时会报错：

```
pydantic_core._pydantic_core.ValidationError: 2 validation errors for VllmConfig
scale_dtype
  Extra inputs are not permitted [type=extra_forbidden, ...]
zp_dtype
  Extra inputs are not permitted [type=extra_forbidden, ...]
```

**`run.sh` 已自动处理**：量化完成后会自动从 config.json 中删除这两个字段。

如果是分步操作或之前量化的模型遇到此问题，手动修复：

```bash
python3 -c "
import json
path = '/models/xxx-FP8-Dynamic/config.json'
with open(path) as f:
    config = json.load(f)
def clean(obj):
    if isinstance(obj, dict):
        for k in ('scale_dtype', 'zp_dtype'):
            obj.pop(k, None)
        for v in obj.values():
            clean(v)
    elif isinstance(obj, list):
        for v in obj:
            clean(v)
clean(config)
with open(path, 'w') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print('Done')
"
```

---

## Patch 原理

transformers 的 `save_pretrained` 方法中有如下判断：

```python
if (
    hasattr(self, "hf_device_map")
    and len(set(self.hf_device_map.values())) > 1
    and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())
):
    # 构建 module_map，处理 offloaded modules
    # 这段代码与 Qwen2.5-VL 的 visual encoder 不兼容
```

patch 将这个条件替换为 `if False:`，直接跳过整个 offloaded modules 处理块。
使用 `device_map='auto'` 时，只要 GPU 显存够用（参数不 offload 到 CPU/disk），跳过该块完全安全。

---

## 适配其他模型

使用 `run.sh` 时通过命令行参数即可适配，无需改代码。

使用 `quant_fp8.py`（分步操作）时，修改脚本中的以下变量：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "4"     # 7B 单卡; 大模型改成 "4,5" 等多卡
MODEL_ID = "/models/xxx/release-v2"           # 改为你的模型路径
SAVE_DIR = "/models/xxx-FP8-Dynamic"          # 改为量化后的保存路径
```

用卡数量通过 `CUDA_VISIBLE_DEVICES` 控制，`device_map='auto'` 会自动分配：

| 模型大小 | 显存需求(bf16) | CUDA_VISIBLE_DEVICES 示例 |
|---------|--------------|-------------------------|
| 7B     | ~17GB        | `"0"`（单卡 4090 即可）    |
| 14B/32B | ~34GB       | `"0,1"`（2 卡 4090）      |
| 72B    | ~150GB       | `"0,1,2,3"`（4+ 卡）     |

`ignore` 列表中已包含 `visual.*` 和 `lm_head.*`，适用于所有 Qwen2.5-VL 系列模型。
