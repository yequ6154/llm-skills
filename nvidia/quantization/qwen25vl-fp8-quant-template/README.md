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

## 完整操作步骤

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
1. 以 xd 用户安装 `llmcompressor` 和 `compressed-tensors`（`--no-deps`）
2. 以 root 用户 patch transformers 源码

### 3. 进入容器运行量化

```bash
docker exec -it -u xd <容器名> bash
cd /home/lei.xiong/project/quant/qwen25vl-fp8-quant-template
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
├── start_container.sh       # 启动容器脚本
├── setup_and_patch.sh       # 安装依赖 + patch transformers（在宿主机执行）
├── patch_transformers.py    # patch 脚本（被 setup_and_patch.sh 调用）
└── quant_fp8.py             # 量化脚本（在容器内执行）
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
由于我们使用 `device_map='cuda:0'`（所有参数在同一张卡上，无 offload），跳过该块完全安全。

---

## 适配其他模型

修改 `quant_fp8.py` 中的以下变量即可：

```python
os.environ["CUDA_VISIBLE_DEVICES"] = "4"     # 改为你要用的 GPU 编号
MODEL_ID = "/models/xxx/release-v2"           # 改为你的模型路径
SAVE_DIR = "/models/xxx-FP8-Dynamic"          # 改为量化后的保存路径
```

`ignore` 列表中已包含 `visual.*` 和 `lm_head.*`，适用于所有 Qwen2.5-VL 系列模型。
