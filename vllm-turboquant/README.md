# vLLM + TurboQuant 镜像使用说明

本目录提供了一套基于 vLLM OpenAI 镜像的 TurboQuant 方案，目标是：

- 不改变基础镜像启动行为（不改 `ENTRYPOINT/CMD`）
- 只在镜像内注入 TurboQuant 适配逻辑
- 后续使用方式尽量和原生 `vllm-openai` 一致

## 目录内脚本说明

- `Dockerfile.vllm-turboquant`  
  基于 `BASE_IMAGE` 构建新镜像，安装 `turboquant`，并通过 `.pth` 自动加载 `tq_vllm_patch.py`。

- `build_turboquant_image.sh`  
  一键构建镜像脚本。

- `run_turboquant_container.sh`  
  一键启动容器脚本（封装了常用环境变量和挂载）。

- `docker_start.sh`  
  兼容入口，内部直接调用 `run_turboquant_container.sh`。

- `turboquant_vllm_patch/sitecustomize.py`  
  TurboQuant 自动注入逻辑源文件（构建时会拷贝进镜像并转为 `tq_vllm_patch.py`）。

## 1) 构建镜像

在当前目录执行：

```bash
cd /home/lei.xiong/script/qwen3.5-27b/vllm-turboquant
./build_turboquant_image.sh
```

默认参数：

- `IMAGE_NAME=registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.20.0__turboquant`
- `BASE_IMAGE=registry.100credit.cn/xdynamic/x86_64/nvidia/vllm-openai:v0.20.0`
- `TURBOQUANT_GIT_URL=https://github.com/0xSero/turboquant.git`
- `TURBOQUANT_GIT_REF=main`

可选自定义示例：

```bash
IMAGE_NAME=my-vllm-tq:v0.20.0 \
BASE_IMAGE=vllm/vllm-openai:v0.20.0 \
TURBOQUANT_GIT_REF=main \
./build_turboquant_image.sh
```

## 2) 启动容器（脚本方式）

```bash
cd /home/lei.xiong/script/qwen3.5-27b/vllm-turboquant
./run_turboquant_container.sh
```

也可以用兼容入口：

```bash
./docker_start.sh
```

启动后查看日志：

```bash
docker logs -f qwen3-5-27b-vllm-tq
```

## 3) 常用运行参数

`run_turboquant_container.sh` 支持通过环境变量覆盖默认值：

- 基础运行
  - `IMAGE_NAME`：镜像名
  - `CONTAINER_NAME`：容器名
  - `GPU_DEVICES`：GPU 列表（如 `0,1`）
  - `HOST_PORT`：服务端口
  - `MODEL_MOUNT`：模型目录挂载（宿主机:容器）
  - `WORK_MOUNT`：工作目录挂载（宿主机:容器）

- vLLM 参数
  - `MODEL_PATH`
  - `SERVED_MODEL_NAME`
  - `TP_SIZE`
  - `GPU_MEMORY_UTILIZATION`
  - `MAX_MODEL_LEN`
  - `MAX_NUM_SEQS`
  - `ENABLE_PREFIX_CACHING`
  - `ENABLE_CHUNKED_PREFILL`
  - `TRUST_REMOTE_CODE`
  - `DTYPE`
  - `SPECULATIVE_CONFIG`

- TurboQuant 参数
  - `TQ_ENABLE`（默认 `1`）
  - `TQ_KEY_BITS`（默认 `3`）
  - `TQ_VALUE_BITS`（默认 `4`）
  - `TQ_BUFFER_SIZE`（默认 `128`）

示例（2 卡 + 改端口）：

```bash
GPU_DEVICES=1,3 HOST_PORT=8322 TP_SIZE=2 ./run_turboquant_container.sh
```

## 4) 与原生 vLLM 镜像用法一致性

本方案不会覆盖基础镜像 `ENTRYPOINT/CMD`。  
核心区别仅在于：镜像内额外安装了 TurboQuant，并通过 Python `.pth` 自动加载补丁模块。

因此你可以像原生 vLLM OpenAI 镜像一样使用该镜像，只需把镜像名替换成你构建后的 `IMAGE_NAME`。

## 5) 如何确认 TurboQuant 已生效

观察容器日志中是否出现以下前缀：

- `[turboquant-autoinstall] patched ...`
- `[turboquant-autoinstall] hooks installed: ...`

例如：

```bash
docker logs qwen3-5-27b-vllm-tq | grep "turboquant-autoinstall"
```

如果看到 `hooks installed`，说明 worker 上已完成 TurboQuant hook 注入。

## 6) 常见问题

- 构建很慢  
  首次需要拉取 `BASE_IMAGE` 对应的大层镜像，耗时较长是正常现象。

- 启动失败提示找不到模型  
  检查 `MODEL_MOUNT` 与 `MODEL_PATH` 是否对应；容器内路径必须能访问到模型目录。

- 没看到 TurboQuant 日志  
  确认 `TQ_ENABLE=1`，并检查容器内是否确实使用了你新构建的镜像（不是基础镜像）。
