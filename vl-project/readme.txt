以下是 vLLM 近期（2025-2026）针对模型启动时间优化以及OCR/视觉多模态模型的重要 PR 和特性，按类别整理：

一、通用模型启动时间优化
1. 多 GPU 并行启动 — PR #18307
将多 GPU 场景下的 Worker 进程从串行启动改为并发启动（ThreadPoolExecutor），直接缩短多卡场景的初始化时间。

2. 多线程权重加载 — PR #23928 (2025.9 已合并)
使用 ThreadPoolExecutor 并行加载 safetensors/pt 权重文件，可配置线程数（默认 8）。在 NVMe SSD 上加载 62GB 模型时实测 1.43x 加速。

--enable-multithread-load --num-load-threads 8
3. Safetensors Eager Loading — PR #24469
针对网络文件系统（Lustre、NFS），引入 --safetensors-load-strategy eager，用顺序读取替代低效的 mmap 懒加载。实测从 94 分钟降到 14 分钟。

4. 分布式权重加载 — PR #3729, #6127
PR #3729：分布式加载，每个 TP rank 只读 1/tp_size 的权重，然后通过 scatter/broadcast 分发，TP≥4 时 40%+ 加速
PR #6127：延迟张量物化，进程先只读元数据，按需物化，减少总磁盘 I/O
5. 远程实例权重加载 — PR #27417
从已运行的 vLLM 实例通过 NVLink/RDMA 高速互联加载权重，适合大规模分布式部署，避免重复从存储读取。

6. CUDA Graph 优化
特性	PR/Issue	状态	效果
分片式 CUDAGraph 捕获加速	PR #10059	已合并	每个 shape 从 ~5s 降到 <0.5s
延迟 CUDAGraph 捕获	Issue #20098, PR #23184	ON HOLD（未合并）	启动时不预先捕获 67 个 graph，按需捕获
torch.compile 热启动优化	Issue #20402	RFC 提出	序列化 Dynamo 字节码，跳过重复编译
7. 优化级别 — PR #26847
引入 -O 优化级别控制启动/运行权衡：

-O0  # 无优化，最快启动（开发调试用）
-O1  # 轻量优化
-O2  # 完整优化（默认，生产环境）
二、OCR/视觉多模态模型专项优化
1. Qwen2/2.5-VL 启动时间优化 — PR #19756 (2025.6 已合并)
问题：Qwen2-VL 启动时处理 dummy 输入耗时约 40 秒（20 秒 × 2 次）。

解决：预计算每个模态的最大 token 数，跳过昂贵的 dummy 输入生成，大幅缩短初始化时间。

2. Vision Encoder torch.compile — PR #33827 + 社区实践
通过 compile_mm_encoder 标志对视觉编码器（ViT）进行 torch.compile 编译优化：

vllm serve model --compile-mm-encoder
模型	GPU	效果
Qwen3-VL	H200	吞吐量提升 3.4%
InternVL3-2B	H100	请求吞吐从 19.53 → 21.81 req/s (+11.6%)
适用场景：固定分辨率的视觉任务（文档 AI、OCR 流水线、视频分析），因为固定分辨率让 torch.compile 能充分特化计算图。

3. Encoder-Prefill-Decode (EPD) 分离 — PR #25233 (2025.11 已合并)
这是对 OCR 类模型影响最大的架构级优化：

                    传统方式 (单 GPU)
    ┌──────────────────────────────────────┐
    │  [Vision Encoder] → [Prefill] → [Decode]  │
    │  三个阶段串行，encoder 阻塞文本请求        │
    └──────────────────────────────────────┘
                    EPD 分离
    ┌─────────────┐   ┌────────────────────┐
    │ Encoder GPU │   │ Prefill+Decode GPU │
    │ 独立处理图片 │──→│ 只做文本生成        │
    │ 可独立扩缩容 │   │ 不被图片处理阻塞    │
    └─────────────┘   └────────────────────┘
解决的核心问题：

Vision encoder（如 ViT）是 compute-bound，decode 是 memory-bound，混在一起资源利用不匹配
图片处理时间不可预测，会阻塞纯文本请求
OCR 场景大量图片处理时，encoder 成为瓶颈
收益：

encoder 和 decode 可独立扩缩容
encoder 输出可跨请求缓存（相同图片不重复编码）
TTFT 显著降低
4. Encoder Cache 优化 — PR #30475 (2025.12 已合并)
优化 encoder 缓存管理器，只存储 embedding 而非完整 token 范围，大幅节省内存：

Qwen3-VL 8xH100：encoder cache 预算从 153,600 降到 12,288 tokens
可用 KV cache 内存从 7.99 GiB 增加到 12.68 GiB
5. GOT-OCR2 支持情况
GOT-OCR2 的 vLLM 支持请求 (Issue #13862) 在 2025.6 因长期无人跟进被关闭。目前 vLLM 官方支持的 OCR 能力主要通过以下多模态模型实现：

模型	OCR 能力	vLLM 支持
Qwen2.5-VL	文档理解、表格识别、手写识别	完整支持
Qwen3-VL	增强的文档 AI 能力	完整支持
InternVL3	通用视觉理解	完整支持 + torch.compile
GOT-OCR2	专用 OCR	未支持
三、OCR 场景启动优化推荐方案
针对 OCR 类模型（如 Qwen2.5-VL），综合以上特性的推荐配置：

vllm serve Qwen/Qwen2.5-VL-7B-Instruct \
  --tensor-parallel-size 2 \
  --compile-mm-encoder \              # 编译视觉编码器（固定分辨率场景）
  -O2 \                               # 完整优化级别
  --enable-multithread-load \          # 多线程权重加载
  --num-load-threads 8 \
  --enable-chunked-prefill             # chunked prefill（V1 默认开启）
如果是大规模部署，进一步使用 EPD 分离将 encoder 独立部署，消除图片处理对文本生成的干扰。