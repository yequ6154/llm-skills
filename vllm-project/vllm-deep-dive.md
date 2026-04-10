# vLLM 深度剖析 — 从请求到推理的全链路技术解析

> 基于 vLLM V1 源码分析 | 2026-04-02

---

## 目录

1. [Chunked Prefill：混合 Batch 的底层处理](#1-chunked-prefill混合-batch-的底层处理)
2. [为什么用 1D 拼接而非 2D Padded Batch](#2-为什么用-1d-拼接而非-2d-padded-batch)
3. [GPU 并行的本质：CUDA 线程，不是张量维度](#3-gpu-并行的本质cuda-线程不是张量维度)
4. [Continuous Batching 调度逻辑](#4-continuous-batching-调度逻辑)
5. [TTFT 优化：Decode 优先会不会饿死 Prefill？](#5-ttft-优化decode-优先会不会饿死-prefill)
6. [inter_prefill_budget 深度解析](#6-inter_prefill_budget-深度解析)
7. [GPU 计算饱和点测试](#7-gpu-计算饱和点测试)
8. [Benchmark 参数详解](#8-benchmark-参数详解)
9. [投机推理 (Speculative Decoding)](#9-投机推理-speculative-decoding)

---

## 1. Chunked Prefill：混合 Batch 的底层处理

**Q: 开启 chunked prefill 后，GPU forward 的输入里既有 prefill 的 token，也有 decode 的 token，模型底层是如何处理这种混合输入的？**

### 1.1 核心理解：模型不区分，Attention 层区分

GPU forward 的时候，**不存在"prefill token"和"decode token"两种不同类型**。从 Transformer 模型的大部分层（Embedding、FFN/MLP、LayerNorm 等）来看，所有 token 都是完全一样的——都是一个 hidden_state 向量，经过相同的矩阵运算。

**唯一不同的地方是 Attention 层**：

- **Prefill token**：需要 attend 到同一请求中它前面的所有 token（因果注意力），并把自己的 KV 写入 KV Cache
- **Decode token**：只有 1 个 query token，需要 attend 到 KV Cache 中该请求的所有历史 KV，并把自己的 KV 写入 KV Cache

### 1.2 混合 Batch 的数据布局

Scheduler 调度了一个混合 batch（请求A: prefill 512 tokens, 请求B: decode 1 token, 请求C: prefill 256 tokens），所有 token 被**拼接为一个一维张量**：

```
输入 token_ids: [A_0, A_1, ..., A_511, B_0, C_0, C_1, ..., C_255]
                |--- 512 tokens ---|  |1|  |--- 256 tokens ---|
                    请求A (prefill)    请求B    请求C (prefill)
                                     (decode)

total_num_scheduled_tokens = 512 + 1 + 256 = 769
```

### 1.3 两个关键元数据

#### query_start_loc — 每个请求的 query 在 batch 中的起始位置

```
query_start_loc = [0, 512, 513, 769]
                   ^    ^    ^    ^
                   A起点 B起点 C起点 总结束

通过 query_start_loc[i+1] - query_start_loc[i] 得到 query 长度:
  请求A: 512 (prefill, 多 query)
  请求B: 1   (decode, 单 query)
  请求C: 256 (prefill, 多 query)
```

#### seq_lens — 每个请求的完整上下文长度（含历史 KV）

```
seq_lens = [512, 1025, 256]
            ^     ^     ^
         A本次512  B已有1024+1=1025  C本次256

请求A: query_len=512, seq_len=512    → 相等, prefill
请求B: query_len=1,   seq_len=1025   → 远小于, decode
请求C: query_len=256, seq_len=256    → 相等, prefill
```

来源代码 `gpu_model_runner.py`：

```python
self.query_start_loc.np[0] = 0
self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens

self.seq_lens[:num_reqs] = (
    self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
)
```

### 1.4 FlashAttention 如何处理混合 Batch

vLLM V1 使用 `flash_attn_varlen_func`（variable length），原生支持混合长度 batch：

```python
flash_attn_varlen_func(
    q=query[:num_actual_tokens],     # 所有 769 个 query
    k=key_cache,                     # 整个 Paged KV Cache
    v=value_cache,
    cu_seqlens_q=query_start_loc,    # [0, 512, 513, 769]
    seqused_k=seq_lens,              # [512, 1025, 256]
    block_table=block_table,         # 每请求的 KV block 位置
    causal=True,                     # 因果 mask
)
```

> **FlashAttention 根本不需要知道哪个是"prefill"哪个是"decode"。** 它只看 `query_len` 和 `seq_len`，自动按因果 mask 处理每个请求。

### 1.5 KV Cache 写入：slot_mapping

`slot_mapping` 是长度为 769 的一维张量，对每个 token 精确指定 KV Cache 存储位置：

```python
reshape_and_cache_flash(
    key, value,             # 当前 batch 所有 token 的 KV
    key_cache, value_cache, # Paged KV Cache
    slot_mapping,           # [slot_A_0, ..., slot_A_511, slot_B_0, slot_C_0, ...]
    kv_cache_dtype,
    k_scale, v_scale,
)
# 不区分 prefill/decode, 统一写入
```

### 1.6 总结

| 环节 | 是否区分 | 机制 |
|------|---------|------|
| Embedding | 不区分 | 统一查表 |
| QKV Projection | 不区分 | 统一矩阵乘法 |
| KV Cache 写入 | 不区分 | slot_mapping 指定位置 |
| **Attention 计算** | **隐式区分** | query_start_loc + seq_lens + block_table |
| MLP/FFN | 不区分 | 统一矩阵乘法 |
| 采样 | 不区分 | 取每请求最后 token 的 logits |

---

## 2. 为什么用 1D 拼接而非 2D Padded Batch

**Q: 一个 batch 的 requests 被拼接为一个一维张量，为什么不是组成二维的张量并行处理？**

### 2.1 2D Padded 的巨大浪费

```
═══ 2D Padded: [3, 512, 4096] ═══

请求A: [实际512个token .........................]  ← 512 有效
请求B: [1个token | pad pad ... 511个pad]           ← 1 有效, 511 浪费!
请求C: [256个token ... | pad ... 256个pad]          ← 256 有效, 256 浪费!

总计算: 3 × 512 = 1536 个 token
有效:   769 个
浪费:   767 个 (49.9%!)

═══ 1D Packed: [769, 4096] ═══

[A_0, A_1, ..., A_511, B_0, C_0, C_1, ..., C_255]

总计算: 769 个 token
浪费:   0 个
```

### 2.2 PyTorch nn.Linear 的真相

你可能以为 `[3, 512, 4096]` 的第一个维度 3 代表"3 个请求并行处理"。但实际上：

```python
# FFN 层的 Linear: weight shape = [4096, 11008]

# 输入是 3D: [3, 512, 4096]
output = F.linear(input, weight)

# PyTorch 内部实际执行:
input_2d = input.reshape(3 * 512, 4096)  # → [1536, 4096]  ← 展平!
output_2d = input_2d @ weight.T           # → [1536, 11008] ← 实际 GEMM
output = output_2d.reshape(3, 512, 11008) # → reshape 回去
```

> **PyTorch 的 `nn.Linear` 在底层永远是把前面的维度展平成一个大的 2D 矩阵来做 GEMM。** 所以 `[3, 512, 4096]` 实际是 `[1536, 4096]`，而 vLLM 的 `[769, 4096]` 只是 M 更小（且没有浪费）。

---

## 3. GPU 并行的本质：CUDA 线程，不是张量维度

**Q: 一维 [769, 4096] 在经过 FFN 层时如何做到并行处理？**

### 3.1 矩阵乘法在 GPU 上的并行方式

GEMM `C = A × B`，其中 `A: [M, K]`, `B: [K, N]`，GPU 把输出矩阵 C 切成 tile，每个 tile 由一个 Thread Block 独立计算：

```
输出矩阵 C: [M, N]
┌──────┬──────┬──────┬──────┬───┐
│TB(0,0)│TB(0,1)│TB(0,2)│TB(0,3)│...│  ← 每个格子是一个 tile
├──────┼──────┼──────┼──────┼───┤     (比如 128×128)
│TB(1,0)│TB(1,1)│TB(1,2)│TB(1,3)│...│
├──────┼──────┼──────┼──────┼───┤     每个 tile 由一个
│TB(2,0)│TB(2,1)│TB(2,2)│TB(2,3)│...│  Thread Block 独立计算
├──────┼──────┼──────┼──────┼───┤
│ ...  │ ...  │ ...  │ ...  │...│     所有 Thread Block 同时执行!
└──────┴──────┴──────┴──────┴───┘

并行度 = ceil(M/tile_M) × ceil(N/tile_N)
```

具体对比：

| 方式 | 矩阵大小 | Thread Block 数 | 有效计算 |
|------|---------|----------------|---------|
| 1D Packed | [769, 4096] × [4096, 11008] | 6 × 86 = 516 | **100%** |
| 2D Padded | [1536, 4096] × [4096, 11008] | 12 × 86 = 1032 | **~50%** |

**516 个 Thread Block 全部同时执行，769 个 token 都在并行计算。** 2D 的 1032 个也是同时执行，但约 50% 在算 padding 的无用数据。

### 3.2 类比理解

```
2D Padded 方式 — 工厂分组:
  A组: 512 个工件, 512 个工人处理         ✓ 全部有效
  B组: 1 个工件 + 511 个空工作台          ✗ 511个工人在摸鱼!
  C组: 256 个工件 + 256 个空工作台        ✗ 256个工人在摸鱼!
  实际利用率: 769/1536 = 50%

1D Packed 方式 — 传送带:
  [A的512个, B的1个, C的256个] = 769个工件
  769 个工人同时处理, 没有任何人闲置
  实际利用率: 769/769 = 100%
```

### 3.3 认知纠正

| 误解 | 事实 |
|------|------|
| "batch 维度让 3 个请求并行" | batch 维度只是逻辑概念，PyTorch 内部展平为 [3×512, hidden] 做计算 |
| "一维 [769] 是串行处理" | 矩阵乘法的行之间天然独立，GPU 对每行分配独立线程块，全部并行 |
| "二维比一维更高效" | 恰恰相反，二维需要 padding，浪费大量 GPU 计算 |

> **一维拼接是一个纯粹的工程优化：消除 padding 浪费，同时不牺牲任何并行性。**

---

## 4. Continuous Batching 调度逻辑

**Q: Continuous Batching 的逻辑是怎样的？核心代码在哪？**

### 4.1 核心代码

文件：`vllm/v1/core/sched/scheduler.py`

`Scheduler.schedule()` 方法每次迭代动态决定哪些请求参与本次 batch：

1. **计算 Token Budget**
   `token_budget = min(max_num_batched_tokens, available_kv_slots)` — 每步可处理的最大 token 数

2. **优先调度 Running 请求 (decode)**
   已在运行的请求优先拿到预算，每个只需 1 token。不够空间时触发 preemption

3. **用剩余预算调度 Waiting 请求 (prefill)**
   新请求用剩余的 token_budget 做 prefill。如果 prompt 太长可 chunk 分片

4. **构建 SchedulerOutput**
   打包所有调度决策发送给 GPU Worker 执行

### 4.2 关键参数

| 参数 | 作用 |
|------|------|
| `max_num_batched_tokens` | 全局 token 预算上限 |
| `max_num_seqs` | 最大并发请求数 |
| `long_prefill_token_threshold` | 超过此长度的 prefill 会被分片 |
| `inter_prefill_budget` | prefill 之间的预算限制（防 HoL blocking） |

---

## 5. TTFT 优化：Decode 优先会不会饿死 Prefill？

**Q: Continuous Batching 优先调度 running 请求，新请求会一直等待，这不就影响 TTFT 了吗？**

### vLLM 的多层防护机制

**方案 1: `max_num_batched_tokens` 上限**
限制单步最大 token 数。Decode 请求多了自然就有 token 预算剩余给 prefill。

**方案 2: Chunked Prefill 分片**
长 prompt 不一次性占完预算，分成多个 chunk 逐步处理，每个 chunk 之间可以插入 decode 请求。

**方案 3: `long_prefill_token_threshold`**
自动对超过阈值的 prefill 进行分片，防止单个长 prefill 霸占 GPU。

**方案 4: `inter_prefill_budget`** (新特性)
解决 prefill 之间的 Head-of-Line blocking 问题（详见第 6 节）。

**方案 5: Preemption 抢占**
KV Cache 不够时，可以 swap out 低优先级的 running 请求，腾出空间给新请求。

**方案 6: PD 分离**
将 prefill 和 decode 放在不同节点，彻底消除互相干扰。

---

## 6. inter_prefill_budget 深度解析

**Q: inter_prefill_budget 解决的是什么问题？怎么理解 Head-of-Line blocking？**

### 6.1 问题场景：Prefill 之间的队头阻塞

```
请求队列: [A: 4096 tokens] [B: 32 tokens] [C: 64 tokens]
max_num_batched_tokens = 8192

没有 inter_prefill_budget 时:
  Step 1: 调度 A(4096) + B(32) + C(64) → 一步搞定
          但 B 和 C 必须等 A 的 4096 token 全部处理完!
          A 的 prefill 耗时巨大 → B, C 的 TTFT 被拉高

有 inter_prefill_budget = 512 时:
  Step 1: A 只做 512 tokens (chunk 1) + B(32) + C(64)
          B, C 很快完成首次输出 → TTFT 极低!
  Step 2: A 继续 512 tokens (chunk 2) + B, C 的 decode
  Step 3: A 继续 512 tokens (chunk 3) + ...
  ...直到 A 的 prefill 全部完成
```

### 6.2 效果

PR #33743 的基准测试（Gemma-3-27B）：

| 指标 | 变化 |
|------|------|
| TTFT (Time To First Token) | **降低 37%** |
| TPOT (Time Per Output Token) | 仅增加 5-10% |

---

## 7. GPU 计算饱和点测试

**Q: 怎么知道 GPU 计算 token 的饱和点是多大？针对不同芯片怎么得到这个饱和点？**

### 7.1 饱和点的含义

**GPU 饱和点**是指 prefill 时，输入 token 数增加到某个值后，单 token 处理时间不再下降（GPU 算力被完全利用）的临界点。

GPU 有固定的计算核心数。当 token 太少时，很多核心闲置（memory-bound），随着 token 增多，核心利用率提高，单 token 耗时下降。当 token 多到完全填满所有核心（compute-bound），再增加 token 就不会更快了——这就是饱和点。

### 7.2 测试方法

```bash
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  vllm bench serve \
    --model your-model \
    --input-len $INPUT_LEN \
    --output-len 1 \
    --num-prompts 5 \
    --request-rate 0.1 \
    --max-num-batched-tokens $INPUT_LEN
done
```

观察 `time_per_token = prefill_time / INPUT_LEN`：

```
INPUT_LEN   prefill_time   time_per_token
32          2.1ms          65.6 μs/token    ← memory-bound
64          2.3ms          35.9 μs/token    ← 在下降
128         2.8ms          21.9 μs/token    ← 在下降
256         3.5ms          13.7 μs/token    ← 在下降
512         5.2ms          10.2 μs/token    ← 接近饱和
1024        9.8ms          9.6 μs/token     ← 基本饱和!
2048        19.5ms         9.5 μs/token     ← 不再下降
4096        38.8ms         9.5 μs/token     ← 确认饱和
```

当 `time_per_token` **不再明显下降**时，那个 `INPUT_LEN` 就是饱和点。

### 7.3 不同 GPU 的参考饱和点

| GPU | 典型饱和点 (7B 模型) | 典型饱和点 (70B 模型) |
|-----|--------------------|--------------------|
| A100 80GB | ~512-1024 | ~256-512 |
| 4090 24GB | ~256-512 | N/A (显存不够) |
| Ascend 910B | 需实测 | 需实测 |

> 注意：饱和点与具体模型的隐藏层维度、注意力头数、MLP 结构等密切相关，必须实际测试。

---

## 8. Benchmark 参数详解

**Q: `--request-rate` 设置更小是不是就能确保每次处理一个请求来测试饱和点？**

### 8.1 request-rate 的含义

```
--request-rate 0.5  → 每秒发 0.5 个请求 = 每 2 秒 1 个
--request-rate 0.1  → 每秒发 0.1 个请求 = 每 10 秒 1 个
--request-rate inf  → 一次性全发（压力测试模式）
```

确保单请求隔离测试的原则：**发送间隔 > 单请求处理时间**

### 8.2 参数互相关系

| 参数 | 含义 | 关系 |
|------|------|------|
| `--max-num-batched-tokens` | 全局 token 预算上限 | 总预算 |
| `--inter-prefill-budget` | prefill 子预算 | ≤ max-num-batched-tokens |
| `--num-prompts` | 总共发几个请求 | 统计样本数 |
| `--request-rate` | 每秒发几个请求 | 控制请求间隔 |

> - **测饱和点时**，设 `--request-rate 0.1` 或更小，确保请求不重叠。
> - **测吞吐量时**，设 `--request-rate inf`，让系统满载。

---

## 9. 投机推理 (Speculative Decoding)

**Q: 投机推理的逻辑是怎样的？大模型怎么验证？什么机制决定选择小模型的输出？**

### 9.1 核心思想

大模型验证 K 个 token 的开销 ≈ 生成 1 个 token 的开销（因为 K 个 token 可以并行 forward），猜对了就"免费"获得多个 token。

```
普通生成 (5个token):
  [Forward] [Forward] [Forward] [Forward] [Forward]
  耗时: 5 × T_large

投机推理 (5个token, 全部猜对):
  [Draft×5] [Verify一次] → 一步出 5+1 个token!
  耗时: T_draft + T_large ≈ 1.x × T_large
```

### 9.2 vLLM 支持的 Drafter 方法

| 方法 | 类名 | 文件 | 原理 |
|------|------|------|------|
| **Draft Model** | `DraftModelProposer` | `draft_model.py` | 独立小模型生成草稿 |
| **EAGLE** | `EagleProposer` | `eagle.py` | 利用大模型 hidden states 的轻量头 |
| **EAGLE3** | `DFlashProposer` | `dflash.py` | EAGLE 的优化变体 |
| **Medusa** | `MedusaProposer` | `medusa.py` | 多头并行预测 |
| **N-gram** | `NgramProposer` | `ngram_proposer.py` | 基于历史 token 模式匹配（无模型） |
| **Suffix** | `SuffixDecodingProposer` | `suffix_decoding.py` | 后缀树匹配 |

### 9.3 完整执行流程

```
┌───────────────────── GPU Worker ─────────────────────┐
│                                                       │
│  Step 1: Scheduler 调度                                │
│  ┌──────────────────────────────────────────┐         │
│  │ 把上一轮 drafter 猜测的 K 个 draft token   │         │
│  │ 连同原始 token 一起放入 scheduled_tokens    │         │
│  │ 例: 请求A 原本 1 个 decode → 变成 1+K 个   │         │
│  └────────────────┬─────────────────────────┘         │
│                   ↓                                    │
│  Step 2: 大模型 Forward (验证)                          │
│  ┌──────────────────────────────────────────┐         │
│  │ 大模型一次 forward 处理 原始+K个draft       │         │
│  │ 在每个 draft position 产出 target logits    │         │
│  │ 这就是"验证"— 一次 forward 搞定            │         │
│  └────────────────┬─────────────────────────┘         │
│                   ↓                                    │
│  Step 3: Rejection Sampling (接受/拒绝)                │
│  ┌──────────────────────────────────────────┐         │
│  │ 逐个比较 draft token vs target token       │         │
│  │ - 匹配: 接受, 继续下一个                    │         │
│  │ - 不匹配: 拒绝, 用 target token 替换        │         │
│  │          后续 draft 全部丢弃               │         │
│  │ - 全部接受: 额外获得一个 bonus token!       │         │
│  └────────────────┬─────────────────────────┘         │
│                   ↓                                    │
│  Step 4: Drafter 生成新草稿                             │
│  ┌──────────────────────────────────────────┐         │
│  │ 用 drafter 为下一轮猜测 K 个新 token        │         │
│  │ (利用大模型刚产出的 hidden states / token)  │         │
│  └────────────────┬─────────────────────────┘         │
│                   ↓                                    │
│              回到 Step 1                               │
└───────────────────────────────────────────────────────┘
```

### 9.4 代码层面的关键路径

#### Step 2: 大模型验证

```python
# gpu_model_runner.py → execute_model
hidden_states = model(input_ids, positions, ...)       # 一次 forward
sample_hidden_states = hidden_states[logits_indices]   # 取验证位置的 hidden
logits = model.compute_logits(sample_hidden_states)    # 计算 logits
```

大模型只做**一次 forward**，draft token 也在输入中，大模型在每个 draft 位置自然产出概率分布。

#### Step 3: 调用 RejectionSampler

```python
# gpu_model_runner.py → _sample
if spec_decode_metadata is not None:
    sampler_output = self.rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,       # N-gram 等无概率分布
        logits=logits,          # 大模型的 logits
        sampling_metadata=...,
    )
```

#### Step 4: Drafter 提出草稿

```python
# gpu_model_runner.py → propose_draft_token_ids
if spec_config.use_eagle():
    draft_token_ids = self.drafter.propose(
        target_token_ids=...,
        target_hidden_states=target_hidden_states,  # EAGLE: 用大模型 hidden states
        next_token_ids=next_token_ids,
    )
elif spec_config.uses_draft_model():
    draft_token_ids = self.drafter.propose(
        target_hidden_states=None,  # Draft Model: 独立运行
    )
```

### 9.5 验证机制 A：Greedy 验证 (temperature=0)

Triton kernel `rejection_greedy_sample_kernel`：

```
对每个 draft 位置 pos = 0, 1, ..., K-1:
    draft_token = 小模型猜的 token
    target_token = argmax(大模型在该位置的 logits)

    if draft_token == target_token:
        接受! output[pos] = target_token
    else:
        拒绝! output[pos] = target_token  ← 用大模型的替换
        后续所有 draft 全部丢弃

if 全部接受:
    output[K] = bonus_token  ← 白赚一个!
```

**举例：**

```
小模型猜: [the, cat, sat, on]    (K=4)
大模型验: [the, cat, is,  ...]

pos=0: draft=the, target=the → 接受 ✓
pos=1: draft=cat, target=cat → 接受 ✓
pos=2: draft=sat, target=is  → 拒绝 ✗ → output[2]=is, 停止

最终输出: [the, cat, is]  (3个token, 比普通的1个多了2个)
```

### 9.6 验证机制 B：Random 验证 (temperature>0)

基于论文 [Leviathan et al. 2022](https://arxiv.org/abs/2211.17192) 的概率接受方法：

```
对每个 draft 位置 pos = 0, 1, ..., K-1:
    draft_token = 小模型猜的 token
    p_draft  = 小模型对 draft_token 的概率
    p_target = 大模型对 draft_token 的概率
    u        = uniform(0, 1) 随机数

    if p_draft > 0 AND p_target / p_draft >= u:
        接受!
    else:
        拒绝! 从调整后分布 max(p_target - p_draft, 0) 采样 recovered token
        后续全部丢弃

if 全部接受:
    output[K] = bonus_token
```

**Recovered Token 的数学保证：**

```
adjusted_prob[v] = max(p_target[v] - p_draft[v], 0)
recovered_token = sample_from(adjusted_prob)
```

从调整后的分布采样，保证投机推理的输出分布**与直接用大模型生成完全一致**——数学上等价，不会降低生成质量。

### 9.7 验证决策总览

```
         Greedy (temperature=0)
         ┌───────────────────────────────┐
         │ draft_token == argmax(target)? │
         │      ↙          ↘            │
         │    Yes          No            │
         │  接受 draft    拒绝, 用 target  │
         └───────────────────────────────┘

         Random (temperature>0)
         ┌───────────────────────────────┐
         │ p_target(draft_token)          │
         │ ───────────────── ≥ uniform ?  │
         │ p_draft(draft_token)           │
         │      ↙          ↘            │
         │    Yes          No            │
         │  接受 draft    拒绝, 从         │
         │             max(p_t-p_d,0) 采样│
         └───────────────────────────────┘
```

### 9.8 性能收益分析

假设 draft K=5 个 token，大模型 forward 时间 T，小模型（EAGLE 头）0.1T：

| 场景 | 接受数 | 产出 token | 耗时 | 加速比 |
|------|--------|-----------|------|--------|
| 全部接受 | 5/5 | 6 (5+bonus) | 1.5T | **4x** |
| 接受 3 个 | 3/5 | 4 (3+recovered) | 1.5T | **2.67x** |
| 全部拒绝 | 0/5 | 1 (recovered) | 1.5T | **0.67x (变慢!)** |

> **投机推理的价值完全取决于小模型的预测准确率。** EAGLE 通过直接使用大模型的 hidden states，通常能达到 70-90% 的接受率，实现 2-3x 加速。

### 9.9 Worker 线程处理

整个投机推理在**同一个 GPU Worker 进程**内完成：

| Drafter 类型 | 运行位置 | 与大模型的关系 |
|-------------|---------|--------------|
| EAGLE | 同 GPU | 共享 embedding/lm_head，使用大模型 hidden states |
| Draft Model | 同 GPU | 独立小模型，参数量更少 |
| Medusa | 同 GPU | 多头并行预测 |
| N-gram | CPU | 纯模式匹配，不需要模型 |

---

*vLLM Deep Dive | 基于 vLLM V1 源码分析 | Generated 2026-04-02*
