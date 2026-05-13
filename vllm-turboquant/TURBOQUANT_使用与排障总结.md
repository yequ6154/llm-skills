# TurboQuant 使用与排障总结

本文结合本次实测（vLLM `v0.20.0` + Qwen3.5-27B-FP8 + 2x4090）总结 TurboQuant 的判断逻辑、参数含义与 OOM 排障结论。

## 1. 本次结论（先看）

- TurboQuant 已成功挂载（`hooks installed: count=16`）。
- 但在当前负载（`prompt_tokens=2000, output_tokens=256`）下，并发提升到 `5/6` 时出现 OOM。
- OOM 触发点在 TurboQuant 反量化链路，而不是 KV cache 填满。
- 因此当前场景下，TurboQuant 并非稳定增益方案，收益窗口在较低并发（如 `1~4`）内。

---

## 2. 关键疑问统一解释

## 2.1 为什么 KV cache usage 不高，还是 OOM？

- `GPU KV cache usage` 是 **KV 池占用比例**。
- `nvidia-smi dmon` 的 `mem%` 是 **显存带宽利用率**，不是 KV 占用，也不是显存容量占比。
- OOM 来自运行时临时张量峰值（反量化/中间 buffer），即使 KV usage 低也会崩（可能原因之一如反量化需要申请很大的额外显存导致）。

## 2.2 反量化路径从哪里看出来？

服务端栈中明确出现：

- `turboquant/integration/vllm.py`
- `turboquant/score.py -> compute_hybrid_attention`
- `turboquant/quantizer.py -> dequantize -> _unpack_indices`
- `torch.OutOfMemoryError`

说明 OOM 发生在 TurboQuant attention 的反量化阶段。

## 2.3 `GPU_MEMORY_UTILIZATION` 和“剩余显存”是什么关系？

- `GPU_MEMORY_UTILIZATION` 是 vLLM 的预算目标（尤其影响 KV 预算），不是硬分区。
- 反量化临时张量、激活、图捕获池都从 CUDA 全局空闲里动态申请。
- 实际是“所有组件共同竞争同一可分配显存”，不是严格“0.9 区 / 0.1 区”。

## 2.4 反量化为什么额外占显存？

量化 KV 不能直接参与计算，需解包/反量化成可算张量，会产生：

- unpack 索引中间张量
- 反量化后的 `k_hist/v_hist`
- kernel 临时 buffer

这些都是运行时额外申请，可能触发瞬时峰值 OOM。

## 2.5 并发越高，反量化显存是否越大？

通常是。并发升高会增大同一 step 的有效 token 与历史块规模，反量化临时张量近似线性放大，峰值更容易触顶。

## 2.6 如何用 `nvidia-smi dmon -s pucm -d 1` 判断瓶颈类型？

先明确：

- `sm%`：计算核心忙碌度（更偏算力）
- `mem%`：显存控制器忙碌度（更偏带宽）
- `fb`：显存容量占用（MB）

判读经验（结合压测同时间窗口看）：

- `sm` 长期高（如 `>=85%`），`mem` 中低（如 `<=60%`）：
  - 更偏 **GEMM/算力瓶颈**
- `mem` 长期高（如 `>=80%`），`sm` 中低（如 `<=60%`）：
  - 更偏 **显存带宽/KV 访存瓶颈**
- `sm`、`mem` 都高（如都 `>=75%`）：
  - **混合瓶颈**
- `fb` 接近卡容量上限，且出现 `CUDA out of memory`：
  - **容量/峰值瓶颈**（常见于高并发下临时张量峰值）

注意：

- `dmon mem%` 不是 KV cache 占比。
- 需结合 vLLM 日志的 `GPU KV cache usage` 和错误栈一起看，单看 `dmon` 可能误判。

---

## 3. 参数含义（本次重点）

- `TQ_ENABLE`：是否开启 TurboQuant（`1` 开，`0` 关）
- `TQ_KEY_BITS`：K 的量化位宽
- `TQ_VALUE_BITS`：V 的量化位宽
- `TQ_BUFFER_SIZE`：TurboQuant 分块缓冲大小（影响峰值与吞吐）

经验规律：

- 位宽越低（如 `3->2`）：更省资源，但质量/稳定性风险更高。
- `TQ_BUFFER_SIZE` 越小：峰值显存通常更低，但吞吐可能下降；不是越小越好。

---

## 4. 为什么“开了 TurboQuant”可能不提速？

TurboQuant 不是必然提速，核心看“净收益”是否为正：

`量化后KV节省的容量/带宽收益` 是否大于 `反量化计算与临时显存成本`

若当前瓶颈是 GEMM/算力（而非 KV），TurboQuant 可能收益小，甚至负收益。

## 4.1 关于“TurboQuant 是否有意义”的总账问题（Q&A）

问题（原始表述）：

> 在这些场景下，是不是：  
> `存储KVcache本身需要的显存 + 反量化需要的额外显存`  
> `< 不使用turboquant时KVcache存储需要的显存`？  
> 不然 TurboQuant 就没有意义吧，且 TurboQuant 还多了反量化计算延迟？

回答：

- 方向是对的，但需要看的是“**系统总账**”，不只是 KV 存储容量账。
- 更完整的判定应是：  
  `量化后KV存储成本 + 反量化临时显存峰值 + 反量化算力/时延成本`  
  是否小于  
  `原生KV存储成本 + 原生访存/计算成本`。
- 如果不满足这个条件，TurboQuant 在该场景就可能没有净收益，甚至负收益（吞吐变差或更容易 OOM）。

实操上建议用以下结果来判定“有没有意义”：

- 吞吐（`tokens/s`）是否提升
- 延迟（`TTFT/ITL`）是否可接受
- 错误率（500/OOM/EngineDead）是否上升
- 稳定并发上限是否提升

结论：

- **是的**，TurboQuant 确实多了反量化计算过程与临时显存开销。
- 只有当它节省的 KV 容量/带宽收益大于这些新增成本时，才有工程意义。

---

## 5. 适用/不适用场景

## 5.1 更可能有收益

- 长上下文 + 长输出（KV 压力大）
- decode-heavy 负载
- 高并发且 KV/带宽是主瓶颈
- 显存受限但算力仍有余量

## 5.2 不太适用或收益不稳

- prefill-heavy（超长 prompt、短输出）
- GEMM/SM 已接近打满
- 反量化链路频繁 OOM
- 低延迟抖动敏感场景

---

## 6. 本次实操建议（按优先级）

1. 将稳定并发目标先限定在 `1~4` 做有效比较。
2. 继续压 `5+` 前先降低峰值参数：
   - `GPU_MEMORY_UTILIZATION=0.82~0.84`
   - `MAX_NUM_SEQS=4`
   - `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
   - `TQ_BUFFER_SIZE` 可试 `64 -> 48 -> 32`
3. 若仍不稳，再考虑 `TQ_VALUE_BITS=3`（优先于继续降 `TQ_KEY_BITS`）。
4. 做 A/B 必须确保镜像和启动脚本不串台（base 组不能加载 TurboQuant patch）。

---

## 7. 建议的决策标准

仅在同时满足以下条件时保留 TurboQuant：

- 同工况下吞吐有正提升（`tokens/s`）
- 延迟不明显恶化（`TTFT/ITL` 可接受）
- 错误率不升高（无大量 500 / EngineDead / ConnectError）
- 并发上限不降低（或至少不更差）

否则应默认关闭 TurboQuant，仅在 decode-heavy 或特定流量做定向启用。
