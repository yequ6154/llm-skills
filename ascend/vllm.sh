# cpu绑核
export CPU_AFFINITY_CONF=1,npu_affine:1

# 内存分配
export LD_PRELOAD=/usr/lib64/libjemalloc.so.2:$LD_PRELOAD # (前提需要安装yum install jemalloc)

# 绝大多数以文本生成为主的大模型推理服务中可以使用这个参数。极致优化解码（Token生成）阶段的重复开销，这是大模型推理耗时的主要部分
--compilation-config '{"cudagraph_mode": "FULL_DECODE_ONLY"}'

# 这是一个低风险、潜在高回报的性能优化选项。 在完成参数调整后，可以通过观察日志中的 Avg generation throughput 来对比优化效果
--async-scheduling