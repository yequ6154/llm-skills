vLLM /v1/chat/completions 请求全链路
总览：9 个阶段，3 个进程，17+ 个关键类
一个 curl 请求的完整生命旅程：

Client ──HTTP──▶ [API Server 进程] ──ZMQ──▶ [EngineCore 进程] ──multiproc──▶ [GPU Worker 进程]
阶段 1: HTTP 请求接收
文件: vllm/entrypoints/openai/api_server.py

FastAPI 注册 POST /v1/chat/completions 路由。请求体解析为 ChatCompletionRequest (Pydantic model，定义在 entrypoints/openai/chat_completion/protocol.py)，然后转发给 ServingChat。

阶段 2: ServingChat 处理
文件: vllm/entrypoints/openai/serving_chat.py 方法: ServingChat.create_chat_completion()

这是最关键的预处理阶段：

验证参数 (model name, temperature, top_p 等)
Chat Template (Jinja2) 将 messages[] 渲染为 prompt 字符串
Tokenize prompt 为 token_ids[]
处理多模态数据 (图片/音频 → feature tensors)
构建 SamplingParams 对象
如有 response_format，配置 GuidedDecodingParams (结构化输出)
如有 tools，配置 tool_call_parser
阶段 3: AsyncLLM 引擎客户端
文件: vllm/v1/engine/async_llm.py

AsyncLLM.generate() 在 API Server 进程内运行：

add_request() 分配唯一 request_id
InputPreprocessor 做最终预处理
构建 EngineCoreRequest 对象
通过 ZMQ socket 发送到 EngineCore 进程
output_handler 异步循环监听返回
阶段 4: EngineCore 调度 (跨进程)
文件: vllm/v1/engine/core.py → vllm/v1/core/scheduler.py → vllm/v1/core/kv_cache_manager.py

EngineCore 独立进程 busy loop：

Scheduler.schedule() 决策本轮 batch 组成
KVCacheManager.get_computed_blocks() 检查 prefix cache
KVCacheManager.allocate_slots() 分配 KV cache blocks
生成 SchedulerOutput (scheduled_requests + block_tables)
阶段 5-6: GPU 执行
文件: vllm/v1/executor/multiproc_executor.py → vllm/v1/worker/gpu_worker.py → vllm/v1/worker/gpu_model_runner.py

GPUModelRunner.execute_model():

组装 input_ids, positions 为 GPU tensors
构建 Attention metadata (FlashAttention/FlashInfer)
CUDAGraph 加速 (piecewise CUDAGraph)
set_forward_context() 传递 attention backend 信息
调用 model.forward()
阶段 7: 模型前向传播
文件: vllm/model_executor/models/llama.py (以 Llama 为例)

embed_tokens → N × [RMSNorm → Attention(Q,K,V) → RMSNorm → MLP] → RMSNorm → hidden_states
Attention 内部使用 PagedAttention kernel，从分页的 KV Cache 中读取历史 K/V。

阶段 8: 采样
文件: vllm/v1/sample/sampler.py

hidden_states → lm_head → logits → temperature/top_p/top_k → sample → token_id

阶段 9: 输出返回
回程路径: EngineCore → ZMQ → AsyncLLM → Detokenizer → ServingChat → HTTP Response

EngineCore 的 update_from_output() 检查 stop 条件
Detokenizer 增量将 token_ids 转为 text
ServingChat 构建 ChatCompletionResponse
流式: SSE data: {...}\n\n; 非流式: 完整 JSON