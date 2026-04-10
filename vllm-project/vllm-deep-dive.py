Q-1:在ServingChat.create_chat_completion函数里面：调用 tokenizer 将 prompt 转为 token_ids。如果我的输入是token_ids，这种是不是就会省略这一步操作？这个逻辑是在代码哪块判断输入是字符串还是token_ids，然后进行tokenizer或者不操作tokenizer
A-1:
'''
三条分支路径
关键代码逻辑如下（简化版）：
class InputPreprocessor:
    def _prompt_to_llm_inputs(self, prompt, tokenization_kwargs=None) -> SingletonInput:
        if "prompt_embeds" in prompt:
            return self._process_embeds(prompt)        # 路径 C: 直接传入 embeddings
        if "prompt_token_ids" in prompt:
            return self._process_tokens(prompt)        # 路径 B: 已有 token_ids → 跳过 tokenizer
        if "prompt" in prompt:
            return self._process_text(prompt, ...)     # 路径 A: 文本字符串 → 调用 tokenizer
判断方式非常直接 — 检查 dict 里有哪个 key：

输入中包含的 key	走哪条路径	是否调用 tokenizer
"prompt" (str)	_process_text()	是，调用 _tokenize_prompt()
"prompt_token_ids" (List[int])	_process_tokens()	否，跳过 tokenizer
"prompt_embeds" (Tensor)	_process_embeds()	否，跳过 tokenizer
路径 A: 传入文本 → 调用 tokenizer
def _process_text(self, parsed_content: TextPrompt, ...) -> TokensInput:
    prompt_text = parsed_content["prompt"]           # 拿到字符串
    prompt_token_ids = self._tokenize_prompt(prompt_text, ...)  # 调用 tokenizer！
    inputs = tokens_input(prompt_token_ids)
    inputs["prompt"] = prompt_text
    return inputs
路径 B: 传入 token_ids → 跳过 tokenizer
def _process_tokens(self, parsed_content: TokensPrompt, ...) -> TokensInput:
    prompt_token_ids = self._truncate_inputs(parsed_content["prompt_token_ids"], ...)
    # ↑ 注意：这里只做 truncation（截断到 max_model_len），不调用 tokenizer！
    inputs = tokens_input(prompt_token_ids)
    return inputs
你说得对：如果输入已经是 token_ids，确实会省略 tokenization 这一步，只做截断处理。

输入类型定义
这些类型定义在 vllm/inputs/llm.py 中：

class TextPrompt(_PromptOptions):
    """文本提示 - 需要 tokenize"""
    prompt: str
class TokensPrompt(_PromptOptions):
    """已 tokenize 的提示 - 跳过 tokenize"""
    prompt_token_ids: list[int]
    prompt: NotRequired[str]        # 可选的原始文本（用于日志/返回）
    token_type_ids: NotRequired[list[int]]
class EmbedsPrompt(_PromptOptions):
    """直接传入 embeddings"""
    prompt_embeds: torch.Tensor
# 统一类型
DecoderOnlyPrompt = str | TextPrompt | list[int] | TokensPrompt | EmbedsPrompt
但是，Chat Completion API 能直接传 token_ids 吗？
这是关键的一点：/v1/chat/completions 端点只接受 messages[] 文本格式，所以在 ServingChat 层面，输入一定会经过 chat template 渲染 + tokenization。

如果你想直接传 token_ids 跳过 tokenizer，有两个方式：

方式 1 — 使用 /v1/completions API（非 chat）：

curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.1-8B",
    "prompt": [1, 15043, 29892, 920, 526, 366],
    "max_tokens": 50
  }'
当 prompt 字段是一个整数数组时，vLLM 会识别为 token_ids，走 _process_tokens() 路径。

方式 2 — 使用离线推理 Python API：

from vllm import LLM, SamplingParams
llm = LLM(model="meta-llama/Llama-3.1-8B")
outputs = llm.generate(
    {"prompt_token_ids": [1, 15043, 29892, 920, 526, 366]},  # TokensPrompt
    SamplingParams(max_tokens=50)
)
完整调用链路
/v1/chat/completions (messages[])
    → ServingChat: chat_template → prompt 字符串 → 必然 tokenize
        → AsyncLLM.generate(TextPrompt)
            → InputPreprocessor._process_text() → _tokenize_prompt() ✅
/v1/completions (prompt = [1,2,3,...])
    → ServingCompletion: 检测到是 List[int]
        → AsyncLLM.generate(TokensPrompt)
            → InputPreprocessor._process_tokens() → 跳过 tokenizer ✅
Python API: llm.generate({"prompt_token_ids": [...]})
    → InputPreprocessor._process_tokens() → 跳过 tokenizer ✅
还有一层：V1 InputProcessor 的额外判断
在 V1 引擎中（vllm/v1/engine/input_processor.py），还有一层上游判断：

class InputProcessor:
    def process_inputs(self, prompt, ...):
        if isinstance(prompt, dict) and "type" in prompt:
            processed_inputs = prompt   # 已经是 EngineInput，完全跳过预处理
        else:
            processed_inputs = self.input_preprocessor.preprocess(prompt)  # 走上面的分支逻辑
如果 prompt 已经是处理过的 EngineInput（带有 "type" key），则连 InputPreprocessor 都不会被调用，直接使用。

总结：判断的核心就是 InputPreprocessor._prompt_to_llm_inputs() 中对 dict key 的检查 — "prompt" key → tokenize，"prompt_token_ids" key → 跳过。但在 chat completion API 场景下，因为输入是 messages[]，必须经过 chat template 渲染成字符串，所以 tokenization 是无法跳过的。
'''

Q-2:Continuous Batching 核心思想
A-2:
'''
传统 static batching 的问题是：一个 batch 里所有请求必须等最长的那个完成，才能释放资源、填入新请求。Continuous Batching 的核心思想是每一个 iteration（step）都重新做调度决策 — 随时加入新请求、随时移除完成的请求。

核心代码位置
vllm/v1/core/sched/scheduler.py → Scheduler 类
两个核心方法构成了 Continuous Batching 的完整循环：

方法	职责
schedule()	每个 step 开始时：决定哪些请求参与本轮 batch
update_from_output()	每个 step 结束后：处理输出、移除完成的请求
schedule() 方法 — 构建本轮 batch
源码中有一段非常关键的设计注释（第 342-351 行）：

def schedule(self) -> SchedulerOutput:
    # NOTE(woosuk) on the scheduling algorithm:
    # There's no "decoding phase" nor "prefill phase" in the scheduler.
    # Each request just has the num_computed_tokens and
    # num_tokens_with_spec. ...
    # At each step, the scheduler tries to assign tokens to the requests
    # so that each request's num_computed_tokens can catch up its
    # num_tokens_with_spec. This is general enough to cover
    # chunked prefills, prefix caching, speculative decoding,
    # and the "jump decoding" optimization in the future.
调度器并不区分 "prefill 阶段" 和 "decode 阶段"，而是用一个统一的模型：每个请求有 num_computed_tokens（已计算）和 num_tokens_with_spec（需要计算到的目标），调度器让每个请求的 computed 追赶 target。

Step 1: 先调度 RUNNING 队列中的请求（decode 优先）
# First, schedule the RUNNING requests.
req_index = 0
while req_index < len(self.running) and token_budget > 0:
    request = self.running[req_index]
    # 计算这个请求需要多少新 tokens
    num_new_tokens = (
        request.num_tokens_with_spec
        + request.num_output_placeholders
        - request.num_computed_tokens
    )
    # 对于 decode 请求，num_new_tokens 通常 = 1
    # 对于 chunked prefill 未完成的请求，num_new_tokens 可能很大
    num_new_tokens = min(num_new_tokens, token_budget)
    # 尝试分配 KV cache blocks
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is None:
        # 资源不够 → 抢占优先级最低的请求
        preempted_req = self.running.pop()  # 移除最低优先级
        self._preempt_request(preempted_req)
    else:
        # 调度成功
        scheduled_running_reqs.append(request)
        token_budget -= num_new_tokens
关键点：running 队列里的请求（正在 decode 的）优先被调度，确保已经开始生成的请求不被饿死。如果 KV cache 不够，会抢占（preempt）优先级最低的请求。

Step 2: 再调度 WAITING 队列中的新请求（prefill）
# Next, schedule the WAITING requests.
while (self.waiting or self.skipped_waiting) and token_budget > 0:
    if len(self.running) == self.max_num_running_reqs:
        break  # 达到最大并发数
    request = request_queue.peek_request()
    # 检查 prefix cache 命中
    new_computed_blocks, num_computed_tokens = (
        self.kv_cache_manager.get_computed_blocks(request)
    )
    # 计算需要 prefill 的 token 数
    num_new_tokens = request.num_tokens - num_computed_tokens
    num_new_tokens = min(num_new_tokens, token_budget)
    # Chunked Prefill: 如果 budget 不够处理全部 prompt，只处理一部分
    if not self.scheduler_config.enable_chunked_prefill and num_new_tokens > token_budget:
        break  # 不启用 chunked prefill 时，必须一次处理完
    # 尝试分配 KV cache blocks
    new_blocks = self.kv_cache_manager.allocate_slots(request, num_new_tokens)
    if new_blocks is None:
        break  # 没有足够的 blocks
    # 成功调度！加入 running 队列
    self.running.append(request)
    request.status = RequestStatus.RUNNING
    scheduled_new_reqs.append(request)
    token_budget -= num_new_tokens
关键点：新请求只要还有 token_budget 就会被加入 batch。Chunked Prefill 允许长 prompt 分多个 step 处理。

update_from_output() 方法 — 处理输出、移除完成请求
每次 GPU forward + sample 完成后调用：

def update_from_output(self, scheduler_output, model_runner_output):
    stopped_running_reqs: set[Request] = set()
    for req_id, num_tokens_scheduled in num_scheduled_tokens.items():
        request = self.requests.get(req_id)
        generated_token_ids = sampled_token_ids[req_index]
        # 检查 stop 条件 (EOS / stop words / max_tokens)
        new_token_ids, stopped = self._update_request_with_output(
            request, generated_token_ids
        )
        if stopped:
            finished = self._handle_stopped_request(request)
            if finished:
                self._free_request(request)  # 释放 KV cache blocks！
            stopped_running_reqs.add(request)
    # 从 running 队列移除已完成的请求
    if stopped_running_reqs:
        self.running = remove_all(self.running, stopped_running_reqs)
关键点：完成的请求立即被移除，其 KV cache blocks 立即被释放，下一个 schedule() 调用就能把这些 blocks 分配给新请求。

Continuous Batching 的完整循环
整个循环在 EngineCore 的 busy loop 中运行（vllm/v1/engine/core.py）：

                    ┌──────────────────────────────────────┐
                    │         EngineCore busy loop          │
                    └──────────────────────────────────────┘
                                     │
        ┌────────────────────────────┼────────────────────────────┐
        ▼                            ▼                            ▼
  ① schedule()              ② execute_model()          ③ update_from_output()
  ┌──────────────┐         ┌──────────────┐         ┌──────────────────┐
  │ 遍历 running  │         │ GPU forward  │         │ 追加新 tokens     │
  │ 每个 decode   │         │   + sample   │         │ 检查 stop 条件    │
  │ 需要 1 token  │   ──►   │              │   ──►   │ 完成的请求:       │
  │              │         │              │         │   释放 KV blocks  │
  │ 遍历 waiting  │         │              │         │   从 running 移除 │
  │ 加入新 prefill │         │              │         │ 新请求到达:       │
  │ 直到 budget=0 │         │              │         │   加入 waiting    │
  └──────────────┘         └──────────────┘         └──────────────────┘
        │                                                     │
        └─────────────────── 下一个 step ◄────────────────────┘
每一轮 step：

schedule()：先给 running 的 decode 请求各分配 1 token 的 budget，再用剩余 budget 接纳 waiting 的新请求做 prefill
execute_model()：GPU 执行一次 forward pass，所有 prefill 和 decode 的 tokens 混合在一个 batch 里
update_from_output()：处理生成的 tokens，完成的请求立即释放资源，腾出的 slots 在下一个 step 马上可用
关键数据结构
结构	文件	作用
self.running: list[Request]	scheduler.py	正在生成的请求列表
self.waiting: RequestQueue	scheduler.py	等待调度的新请求队列
token_budget: int	schedule() 内	本轮可分配的 token 总预算
num_scheduled_tokens: dict	schedule() 内	每个请求本轮分配的 token 数
Request.num_computed_tokens	request.py	已计算的 token 数
Request.num_tokens_with_spec	request.py	需要计算到的目标 token 数
与 Static Batching 的关键区别
Static Batching:                    Continuous Batching (vLLM):
┌─────────────────────┐             ┌─────────────────────┐
│ Req A: ████████████  │             │ Req A: ████████  ←完成,释放
│ Req B: ████░░░░░░░░  │             │ Req B: ████████████████
│ Req C: ██████░░░░░░  │             │ Req C: ██████████
│                     │             │ Req D:     ████████████  ←新加入
│ 必须等 A 完成才能    │             │ Req E:         ████████  ←新加入
│ 释放整个 batch       │             │ 每个 step 动态调整     │
└─────────────────────┘             └─────────────────────┘
░ = GPU 空闲浪费                     没有浪费，GPU 始终满载
核心就在 vllm/v1/core/sched/scheduler.py 的 schedule() + update_from_output() 两个方法，配合 vllm/v1/engine/core.py 的 busy loop 驱动整个循环。
'''

Q-3:在上面说的Continuous Batching 时，优先调度running中的请求，如果一直没有足够多的显存，新请求会一直等待，这样是不是就会影响请求的TTFT，针对特殊场景（要求TTFT低，TPOT高无所谓），这不就有问题了吗？
A-3:
'''
问题根源
你观察到的问题确实存在。在 schedule() 的代码中：

# 第一步: 先调度 RUNNING 请求 (decode)
while req_index < len(self.running) and token_budget > 0:
    ...
    token_budget -= num_new_tokens  # decode 请求吃掉 budget
# 第二步: 用剩余 budget 调度 WAITING 请求 (prefill)
while self.waiting and token_budget > 0:
    ...  # 如果 budget 被 decode 吃完了，新请求进不来
如果 running 队列中有大量 decode 请求，它们会优先消耗 token_budget。极端情况下，decode 请求把 budget 吃光，新请求永远进不了 batch，TTFT 无限增长。

vLLM 的解决方案（从源码层面分析）
vLLM 提供了多层机制来缓解这个问题：

方案 1: max_num_batched_tokens — token 预算远大于 decode 消耗
这是最基础的控制手段。

token_budget = self.max_num_scheduled_tokens
# 默认值 = max_num_batched_tokens，通常设为 2048~8192
关键洞察：decode 阶段每个请求只消耗 1 个 token。假设有 256 个并发 decode 请求，它们只消耗 256 token budget。如果 max_num_batched_tokens=2048，还剩 1792 个 token budget 给新 prefill 请求。所以在实践中，decode 请求很难真正吃光 budget。

token_budget = 2048
256 个 decode 请求 × 1 token = 256 consumed
剩余 1792 tokens → 足够一个 1792 token 的 prefill
方案 2: Chunked Prefill — 让 prefill 和 decode 混合在同一个 batch
vllm serve model --enable-chunked-prefill
启用后，长 prompt 不需要等一整块 budget，而是可以分 chunk 插入到 decode batch 中：

# schedule() 中的关键代码：
num_new_tokens = request.num_tokens - num_computed_tokens
num_new_tokens = min(num_new_tokens, token_budget)  # chunk 到当前剩余 budget
# 即使 budget 只剩 512，也能处理 prompt 的前 512 个 token
效果：新请求不用等所有 decode 完成，而是在每个 step 中"夹塞"部分 prefill tokens，逐步完成 prefill。

Step 1: [decode×256 (256 tokens)] + [new_req prefill chunk (1792 tokens)]
Step 2: [decode×256 (256 tokens)] + [new_req prefill chunk (1792 tokens)]
Step 3: [decode×256 (256 tokens)] + [new_req decode (1 token)]  ← prefill 完成
方案 3: long_prefill_token_threshold — 拆分超长 prefill
源码中的关键逻辑：

if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
    num_new_tokens = self.scheduler_config.long_prefill_token_threshold
这个参数限制单次 prefill 的最大 token 数。假设设为 1024，一个 4096 token 的 prompt 会被分 4 个 step 完成 prefill，避免单个 prefill 独占 GPU。

方案 4: inter_prefill_budget (新特性) — 防止 Prefill 之间的队头阻塞
这是 2026 年的新 PR，专门解决多个 prefill 请求互相阻塞的问题：

vllm serve model --inter-prefill-budget 1024
问题场景：当 max_num_batched_tokens=4096 且同时来了 2 个 1024-token 请求，调度器会把它们放到同一个 batch。但实际上 1024 token 就够让 GPU 饱和了，第二个请求白白等了一倍时间。

解决：限制单个 batch 中给 prefill 分配的 budget，每个 step 只处理一个 prefill 请求，避免 GPU 已饱和时还塞更多 prefill。

实测效果：在 Gemma-3-27B 上 TTFT 降低 37%，TPOT 仅增加约 5-10%。

方案 5: max_num_seqs — 限制最大并发请求数
if len(self.running) == self.max_num_running_reqs:
    break  # 达到上限，不再接纳新请求
间接保护了 token budget — 如果 running 数有上限，decode 消耗的 budget 就有上限，必然留出 prefill 空间。

方案 6: 抢占（Preemption）— KV cache 不够时回收 decode 请求
# KV cache 分配失败时
if new_blocks is None:
    preempted_req = self.running.pop()  # 抢占最低优先级的 running 请求
    self._preempt_request(preempted_req)  # 释放其 KV cache
    token_budget += num_scheduled_tokens.pop(preempted_req_id)  # 归还 budget
如果是显存（KV cache blocks）不够而不是 token budget 不够，调度器会抢占正在 decode 的低优先级请求来给新请求腾位置。

你说的场景：要求低 TTFT、TPOT 不敏感
针对这个场景的推荐配置：

vllm serve model \
  --enable-chunked-prefill \                    # 允许 prefill 和 decode 混合
  --max-num-batched-tokens 4096 \               # 大 budget 确保 prefill 有空间
  --max-num-seqs 128 \                          # 限制 decode 并发数
  --long-prefill-token-threshold 1024 \         # 拆分长 prefill
  --inter-prefill-budget 1024                   # 防止 prefill 互相阻塞
或者终极方案 — PD 分离：

# Prefill 节点：只负责处理 prompt，不做 decode
vllm serve model --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer"}'
# Decode 节点：只负责 decode，不做 prefill
vllm serve model --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer"}'
这样 Prefill 节点没有任何 decode 请求竞争资源，新请求一来立刻处理，TTFT 最优。

总结
问题	机制	代码位置
decode 吃光 token budget	max_num_batched_tokens 大于 decode 总消耗	scheduler.py token_budget
长 prefill 独占 GPU	Chunked Prefill 分批处理	scheduler.py enable_chunked_prefill
超长 prompt 阻塞	long_prefill_token_threshold 拆分	scheduler.py 第 402/659 行
多个 prefill 互相阻塞	inter_prefill_budget 限制	PR #33743 (新特性)
KV cache 显存不足	抢占 decode 请求归还 blocks	scheduler.py _preempt_request()
极致 TTFT	PD 分离，Prefill 独立节点	kv_transfer/kv_connector/
你的直觉完全正确 — "先 decode 后 prefill"的默认策略确实存在 TTFT 风险。但 vLLM 通过上述多层机制，在实际部署中让 TTFT 保持可控。而当业务对 TTFT 极度敏感时，PD 分离就是从架构层面彻底解决这个矛盾的方案。

'''

Q-4:能不能详细说下上面方案4，怎么理解说的问题场景，以及inter_prefill_budget是怎么解决的？在 Gemma-3-27B 上 TTFT 降低 37%，TPOT 仅增加约 5-10%--这个结果有没有链接给我看看，这个新的PR链接有没有
A-4:
'''
方案 4: inter_prefill_budget 详解
问题场景是什么？
先理解核心矛盾。假设这样的配置：

max_num_batched_tokens = 4096  (为了高吞吐量设的比较大)
现在同时来了 2 个请求，每个 prompt 长度 1024 tokens。

没有 inter_prefill_budget 时的调度行为：

Step 1 的 schedule():
  token_budget = 4096
  RUNNING 队列: 空 (没有正在 decode 的请求)
  WAITING 队列: [Req_A(1024), Req_B(1024)]
  → 调度 Req_A: token_budget = 4096 - 1024 = 3072  ✓
  → 调度 Req_B: token_budget = 3072 - 1024 = 2048  ✓
  → 两个请求被塞进同一个 batch!
问题在于：GPU 处理 1024 个 prefill tokens 时已经计算饱和了（prefill 是 compute-bound）。再往 batch 里塞第二个 1024-token 请求，GPU 不会更快，只会线性增加计算时间。

结果：

┌─────────────────────────────────────┐
│ 没有 inter_prefill_budget:          │
│                                     │
│ Step 1: [Req_A 1024tok + Req_B 1024tok] = 2048 tokens prefill
│         GPU 已饱和，计算时间 ≈ 2秒 (而非1秒)
│                                     │
│ Req_A TTFT = 2秒  ← 被 Req_B 拖慢了!│
│ Req_B TTFT = 2秒                    │
│ 平均 TTFT = 2秒                     │
└─────────────────────────────────────┘
Req_A 本来 1 秒就能出第一个 token，但因为和 Req_B 塞在同一个 batch 里，硬生生等了 2 秒。Req_A 白白多等了 1 秒，而这 1 秒并没有带来任何吞吐量提升（因为 GPU 在 1024 tokens 时就饱和了）。

inter_prefill_budget 怎么解决？
vllm serve model --inter-prefill-budget 1024
它做的事很简单：限制每个 step 分配给 prefill 请求的 token 预算上限。

Step 1 的 schedule():
  token_budget = 4096
  inter_prefill_budget = 1024  ← 新约束
  
  → 调度 Req_A: 消耗 1024 tokens
  → prefill 已用 1024 ≥ inter_prefill_budget(1024)，停止接纳更多 prefill!
  → Req_B 留在 WAITING 队列
Step 1 执行: [Req_A 1024tok] → GPU 刚好饱和, 耗时 ≈ 1秒
  Req_A TTFT = 1秒 ✓
Step 2 的 schedule():
  → Req_A 进入 RUNNING 队列 (decode, 只需 1 token)
  → Req_B 从 WAITING 被调度 (prefill 1024 tokens)
Step 2 执行: [Req_A decode(1tok) + Req_B prefill(1024tok)] ≈ 1秒
  Req_B TTFT = 2秒 (等了1秒 + 处理1秒)
结果：

┌─────────────────────────────────────┐
│ 有 inter_prefill_budget = 1024:     │
│                                     │
│ Step 1: [Req_A prefill 1024tok]     │
│ Step 2: [Req_A decode 1tok + Req_B prefill 1024tok] │
│                                     │
│ Req_A TTFT = 1秒  ← 快了1秒!       │
│ Req_B TTFT = 2秒  (和之前一样)      │
│ 平均 TTFT = 1.5秒  ← 降低了25%     │
└─────────────────────────────────────┘
为什么 TPOT 只会略微增加？
没有 inter_prefill_budget:
  Step 1: [Req_A prefill + Req_B prefill]  → 2秒
  Step 2: [Req_A decode + Req_B decode]    → 很快 (2 tokens)
  Step 3: [Req_A decode + Req_B decode]    → 很快
有 inter_prefill_budget:
  Step 1: [Req_A prefill]                  → 1秒
  Step 2: [Req_A decode + Req_B prefill]   → ≈ 1秒 ← Req_A 的 decode 和 Req_B 的 prefill 混在一起
  Step 3: [Req_A decode + Req_B decode]    → 很快
Step 2 中 Req_A 的 decode（1 个 token）被 Req_B 的 prefill（1024 tokens）拖慢了——这就是 TPOT 增加的来源。但因为：

decode 本身是 memory-bound，开销极小
只是偶尔有一两个 step 受影响，平摊后影响很小
输出 token 越多，这个开销越被稀释
所以 TPOT 增幅只有 5-10%。

PR 链接和 benchmark 数据
PR 链接：https://github.com/vllm-project/vllm/pull/33743

PR 标题：[Core] Reduce median/average TTFT by up to ~37% with Inter-Prefill-Budget

下面是 PR 中的实测数据（单 A100，Gemma-3-27B-IT）：

场景 (Input/Output/请求数)	指标	基线 (v0.15.0)	加 budget(2048)	变化
1024 / 350 / 16	Median TTFT	4213ms	2648ms	-37%
400 / 400 / 20	Median TTFT	2020ms	1319ms	-35%
50 / 150 / 100	Median TTFT	1269ms	1194ms	-6%
1024 / 350 / 16	Mean TPOT	46ms	51ms	+10%
400 / 400 / 20	Mean TPOT	45ms	47ms	+4%
50 / 150 / 100	Mean TPOT	62ms	63ms	+1%
可以看到：

长 prompt (1024 tokens) 场景效果最明显，TTFT 降 37%，TPOT 仅增 10%
短 prompt (50 tokens) 场景效果较小（因为短 prompt 不太会饱和 GPU）
注意：这个 PR 状态目前仍是 open（未合并），还在 review 中，但基准测试数据是可靠的。
'''

Q-5:我怎么知道GPU计算token的饱和点是多大？针对不同的芯片，比如ascend 910B、A100、4090等卡，怎么得到这个饱和点呢？
A-5:
'''
这是一个非常实际且关键的问题。GPU 的"prefill 饱和点"取决于模型大小、GPU 算力和显存带宽的交叉点。让我从原理和实测两个维度来解答。

原理：为什么会有"饱和点"？
Prefill 阶段的计算本质是矩阵乘法（GEMM），其计算量和 token 数成正比：

计算量 (FLOPs) ≈ 2 × num_params × num_tokens
而 GPU 有两个物理极限：

计算上限 = GPU FLOPS (算力天花板)
带宽上限 = HBM Bandwidth (显存带宽天花板)
当 token 数少时：矩阵太小，GPU 的大量计算单元闲置，瓶颈在内存读取（memory-bound） 当 token 数多时：矩阵足够大，计算单元满载，瓶颈在算力（compute-bound）

饱和点就是从 memory-bound 转变为 compute-bound 的临界 token 数。超过这个点，再加 token 只会线性增加计算时间，不会提高 GPU 利用率。

理论计算方法
关键指标是 Arithmetic Intensity（算术强度），也叫 operational intensity：

算术强度 = FLOPs / Bytes_transferred
GPU 的分界线叫 ops:byte ratio：

ops:byte ratio = GPU_FLOPS / HBM_Bandwidth
GPU	BF16 FLOPS	HBM 带宽	ops:byte ratio
A100 80GB	312 TFLOPS	2.0 TB/s	156
H100 SXM	989 TFLOPS	3.35 TB/s	295
RTX 4090	165 TFLOPS (BF16)	1.01 TB/s	163
Ascend 910B	256 TFLOPS (BF16)	1.18 TB/s	217
对于 Transformer 的 prefill，单层的 GEMM 算术强度大约为：

arithmetic_intensity ≈ num_tokens (对于大模型的线性层)
所以理论饱和点大约就是：

饱和 token 数 ≈ ops:byte ratio
GPU	理论饱和点 (大约)
A100 80GB	~150-200 tokens
H100 SXM	~300 tokens
RTX 4090	~160-200 tokens
Ascend 910B	~200-250 tokens
这意味着对于 A100，大约 150-200 个 prefill tokens 就能让 GPU 计算单元接近满载。

但理论值偏低，实际更高
实际饱和点通常高于理论值，因为：

Attention 的 GEMM 形状不规则（Q×K^T 矩阵不如线性层那么大）
存在 kernel launch overhead、内存碎片
FlashAttention 等优化的实际效率
PyTorch/CUDA 的软件栈开销
实际经验值通常在 512-2048 tokens 范围。

实测方法（最可靠）
方法 1：vLLM 自带 benchmark 工具
# 测不同 prefill 长度的吞吐量
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 20 \
    --request-rate inf
done
观察输出的 token throughput (tok/s)。当 throughput 随 INPUT_LEN 增大而不再增长（曲线变平），那个拐点就是饱和点。

方法 2：写脚本直接测 prefill 时间
import time
import torch
from vllm import LLM, SamplingParams
model = LLM(model="your-model", max_model_len=8192)
sampling = SamplingParams(max_tokens=1)  # 只生成1个token，只看prefill
results = []
for num_tokens in [32, 64, 128, 256, 512, 1024, 2048, 4096]:
    prompt_ids = list(range(num_tokens))  # dummy tokens
    
    start = time.perf_counter()
    model.generate({"prompt_token_ids": prompt_ids}, sampling)
    elapsed = time.perf_counter() - start
    
    throughput = num_tokens / elapsed
    results.append((num_tokens, elapsed, throughput))
    print(f"tokens={num_tokens:5d}  time={elapsed:.4f}s  "
          f"throughput={throughput:.0f} tok/s  "
          f"time_per_token={elapsed/num_tokens*1000:.3f} ms/tok")
你会看到类似这样的输出：

tokens=   32  time=0.0150s  throughput=  2133 tok/s  time_per_token=0.469 ms/tok
tokens=   64  time=0.0155s  throughput=  4129 tok/s  time_per_token=0.242 ms/tok
tokens=  128  time=0.0162s  throughput=  7901 tok/s  time_per_token=0.127 ms/tok
tokens=  256  time=0.0180s  throughput= 14222 tok/s  time_per_token=0.070 ms/tok
tokens=  512  time=0.0230s  throughput= 22261 tok/s  time_per_token=0.045 ms/tok  ← 吞吐还在涨
tokens= 1024  time=0.0380s  throughput= 26947 tok/s  time_per_token=0.037 ms/tok  ← 接近饱和
tokens= 2048  time=0.0720s  throughput= 28444 tok/s  time_per_token=0.035 ms/tok  ← 基本饱和
tokens= 4096  time=0.1410s  throughput= 29049 tok/s  time_per_token=0.034 ms/tok  ← 完全饱和
看 time_per_token：当这个值不再下降时，就是饱和点。上例中大约在 1024-2048 tokens 处饱和。

方法 3：用 roofline model 工具
# 使用 NVIDIA Nsight Compute profiling
ncu --set full -o profile python your_script.py
# 然后在 Nsight Compute GUI 中查看 Roofline 图
# 可以直观看到每个 kernel 在 compute-bound 还是 memory-bound 区域

不同模型 × 不同 GPU 的经验参考值
GPU	7B 模型	13B 模型	70B 模型 (TP=4)
A100 80GB	~512-1024	~256-512	~128-256
H100 SXM	~1024-2048	~512-1024	~256-512
RTX 4090	~512-1024	N/A (显存不够)	N/A
Ascend 910B	~512-1024	~256-512	~128-256
规律：

模型越大 → 每层的 GEMM 矩阵越大 → 更少的 tokens 就能填满计算单元 → 饱和点更低
GPU 算力越强 → 需要更多 tokens 才能喂饱 → 饱和点更高
TP 并行 → 每卡的矩阵变小 → 和模型变小类似 → 饱和点稍高

回到 inter_prefill_budget 的设置建议
知道饱和点后，inter_prefill_budget 的最优设置就是：
--inter-prefill-budget <你实测的饱和点>

例如 A100 + 27B 模型，饱和点大约在 512-1024，那就设 --inter-prefill-budget 1024。这样每个 step 最多给 prefill 分配 1024 tokens，刚好填满 GPU 算力，多余的 prefill 请求留到下个 step，既不浪费算力又降低 TTFT。
'''

Q-6:vLLM 自带 benchmark 工具测试获取GPU计算token的饱和点时，--num-prompts怎么设置？确定好这个饱和点后，vllm启动参数中，--max-num-batched-tokens又怎么设置，与--inter-prefill-budget有没有制约关系？
A-6:
'''
1. --num-prompts 怎么设置
测饱和点时，--num-prompts 的目的是确保同时有足够多请求同时打到 GPU。关键原则：

必须确保请求是"同时到达"的，否则前一个做完了后一个才来，永远只有 1 个请求在 GPU 上，测的是单请求延迟而非饱和行为。

# 关键参数: --request-rate inf  ← 所有请求同时发出
vllm bench serve \
  --model your-model \
  --dataset-name random \
  --random-input-len 512 \
  --random-output-len 1 \
  --num-prompts 1 \          # ← 这里设 1
  --request-rate inf
测饱和点时 --num-prompts 应该设为 1。

原因：你测的是 "多少个 prefill tokens 能让 GPU 饱和"，不是 "多少个请求能让 GPU 饱和"。只需要 1 个请求，改变它的 --random-input-len 就够了。

# 正确的测法: 固定 1 个请求，变化 input 长度
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 1 \
    --request-rate inf
done
但更准确的做法是多跑几次取平均，所以设 5-10 个（串行执行，避免并发干扰）：

for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 5 \
    --request-rate 0.5
    # request-rate=0.5 表示每秒发0.5个请求(2秒一个)
    # 确保前一个跑完后一个才来，不互相干扰
done
然后看输出中的 Mean TTFT，当 TTFT / INPUT_LEN（每 token 耗时）不再下降时就是饱和点。

2. --max-num-batched-tokens 怎么设置
这是 vLLM 调度器每个 step 允许处理的最大 token 总数（包含所有请求的 prefill + decode tokens）。

它的作用
# scheduler.py 中
token_budget = self.max_num_scheduled_tokens  # 就是 max_num_batched_tokens
每个 step：

所有 running 请求的 decode tokens + 新请求的 prefill tokens ≤ max_num_batched_tokens
设置原则
max_num_batched_tokens = max_concurrent_decode + 最大单次prefill_tokens
实际推荐值：

场景	推荐值	理由
低延迟优先	2048-4096	够用就行，避免 batch 太大增加延迟
高吞吐优先	8192-16384	大 batch 提高 GPU 利用率
超长上下文	32768+	适配长 prompt
默认值 (V1)	8192	vLLM 默认
一个具体的推算示例
假设：

A100 80GB，部署 Llama-3.1-8B
目标：同时支持 256 个并发 decode + 新请求 prefill
测出的饱和点：~1024 tokens
每个 step 的 token 分布：
  256 个 decode 请求 × 1 token = 256 tokens
  新请求 prefill（希望一次做完 2048 tokens 的 prompt）= 2048 tokens
  总计 = 256 + 2048 = 2304 tokens
→ max_num_batched_tokens 至少设 2304，建议设 4096 留余量
3. max_num_batched_tokens 与 inter_prefill_budget 的制约关系
这两个参数之间有明确的层级关系：

max_num_batched_tokens (总预算)
├── decode tokens 消耗 (running 队列，优先)
└── prefill tokens 消耗 (waiting 队列)
    └── 受 inter_prefill_budget 进一步限制
制约关系图
                max_num_batched_tokens = 4096 (总上限)
                ┌──────────────────────────────────────┐
                │                                      │
  ┌─────────────┴──┐              ┌────────────────────┴───┐
  │  decode tokens  │              │   prefill tokens        │
  │  (动态, 按需)    │              │   (受两层限制)            │
  │  如: 256 tokens │              │                          │
  └────────────────┘              │  上限1: token_budget 剩余  │
                                  │        = 4096 - 256 = 3840│
                                  │                          │
                                  │  上限2: inter_prefill_budget│
                                  │        = 1024 (如果设了)   │
                                  │                          │
                                  │  实际 = min(3840, 1024)   │
                                  │        = 1024             │
                                  └──────────────────────────┘
也就是说 inter_prefill_budget 是 max_num_batched_tokens 的子约束：

实际可用的 prefill budget = min(
    max_num_batched_tokens - decode已消耗的tokens,
    inter_prefill_budget   # 如果设了的话
)
必须满足的约束
inter_prefill_budget ≤ max_num_batched_tokens
如果 inter_prefill_budget > max_num_batched_tokens，那 inter_prefill_budget 等于没设。

不同配置组合的效果
max_num_batched_tokens	inter_prefill_budget	效果
4096	不设	所有 prefill 可用 4096-decode 消耗的全部 budget。多个 prefill 可能挤在一个 batch
4096	1024	每个 step 最多给 prefill 分 1024 tokens。GPU 刚好饱和，不浪费
4096	4096	等于没设 inter_prefill_budget
2048	1024	prefill 最多 1024，总 batch 最多 2048
1024	1024	非常保守，decode 几乎没 budget（不推荐）
推荐配置
# 低 TTFT 场景
vllm serve model \
  --max-num-batched-tokens 4096 \
  --inter-prefill-budget 1024 \      # = GPU饱和点
  --enable-chunked-prefill
# 高吞吐场景
vllm serve model \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill
  # 不设 inter-prefill-budget，让 GPU 满载
实际决策流程
1. 先测你的 GPU + 模型的饱和点 → 得到 S (如 1024)
2. 估算最大并发 decode 数 → 得到 D (如 256)
3. 设置:
   inter_prefill_budget = S                     (如 1024)
   max_num_batched_tokens = D + S × 2           (如 256 + 2048 = 2304, 向上取到 4096)
                           ↑
                    留 2 倍余量给 chunked prefill 和波动
4. 如果 TTFT 不敏感，不设 inter_prefill_budget
   如果 TTFT 敏感，inter_prefill_budget = S
简而言之：max_num_batched_tokens 是总水管口径，inter_prefill_budget 是 prefill 分支上的阀门，阀门不能比水管粗。
'''
Q-7:
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 5 \
    --request-rate 0.5
    # request-rate=0.5 表示每秒发0.5个请求(2秒一个)
    # 确保前一个跑完后一个才来，不互相干扰
done
怎么理解“# request-rate=0.5 表示每秒发0.5个请求(2秒一个)
    # 确保前一个跑完后一个才来，不互相干扰”
如果我不知道prefill处理多久，那我是不是尽量把--request-rate设置更小，这样就能确保每次处理的都是一个请求，这样就可以测试这个饱和点？
A-7:
'''
--request-rate 的含义
--request-rate 控制的是请求发送速率，单位是 请求数/秒，遵循泊松分布：

--request-rate 0.5  → 平均每秒发 0.5 个请求 → 平均 2 秒发 1 个
--request-rate 0.1  → 平均每秒发 0.1 个请求 → 平均 10 秒发 1 个
--request-rate inf  → 所有请求同时发出
为什么要控制速率？
测饱和点的目标是：测量"单个请求独占 GPU 时，不同 prompt 长度的 prefill 耗时"。

如果两个请求同时在 GPU 上，它们会共享算力，测出来的时间不是单请求的真实 prefill 时间，而是被干扰后的时间。

你的思路完全正确
如果我不知道 prefill 处理多久，那我是不是尽量把 --request-rate 设置更小？

**对的。**设得越小，越能保证每个请求独占 GPU，测出的是纯净的单请求 prefill 时间。

但也没必要设得太离谱（比如 0.01），否则整个测试耗时太长。实际操作中，可以这样做：

最简单的方法：直接设 --request-rate inf + --num-prompts 1
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  echo "=== INPUT_LEN=$INPUT_LEN ==="
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 1 \
    --request-rate inf
done
每轮只发 1 个请求，不存在并发干扰的问题。--request-rate inf 在只有 1 个请求时没有影响。

这是最干净的测法，但只有 1 次采样，结果可能有波动。

更稳定的方法：多次采样但确保串行
for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  echo "=== INPUT_LEN=$INPUT_LEN ==="
  # 跑 5 轮，每轮 1 个请求
  for i in 1 2 3 4 5; do
    vllm bench serve \
      --model your-model \
      --dataset-name random \
      --random-input-len $INPUT_LEN \
      --random-output-len 1 \
      --num-prompts 1 \
      --request-rate inf
  done
done
每个 INPUT_LEN 跑 5 次，取平均值，完全没有并发干扰。

或者用低速率批量发
如果不想跑那么多次命令，用低速率也可以：

for INPUT_LEN in 32 64 128 256 512 1024 2048 4096; do
  echo "=== INPUT_LEN=$INPUT_LEN ==="
  vllm bench serve \
    --model your-model \
    --dataset-name random \
    --random-input-len $INPUT_LEN \
    --random-output-len 1 \
    --num-prompts 5 \
    --request-rate 0.05
done
--request-rate 0.05 表示平均 20 秒发 1 个请求。即使是 4096 tokens 的 prefill 在任何卡上也远不到 20 秒，所以能确保串行。

最终看什么指标判断饱和点
看输出中的 Mean TTFT，然后算 每 token 耗时：

time_per_token = Mean_TTFT / INPUT_LEN
INPUT_LEN	Mean TTFT (ms)	time_per_token (ms)	状态
32	15	0.469	memory-bound，GPU 没吃饱
64	16	0.250	memory-bound
128	17	0.133	memory-bound
256	20	0.078	开始接近饱和
512	28	0.055	接近饱和
1024	48	0.047	基本饱和 ← 拐点
2048	92	0.045	完全饱和
4096	182	0.044	完全饱和
当 time_per_token 不再明显下降，那个 INPUT_LEN 就是饱和点。上例中是 ~1024 tokens。
'''

Q-8:开启chunked prefill后，处理的请求可能是部分prefill + decode的混合，模型底层是如何处理这种混合输入的？比如说，GPU forward的时候，输入里既有prefill的token，也有decode的token，这些token是怎么区分开来进行处理的？在模型层面，是不是需要特殊的机制来区分哪些token是prefill的，哪些是decode的,以确保同一个batch中，做prefill的记录KVcache，做decode的，输出下一个token
A-8:
'''
这是一个非常核心的问题。在 chunked prefill 下，模型本身 (Transformer 层) 不需要区分 prefill 和 decode。区分的关键完全在 Attention 层，通过两个核心元数据实现：query_start_loc 和 seq_lens。让我从底层详细解释。

一、核心理解：模型不区分，Attention 层区分
首先要打破一个误解：GPU forward 的时候，不存在"prefill token"和"decode token"两种不同类型。从 Transformer 模型的大部分层（Embedding、FFN/MLP、LayerNorm、RMSNorm 等）来看，所有 token 都是完全一样的 —— 都是一个 hidden_state 向量，经过相同的矩阵运算。

唯一不同的地方是 Attention 层，因为：

Prefill token：需要 attend 到同一请求中它前面的所有 token（因果注意力），并且需要把自己的 KV 写入 KV Cache
Decode token：只有 1 个 query token，需要 attend 到 KV Cache 中该请求的所有历史 KV，并且需要把自己的 KV 写入 KV Cache
二、混合 Batch 的数据布局
当 Scheduler 调度了一个混合 batch，例如：

请求 A：prefill 阶段，本次调度 512 个 token（可能是 chunked，总共 2048 但这次只做 512）
请求 B：decode 阶段，本次调度 1 个 token
请求 C：prefill 阶段，本次调度 256 个 token
所有 token 被拼接为一个一维张量（flattened 1D），送入模型：

输入 token_ids: [A_0, A_1, ..., A_511, B_0, C_0, C_1, ..., C_255]
                |--- 512 tokens ---|  |1|  |--- 256 tokens ---|
                    请求A (prefill)    请求B    请求C (prefill)
                                     (decode)
total_num_scheduled_tokens = 512 + 1 + 256 = 769
gpu_model_runner.py 中 _prepare_inputs 构建这个布局的关键代码：


5e2600f4-d433-46a6-ba25-0761c2a6f70a.txt
Lines 1801-1809
        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)
        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # self.query_pos.np[:10]: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens = self._get_cumsum_and_arange(
            num_scheduled_tokens, self.query_pos.np
        )
三、两个关键元数据：如何让 Attention 层区分
Attention 层靠 CommonAttentionMetadata 中的两个核心字段来区分：

1. query_start_loc — 每个请求的 query 在 batch 中的起始位置
query_start_loc = [0, 512, 513, 769]
                   ^    ^    ^    ^
                   A起点 B起点 C起点 总结束
通过 query_start_loc[i+1] - query_start_loc[i] 就知道每个请求的 query 长度：

请求 A：512（prefill，多个 query token）
请求 B：1（decode，单个 query token）
请求 C：256（prefill，多个 query token）
2. seq_lens — 每个请求的完整上下文长度（含历史 KV）
seq_lens = [512, 1025, 256]
            ^     ^     ^
         A本次共512  B已有1024+本次1=1025  C本次共256
这是关键差异：

请求 A（prefill）：query_len=512, seq_len=512, 它们相等（或近似），说明是 prefill
请求 B（decode）：query_len=1, seq_len=1025, query_len 远小于 seq_len，说明大量历史 KV 在缓存中
请求 C（prefill）：query_len=256, seq_len=256
构建代码：


5e2600f4-d433-46a6-ba25-0761c2a6f70a.txt
Lines 1893-1899
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1 : num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[: num_reqs + 1]

5e2600f4-d433-46a6-ba25-0761c2a6f70a.txt
Lines 1995-1998
        self.seq_lens[:num_reqs] = (
            self.num_computed_tokens[:num_reqs] + num_scheduled_tokens_gpu
        )
        self.seq_lens[num_reqs:].fill_(0)
四、FlashAttention 如何处理混合 Batch
vLLM V1 使用的是 flash_attn_varlen_func（variable length），它原生支持混合长度的 batch。看实际调用代码：


0e51d1a1-8d24-42d1-94f7-08318365031e.txt
Lines 738-761
                    flash_attn_varlen_func(
                        q=query[:num_actual_tokens],
                        k=key_cache,
                        v=value_cache,
                        out=output[:num_actual_tokens],
                        cu_seqlens_q=cu_seqlens_q,
                        max_seqlen_q=max_seqlen_q,
                        seqused_k=seqused_k,
                        max_seqlen_k=max_seqlen_k,
                        softmax_scale=self.scale,
                        causal=attn_metadata.causal,
                        alibi_slopes=self.alibi_slopes,
                        window_size=sliding_window_size,
                        block_table=block_table,
                        softcap=self.logits_soft_cap,
                        scheduler_metadata=scheduler_metadata,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=q_descale,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        num_splits=attn_metadata.max_num_splits,
                        s_aux=self.sinks,
                    )
                    return output
注意关键参数：

q=query[:num_actual_tokens]：所有 769 个 query token 拼在一起
k=key_cache, v=value_cache：这是整个 Paged KV Cache，不是当前 batch 的 KV
cu_seqlens_q=query_start_loc：[0, 512, 513, 769] 告诉 FlashAttention 每个请求的 query 边界
seqused_k=seq_lens：[512, 1025, 256] 告诉 FlashAttention 每个请求要 attend 的 KV 长度
block_table：告诉 FlashAttention 每个请求的 KV 存在 KV Cache 的哪些 block 里
causal=True：因果注意力 mask
FlashAttention 内部的处理逻辑：

FlashAttention varlen kernel 内部对每个请求独立处理：

对于请求 A (prefill, query_len=512, seq_len=512):
  - 512 个 query token，每个只 attend 到它之前的 token（因果 mask）
  - Q 来自当前输入，K/V 从 KV Cache 中的 block_table[A] 指向的 block 读取
对于请求 B (decode, query_len=1, seq_len=1025):
  - 1 个 query token，attend 到 1025 个 KV token
  - Q 来自当前输入，K/V 从 KV Cache 中的 block_table[B] 读取
对于请求 C (prefill, query_len=256, seq_len=256):
  - 256 个 query token，每个只 attend 到它之前的 token（因果 mask）
FlashAttention 根本不需要知道哪个是 "prefill"，哪个是 "decode"。 它只看 query_len 和 seq_len 的关系，自动按因果 mask 处理。

五、KV Cache 写入：slot_mapping 的作用
在做 attention 计算之前（或之后，取决于 backend），需要把当前 batch 中所有 token 的 KV 写入 KV Cache。这通过 do_kv_cache_update 实现：


0e51d1a1-8d24-42d1-94f7-08318365031e.txt
Lines 792-823
    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return
        key_cache, value_cache = kv_cache.unbind(0)
        # Reshape the input keys and values and store them in the cache.
        # Skip this if sharing KV cache with an earlier attention layer.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )
slot_mapping 是一个长度为 769 的一维张量，对于 batch 中的每一个 token（无论 prefill 还是 decode），都精确指定了这个 token 的 KV 应该存到 KV Cache 的哪个 slot。

slot_mapping: [slot_A_0, slot_A_1, ..., slot_A_511, slot_B_0, slot_C_0, ..., slot_C_255]
reshape_and_cache_flash 是一个 CUDA kernel，把 769 个 token 的 KV 一次性写入到 KV Cache 的对应位置，不区分 prefill 还是 decode，统一写入。

六、完整执行流程图
Scheduler 调度:
  请求A: prefill chunk 512 tokens
  请求B: decode 1 token
  请求C: prefill chunk 256 tokens
                    ┌─────────────────────────────────────┐
                    │         拼接为 1D 张量                │
                    │  [A×512, B×1, C×256] = 769 tokens    │
                    └──────────────┬──────────────────────┘
                                   │
                    ┌──────────────▼──────────────────────┐
                    │       Embedding Layer               │
                    │  input: [769] token_ids              │
                    │  output: [769, hidden_dim]           │
                    │  ── 所有 token 完全相同处理 ──        │
                    └──────────────┬──────────────────────┘
                                   │
              ┌────────────────────▼────────────────────────┐
              │          每个 Transformer Layer              │
              │                                             │
              │  1. RMSNorm → 统一处理 769 个 token          │
              │  2. QKV Projection → 统一矩阵乘法            │
              │     Q: [769, num_heads, head_dim]            │
              │     K: [769, num_kv_heads, head_dim]         │
              │     V: [769, num_kv_heads, head_dim]         │
              │                                             │
              │  3. KV Cache 写入 (reshape_and_cache_flash)  │
              │     根据 slot_mapping 把 769 个 KV           │
              │     写入 Paged KV Cache 对应位置             │
              │     ── 不区分 prefill/decode ──              │
              │                                             │
              │  4. Attention (flash_attn_varlen_func)       │
              │     ┌──────────────────────────────────┐     │
              │     │ query_start_loc=[0,512,513,769]  │     │
              │     │ seq_lens=[512, 1025, 256]        │     │
              │     │ block_table=[A的blocks,          │     │
              │     │              B的blocks,          │     │
              │     │              C的blocks]          │     │
              │     │                                  │     │
              │     │ FlashAttention 内部:              │     │
              │     │  请求A: 512 Q × 512 KV (因果)    │     │
              │     │  请求B: 1 Q × 1025 KV (全attend) │     │
              │     │  请求C: 256 Q × 256 KV (因果)    │     │
              │     │  ── 统一一个 kernel 完成 ──       │     │
              │     └──────────────────────────────────┘     │
              │                                             │
              │  5. O_proj → 统一矩阵乘法                    │
              │  6. MLP/FFN → 统一处理                       │
              └────────────────────┬────────────────────────┘
                                   │  × N layers
                    ┌──────────────▼──────────────────────┐
                    │         Final RMSNorm + LM Head      │
                    │  output logits: [769, vocab_size]    │
                    │                                      │
                    │  但只采样最后一个 token:               │
                    │    请求A: logits[511] → 采样下一token  │
                    │    请求B: logits[512] → 采样下一token  │
                    │    请求C: logits[768] → 采样下一token  │
                    └──────────────────────────────────────┘
七、总结
环节	是否区分 prefill/decode	机制
Embedding	不区分	统一查表
QKV Projection	不区分	统一矩阵乘法
KV Cache 写入	不区分	slot_mapping 指定每个 token 写入位置
Attention 计算	隐式区分	query_start_loc 确定 query 长度，seq_lens 确定 KV 长度，block_table 定位 KV；FlashAttention varlen kernel 按因果 mask 统一计算
MLP/FFN	不区分	统一矩阵乘法
采样	不区分	只取每个请求的最后一个 token 的 logits
核心设计哲学：vLLM 不需要显式标记"这是 prefill token""这是 decode token"。
它通过 query_start_loc（每个请求多少个新 query）和 seq_lens（每个请求总共要 attend 多长的 KV）这两个元数据，让 FlashAttention 的 variable-length kernel 自然地、在同一个 CUDA kernel 调用中处理混合 batch。这就是 chunked prefill 能高效工作的根本原因。
'''

Q-9:投机推理是怎样的逻辑，梳理下代码及底层work线程怎么处理的，以及大模型是怎么验证的，什么机制决定选择小模型的输出
A-9:
'''
一、投机推理的核心思想
普通自回归生成：大模型每次 forward 只产出 1 个 token，速度受限于大模型的推理延迟。

投机推理的核心洞察：大模型验证 K 个 token 的开销 ≈ 生成 1 个 token 的开销（因为 K 个 token 可以并行 forward），但如果猜对了就等于"免费"获得了多个 token。

普通生成 (5个token):  [Forward] [Forward] [Forward] [Forward] [Forward]
                      ═══════╪═══════╪═══════╪═══════╪═══════╪
                      耗时: 5 × T_large
投机推理 (5个token):  [Draft×5] [Verify一次] → 如果全对,一步出5个token
                      ═══════╪══════════╪
                      耗时: T_draft + T_large ≈ 1.x × T_large
二、vLLM 中的投机推理方法
vLLM V1 支持多种 drafter（草稿生成器）：

方法	类名	文件	原理
Draft Model	DraftModelProposer	draft_model.py	独立小模型生成草稿
EAGLE	EagleProposer	eagle.py	利用大模型 hidden states 的轻量头
EAGLE3 / DFlash	DFlashProposer	dflash.py	EAGLE 的优化变体
Medusa	MedusaProposer	medusa.py	多头并行预测
N-gram	NgramProposer	ngram_proposer.py	基于历史 token 模式匹配（无模型）
N-gram GPU	NgramProposerGPU	ngram_proposer_gpu.py	GPU 加速的 N-gram
Suffix	SuffixDecodingProposer	suffix_decoding.py	后缀树匹配
三、完整执行流程
                        一个完整的投机推理迭代
    ┌──────────────────────────────────────────────────────┐
    │                                                      │
    │  Step 1: Scheduler 调度                               │
    │  ┌─────────────────────────────────────────────┐     │
    │  │ 把上一轮 drafter 猜测的 K 个 draft token      │     │
    │  │ 连同原始 token 一起放入 scheduled_tokens       │     │
    │  │ 例如: 请求A 原本只有 1 个 decode token         │     │
    │  │       现在变成 1 + K 个 token (1原始 + K草稿)  │     │
    │  └─────────────────────────────────────────────┘     │
    │                      ↓                               │
    │  Step 2: 大模型 Forward (验证)                        │
    │  ┌─────────────────────────────────────────────┐     │
    │  │ 大模型一次 forward 处理所有 1+K 个 token       │     │
    │  │ 在每个 draft position 产生 target logits       │     │
    │  │ 这就是"验证" — 大模型告诉你它在每个位置         │     │
    │  │ 会选择什么 token                               │     │
    │  └─────────────────────────────────────────────┘     │
    │                      ↓                               │
    │  Step 3: Rejection Sampling (接受/拒绝)               │
    │  ┌─────────────────────────────────────────────┐     │
    │  │ 逐个比较 draft token vs target token           │     │
    │  │ - 匹配: 接受,继续验证下一个                     │     │
    │  │ - 不匹配: 拒绝,用 target 的 token 替换          │     │
    │  │          后续 draft 全部丢弃                    │     │
    │  │ - 全部接受: 额外获得一个 bonus token!           │     │
    │  └─────────────────────────────────────────────┘     │
    │                      ↓                               │
    │  Step 4: Draft 新一轮草稿                             │
    │  ┌─────────────────────────────────────────────┐     │
    │  │ 用 drafter 为下一次迭代猜测 K 个新 token        │     │
    │  │ (利用大模型刚产出的 hidden states / token)      │     │
    │  └─────────────────────────────────────────────┘     │
    │                      ↓                               │
    │              回到 Step 1,循环                         │
    └──────────────────────────────────────────────────────┘
四、代码层面的详细流程
Step 1: 输入准备 — 把 draft token 塞入 batch
在 gpu_model_runner.py 的 _prepare_inputs 中，当存在 scheduled_spec_decode_tokens 时，会调用 _calc_spec_decode_metadata：

# gpu_model_runner.py
def _calc_spec_decode_metadata(self, ...):
    # 计算每个请求的 draft token 数量
    # 构建 target_logits_indices: 大模型需要在哪些位置产出 logits 来验证
    # 构建 bonus_logits_indices: 如果全部接受,bonus token 从哪个位置采样
    # 构建 draft_token_ids: 每个 draft 位置对应的小模型猜测的 token
假设请求 A 有 3 个 draft token [d1, d2, d3]，输入 batch 变成：

原始 token:  [A_current]
加入 draft:  [A_current, d1, d2, d3]
大模型 forward 后产出 4 个位置的 logits:
  logits[0] → 验证 d1 (在 A_current 之后应该是什么)
  logits[1] → 验证 d2 (在 d1 之后应该是什么)
  logits[2] → 验证 d3 (在 d2 之后应该是什么)
  logits[3] → bonus (在 d3 之后应该是什么,如果全部接受)
Step 2: 大模型 Forward — 一次验证所有 draft
# gpu_model_runner.py -> execute_model
hidden_states = model(input_ids, positions, ...)  # 一次 forward
sample_hidden_states = hidden_states[logits_indices]  # 取验证位置
logits = model.compute_logits(sample_hidden_states)   # 计算 logits
关键：大模型只做一次 forward，但由于 draft token 也在输入中，大模型在每个 draft 位置都会产出一个概率分布。这就是"验证" — 不是单独跑一次验证，而是通过正常 forward 自然产出。

Step 3: Rejection Sampling — 核心验证代码
_sample 方法检测到有 spec_decode_metadata 时，调用 RejectionSampler：

# gpu_model_runner.py
def _sample(self, logits, spec_decode_metadata):
    if spec_decode_metadata is None:
        return self.sampler(logits=logits, ...)  # 普通采样
    
    # 投机推理的验证+采样
    sampler_output = self.rejection_sampler(
        spec_decode_metadata,
        draft_probs=None,     # N-gram 等无概率分布的 drafter
        logits=logits,        # 大模型的 logits
        sampling_metadata=...,
    )
    return sampler_output
RejectionSampler.forward 的核心流程：

# rejection_sampler.py -> forward
def forward(self, metadata, draft_probs, logits, sampling_metadata):
    # 1. 采样 bonus token (如果全部 draft 被接受才用到)
    bonus_logits = logits[bonus_logits_indices]
    bonus_token_ids = self.sampler(logits=bonus_logits, predict_bonus_token=True)
    
    # 2. 获取每个 draft 位置的 target logits
    target_logits = logits[target_logits_indices]  # [num_draft_tokens, vocab]
    
    # 3. 应用采样约束 (temperature, top-k, top-p)
    target_logits = apply_sampling_constraints(target_logits, ...)
    
    # 4. 执行 rejection sampling
    output_token_ids = rejection_sample(
        draft_token_ids,    # 小模型猜的 token
        target_logits,      # 大模型在每个位置的概率分布
        bonus_token_ids,    # bonus token
        ...
    )
    return SamplerOutput(sampled_token_ids=output_token_ids)
Step 4: Drafter 提出新草稿
# gpu_model_runner.py -> propose_draft_token_ids
def propose_draft_token_ids(self, ...):
    if spec_config.use_eagle():
        # EAGLE: 利用大模型 hidden_states 驱动轻量头
        draft_token_ids = self.drafter.propose(
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,  # 关键!
            next_token_ids=next_token_ids,
            ...
        )
    elif spec_config.uses_draft_model():
        # 独立小模型
        draft_token_ids = self.drafter.propose(
            target_token_ids=...,
            target_hidden_states=None,  # 不传 hidden states
            ...
        )
    elif "ngram" in method:
        # N-gram: 纯模式匹配,不用模型
        draft_token_ids = self.drafter.propose(self.input_batch, ...)
五、两种验证机制的详细算法
机制 A: Greedy 验证 (temperature=0)
对应 Triton kernel rejection_greedy_sample_kernel：

对于每个 draft 位置 pos = 0, 1, ..., K-1:
    draft_token = 小模型猜的 token
    target_token = argmax(大模型在该位置的 logits)  ← 大模型会选什么
    
    if draft_token == target_token:
        接受! output[pos] = target_token
    else:
        拒绝! output[pos] = target_token  ← 用大模型的替换
        后续所有 draft 全部丢弃,结束
        
if 全部接受:
    output[K] = bonus_token  ← 额外白赚一个 token!
代码实现：


9fd857f9-f048-4030-9dc1-56d4723bcd96.txt
Lines 674-693
    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            target_argmax_id = tl.load(target_argmax_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos,
                target_argmax_id,
            )
            if draft_token_id != target_argmax_id:
                # Reject.
                rejected = True
    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
举例：

小模型猜: [the, cat, sat, on]    (K=4)
大模型验: [the, cat, is,  ...]
pos=0: draft=the, target=the → 接受 ✓
pos=1: draft=cat, target=cat → 接受 ✓
pos=2: draft=sat, target=is  → 拒绝 ✗ → output[2]=is, 停止
最终输出: [the, cat, is]  (3个token, 比普通生成的1个多了2个)
机制 B: Random 验证 (temperature>0)
基于论文 Leviathan et al. 2022 的概率接受方法。

对应 Triton kernel rejection_random_sample_kernel：

对于每个 draft 位置 pos = 0, 1, ..., K-1:
    draft_token = 小模型猜的 token
    p_draft  = 小模型对 draft_token 的概率 (如果有)
    p_target = 大模型对 draft_token 的概率
    u        = uniform(0, 1) 随机数
    
    if p_draft > 0 AND p_target / p_draft >= u:
        接受! output[pos] = draft_token
    else:
        拒绝! output[pos] = recovered_token  ← 从调整后的分布采样
        后续所有 draft 全部丢弃
        
if 全部接受:
    output[K] = bonus_token
代码实现：


9fd857f9-f048-4030-9dc1-56d4723bcd96.txt
Lines 722-755
    rejected = False
    for pos in range(num_draft_tokens):
        if not rejected:
            draft_token_id = tl.load(draft_token_ids_ptr + start_idx + pos)
            if NO_DRAFT_PROBS:
                draft_prob = 1
            else:
                draft_prob = tl.load(
                    draft_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
                )
            target_prob = tl.load(
                target_probs_ptr + (start_idx + pos) * vocab_size + draft_token_id
            )
            uniform_prob = tl.load(uniform_probs_ptr + start_idx + pos)
            # NOTE(woosuk): While the draft probability should never be 0,
            # we check it to avoid NaNs. If it happens to be 0, we reject.
            if draft_prob > 0 and target_prob / draft_prob >= uniform_prob:
                # Accept.
                token_id = draft_token_id
            else:
                # Reject. Use recovered token.
                rejected = True
                token_id = tl.load(recovered_token_ids_ptr + start_idx + pos)
            tl.store(
                output_token_ids_ptr + req_idx * (max_spec_len + 1) + pos, token_id
            )
    if not rejected:
        # If all tokens are accepted, append the bonus token.
        bonus_token_id = tl.load(bonus_token_ids_ptr + req_idx)
        tl.store(
            output_token_ids_ptr + req_idx * (max_spec_len + 1) + num_draft_tokens,
            bonus_token_id,
        )
Recovered token（恢复 token）的计算逻辑：

当 draft 被拒绝时，不是简单用 target 的 argmax，而是从调整后的概率分布采样：

adjusted_prob[v] = max(p_target[v] - p_draft[v], 0)   对每个 vocab token v
recovered_token = sample_from(adjusted_prob)

9fd857f9-f048-4030-9dc1-56d4723bcd96.txt
Lines 832-833
                    prob = tl.maximum(target_prob - draft_prob, 0.0)
                # NOTE(woosuk): We don't need `prob = prob / tl.sum(prob)` here because
这保证了投机推理的输出分布与直接用大模型生成完全一致 — 数学上等价，不会降低生成质量。

六、Worker 线程如何处理
整个投机推理在同一个 GPU Worker 进程内完成，不需要额外的 worker：

┌────────────────────── GPU Worker ──────────────────────┐
│                                                        │
│  gpu_model_runner.execute_model():                     │
│                                                        │
│  ┌──────────────────────────────────────┐              │
│  │ 1. _prepare_inputs()                 │              │
│  │    - 把上轮 draft tokens 加入 batch   │              │
│  │    - 构建 SpecDecodeMetadata          │              │
│  │    - 计算 logits_indices              │              │
│  └───────────────┬──────────────────────┘              │
│                  ↓                                      │
│  ┌──────────────────────────────────────┐              │
│  │ 2. model.forward()                   │ ← 大模型      │
│  │    - 一次 forward 处理 原始+draft     │   一次forward  │
│  │    - 产出所有位置的 hidden_states     │              │
│  └───────────────┬──────────────────────┘              │
│                  ↓                                      │
│  ┌──────────────────────────────────────┐              │
│  │ 3. compute_logits() + _sample()      │              │
│  │    - RejectionSampler 验证 draft      │ ← Triton     │
│  │    - Greedy/Random 接受/拒绝          │   Kernel      │
│  │    - 确定最终 output tokens           │              │
│  └───────────────┬──────────────────────┘              │
│                  ↓                                      │
│  ┌──────────────────────────────────────┐              │
│  │ 4. propose_draft_token_ids()         │              │
│  │    - drafter.propose()               │ ← 小模型      │
│  │    - 为下一轮生成 K 个 draft token     │   K次forward  │
│  │    - 利用大模型的 hidden_states       │  (或1次tree)  │
│  └───────────────┬──────────────────────┘              │
│                  ↓                                      │
│  返回 output + draft_token_ids 给 EngineCore            │
│                                                        │
└────────────────────────────────────────────────────────┘
关键点：

大模型和小模型共享同一个 GPU（EAGLE/Medusa 的参数是大模型的子集 + 轻量头）
Draft Model 方式的小模型也加载在同一个 GPU 上，只是参数量更少
N-gram 方式不需要额外模型，在 CPU 上做模式匹配
七、什么决定选择小模型的输出
总结验证决策逻辑：

                    Greedy (temperature=0)
                    ┌───────────────────────────────────┐
                    │                                   │
                    │  draft_token == argmax(target) ?   │
                    │         ↙          ↘              │
                    │       Yes          No              │
                    │    接受 draft    拒绝, 用 target    │
                    │                                   │
                    └───────────────────────────────────┘
                    Random (temperature>0)
                    ┌───────────────────────────────────┐
                    │                                   │
                    │  p_target(draft_token)             │
                    │  ─────────────────── ≥ uniform ?   │
                    │  p_draft(draft_token)              │
                    │         ↙          ↘              │
                    │       Yes          No              │
                    │    接受 draft    拒绝, 从           │
                    │               max(p_t-p_d, 0) 采样 │
                    │                                   │
                    └───────────────────────────────────┘
直觉理解：

如果大模型也给了这个 token 很高的概率 → 接受（大小模型一致）
如果大模型对这个 token 概率低但小模型概率高 → p_target/p_draft 很小 → 大概率拒绝
如果小模型"碰巧"猜到了大模型也倾向的 token → 接受
接受率取决于：

小模型与大模型的分布匹配度（越接近，接受率越高）
Temperature（温度越高，分布越平，接受率通常越高）
任务难度（简单的补全场景，小模型更容易猜对）
八、性能收益分析
假设 draft 生成 K=5 个 token，大模型 forward 时间为 T，小模型（EAGLE 头）为 0.1T：

场景	接受数	产出 token	耗时	等效加速
全部接受	5/5	6 (5+bonus)	T + 0.1T×5 = 1.5T	6/1.5 = 4x
接受 3 个	3/5	4 (3+recovered)	1.5T	4/1.5 = 2.67x
全部拒绝	0/5	1 (recovered)	1.5T	1/1.5 = 0.67x (变慢!)
所以投机推理的价值完全取决于小模型的预测准确率。vLLM 中 EAGLE 通过直接使用大模型的 hidden states，通常能达到 70-90% 的接受率，实现 2-3x 的加速。
'''

Q-10:Draft 新一轮草稿，下一轮生成 K 个 draft token--这个代码逻辑在哪可以看到？是每次验证完draft的生成后，就开始下一轮的生成吗，这个逻辑链路是怎样的
A-10:
'''
是的，每次大模型验证完成后，在同一个 sample_tokens() 调用中，紧接着就会生成下一轮的 K 个 draft token。整个链路如下：

完整调用链
gpu_worker.execute_model()
  └→ gpu_model_runner.execute_model()     ← 大模型 forward (返回 None, 存状态)
  └→ gpu_model_runner.sample_tokens()     ← 采样 + draft，一切在这里发生
       │
       ├─ 1. _sample(logits, spec_decode_metadata)
       │     └→ rejection_sampler(...)     ← 验证上一轮 draft，得到 sampled_token_ids
       │
       ├─ 2. propose_draft_token_ids(sampled_token_ids)  ← 立刻生成下一轮 draft
       │     └→ self.drafter.propose(...)
       │           ├─ 第1步: set_inputs_first_pass()    ← 用大模型 hidden states 准备输入
       │           ├─ 第1步: model(**model_kwargs)       ← draft 模型 forward #1
       │           ├─ 第1步: _greedy_sample()            ← 拿到第 1 个 draft token
       │           │
       │           ├─ for token_index in range(K-1):     ← 自回归循环
       │           │     ├─ 更新 input_ids, positions, slot_mapping, seq_lens
       │           │     ├─ model(**model_kwargs)         ← draft 模型 forward #2, #3, ...
       │           │     └─ _greedy_sample()              ← 拿到后续 draft token
       │           │
       │           └─ return torch.stack(draft_token_ids_list)  ← [batch, K]
       │
       ├─ 3. _copy_draft_token_ids_to_cpu()   ← 异步拷贝到 CPU
       │
       ├─ 4. _bookkeeping_sync()              ← 更新请求状态
       │
       └─ 5. 返回 ModelRunnerOutput            ← 包含验证后的 token + draft ids
            (draft ids 交给 Scheduler 用于下一轮调度)
关键代码逐段解读
第一环：sample_tokens() — 验证后立即生成 draft
在 gpu_model_runner.py 的 sample_tokens() 中，验证和 draft 生成是紧密串联的：

# gpu_model_runner.py: sample_tokens()
# ===== 第1步: 验证上一轮的 draft =====
sampler_output = self._sample(logits, spec_decode_metadata)
# ===== 第2步: 立即定义 draft 生成闭包 =====
def propose_draft_token_ids(sampled_token_ids):
    self._draft_token_ids = self.propose_draft_token_ids(
        scheduler_output,
        sampled_token_ids,          # 刚刚验证通过的 token
        self.input_batch.sampling_metadata,
        hidden_states,              # 大模型的 hidden states
        sample_hidden_states,       # 大模型采样位置的 hidden
        aux_hidden_states,
        spec_decode_metadata,
        spec_decode_common_attn_metadata,
        slot_mappings,
    )
    self._copy_draft_token_ids_to_cpu(scheduler_output)
# ===== 第3步: 根据 drafter 类型决定何时调用 =====
if use_gpu_toks:
    # EAGLE/DraftModel: 直接用 GPU 上的 sampled_token_ids，不等 bookkeeping
    propose_draft_token_ids(sampler_output.sampled_token_ids)   # ← 立即!
else:
    propose_drafts_after_bookkeeping = True  # ngram 等需要 CPU token，等 bookkeeping
# ===== 第4步: bookkeeping (CPU 同步) =====
valid_sampled_token_ids = self._bookkeeping_sync(...)
if propose_drafts_after_bookkeeping:
    # ngram: 需要 CPU 侧的 token 历史
    propose_draft_token_ids(valid_sampled_token_ids)             # ← bookkeep 后
第二环：propose_draft_token_ids() — 准备输入并调用 drafter
# gpu_model_runner.py: propose_draft_token_ids()
if spec_config.use_eagle() or spec_config.uses_draft_model():
    # 准备 next_token_ids (验证通过的最新 token)
    next_token_ids, valid_sampled_tokens_count = (
        self.drafter.prepare_next_token_ids_padded(sampled_token_ids, ...)
    )
    # 准备大模型的 hidden states 和 token ids 作为 drafter 输入
    target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
    target_positions = self._get_positions(num_scheduled_tokens)
    target_hidden_states = hidden_states[:num_scheduled_tokens]
    # 调用 drafter.propose() 生成 K 个 draft token
    draft_token_ids = self.drafter.propose(
        target_token_ids=target_token_ids,
        target_positions=target_positions,
        target_hidden_states=target_hidden_states,    # 大模型的 hidden states
        next_token_ids=next_token_ids,                 # 验证通过的最新 token
        common_attn_metadata=common_attn_metadata,     # attention 元数据
        slot_mappings=slot_mappings,
    )
第三环：SpecDecodeBaseProposer.propose() — K 步自回归循环的核心
这是生成 K 个 draft token 的核心代码，位于 vllm/v1/spec_decode/eagle.py：

# eagle.py: SpecDecodeBaseProposer.propose()
def propose(self, target_token_ids, target_positions,
            target_hidden_states, next_token_ids, ...):
    # ====== 第一次 forward: 用大模型的 hidden states 做输入 ======
    num_tokens, token_indices_to_sample, common_attn_metadata = (
        self.set_inputs_first_pass(
            target_token_ids, next_token_ids,
            target_positions, target_hidden_states, ...
        )
    )
    # 构建 attention metadata
    per_group_attn_metadata, per_layer_attn_metadata = (
        self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
    )
    # 构建模型输入 (input_ids, positions, hidden_states)
    model_kwargs = self.build_model_inputs_first_pass(num_tokens, ...)
    # draft 模型第 1 次 forward
    with set_forward_context(per_layer_attn_metadata, ...):
        ret_hidden_states = self.model(**model_kwargs)       # ← draft 模型 forward
    # 采样第 1 个 draft token
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    draft_token_ids = self._greedy_sample(sample_hidden_states)  # ← argmax
    # 如果只需 1 个 draft token，直接返回
    if self.num_speculative_tokens == 1:
        return draft_token_ids.view(-1, 1)
    # ====== 自回归循环: 生成第 2~K 个 draft token ======
    draft_token_ids_list = [draft_token_ids]
    for token_index in range(self.num_speculative_tokens - 1):
        # (a) 输入 = 上一步的 draft token
        input_ids = draft_token_ids_list[-1].int()
        # (b) 更新 position、slot_mapping、seq_lens (每步+1)
        eagle_step_update_slot_mapping_and_metadata(
            positions, block_table, seq_lens, block_size, ...
        )
        common_attn_metadata.max_seq_len += 1
        # (c) 重建 attention metadata
        _, per_layer_attn_metadata = self.build_per_group_and_layer_attn_metadata(
            common_attn_metadata, draft_index=token_index + 1
        )
        # (d) 设置模型输入
        self.input_ids[:batch_size] = input_ids
        self.hidden_states[:batch_size] = hidden_states  # 上一步的 hidden
        model_kwargs = {
            "input_ids": self.input_ids[:input_batch_size],
            "positions": self._get_positions(input_batch_size),
            "hidden_states": self.hidden_states[:input_batch_size],  # EAGLE 特有
        }
        # (e) draft 模型第 2, 3, ..., K 次 forward
        with set_forward_context(per_layer_attn_metadata, ...):
            ret_hidden_states = self.model(**model_kwargs)          # ← draft forward
        # (f) 采样下一个 draft token
        draft_token_ids = self._greedy_sample(last_hidden_states[:batch_size])
        draft_token_ids_list.append(draft_token_ids)
    # 拼接所有 draft token: [batch_size, K]
    return torch.stack(draft_token_ids_list, dim=1)
时间线图
一次完整迭代的时间线:
时间 ─────────────────────────────────────────────────────→
     execute_model()              sample_tokens()
     ┌────────────────┐ ┌─────────────────────────────────────────┐
     │                │ │                                         │
     │  大模型 forward │ │ _sample()    propose_draft_token_ids()  │
     │  (验证 draft   │ │ ┌────────┐  ┌──────────────────────┐    │
     │   + 正常推理)   │ │ │验证:    │  │ draft forward #1     │    │
     │                │ │ │rejection│  │ → greedy → token_1   │    │
     │  1 次 forward  │ │ │sampler │  │ draft forward #2     │    │
     │  产出所有位置   │ │ │        │  │ → greedy → token_2   │    │
     │  的 logits     │ │ │得到     │  │ draft forward #3     │    │
     │                │ │ │accepted │  │ → greedy → token_3   │    │
     │  同时处理:     │ │ │tokens  │  │ ...                  │    │
     │  · 原始 token  │ │ │+ bonus │  │ draft forward #K     │    │
     │  · 上轮 draft  │ │ │        │  │ → greedy → token_K   │    │
     │                │ │ └────────┘  └──────────────────────┘    │
     │                │ │                                         │
     │  return None   │ │ bookkeeping → return output + drafts    │
     └────────────────┘ └─────────────────────────────────────────┘
     │                │ │            │ │                          │
     ├── T_large ─────┤ ├─ T_verify ─┤ ├──── K × T_draft ───────┤
     draft 结果交给 Scheduler → 下一轮 execute_model 时作为输入验证
关键设计要点
问题	答案
draft 在哪里生成？	在 sample_tokens() 中，验证完成后立即调用 propose_draft_token_ids()
是 K 次 forward 还是 1 次？	K 次 forward（自回归循环），每次产出 1 个 draft token
EAGLE 为什么快？	draft 模型极小（只有几层 + 轻量头），共享大模型的 embedding/lm_head，单次 forward 很快
draft 用什么采样？	贪心 (argmax)，不用 temperature/top-p，因为 draft 只需要"猜测最可能的"即可
下一轮怎么拿到 draft？	draft ids 存在 self._draft_token_ids，异步拷贝到 CPU → Scheduler 拿到后放入下一轮的 scheduled_spec_decode_tokens → 下一次 execute_model() 的 _prepare_inputs() 中拼入 batch
如果没有 spec decode？	sample_tokens() 中 spec_config is None 时跳过所有 draft 逻辑，直接返回

'''

