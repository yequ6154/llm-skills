1、量化最重要的两个文件是：量化后的模型权重文件+量化描述文件（quant_model_description.json）
量化后，对应的目录下可能就只有这两个文件，其他文件，可直接用原模型目录下拷贝过来

2、quant_model_description.json文件主要是key-value的形式
模型加载的过程，是按照quant_model_description.json中的key进行加载

所以注意：
1)在量化过程，遍历模型层的名称会是model.*，所以quant_model_description.json中记录的也会是model.*
2)然而，vllm等相关引擎加载模型权重时，vLLM 对 LLM 主干有硬编码的前缀兼容规则，对 Vision Tower 没有，
3）所以：model.layers.*能被vllm识别；对vision层加载，需要手动去了model.前缀（需要修改quant_model_description.json文件vision的key，去掉model.前缀）




