1、量化最重要的两个文件是：量化后的模型权重文件+量化描述文件（quant_model_description.json）
量化后，对应的目录下可能就只有这两个文件，其他文件，可直接用原模型目录下拷贝过来

2、quant_model_description.json文件主要是key-value的形式
模型加载的过程，是按照quant_model_description.json中的key进行加载

所以注意：
1)在量化过程，遍历模型层的名称会是model.*，所以quant_model_description.json中记录的也会是model.*
2)然而，vllm等相关引擎加载模型权重时，vLLM 对 LLM 主干有硬编码的前缀兼容规则，对 Vision Tower 没有，
3）所以：model.layers.*能被vllm识别；对vision层加载，需要手动去了model.前缀（需要修改quant_model_description.json文件vision的key，去掉model.前缀）


量化gemma3-27b-it-w8a8时，
1、先git clone https://gitcode.com/Ascend/msit.git
2、进入到msit/msmodelslim的目录并运行安装脚本
cd msit/msmodelslim
bash install.sh

3、
1)将w8a8权重目录下的quant_model_description.json中的model.language_model.替换为model.
2)model.vision_tower.vision_model.替换为vision_tower.vision_model.
3)将w8a8权重目录下的model.safetensors.index.json文件中的权重文件名称全部改为量化的权重文件名称



