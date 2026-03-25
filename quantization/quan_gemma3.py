#量化工具modelslim
https://gitcode.com/Ascend/msit/blob/master/msmodelslim/docs/%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/Python-API%E6%8E%A5%E5%8F%A3%E8%AF%B4%E6%98%8E/%E9%87%8F%E5%8C%96%E6%8E%A5%E5%8F%A3/%E8%AE%AD%E7%BB%83%E5%90%8E%E9%87%8F%E5%8C%96%EF%BC%88PyTorch%EF%BC%89/QuantConfig.md  


# Copyright Huawei Technologies Co., Ltd. 2025. All rights reserved.
import os
import argparse
import functools
import sys
from transformers import AutoTokenizer, AutoModel, Gemma3ForConditionalGeneration, AutoProcessor, AutoConfig
import torch

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.abspath(os.path.join(current_directory, '..', "..",".."))
print(f"{parent_directory=}")
sys.path.append(parent_directory)

from example.common.utils import cmd_bool
from example.common.security.path import get_valid_read_path, get_write_directory
from example.common.copy_config_files import copy_config_files, modify_config_json
from msmodelslim.pytorch.llm_ptq.anti_outlier import AntiOutlierConfig, AntiOutlier
from msmodelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from example.multimodal_sd.utils import get_disable_layer_names

CPU = "cpu"
NPU = "npu"


def model_generate_test(model, processor, text):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful assistant."}]
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": text}
            ]
        }
    ]

    inputs = processor.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device, dtype=torch.bfloat16)

    input_len = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        generation = model.generate(**inputs, max_new_tokens=10240, do_sample=False)
        generation = generation[0][input_len:]

    decoded = processor.decode(generation, skip_special_tokens=True)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/s30048949/LLM-Research/gemma-3-27b-it')
    parser.add_argument('--calib_images', type=str,
                        default='')
    parser.add_argument('--save_directory', type=str,
                        default='/home/s30048949/weights/LLM-Research/gemma-3-27b-it_w8a8_test')
    parser.add_argument('--part_file_size', type=int, default=None)
    parser.add_argument('--w_bit', type=int, default=8)
    parser.add_argument('--a_bit', type=int, default=8)
    parser.add_argument('--device_type', type=str, choices=[CPU, NPU], default=NPU)
    parser.add_argument('--trust_remote_code', type=cmd_bool, default=True)
    parser.add_argument('--anti_method', type=str, choices=['m2', 'm4'], default='m4')
    parser.add_argument('--act_method', type=int, default=2)
    parser.add_argument('--open_outlier', type=cmd_bool, default=True)
    parser.add_argument('--is_dynamic', type=cmd_bool, default=True)
    parser.add_argument('--is_lowbit', type=cmd_bool, default=False)
    parser.add_argument('--group_size', type=int, choices=[64, 128, 256, 512], default=64)
    parser.add_argument('--mindie_format', action="store_true", default=False, help="Enable only mindie config save")
    return parser.parse_args()

def custom_hook(model_config):
    model_config["quantize"] = "w8a8_dynamic"

if __name__ == '__main__':
    import os
    import shutil

    args = parse_args()
    source_file = os.path.abspath(__file__)
    os.makedirs(args.save_directory, exist_ok=True)
    shutil.copy(source_file, args.save_directory)
    # check args
    args.save_directory = get_write_directory(args.save_directory, write_mode=0o750)

    # 1.加载模型
    device_map = CPU if args.device_type == CPU else "auto"

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=args.model_path, local_files_only=True)

    model = Gemma3ForConditionalGeneration.from_pretrained(args.model_path,
                                                           device_map=device_map,
                                                           trust_remote_code=args.trust_remote_code,
                                                           torch_dtype="auto",
                                                           local_files_only=True).eval()

    # exit()
    config = AutoConfig.from_pretrained(args.model_path,
                                        trust_remote_code=args.trust_remote_code,
                                        local_files_only=True)

    # 2.加载处理器
    processor = AutoProcessor.from_pretrained(args.model_path, local_files_only=True)

    # 3.设置回退层
    disable_names = []
    '''
    (model: nn.Module,
                            layer_include: Union[List[str], Tuple[str], str],
                            layer_exclude: Union[List[str], Tuple[str], str]) -> List[str]:
    '''
    vision_name = get_disable_layer_names(model, None, ["*vision*"])
    llm_name = []


    disable_names.extend(vision_name)
    # 4.加载校准集
    calib_dataset = []
    calib_list = [
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nAn experiment was designed to test the effects of three different types of paint on the durability of wooden toys. Because boys and girls tend to play differently with toys, a randomly selected group of children was divided into two groups by sex. Which of the following statements about this experiment is true?\n\nOptions:\n\nA. There are only three treatment combinations in this experiment.\nB. This is a completely randomized design.\nC. The sex of the children does not affect the experiment results.\nD. Sex is a blocking factor.\nE. The type of wooden toys is a blocking factor.\nF. Type of paint is a blocking factor.\nG. The durability of toys is a blocking factor.\nH. This experiment does not have any blocking factors.",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nFor $p(x)=f(x)g(x)$, if $f(2)=3$, $f'(2)=-4$, $g(2)=1$, and $g'(2)=6$, what is $p'(2)$?\n\nOptions:\n\nA. 12\nB. 10\nC. 5\nD. -2\nE. 14\nF. 8\nG. 16\nH. 18\nI. 20\nJ. 9",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nHow many ways are there to divide a set of 6 elements into 3 non-empty ordered subsets?\n\nOptions:\n\nA. 300\nB. 1100\nC. 1000\nD. 540\nE. 1200\nF. 800\nG. 720\nH. 900\nI. 460\nJ. 1500",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nHow many ways are there to color the vertices of a cube with two colors, up to rotation?\n\nOptions:\n\nA. 10\nB. 30\nC. 23\nD. 15\nE. 18\nF. 36\nG. 42\nH. 48\nI. 33\nJ. 26",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nWhen 30! is computed, it ends in 7 zeros. Find the digit that immediately precedes these zeros.\n\nOptions:\n\nA. 0\nB. 2\nC. 7\nD. 1\nE. 8\nF. 6\nG. 4\nH. 9\nI. 5\nJ. 3",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nmatrix $A=(\\begin{array}{rrrr} -2 & -1 & -1 & -1 \\ 2 & 1 & 3 & 2 \\ 1 & 1 & 0 & 1 \\ -1 & -1 & -2 & -2 \\end{array})$. Suppose f is the minimal polynomial of A. What is f(99)? Return the numeric without explanation.\n\nOptions:\n\nA. 1000000.0\nB. 1010000.0\nC. 980000.0\nD. 100000.0\nE. 989000.0\nF. 990100.0\nG. 980001.0\nH. 980100.0\nI. 990000.0\nJ. 999000.0",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nA poker hand is defined as drawing 5 cards at random without replacement from a deck of 52 playing cards. Find the probability of four of a kind (four cards of equal face value and one card of a different value).\n\nOptions:\n\nA. 0.00012\nB. 0.00009\nC. 0.00015\nD. 0.00006\nE. 0.00018\nF.  0.00024\nG. 0.00048\nH. 0.00003\nI. 0.00036\nJ. 0.00030",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nStatement 1 |  Suppose ∑|a_i| diverges and ∑ a_i = 2. There is a rearrangement a_i_k of the terms such that ∑ a_i_k = 4. Statement 2 | There exists metric spaces X and Y with X closed and bounded and a continuous mapping f : X → Y such that f(X) is NOT “closed and bounded”.\n\nOptions:\n\nA. False, Not Sure\nB. False, False\nC. Not Sure, True\nD. True, Not Sure\nE. Not Sure, Not Sure\nF. True, False\nG. False, True\nH. True, Cannot be determined\nI. Not Sure, False\nJ. True, True",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nA manufacturer of ready-bake cake mixes is interested in designing an experiment to test the effects of four different temperature levels (300, 325, 350, and 375F), two different types of pans (glass and metal), and three different types of ovens (gas, electric, and microwave) on the texture of its cakes, in all combinations. Which of the following below is the best description of the design of the necessary experiment?\n\nOptions:\n\nA. A randomized block design, blocked on temperature and type of pan, with 12 treatment groups\nB. A completely randomized design with 6 treatment groups\nC. A randomized block design, blocked on type of oven, with 24 treatment groups\nD. A randomized block design, blocked on temperature, with six treatment groups\nE. A completely randomized design with 18 treatment groups\nF. A randomized block design, blocked on type of oven, with 12 treatment groups\nG. A completely randomized design with 24 treatment groups\nH. A randomized block design, blocked on type of pan, with 12 treatment groups\nI. A completely randomized design with 12 treatment groups\nJ. A completely randomized design with nine treatment groups",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nUse the Runge-Kutta method with $h=0.1$ to find approximate values of the solution of $(y-1)^2 * y' = 2x + 3$ with y(1) = 4. What is y(0)?\n\nOptions:\n\nA. 3.21098765\nB. 4.90876123\nC. 2.98765432\nD. 3.78543210\nE. 2.65432109\nF. 4.56789012\nG. 5.01234567\nH. 3.46621207\nI. 4.12345678\nJ. 5.24681357",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nThe circle $2x^2 = -2y^2 + 12x - 4y + 20$ is inscribed inside a square which has a pair of sides parallel to the x-axis. What is the area of the square?\n\nOptions:\n\nA. 60\nB. 160\nC. 2\\sqrt{20}\nD. 10\\sqrt{20}\nE. 4\\sqrt{20}\nF. \\sqrt{20}\nG. 80\nH. 40\nI. 20\nJ. 100",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\n5.3-19. Two components operate in parallel in a device, so the device fails when and only when both components fail. The lifetimes, $X_1$ and $X_2$, of the respective components are independent and identically distributed with an exponential distribution with $\\theta=2$. The cost of operating the device is $Z=2 Y_1+Y_2$, where $Y_1=\\min \\left(X_1, X_2\\right)$ and $Y_2=\\max \\left(X_1, X_2\\right)$. Compute $E(Z)$.\n\n\nOptions:\n\nA. $9$\nB. $8$\nC. $4$\nD.  $5$\nE. $12$\nF. $3$\nG. $10$\nH. $4.5$\nI. $7$\nJ. $6$",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nA tank contains 100 gal of water and $50 \\mathrm{oz}$ of salt. Water containing a salt concentration of $\\frac{1}{4}\\left(1+\\frac{1}{2} \\sin t\\right) \\mathrm{oz} / \\mathrm{gal}$ flows into the tank at a rate of $2 \\mathrm{gal} / \\mathrm{min}$, and the mixture in the tank flows out at the same rate.\nThe long-time behavior of the solution is an oscillation about a certain constant level. What is the amplitude of the oscillation?\n\nOptions:\n\nA. 0.14995\nB.  0.24995\nC. 0.34995\nD. 0.29995\nE. 0.50000\nF. 0.44995\nG. 0.39995\nH. 0.19995\nI. 0.59995\nJ. 0.10000",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nWhat is the probability that a randomly selected integer in the set $$\\{1,2,3,\\ldots,100\\}$$ is divisible by 2 and not divisible by 3? Express your answer as a common fraction.\n\nOptions:\n\nA. \\frac{17}{31}\nB. \\frac{33}{66}\nC. \\frac{33}{100}\nD. \\frac{17}{50}\nE. \\frac{1}{6}\nF. \\frac{17}{66}\nG. \\frac{1}{3}\nH. \\frac{31}{66}\nI. \\frac{31}{100}\nJ. \\frac{17}{100}",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nEvaluate $\\int_c z^2 / (z - 5) dz$, where c is the circle that $|z| = 2$.\n\nOptions:\n\nA. $2\\pi i$\nB. 0\nC. $-2\\pi i$\nD. 1\nE. $4\\pi i$\nF. -1\nG. $5$\nH. $10\\pi i$\nI. $-5$\nJ. 2",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nFind the number that makes the statement true: 0.32 g = _ cg.\n\nOptions:\n\nA. 3.20\nB. 0.32\nC. 3200\nD. 32\nE. 3.2\nF. 32000\nG. 0.0032\nH. 320\nI. 3,200\nJ. 0.032",
        "Answer the following multiple choice question. The last line of your response should be of the following format: 'ANSWER: $LETTER' (without quotes) where LETTER is one of Options(e.g. one of ABCDEFGHIJKLMNOP). Think step by step before answering.\n\nQuestion:\n\nStatement 1 | The external direct product of cyclic groups is cyclic. Statement 2 | The external direct product of D_3 and D_4 is isomorphic to D_12.\n\nOptions:\n\nA. Statement 1 is an example of Statement 2, False\nB. True, True\nC. True, False\nD. Both statements are true, but unrelated\nE. True, Statement 2 is an example of Statement 1\nF. Both statements are true, but Statement 1 is sometimes false\nG. False, True\nH. Statement 1 is dependent on the conditions of Statement 2, False\nI. False, False\nJ. Both statements are false, but Statement 2 is occasionally true"]

    for calib_data in calib_list:
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": calib_data}
                ]
            }
        ]

        inputs = processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt"
        ).to(model.device, dtype=torch.bfloat16)
        calib_dataset.append(
            [inputs['input_ids'], None, inputs['attention_mask'], None, None, None, None, None, None, None, None, None])


    # 6.模型量化

    quant_config = QuantConfig(
        w_bit=args.w_bit,
        a_bit=args.a_bit,
        disable_names=disable_names,
        dev_type=args.device_type,
        dev_id=model.device.index,
        act_method=args.act_method,
        mm_tensor=False,
        open_outlier=args.open_outlier,
        is_dynamic=args.is_dynamic,
        is_lowbit=args.is_lowbit,
        group_size=args.group_size
    )
    calibrator = Calibrator(model, quant_config, calib_data=calib_dataset, disable_level='L0')
    calibrator.run()
    # 7.保存权重
    # exit()
    if args.mindie_format:
        quant_model_description_json_name = "quant_model_description_w8a8_dynamic.json"
    else:
        quant_model_description_json_name = "quant_model_description.json"

    save_type = "safe_tensor" if args.mindie_format else "ascendV1"
    calibrator.save(args.save_directory,
                    save_type=[save_type], 
                    part_file_size=args.part_file_size)
    
    custom_hooks = {
        'config.json': functools.partial(modify_config_json, custom_hook=custom_hook)
    }
    copy_config_files(input_path=args.model_path, output_path=args.save_directory, quant_config=quant_config,
                      mindie_format=args.mindie_format, custom_hooks=custom_hooks)


