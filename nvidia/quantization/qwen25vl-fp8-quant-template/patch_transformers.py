"""
Patch transformers save_pretrained to fix Qwen2.5-VL compatibility.

Bug: transformers >= 4.52 的 save_pretrained 中, offloaded modules 处理逻辑
与 Qwen2.5-VL 的 visual encoder 模块命名不兼容, 导致 KeyError。

Fix: 将 offloaded modules 的判断条件设为 False, 跳过整个有问题的代码块。
使用 device_map='cuda:0' 时模型全在 GPU 上, 跳过该块完全安全。
"""

import sys

path = "/usr/local/lib/python3.12/dist-packages/transformers/modeling_utils.py"

try:
    with open(path) as f:
        code = f.read()
except FileNotFoundError:
    print(f"ERROR: {path} not found")
    sys.exit(1)

old = (
    '            if (\n'
    '                hasattr(self, "hf_device_map")\n'
    '                and len(set(self.hf_device_map.values())) > 1\n'
    '                and ("cpu" in self.hf_device_map.values() or "disk" in self.hf_device_map.values())\n'
    '            ):'
)
new = '            if False:  # PATCHED: skip offloaded modules handling for VLM compatibility'

if "if False:  # PATCHED" in code:
    print("Already patched, skipping.")
    sys.exit(0)

count = code.count(old)
if count == 0:
    print("ERROR: target pattern not found. transformers version may differ.")
    print("Check the save_pretrained method in modeling_utils.py manually.")
    sys.exit(1)
elif count > 1:
    print(f"WARNING: found {count} occurrences, patching only the first one.")

code = code.replace(old, new, 1)
with open(path, "w") as f:
    f.write(code)

print("Patched successfully.")
