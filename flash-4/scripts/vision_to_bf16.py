#!/usr/bin/env python3
"""Rewrite flash-3's floating vision tower (vision-f16-4.05.safetensors, F16, `vision_tower.*` names with Hugging Face's
channels-first patch embedding) as bf16 `model.visual.*` tensors, the layout TensorFold's native tower loader reads (its
`vision_tower.*` spelling is MLX's channels-last patch embedding). Run in a container that has
torch and safetensors (the recipe's build image does): vision_to_bf16.py IN OUT.   F16 -> bf16 rounds to nearest even;
the tower was decoded from 6-bit EXL3, so the three dropped mantissa bits are far below its quantization error."""
import sys
from safetensors import safe_open
from safetensors.torch import save_file
import torch

src, dst = sys.argv[1], sys.argv[2]
out = {}
with safe_open(src, "pt") as f:
    for k in f.keys():
        t = f.get_tensor(k)
        name = "model.visual." + k[len("vision_tower."):] if k.startswith("vision_tower.") else k
        out[name] = t.to(torch.bfloat16).contiguous() if t.is_floating_point() else t.contiguous()
save_file(out, dst, metadata={"format": "pt", "source": src, "converted": "f16 -> bf16"})
print(f"{len(out)} tensors -> {dst}")
