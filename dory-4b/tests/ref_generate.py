"""Greedy generation with the reference (full recompute each step): a sanity check that ref.py reads Dory right."""
import sys, glob, torch
from tokenizers import Tokenizer
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from ref import Ref
d = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 40
dtype = torch.bfloat16 if "--bf16" in sys.argv else torch.float32
tok = Tokenizer.from_file(d + "/tokenizer.json")
prompt = "<|im_start|>user\nWhat is the capital of France? Answer in one sentence.<|im_end|>\n<|im_start|>assistant\n<think></think>"
ids = tok.encode(prompt, add_special_tokens=False).ids
print("prompt ids", len(ids), ids[:8])
m = Ref(d, dtype=dtype)
out = []
for _ in range(n):
    lg = m.forward(ids + out)
    t = int(lg[-1].argmax())
    out.append(t)
    if t in (2, 11): break
print(repr(tok.decode(out, skip_special_tokens=False)))
print(out)
