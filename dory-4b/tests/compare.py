"""Compare libdory against the PyTorch reference (tests/ref.py) on one prompt.

    python tests/compare.py MODEL_DIR [LIB] [--loops N]

Checks: (1) one prompt block with every row's logits, (2) the same tokens as a short prompt then one-row decode steps
and a verify window (record blocks), (3) row invariance: a window's rows alone and among other sequences give identical
candidates. Prints the max |logit| difference against the fp32 reference and the top-1 agreement.
"""
import ctypes as C
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
from ref import Ref

d = sys.argv[1]
lib_path = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else "/cache/build/libdory.so"
loops = int(sys.argv[sys.argv.index("--loops") + 1]) if "--loops" in sys.argv else None
if loops:
    os.environ["DORY_LOOPS"] = str(loops)
os.environ.setdefault("DORY_SEQS", "4")
os.environ.setdefault("DORY_NOCAL", "1")
V = 131072

from tokenizers import Tokenizer
tok = Tokenizer.from_file(d + "/tokenizer.json")
text = ("<|im_start|>user\nExplain in detail why the sky is blue, and mention Rayleigh scattering, the role of the sun's "
        "spectrum and why sunsets look red. Use three paragraphs.<|im_end|>\n<|im_start|>assistant\n<think>\n")
ids = tok.encode(text, add_special_tokens=False).ids
ids = (ids * 4)[:100]
n = len(ids)

L = C.CDLL(lib_path)
L.dory_open.restype = C.c_void_p
L.dory_open.argtypes = [C.c_char_p, C.c_int]
for f in ("dory_prefill_all", "dory_prefill_seq", "dory_forward_seq", "dory_forward_multi", "dory_seq_alloc"):
    getattr(L, f).restype = C.c_int
L.dory_seq_alloc.argtypes = [C.c_void_p]
L.dory_prefill_all.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int]
L.dory_prefill_seq.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int]
L.dory_forward_seq.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int]
L.dory_forward_multi.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int]
L.dory_keep_seq.argtypes = [C.c_void_p, C.c_int, C.c_int]
L.dory_seq_release.argtypes = [C.c_void_p, C.c_int]
L.dory_logits_row.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float)]
L.dory_candidates.argtypes = [C.c_void_p, C.c_int, C.c_int, C.POINTER(C.c_float), C.POINTER(C.c_int)]

m = L.dory_open(d.encode(), 8192)
assert m, "dory_open failed"


def arr(xs):
    return (C.c_int * len(xs))(*xs)


def logits(row):
    buf = (C.c_float * V)()
    L.dory_logits_row(m, row, buf)
    return torch.frombuffer(bytearray(buf), dtype=torch.float32).clone()


def cands(row, k=8):
    v = (C.c_float * 64)(); i = (C.c_int * 64)()
    L.dory_candidates(m, row, k, v, i)
    return list(i[:k]), list(v[:k])


ref = Ref(d, loops=loops)
R = ref.forward(ids).cpu()
del ref
torch.cuda.empty_cache()


def report(name, got, want):
    diff = (got - want).abs().max().item()
    agree = (got.argmax(-1) == want.argmax(-1)).float().mean().item()
    scale = want.abs().max().item()
    print(f"{name}: max|diff| {diff:.4f} (logits up to {scale:.1f}), top-1 agreement {agree * 100:.1f}%")


# (1) one prompt block, every row
s0 = L.dory_seq_alloc(m)
assert L.dory_prefill_all(m, s0, arr(ids), n) == 0
G = torch.stack([logits(r) for r in range(n)])
report("prompt block, all rows", G, R)
L.dory_seq_release(m, s0)

# (2) prompt of n-24 tokens, then 8 one-row steps, then a window of 8 and one of 8 kept partially
s1 = L.dory_seq_alloc(m)
p = n - 24
assert L.dory_prefill_seq(m, s1, arr(ids[:p]), p, 8) == 0
rows = [logits(0)]
for t in range(p, p + 8):
    assert L.dory_forward_seq(m, s1, arr([ids[t]]), 1, 1, 8) == 0
    L.dory_keep_seq(m, s1, 1)
    rows.append(logits(0))
assert L.dory_forward_seq(m, s1, arr(ids[p + 8:p + 16]), 8, 1, 8) == 0
rows += [logits(r) for r in range(8)]
L.dory_keep_seq(m, s1, 3)   # roll back 5 rows, then redo them
assert L.dory_forward_seq(m, s1, arr(ids[p + 11:p + 16]), 5, 1, 8) == 0
redo = torch.stack([logits(r) for r in range(5)])
D = torch.stack(rows)
report("prompt + decode steps + window", D, R[p - 1:p + 16])
report("rolled-back window redone", redo, R[p + 11:p + 16])

# (3) row invariance: the window again on a fresh sequence alone, then among two other sequences
def fresh_window(others):
    seqs = [L.dory_seq_alloc(m) for _ in range(1 + len(others))]
    for s_, pre in zip(seqs, [p] + [o for o in others]):
        assert L.dory_prefill_seq(m, s_, arr(ids[:pre]), pre, 1) == 0
    toks, cnt = [], []
    for s_, pre in zip(seqs, [p] + others):
        w = ids[pre:pre + 6]
        toks += w; cnt.append(len(w))
    assert L.dory_forward_multi(m, len(seqs), arr(seqs), arr(cnt), arr(toks), 64) == 0
    out = [cands(r) for r in range(6)]
    for s_ in seqs:
        L.dory_seq_release(m, s_)
    return out

alone = fresh_window([])
mixed = fresh_window([40, 70])
print("row invariance (window alone vs among 3 streams):", "identical" if alone == mixed else "DIFFERENT")
greedy = [c[0][0] for c in alone]
print("window greedy", greedy, "ref", R[p - 1 + 1:p + 6].argmax(-1).tolist())
