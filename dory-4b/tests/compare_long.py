"""Long-context check: N tokens of real text prefilled in TMAX chunks (the sliding ring wraps), then 8 decode steps and a
verify window, against the fp32 reference's last rows.   python tests/compare_long.py MODEL_DIR [N]"""
import ctypes as C, os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
from ref import Ref
from tokenizers import Tokenizer
d = sys.argv[1]; N = int(sys.argv[2]) if len(sys.argv) > 2 else 5000
os.environ.setdefault("DORY_SEQS", "2"); os.environ.setdefault("DORY_NOCAL", "1")
tok = Tokenizer.from_file(d + "/tokenizer.json")
src = open("/work/dory-4b/engine/dory_host.cu").read() + open("/work/dory-4b/engine/dory_attn.cu").read()
ids = tok.encode(src, add_special_tokens=False).ids
while len(ids) < N + 16: ids += ids
ids = ids[:N + 16]
L = C.CDLL("/cache/build/libdory.so"); L.dory_open.restype = C.c_void_p; L.dory_open.argtypes = [C.c_char_p, C.c_int]
for f, a in (("dory_seq_alloc", [C.c_void_p]), ("dory_prefill_seq", [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int]),
             ("dory_forward_seq", [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int])):
    getattr(L, f).argtypes = a; getattr(L, f).restype = C.c_int
L.dory_keep_seq.argtypes = [C.c_void_p, C.c_int, C.c_int]; L.dory_logits_row.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_float)]
m = L.dory_open(d.encode(), 16384)
arr = lambda xs: (C.c_int * len(xs))(*xs)
def lg(r):
    b = (C.c_float * 131072)(); L.dory_logits_row(m, r, b); return torch.frombuffer(bytearray(b), dtype=torch.float32).clone()
s = L.dory_seq_alloc(m)
assert L.dory_prefill_seq(m, s, arr(ids[:N]), N, 8) == 0
rows = [lg(0)]
for t in range(N, N + 8):
    assert L.dory_forward_seq(m, s, arr([ids[t]]), 1, 1, 8) == 0; L.dory_keep_seq(m, s, 1); rows.append(lg(0))
assert L.dory_forward_seq(m, s, arr(ids[N + 8:N + 16]), 8, 1, 8) == 0
rows += [lg(r) for r in range(8)]
G = torch.stack(rows)
R = Ref(d).forward(ids[:N + 16]).cpu()[N - 1:N + 16]
print(f"N={N}: max|diff| {(G - R).abs().max():.4f}, top-1 agreement {(G.argmax(-1) == R.argmax(-1)).float().mean() * 100:.1f}%")
