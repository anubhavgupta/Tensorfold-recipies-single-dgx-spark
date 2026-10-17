"""A few 1-row decode steps (and an 8-stream round) for nsys: DORY_NO_GRAPHS=1 nsys profile python tests/prof_decode.py MODEL_DIR"""
import ctypes as C, os, sys
os.environ.setdefault("DORY_SEQS", "8"); os.environ.setdefault("DORY_NOCAL", "1")
L = C.CDLL(sys.argv[2] if len(sys.argv) > 2 else "/cache/build/libdory.so"); L.dory_open.restype = C.c_void_p; L.dory_open.argtypes = [C.c_char_p, C.c_int]
for f in ("dory_seq_alloc", "dory_prefill_seq", "dory_forward_seq", "dory_forward_multi"): getattr(L, f).restype = C.c_int
L.dory_seq_alloc.argtypes = [C.c_void_p]; L.dory_prefill_seq.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int]
L.dory_forward_seq.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int]
L.dory_forward_multi.argtypes = [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.POINTER(C.c_int), C.POINTER(C.c_int), C.c_int]
L.dory_keep_seq.argtypes = [C.c_void_p, C.c_int, C.c_int]
m = L.dory_open(sys.argv[1].encode(), 8192); arr = lambda xs: (C.c_int * len(xs))(*xs)
seqs = [L.dory_seq_alloc(m) for _ in range(8)]
for s in seqs: L.dory_prefill_seq(m, s, arr(list(range(1000, 1500))), 500, 1)
L.dory_prof_report.argtypes = [C.c_void_p]
if os.environ.get("DORY_PROF"): L.dory_prof_report(m)  # drop the prompt pass
for _ in range(6): L.dory_forward_seq(m, seqs[0], arr([5]), 1, 1, 1); L.dory_keep_seq(m, seqs[0], 1)
if os.environ.get("DORY_PROF"): print("--- 6 one-row steps", flush=True); L.dory_prof_report(m)
for _ in range(4):
    L.dory_forward_multi(m, 8, arr(seqs), arr([1] * 8), arr([5] * 8), 64)
    for s in seqs: L.dory_keep_seq(m, s, 1)
if os.environ.get("DORY_PROF"):
    L.dory_prof_report.argtypes = [C.c_void_p]
    L.dory_prof_report(m)
