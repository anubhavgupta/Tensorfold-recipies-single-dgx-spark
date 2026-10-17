"""libdory timings without the server: calibration (window and shared-round ms), prefill tok/s, 1-row decode ms."""
import ctypes as C, os, sys, time
d = sys.argv[1]; lib = sys.argv[2] if len(sys.argv) > 2 else "/cache/build/libdory.so"
os.environ.setdefault("DORY_SEQS", "8")
L = C.CDLL(lib); L.dory_open.restype = C.c_void_p; L.dory_open.argtypes = [C.c_char_p, C.c_int]
for f, a in (("dory_seq_alloc", [C.c_void_p]), ("dory_prefill_seq", [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int]),
             ("dory_forward_seq", [C.c_void_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.c_int, C.c_int])):
    getattr(L, f).argtypes = a; getattr(L, f).restype = C.c_int
L.dory_keep_seq.argtypes = [C.c_void_p, C.c_int, C.c_int]; L.dory_seq_release.argtypes = [C.c_void_p, C.c_int]
m = L.dory_open(d.encode(), 65536)
arr = lambda xs: (C.c_int * len(xs))(*xs)
for n in (4096, 16384, 32768):
    s = L.dory_seq_alloc(m); toks = [1000 + (i * 7919) % 90000 for i in range(n)]
    t0 = time.time(); assert L.dory_prefill_seq(m, s, arr(toks), n, 1) == 0; dt = time.time() - t0
    for _ in range(3): L.dory_forward_seq(m, s, arr([5]), 1, 1, 1); L.dory_keep_seq(m, s, 1)
    t1 = time.time()
    for _ in range(50): L.dory_forward_seq(m, s, arr([5]), 1, 1, 1); L.dory_keep_seq(m, s, 1)
    print(f"prefill {n}: {n / dt:.0f} tok/s ({dt:.2f} s); decode at {n}: {(time.time() - t1) / 50 * 1000:.2f} ms/token")
    L.dory_seq_release(m, s)
