"""Micro-benchmark of decode-verify GEMM shapes (M = verify rows), run inside the SGLang container.

FP8 per-tensor via torch._scaled_mm (what SGLang uses for ModelOpt per-tensor FP8) against the same weights with the
scale expanded per channel (sgl-kernel CUTLASS fp8_scaled_mm, triton_scaled_mm), and bf16 F.linear for the small bf16
projections. CUDA-graph timed; prints us per call and GB/s of weight read.
usage: docker exec <ctr> python3 /path/gemm_bench.py [M=6]"""
import sys, torch
import torch.nn.functional as F

M = int(sys.argv[1]) if len(sys.argv) > 1 else 6


def bench(fn, it=200):
    for _ in range(10): fn()
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(it): fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record(); g.replay(); e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) * 1000 / it


from sglang.kernels.ops.gemm.fp8_kernel import triton_scaled_mm
try:
    from sglang.kernels.ops.gemm import fp8_scaled_mm
except Exception as ex:
    fp8_scaled_mm = None; print("no fp8_scaled_mm:", ex)

print(f"M={M}")
for name, K, N in [("gdn in_proj_qkvz", 3072, 20480), ("gdn out_proj", 8192, 3072), ("attn q_proj", 3072, 16384),
                   ("attn o_proj", 8192, 3072), ("attn k/v", 3072, 512)]:
    copies = max(2, int(300e6 // (N * K)) + 1)      # rotate > L2 worth of weights so every call reads DRAM, as in decode
    ws = [(torch.randn(N, K, device="cuda") * 0.05).to(torch.float8_e4m3fn) for _ in range(copies)]
    it = [0]
    def nxt():
        it[0] = (it[0] + 1) % copies
        return ws[it[0]].t()
    a = torch.randn(M, K, device="cuda").to(torch.float8_e4m3fn)
    one = torch.ones((), device="cuda")
    sw = torch.ones((N, 1), device="cuda"); sa = torch.ones((M, 1), device="cuda")
    r = {"torch._scaled_mm": bench(lambda: torch._scaled_mm(a, nxt(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16))}
    if fp8_scaled_mm is not None:
        try: r["cutlass"] = bench(lambda: fp8_scaled_mm(a, nxt(), sa, sw, out_dtype=torch.bfloat16))
        except Exception as ex: r["cutlass"] = f"err {str(ex)[:80]}"
    try: r["triton"] = bench(lambda: triton_scaled_mm(a, nxt(), sa, sw, torch.bfloat16, None))
    except Exception as ex: r["triton"] = f"err {str(ex)[:80]}"
    gb = N * K / 1e9
    print(f"  FP8 {name:17s} {K}x{N} {gb*1e3:4.0f} MB: " + "  ".join(
        f"{k} {v:6.1f}us {gb/v*1e6:4.0f}GB/s" if isinstance(v, float) else f"{k} {v}" for k, v in r.items()), flush=True)

for name, K, N in [("gdn in_proj_ba", 3072, 128), ("router", 3072, 256)]:
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16); x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    t = bench(lambda: F.linear(x, w)); gb = N * K * 2 / 1e9
    print(f"  BF16 {name:16s} {K}x{N}: F.linear {t:.1f}us ({gb/t*1e6:.0f} GB/s)")
