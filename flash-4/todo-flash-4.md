# flash-4 Optimization TODOs (Post-Patches 0001–0012)

Baseline on NVIDIA DGX Spark (GB10, 128 GB LPDDR5X, 273 GB/s, 48 SMs, 24 MiB L2) at commit `605f25c`:
- **Cold Prefill:** 4K `2,053` / 16K `2,050–2,101` / 65K `2,004–2,078` / **248K `1,928 tok/s` (`128.45 s`)**
- **Decode N=1 (steady):** `code 96.9–110`, `prose 52.7–54.4`, `devops 88.3`, `json 109.5–116.3` tok/s (`verify 35.6–54.4 ms` + `draft 3.1 ms`)
- **Decode N=4 (steady):** `code 188.8`, `prose 108.7`, `devops 180.2`, `json 209.8` tok/s (`experts` at 236 GB/s = 86% peak)
- **Capacity:** 8 full 262K streams max (`5.10 GiB`/stream, MXFP8 KV); 6 full 262K streams + 8 GiB warm cache live.

---

## 1. Decode (N=1) — Hyper-Connections (`hc_attn + hc_mlp`: `9.9 ms` = 20% of round, ~128 GB/s)

- [ ] **100% Exact BF16 Fused Decode `W_up [10240, 320] + hcDenseMix` Kernel**
  - **Root cause:** Prefill already uses `readoutFused`, but decode (`n <= 16`) still runs `W_up` through `qmmf`'s 4-stage `cp.async` pipeline for only 5 K-steps (`320 / 64`), writes `hc_up [n, 10240]` to DRAM, and launches a separate `hcDenseMix` kernel (96× per round).
  - **Goal:** Stream `W_up` with coalesced `uint4` loads, keep `act [n, 320]` in SMEM (~10 KB), fuse `hcDenseMix` in registers, and write `mixed [n, 2560] + xsm` directly.
  - **Est. gain:** `-1.5 to -2.5 ms/round` (**+4–6% N=1 decode** across `prose`, `code`, `devops`, `json`), 100% bit-exact BF16.

- [ ] **INT8 (7-bit Mantissa, Group-32/64) Hyper-Connections Weights (`W_down` + `W_up`)**
  - **Root cause:** 96 HCs read `1.27 GB` of BF16 weights on every decode round. Earlier 8-bit MXFP8 HC (`E4M3`, 3-bit mantissa) saved **-4.5 ms/round (+8–18% N=1 decode)**, proving bandwidth is the lever, but 3-bit mantissa across 96 residual-mixing steps broke 120K needle retrieval.
  - **Goal:** Prototype symmetric INT8 (7-bit mantissa = 16× finer precision than E4M3) with per-channel or group-32 FP32 scales, cutting `1.27 GB → 0.64 GB`.
  - **Validation gate:** **Must pass 120K needle retrieval (`FALCON-9472-OMEGA`)**, 5/5 greedy & T=0.7 `draft_on == draft_off` exactness, and zero prefill regression.
  - **Est. gain:** `-3.0 to -4.0 ms/round` (**+8–10% N=1 decode** across all workloads).

- [ ] **Column-Parallel Fused `hcWriteback + hcNormalize` for Decode (`n < 64`)**
  - **Root cause:** Stock's fused `hcWritebackNorm` launches only **1 CTA per row**, which under-utilizes GB10's 48 SMs when `n = 1..8` (which is why stock gates it to `n >= 64` and why forcing `n >= 1` regressed decode by `-1.5–3%`). As a result, `HC MLP` + `HC Attn` run separate `hcWriteback` (including FP32 `s.part` reduction on `HC MLP`) and `hcNormalize` kernels (`96` extra launches + `h [n, 10240]` roundtrip).
  - **Goal:** Split the 10,240 columns across multiple CTAs per row (with a tiny block/grid RMS reduction) so decode (`n < 64`) gets both kernel fusion and full 48-SM occupancy.
  - **Est. gain:** `-0.5 to -0.8 ms/round` (**+1–2% N=1 decode**), 100% bit-exact BF16.

---

## 2. Long-Context Prefill (245K) — QSA + MXFP8 Attention (`318.3 ms` = 19.5% of chunk)

- [ ] **SMEM-Compacted 2-Pass `fn_qsa_select` (`64.8 ms` @ 245K)**
  - **Root cause:** Patch `0011` cut radix passes from 4 → 3 (`69.4 → 64.8 ms`), but `fn_qsa_select_body` still performs **5 full global reads** over `9.03 GB` of `SC[3072, 61250]` (`45.1 GB` total: 3 radix + count + write).
  - **How to fix (100% bit-exact):**
    1. **Global Pass 1:** Run the 11-bit top histogram (`shift = 21`, `2,048` bins, `__match_any_sync`) to find boundary bin `top11` and `cnt_eq`.
    2. **Global Pass 2:** Stream `row` once more, compacting only the `< 4,096` candidates with `(k >> 21) >= top11` into **40 KiB SMEM** (`sh_k[4096]` + `sh_b[4096]`, keeping fallback to global passes if `> 4,096`).
    3. **SMEM-only finish:** Run radix passes 1–2 (`shift = 10, 0`), exact tie-breaking (smallest block IDs for `== cut`), and a 512-element bitonic sort in SMEM to emit ascending block IDs with zero further global reads (cuts DRAM traffic `45.1 GB → 18.1 GB`).
  - **Est. gain:** `-20 to -30 ms/chunk` (**-2.0 s** off 248K TTFT), 100% bit-exact.

- [ ] **Native Blackwell `mma.sync.aligned.kind::mxf8f6f4.block_scale` in `attn_core` (`182.4 ms` @ 245K)**
  - **Root cause:** Patch `0012` halved KV reads (`77 GB → 38.5 GB`, cutting `attn_core` `298 → 182.4 ms`), but still converts MXFP8 → BF16 in registers via BF16x2 SIMD before running BF16 `mma16816`.
  - **Goal:** Use native `sm_121` block-scaled FP8 Tensor Core MMA directly on the staged `E4M3 + E8M0/32` KV tiles (eliminating conversion instructions and increasing occupancy).
  - **Est. gain:** `-25 to -35 ms/chunk` (**-2.5 s** off 248K TTFT).

---

## 3. Dense 6-bit EXL3 (`gdn_proj` + `gdn_out` + `attn_proj` + `attn_out`: `432 ms` prefill / `11.7 ms` N=1 decode)

- [ ] **Custom 3-Stage `cp.async` Transposed 6-bit Dense Prompt Kernel (Prefill)**
  - **Root cause:** `exl3_linear_wc` runs in 128-row blocks (24× weight re-reads per 3,072-row chunk, ALU-bound on 6-bit trellis decode). Unpacking to DRAM BF16 + `qmmf` was slower (`-3%` to `-7%`) due to 5.4 GB unpacked DRAM traffic.
  - **Goal:** Apply the same transposed 3-stage `cp.async` SMEM design as `exl3_prompt.cuh` (Patch `0007`/`0010`) directly to 6-bit dense projections so each 6-bit weight tile is decoded once in SMEM/registers across 64–128 rows without ever writing unpacked weights to DRAM, and write directly into stitched output columns.
  - **Est. gain:** `-100 to -150 ms/chunk` (**+8–11% cold prefill**, toward `~2,250+ tok/s`), 100% lossless.

- [ ] **Zero-Copy Strided Output (`y_stride` + `col` offset) in `tf_exl3_group_12_2` (Decode N=1)**
  - **Root cause:** `cuda_exl3_lane.zig` currently writes stitched 6-bit decode projections (`gdn_proj`: `qkv + z`, `attn_proj`: `q + k + v`) into `exl3_tmp` and then launches **108 separate `copyCols` kernels per round** (`36 × 2` GDN + `12 × 3` Attn) to copy them into `pa`.
  - **Goal:** Pass `y_stride = g.n` and `y_ptr = out + p.col * 2` directly into `LinearArgs` / `tf_exl3_group_12_2`, eliminating all 108 `copyCols` launches and `exl3_tmp` DRAM traffic.
  - **Est. gain:** `-0.8 to -1.5 ms/round` (**+2–3% N=1 decode**), 100% bit-exact.

- [ ] **Fuse EXL3 `rot_in` (`suh * FWHT128`) into `gdn_norm` (`GDN Out`, 36×)**
  - **Root cause:** `gdn_core` writes `gated_v [n, 6144]` BF16, then `gdn_out` launches a separate `tf_exl3_rot_in` kernel (`36` extra launches + `gated_v` roundtrip per pass) to rotate it into FP16 `xh`.
  - **Goal:** Fuse `suh * FWHT128` directly into the end of `gdn_norm` so `gdn_out` consumes pre-rotated FP16 `xh` immediately (same pattern can also fuse `attn_gate` + `rot_in` for `attn_out`).
  - **Est. gain:** `-0.3 to -0.5 ms/round` in N=1 decode and `~-5 ms/chunk` in prefill, 100% bit-exact.

---

## 4. MoE Routed 512×4-bit EXL3 (`453 ms` prefill / `16.5 ms` N=1 decode)

- [ ] **Eliminate Split-K `z` Partials / Fuse `up` Finish + SwiGLU + `down rot_in` on Decode (`n <= 16`)**
  - **Root cause:** In N=1 decode (`16.5 ms` @ ~185–200 GB/s vs N=4's 236 GB/s), `~2.8 ms` (17% of experts) is spent in 4 separate small rotation & split-K finish launches per layer (`192` launches/round) plus fp32 `z` partial write/read.
  - **Goal:** Eliminate split-K `z` partials via in-block/cluster reduction and fuse `up` finish + SwiGLU + `down rot_in` directly on decode (`n <= 16`).
  - **Est. gain:** `-1.5 to -2.5 ms/round` (**+3–5% N=1 decode**), 100% bit-exact.

- [ ] **EXL3 `exl3_prompt.cuh` Instruction-Scheduling & Fusion Tuning (Prefill, Stay on EXL3)**
  - **Root cause:** Without leaving EXL3 4.05 bpw (avoiding double-quantization), `exl3_prompt.cuh` (`453 ms` @ 245K) runs at `~48 eff. TFLOPS` where 4-bit trellis bit-unpacking + `mul1` `__dp4a` codebook ALU shares cycles with FP16 MMA.
  - **Goal:** Profile and tune inner-loop `mul1` decode instruction scheduling / `rot_in` fusion in `exl3_prompt.cuh` while staying 100% on EXL3.
  - **Est. gain:** Modest `~5–10%` on `routed4` (`-20 to -45 ms/chunk`), 100% bit-exact.

---

## 5. Lower-Priority / Exploratory Prefill Items (from `architecture-3d.html`)

- [ ] **Chunked WY Tensor-Core `gdn_core` Prefill (`100.0 ms` = 6.1% @ 245K / 7.0% @ 16K)**
  - **Root cause:** `gdn_prefill.cu` runs exact step-by-step FP32 recurrence across 3,072 rows (`2.77 ms/layer × 36 layers = 100 ms`) to match decode rounding bit-for-bit.
  - **Goal:** Evaluate a chunked WY Tensor-Core prefill formulation (preserving FP32 state updates across chunks) and test against 120K needle retrieval + greedy parity.
  - **Est. gain:** `-40 to -60 ms/chunk` (**+3–4% cold prefill**); note prefill rounding will differ slightly from step-by-step decode.

- [ ] **Optional Startup `madvise(MADV_WILLNEED)` Pre-Fault for PLE N-Gram Table (`16.6 ms` cold 16K / `9.4 ms` 245K)**
  - **Root cause:** Even with Patch `0011` (8-thread host gather), the very first cold prompt hits minor page faults on untouched mmap pages of the host n-gram table (`16.6 ms` vs `9.4 ms` warm).
  - **Goal:** Add an opt-in startup pre-fault knob (only when host RAM headroom allows without squeezing KV cache) to eliminate first-request cold-fault spikes.
  - **Est. gain:** `-5 to -7 ms` on first cold prompt chunk only (0.5% overall), 100% bit-exact.

---

## Reference
- Interactive 3D architecture & latency profiler: `flash-4/docs/architecture-3d.html`
- Serve locally: `python3 -m http.server 8090 --directory flash-4/docs`
