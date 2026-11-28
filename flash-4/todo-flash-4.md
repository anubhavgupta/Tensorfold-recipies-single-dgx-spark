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
  - **Root cause:** Patch `0011` cut radix passes from 4 → 3 (`69.4 → 64.8 ms`), but `fn_qsa_select_body` still performs **5 full global reads** over `9.03 GB` of `SC[3072, 61250]` (3 radix + count + write).
  - **Goal:** After pass 0 (11-bit `shift = 21`), compact only the `< 2,000` candidates with `(k >> 21) >= (cut >> 21)` into SMEM so passes 1–2 and output writing never re-read global memory.
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

---

## Reference
- Interactive 3D architecture & latency profiler: `flash-4/docs/architecture-3d.html`
- Serve locally: `python3 -m http.server 8090 --directory flash-4/docs`
