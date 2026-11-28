# flash-4 Optimization TODOs (Post-Patches 0001–0012)

Baseline on NVIDIA DGX Spark (GB10, 128 GB LPDDR5X, 273 GB/s, 48 SMs, 24 MiB L2) at commit `605f25c`:
- **Cold Prefill:** 4K `2,053` / 16K `2,050–2,101` / 65K `2,004–2,078` / **248K `1,928 tok/s` (`128.45 s`)**
- **Decode N=1 (steady):** `code 96.9–110`, `prose 52.7–54.4`, `devops 88.3`, `json 109.5–116.3` tok/s (`verify 35.6–54.4 ms` + `draft 3.1 ms`)
- **Decode N=4 (steady):** `code 188.8`, `prose 108.7`, `devops 180.2`, `json 209.8` tok/s (`experts` at 236 GB/s = 86% peak)
- **Capacity:** 8 full 262K streams max (`5.10 GiB`/stream, MXFP8 KV); 6 full 262K streams + 8 GiB warm cache live.

---

## 1. Decode (N=1) — Hyper-Connections (`hc_attn + hc_mlp`: `9.9 ms` = 20% of round, ~128 GB/s)

- [x] **100% Exact BF16 Fused Decode `W_up [10240, 320] + hcDenseMix` Kernel — SHIPPED in Patch `0013`**
  - **Verified:** Instantiated `fn_hcn_up1` (`hcn::upmix_body<1, 5>`, 80 CTAs, 14.2 KiB SMEM) in `readoutNative` (`TF_FLASHNEXT_HC_UPMIX_FUSED=1`). Saved **`-2.6 to -4.1 ms/round`** across 2–8 row windows, boosting N=1 decode **+6.9% to +9.7%** (`code 91.0→97.3`, `prose 49.3→53.4`, `devops 82.9→88.7`, `json 99.7→109.4` tok/s) and N=4 decode **+4.1% to +8.0%** (`prose 101.9→110.1`, `json 195.6→211.1` tok/s). 4/4 greedy & T=0.7 SHAs 100% byte-identical.

- [x] **Column-Parallel Fused `hcWriteback + hcNormalize` for Decode (`n < 64`) — ANALYZED & CLOSED**
  - **Finding:** `hcWriteback` runs column-parallel (`10` CTAs/row) to compute partial sums `pss[10]`, while `hcNormalize` requires the sum across all 10 column blocks before normalizing — so single-launch column-parallel fusion requires a grid sync (not faster than `fn_hc_normalize_cols`, which is already column-parallel at `0.7 ms` total). Patch `0013` (`fn_hcn_up1`) also eliminated `norm` reads from `W_up + hcDenseMix`.

---

## 2. Long-Context Prefill (245K) — QSA + MXFP8 Attention (`318.3 ms` = 19.5% of chunk)

- [x] **SMEM-Compacted 2-Pass `fn_qsa_select` (`64.8 ms → 51.0 ms` @ 245K — SHIPPED in Patch `0011`)**
  - **Verified:** Cut 245K `qsa_select` from `64.8 ms → 51.0 ms` (-13.8 ms/chunk), pushing **248K cold prefill to `125.85 s` (`1,966.4 tok/s`, -2.6 s)** and **65K to `2,074.7 tok/s`**. 120K needle byte-identical to BF16 (`sha ae38468a7f52`).

- [ ] **Native Blackwell `mma.sync.aligned.kind::mxf8f6f4.block_scale` in `attn_core` (`182.4 ms` @ 245K)**
  - **Root cause:** Patch `0012` halved KV reads (`77 GB → 38.5 GB`, cutting `attn_core` `298 → 182.4 ms` @ ~211 GB/s = 77% peak), but (1) still runs software `kv8_convert` + SMEM roundtrip into `tiles_kv` with an extra `__syncthreads()`, and (2) holding both raw MXFP8 stage (`17.4 KB`) and BF16 `tiles_kv` (`26.6 KB`) raises SMEM to `44.0 KB`, dropping occupancy from `3 → 2 CTAs/SM`.
  - **How to fix:** Quantize the CTA's small `Q` tile to MXFP8 once at CTA start, and execute `mma.sync.aligned.kind::mxf8f6f4.block_scale` directly on staged `E4M3 + E8M0/32` KV tiles — eliminating `kv8_convert` and `tiles_kv`, dropping SMEM to `~26 KB` (restoring **3 CTAs/SM**), and doubling TC throughput.
  - **Est. gain:** `-25 to -35 ms/chunk` (`182.4 → ~145–155 ms`, **-2.5 s** off 248K TTFT, >90% GB10 BW ceiling).

---

## 3. Dense 6-bit EXL3 (`gdn_proj` + `gdn_out` + `attn_proj` + `attn_out`: `432 ms` prefill / `11.7 ms` N=1 decode)

- [ ] **ExLlamaV3-Style Dense Prefill Staging (`FP16 Unpack + cuBLAS` for `M = 3,072`) — OR Custom 3-Stage `cp.async` Transposed 6-bit Dense Prompt Kernel (Prefill)**
  - **Root cause:** `exl3_linear_wc` runs in 128-row blocks (24× weight re-reads per 3,072-row chunk, ALU-bound on 6-bit trellis decode). Earlier Wave-3 `unpack + BF16 qmmf` was slower due to 1,024-row blocks and a layout mismatch.
  - **Goal:** Either (a) **ExLlamaV3 default (`Prefill staging`)**: unpack each 6-bit dense `W_q` once per 3,072-row chunk into FP16 scratch and run native **FP16 cuBLAS `cublasGemmEx`** (FP32 acc, within 1–3% of native FP16), or (b) apply `exl3_prompt.cuh`'s 3-stage `cp.async` transposed SMEM design directly to 6-bit dense projections without writing unpacked weights to DRAM.
  - **Est. gain:** `-100 to -150 ms/chunk` (**+8–11% cold prefill**, toward `~2,250+ tok/s`), 100% lossless.

- [x] **Zero-Copy Strided Output (`y_stride` + `col` offset) in `tf_exl3_group_12_2` (Decode N=1 — SHIPPED in Patch `0008`)**
  - **Verified:** Eliminated all 108 `copyCols` launches and `exl3_tmp` traffic in decode (`TF_EXL3_GROUP_STRIDED=1`); consistent N=1 gain across all 4 workloads (`code 97.2→97.5`, `prose 53.0→53.2`, `devops 88.8→89.0`, `json 109.2→109.7` tok/s), 4/4 greedy & T=0.7 `draft_on == draft_off` bit-exact.

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

- [x] **Optional Startup `madvise(MADV_WILLNEED)` Pre-Fault for PLE N-Gram Table — TESTED & REJECTED**
  - **Verified:** Built-in `TF_FLASHNEXT_TABLE_WARM=1` already implements `MADV_WILLNEED`, but warming the large host n-gram table causes unified-memory page-reclaim pressure on 128 GB GB10 at 245K context (`experts` spiked to `1,195 ms` on chunk 78). Keeping `TF_FLASHNEXT_TABLE_WARM=0` (8-thread `Gather.main` already keeps `ple` at `9–15 ms` with zero memory pressure).

---

## Reference
- Interactive 3D architecture & latency profiler: `flash-4/docs/architecture-3d.html`
- Serve locally: `python3 -m http.server 8090 --directory flash-4/docs`
