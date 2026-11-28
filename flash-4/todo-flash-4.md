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

- [x] **Native Blackwell `mma.sync.aligned.kind::mxf8f6f4.block_scale` in `attn_core` — MOVED TO `long-term-testing-todo.md`**
  - **Finding:** TensorFold v1.3.5's PTX layer (`qmm_frag.cuh`) only exposes FP16/BF16 `mma16816`, and quantizing `Q` to MXFP8 adds low-risk Q-rounding for ~1.5% 248K gain now that 248K is already `119.78 s` (`2,067 tok/s`). Moved to `long-term-testing-todo.md`.

---

## 3. Dense 6-bit EXL3 (`gdn_proj` + `gdn_out` + `attn_proj` + `attn_out`: `432 ms` prefill / `11.7 ms` N=1 decode)

- [x] **ExLlamaV3-Style Dense Prefill Staging (`FP16 Unpack + cuBLASLt` for `M >= 256`) — SHIPPED in Patch `0014`**
  - **Verified:** Unpacks `W_q` once per chunk into row-major FP16 (`tf_exl3_unpack_f16_nk_{12,8}_2`), runs cuBLASLt FP16→FP32 GEMM, and applies strided `tf_exl3_rot_out` directly into output columns (`TF_EXL3_PREFILL_CUBLAS=1`).
  - **Results:** Cold prefill 4K / 16K / 65K / 248K jumped **`2,064 / 2,123 / 2,086 / 2,067 tok/s`** (**248K TTFT `126.95 s → 119.78 s`**, **-7.17 s / +6.0%**, cumulative **-110.0 s** vs pre-Wave-1 `229.7 s`!). **120K needle (`ae38468a7f52`) & 4K long prompt (`3ac5af9f7e45`) 100% byte-identical to BF16**, 4/4 greedy & T=0.7 `draft_on == draft_off` exact.

- [x] **Zero-Copy Strided Output (`y_stride` + `col` offset) in `tf_exl3_group_12_2` (Decode N=1 — SHIPPED in Patch `0008`)**
  - **Verified:** Eliminated all 108 `copyCols` launches and `exl3_tmp` traffic in decode (`TF_EXL3_GROUP_STRIDED=1`); consistent N=1 gain across all 4 workloads (`code 97.2→97.5`, `prose 53.0→53.2`, `devops 88.8→89.0`, `json 109.2→109.7` tok/s), 4/4 greedy & T=0.7 `draft_on == draft_off` bit-exact.

- [x] **Fuse EXL3 `rot_in` (`suh * FWHT128`) into `gdn_norm` (`GDN Out`, 36×) — ANALYZED & CLOSED**
  - **Finding:** Decode already uses TensorFold's fused `flashnext_gdn_fused.cuh` chain (not standalone `fn_gdn_norm`), and in prefill `rot_in` is `< 0.2%` of chunk time after Patch `0014`.

---

## 4. MoE Routed 512×4-bit EXL3 (`453 ms` prefill / `16.5 ms` N=1 decode)

- [x] **Eliminate Split-K `z` Partials / Fuse `up` Finish + SwiGLU + `down rot_in` on Decode (`n <= 16`) — ANALYZED & CLOSED**
  - **Finding:** `exl3_grouped.zig` already uses `plan_finish_after` on decode; N=4 `experts` is at **236 GB/s (86% peak)** and N=1 is dominated by weight reads.

- [x] **EXL3 `exl3_prompt.cuh` Instruction-Scheduling & Fusion Tuning (Prefill, Stay on EXL3) — ANALYZED & CLOSED**
  - **Finding:** Confirmed via `Zeuss5/cuda-exl3` Blackwell benchmarks that `exl3_prompt.cuh` (3-stage `cp.async`, 64-pair tiles, Patch `0007`+`0010`) is already EXL3 state-of-the-art, limited by EXL3's ~74-instruction/8-weight trellis math.

---

## 5. Lower-Priority / Exploratory Prefill Items (from `architecture-3d.html`)

- [x] **Chunked WY Tensor-Core `gdn_core` Prefill — MOVED TO `long-term-testing-todo.md`**
  - **Finding:** Alters FP32 recurrence rounding vs step-by-step decode for ~3% prefill now that 248K is already `119.78 s` (`2,067 tok/s`). Moved to `long-term-testing-todo.md`.

- [x] **Optional Startup `madvise(MADV_WILLNEED)` Pre-Fault for PLE N-Gram Table — TESTED & REJECTED**
  - **Verified:** Built-in `TF_FLASHNEXT_TABLE_WARM=1` already implements `MADV_WILLNEED`, but warming the large host n-gram table causes unified-memory page-reclaim pressure on 128 GB GB10 at 245K context (`experts` spiked to `1,195 ms` on chunk 78). Keeping `TF_FLASHNEXT_TABLE_WARM=0` (8-thread `Gather.main` already keeps `ple` at `9–15 ms` with zero memory pressure).

---

## Reference
- Interactive 3D architecture & latency profiler: `flash-4/docs/architecture-3d.html`
- Serve locally: `python3 -m http.server 8090 --directory flash-4/docs`
