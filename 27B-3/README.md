# Qwen3.8-27B EXL3 4.00bpw (standard) + z-lab DFlash2 drafter on TensorFold 0.6.6 (one DGX Spark)

A copy of the former 27B-2 recipe ([its README](docs/27B-2.md); removed 2026-10-22) with the same 11 patches, scripts and
defaults, for the other 4.00 bpw quant:

| | 27B-3 (this) | 27B-2 |
|---|---|---|
| Target | `turboderp/Qwen3.8-27B-exl3 @ 4.00bpw` | `turboderp/Qwen3.8-27B-exl3 @ SC_4.00bpw_H5_V6` |
| Quantization | 4.00 bpw, 6-bit head, calibrated | 4.00 bpw, 5-bit head, self-calibrated (SC) |
| Vision tower | bf16 in the shards (identical to `Qwen/Qwen3.8-27B`'s, checked 333/333): loaded directly | 6-bit EXL3 in the shards, so a bf16 sidecar (`vision_bf16.safetensors`) is served instead |
| Model folder | `~/projects/models/Qwen3.8-27B-exl3-4.00bpw` (`build_model_dir.sh`) | `anubhav200/Qwen3.8-27B-exl3-4.00bpw-bf16vision` |
| Chat template | froggeric v22.5 (upstream kept as `chat_template.upstream.jinja`) | froggeric v22.5 |
| Container / port | `tf-27b-3` / 8888 | `tf-27b-2` / 8888 |

- Drafter: `z-lab/Qwen3.8-27B-DFlash2` (bf16; 27B-2 uses igor255's EXL3 quant of it). `DRAFT_REPO=igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` switches back (patch 0005 loads it; download it first, it is
  no longer kept on this box: z-lab's measured the same acceptance and 0-8% faster, below).

```bash
./27B-3/build_model_dir.sh  # once: copies the 4.00bpw pack from the HF cache (downloads it if missing) + the chat template
hf download z-lab/Qwen3.8-27B-DFlash2
./27B-3/start.sh            # port 8888, container tf-27b-3, PARALLEL=9
PARALLEL=1 ./27B-3/start.sh # best single-stream decode ("wide" draft trees)
./27B-3/end.sh
```

Knobs are the same as 27B-2's (see the header of `start.sh`): `PARALLEL` (9), `CONTEXT` (262144), `KV_DTYPE` (fp8),
`VISION` (1), `THINKING` (1), `MAX_TOKENS` (131072), `TENSORFOLD_KEEP_MAX` (128), `MODEL_PATH`, ...

## Measured (27B-3 vs 27B-2, back to back)

Startup: estimate 39.9 GiB, "up to 9 streams, each growing to 262144 tokens"; verify ms by rows 1: 64.2, 16: 71.0,
32: 79.3, 64: 116.5, 128: 226.5 (the same curve as 27B-2).

Decode, 1 client, `bench/same.py` (same prompt, greedy, thinking off, 200-token replies), `PARALLEL=9`:

| | 27B-3 (4.00bpw) | 27B-2 (SC_4.00bpw_H5_V6) |
|---|---:|---:|
| Prose | 51.1-51.6 tok/s (47 rounds, 152/696 accepted) | 49.4 tok/s (49 rounds, 150/711) |
| Code | 74.7-75.9 tok/s (32 rounds, 167/472) | 70.8-71.2 tok/s (34 rounds, 165/491) |

About 3-5% faster here, within run-to-run noise; it comes from a few fewer draft rounds on these two prompts (the two
quants produce different text). Prefill (unique prompts): 954 tok/s at 9.6K tokens, 936 tok/s at 40K (27B-2: 931 at
9.7K, 926 at 40K). Smoke tests: an image is described correctly, a follow-up on the same image reused 243 of 334 prompt
tokens from cache, and text answers are correct.

Not re-measured for 27B-3 (expected to match 27B-2, which has the same engine, patches and memory use): multi-stream
throughput and full-context concurrency; see below.

## Drafter: igor255 EXL3 vs z-lab bf16 (27B-3, back to back, `PARALLEL=9`)

`igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw` against `z-lab/Qwen3.8-27B-DFlash2` (now the default).
Both end up as the same affine 4-bit drafter on the GPU. Aggregate tok/s at 1 / 2 / 4 / 8 clients.

| Test | igor255 (EXL3) | z-lab (bf16, default) |
|---|---|---|
| `same.py` prose (same prompt, greedy, 200 tok) | 50.9 / 89.3 / 144.9 / 205.1 | 52.6 / 91.0 / 143.5 / 209.2 |
| `same.py` code | 74.7 / 124.8 / 209.3 / 280.3 | 73.6 / 119.8 / 206.0 / 293.9 |
| `tput.py` essays (distinct prompts, T=1, 800 tok) | 41.5 / 74.2 / 122.3 / 173.7 | 43.7 / 77.0 / 130.9 / 181.6 |
| `tput.py --code` | 81.7 / 145.2 / 208.9 / 269.9 | 88.8 / 157.2 / 210.5 / 279.6 |
| All requests: tokens per round, drafts accepted | 4.39, 47% (23,749 / 50,394) | 4.37, 47% (23,399 / 49,484) |

Acceptance is the same; z-lab came out 0-8% ahead in most cells (z-lab ran second), close to run-to-run noise (~5-10%).

## Results measured on 27B-2 (same engine and patches)

### Patches

| Patch | What |
|---|---|
| 0001-0004 | From `qwen38-27b`: fp8 KV, larger media limits, etc. |
| 0005 | Load an EXL3 (trellis) drafter: dequantize to bf16, then the engine's affine 4-bit. Affects draft acceptance only, never target output. Verified vs the reference dequantizer (corr 0.9999986). |
| 0006 | EXL3 prefill GEMM tile BK 32 to 64: bit-identical output, 3-20% faster matmuls, +8.6% on a 4096-row chunk. |
| 0007 | Vision prefix cache: image/video prompts resume from and keep states, keyed by media hashes (as in flash-2). Single-GPU path and scheduler path. |
| 0008 | Elastic kept states: up to `TENSORFOLD_KEEP_MAX`, shed oldest-first when `MemAvailable` falls under the headroom. |
| 0009 | EXL3 verify linear: up to 4 passes of 16 rows share one weight decode (stock re-read and re-decoded the weights for every 16 rows); at 49-64 and 97-128 rows a 128-column block is split over 2 thread blocks to keep registers in bounds. Every row bit-identical to stock (layer test at every row count, and greedy 8-stream outputs identical); `TENSORFOLD_EXL3_MP=1` restores the stock kernel. |
| 0010 | Drafter 4-bit matmul: GB10 uses block 11 past 16 rows (10-40% faster drafter GEMMs at 24-192 rows; blocks never change bits). |
| 0011 | Scheduler: a round's GDN state replay runs on a side CUDA stream beside the next round's draft (same outputs; `TENSORFOLD_OVERLAP_REPLAY=0` turns it off). |

### Results (fp8 KV, 262144 context)

| | PARALLEL=8 | PARALLEL=1 |
|---|---|---|
| Decode code T=0 / T=1 | 68.5 / 76.8 tok/s | 80.3 / 75.2 |
| Decode chat T=0 / T=1 | 52.7 / 48.2 | 54.5 / 48.2 |
| Prefill 5-19K | 860-948 tok/s | 972-1049 |
| Prefill 80-105K | 806 | 914 |

Decode is at the memory-bandwidth floor: one verify row streams 15.3 GB of weights in ~62 ms (~245 GB/s), so only
draft acceptance can improve it. Prefill is compute-bound (Triton EXL3 GEMM 63-80 TFLOPS, attention ~76 TFLOPS at depth).

Vision: an image repeated gives `cached_tokens` 277 of 284 and identical output; a different question after the image
only reuses the text before the image (same as flash-2).

### Throughput by number of streams (short prompts, 800 new tokens each, T=1)

`bench/tput.py 1 2 4 8`: N requests sent at once; aggregate = all completion tokens / wall time.

Stock TensorFold's EXL3 verify kernel decoded the weights once per 16 rows, so a verify of 17-32 rows cost two full
weight passes and 64 rows four (gate_proj: 16 rows 204 us, 32 rows 388 us, 64 rows 730 us). With several streams the
drafted rows of all streams share one verify, so that cliff capped multi-stream throughput. Patch 0009 makes up to 48
rows share one decode (32 rows 222-247 us, 48 rows 222 us, 64 rows 343 us), and the verify curve became:

| Rows | 16 | 24 | 32 | 48 | 64 | 128 |
|---|---|---|---|---|---|---|
| Stock (ms) | 69 | 110 | 114 | 161 | 209 | 408 |
| 0009 (ms) | 70 | 75 | 78 | 93 | 132 | 244 |

| Streams | Essays: stock | Essays: 0009 | Code: stock | Code: 0009 |
|---|---|---|---|---|
| 1 | 39 tok/s | 41 tok/s | 85 tok/s | 88 tok/s |
| 2 | 69 | 71 | 129 | 149 |
| 4 | 104 | 119 | 157 | 219 |
| 6 | 114 | 141 | - | - |
| 8 | 117 | 165 | 170 | 257 |
| 12 | - | 200 | - | 277 |
| 16 | - | 214 | - | 293 |

(Aggregate; per-stream speed is aggregate / streams. 12 and 16 streams need `PARALLEL=16` (the default is 9: only 8-9 full 262K-token streams fit in memory, ~7.8 GiB each); a lone stream
is as fast at 16 as at 8.)

### What limits it now (16 streams, ~60 verify rows a round, ~190 ms a round)

| Part | ms / round | Why it is not lower |
|---|---|---|
| Verify forward | ~140 | EXL3 linears are ~110 ms: 64 rows cost ~1.5x one row. Weight decode and mma are not the limit (removing either changes nothing); a block of 8 warps at 228-255 registers is one per SM, so memory latency is poorly hidden. A rewrite with a warp per pass sharing decoded tiles through shared memory was bit-exact but slower, so it was dropped. |
| Draft (DFlash2, 5 layers x ~160 rows) | ~22 | Compute-bound 4-bit GEMMs (0010 helped ~10%). Capping the draft block at 8 or 6 changed nothing; a bf16 drafter accepts the same (6.06 vs 6.00 tok/round) at 3x the draft time. |
| GDN state replay | ~20 | Each stream's fp32 recurrent state (~150 MB) is read and rewritten; 0011 overlaps it with the draft (~4%). |
| Sampling, tree planning, Python | ~10 | |

Second round of tuning (0009 column split, 0010, 0011), same benchmark: 16 code streams 285-296 tok/s (was 293),
16 essay streams 205-208 (was 214), 8 code streams 260-272 (was 257) - within run-to-run noise (~7%, GPU temperature
moves the one-row verify between 64 and 72 ms). The kernel-level gains are real but small next to the fixed costs.

### Concurrency at the full window

Startup: "up to 8 streams, each growing to 262144 tokens". Tested with 8 simultaneous requests of ~249K tokens each
(`bench/bench.py conc 8 155000 1200`, each asked for a 1200-token essay): **8 of 8 completed, no errors or OOM**, memory
steady at about 90 GiB used. Prompts are prefilled one at a time (~225-390 s each, ~640-1100 tok/s at that depth);
decode of a stream running beside another's prefill is slow (~1.8 tok/s), and a lone stream decodes at ~22 tok/s with
250K of context. Each 262K stream costs about 8 GiB of fp8 KV.
Note: all eight were not provably decoding at the same instant, because prefills are serial.
