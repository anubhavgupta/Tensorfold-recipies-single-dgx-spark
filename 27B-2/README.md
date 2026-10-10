# Qwen3.8-27B EXL3 4.00bpw + EXL3 DFlash2 drafter on TensorFold 0.6.6 (one DGX Spark)

- Target: `turboderp/Qwen3.8-27B-exl3 @ SC_4.00bpw_H5_V6` (about 16 GiB)
- Drafter: `igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw`
- TensorFold only (no TabbyAPI). A derived image is built from `patches/*.patch` (`--tf-patches`), rebuilt when they change.

```bash
./27B-2/convert_vision.sh   # once: the pack's vision tower is EXL3-quantized; writes ~/.cache/huggingface/vision-f16-27b.safetensors
./27B-2/start.sh            # port 8888, container tf-27b-2
PARALLEL=1 ./27B-2/start.sh # best single-stream decode ("wide" draft trees)
./27B-2/end.sh
```

Knobs (env): `PARALLEL` (16), `CONTEXT` (262144), `KV_DTYPE` (fp8), `VISION` (1), `VISION_MAX_IMAGES` (50),
`THINKING` (1), `TENSORFOLD_KEEP_MAX` (32), `TENSORFOLD_KEEP_HEADROOM_GIB` (8), `MEMORY_RESERVE_GIB`, `CHECKPOINT_SLOTS`.
Media limits as in flash-2 (body 192 MiB, images 64 MiB, video 64/192 MiB, 16384 image/video tokens).

## Patches

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

## Results (fp8 KV, 262144 context)

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

## Throughput by number of streams (short prompts, 800 new tokens each, T=1)

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

(Aggregate; per-stream speed is aggregate / streams. 12 and 16 streams need `PARALLEL=16`, now the default; a lone stream
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

## Concurrency at the full window

Startup: "up to 8 streams, each growing to 262144 tokens". Tested with 8 simultaneous requests of ~249K tokens each
(`bench/bench.py conc 8 155000 1200`, each asked for a 1200-token essay): **8 of 8 completed, no errors or OOM**, memory
steady at about 90 GiB used. Prompts are prefilled one at a time (~225-390 s each, ~640-1100 tok/s at that depth);
decode of a stream running beside another's prefill is slow (~1.8 tok/s), and a lone stream decodes at ~22 tok/s with
250K of context. Each 262K stream costs about 8 GiB of fp8 KV.
Note: all eight were not provably decoding at the same instant, because prefills are serial.
