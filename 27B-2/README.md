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

Knobs (env): `PARALLEL` (8), `CONTEXT` (262144), `KV_DTYPE` (fp8), `VISION` (1), `VISION_MAX_IMAGES` (50),
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

## Throughput by number of streams (PARALLEL=8, short prompts, 800 new tokens each, T=1)

`bench/tput.py 1 2 4 8`: N requests sent at once; aggregate = all completion tokens / wall time.

| Streams | Essays: aggregate | Essays: per stream | Code: aggregate | Code: per stream |
|---|---|---|---|---|
| 1 | 39 tok/s | 40 | 85 tok/s | 90 |
| 2 | 69 | 41 | 129 | 70 |
| 4 | 104 | 30 | 157 | 42 |
| 6 | 114 | 22 | - | - |
| 8 | 117 | 16 | 170 | 24 |

Aggregate throughput rises about 2-3x up to 4 streams and flattens after that: more streams leave room for fewer draft rows
each (a verify of up to 16 rows costs about the same as 1, while 24+ rows cost much more).

## Concurrency at the full window

Startup: "up to 8 streams, each growing to 262144 tokens". Tested with 8 simultaneous requests of ~249K tokens each
(`bench/bench.py conc 8 155000 1200`, each asked for a 1200-token essay): **8 of 8 completed, no errors or OOM**, memory
steady at about 90 GiB used. Prompts are prefilled one at a time (~225-390 s each, ~640-1100 tok/s at that depth);
decode of a stream running beside another's prefill is slow (~1.8 tok/s), and a lone stream decodes at ~22 tok/s with
250K of context. Each 262K stream costs about 8 GiB of fp8 KV.
Note: all eight were not provably decoding at the same instant, because prefills are serial.
