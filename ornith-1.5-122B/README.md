# Ornith-1.5-122B (NVFP4) + DFlash2 on SGLang, one DGX Spark

Serves [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) with
[SGLang](https://github.com/sgl-project/sglang), drafting with the DFlash2 block-diffusion drafter
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2), on one DGX Spark
(GB10, 128 GB unified memory). OpenAI-compatible API on port 8888, model name `Ornith-1.5-122B`, with reasoning
(`reasoning_content`), tool calling (`qwen3_coder`) and image input.

```bash
hf download ornith-ai/Ornith-1.5-122B-NVFP4      # 77.6 GB
hf download incoai/Ornith-1.5-122B-DFlash2       # 1.9 GB
./ornith-1.5-122B/start.sh                       # ~70 s to ready (weights ~33 s, patches/0001)
python3 ornith-1.5-122B/bench/tools_check.py     # functional tool-call check
./ornith-1.5-122B/end.sh
```

Every knob is an environment variable (or a line in `ornith-1.5-122B/.env`), documented in the header of
`start.sh`; extra SGLang flags can be appended to `start.sh` and win over the defaults.

## Recipe

| Setting | Value | Why |
|---|---|---|
| Image | `lmsysorg/sglang:dev-cu13` @ `035f29e9` (main `65f759144`) + `patches/0001` (fast weight load) | has `DFlash2DraftModel`, DFLASH for `Qwen3_5MoeForConditionalGeneration`, quantized `lm_head` |
| Speculation | DFLASH, `DF_BLOCK=6` verified tokens per round | measured optimum on GB10, see below (the drafter trained at 16) |
| Streams | `PARALLEL=4` x `CONTEXT=262144` | every stream can hold a full 262K context at once with `FAST_LOAD=0` (verified, below); ~252K each with the fast load (see Boot) |
| KV cache | fp8 e4m3; target 12 KiB/token + draft 12 KiB/token | 1,048,600 tokens = 24 GiB (1.00-1.01M with the fast load) |
| GDN state | bf16 SSM, `extra_buffer_lazy`, 90 slots (16 for running requests, the rest are the hybrid prefix cache) | 74.5 MiB per slot |
| `MEM_FRACTION` | 0.91 | leaves ~8 GB `MemAvailable` at idle, >= 3.9 GB under 4 x 245K prompts |
| MoE kernels | `flashinfer_cutlass` | `marlin` decodes the same and prefills 40% slower |
| Attention | flashinfer (target and draft) | `trtllm_mha` (the drafter card's) is SM100-only |
| Prefill | 8192-token chunks, `PREFILL_DECODE_INTERVAL=16` | 16384 stalled a 32K prefill with `MemAvailable` at 2 GB; the interval keeps other streams decoding |
| Sampling defaults | temperature 0.6, top_p 0.95, top_k 20 (the card's coding profile, also the drafter's eval setting) | request values win |
| Thinking | on (`preserve_thinking`), off per request with `chat_template_kwargs: {"enable_thinking": false}` | |

The model is a Qwen3.5-122B-A10B-shaped hybrid: 48 layers (36 Gated DeltaNet + 12 full attention, 2 KV heads x
256), 256 experts with 8 routed + 1 shared, vision tower in bf16. Weights load as 68.1 GB (MTP head not loaded),
the drafter as 1.1 GB.

## Results

DGX Spark (GB10), 2026-10-21, `PARALLEL=4`, full 262K reserved for every stream. Method (`bench/bench.py`): N
simultaneous streaming chat requests with distinct prompts (no prefix-cache hits), 512 new tokens each with
`ignore_eos`. Total = all output tokens / wall time; per-stream = mean decode rate after the first token; accept
length = generated tokens / verify rounds from the server's counters. Code = 8 programming prompts, mixed = 4 prose
+ 2 code + 2 math (at N=1 only the first prompt of the set runs). Noise between runs is ~5-10% (acceptance moves
with the prompt).

### Decode (final config, `DF_BLOCK=6`)

| Workload | N=1 tok/s | N=2 total (per stream) | N=4 total (per stream) | accept len N=1 |
|---|---:|---:|---:|---:|
| code, greedy, thinking off | 49.1 | 55.6 (32.5) | 93.3 (24.6) | 3.71 |
| mixed, greedy, thinking off | 39.9 | 50.1 (26.7) | 61.4 (17.6) | 3.07 |
| mixed, T=0.6, thinking on | 34.1 | 51.5 (27.0) | 64.8 (17.5) | 2.60 |
| code, T=0.6, thinking on | 46.3 | | 78.5 (20.2) | 3.56 |
| no speculation (`SPEC=off`), greedy | 30.1 | 43.2 (21.7) | 62.9-64.6 (15.8-16.3) | |

TTFT for these short prompts is 0.17-0.32 s.

### Choosing the block size

Same method, greedy, thinking off, one boot per row, back to back:

| `DF_BLOCK` | code N=1 | mixed N=1 | code N=4 total | mixed N=4 total | ms per round (N=1) |
|---:|---:|---:|---:|---:|---:|
| off | 30.2 | 30.1 | 64.6 | 62.9 | 33 |
| 4 | 45.6 | 38.6 | 87.9 | 65.1 | 65 |
| **6** | **54.5** | **36.9** | **94.4** | **62.7** | 75 |
| 8 | 50.6 | 34.9 | 100.0 | 60.1 | 82 |
| 16 (drafter card) | 54.3 | 26.9 | 72.8 | 47.4 | 110 |

A verify round costs ~33 ms + ~4.8 ms per verified token: each extra token routes to more of the 256 experts, and
reading their NVFP4 weights is what the round spends its time on (profile of a D=6 round, 82 ms: MoE grouped GEMM
45%, FP8 attention/GDN projections 26%, bf16 drafter GEMMs 12%, W4A16 `lm_head` 10%). Code accepts ~4 tokens per
round with 6 or 16 verified; prose and reasoning accept ~3 either way, so 16 wastes ~35 ms per round on rejected
tokens. 6 is the best single-stream compromise; `DF_BLOCK=8` gives a little more total throughput on code at 4
streams, `DF_BLOCK=4` a little more on prose.

### Prefill

One request at a time, unique word-salad prompt, `max_tokens=1`, nothing else running:

| Prompt tokens | TTFT | Prefill tok/s |
|---:|---:|---:|
| 1,997 | 0.72 s | 2,762 |
| 7,877 | 2.43 s | 3,243 |
| 31,397 | 12.26 s | 2,561 |
| 98,037 | 63.9 s | 1,535 |

While other streams are decoding, `PREFILL_DECODE_INTERVAL=16` runs 16 decode rounds between prefill chunks
(`bench.py interleave 100000 3000`, one stream decoding while a 98K prompt prefills):

| Interval | 98K prefill | Decoding stream during that prefill |
|---:|---:|---:|
| 0 (stock) | 67.2 s (1,458 tok/s) | 0.6 tok/s |
| **16** | 78.3-79.2 s (1,237-1,252 tok/s) | 10.3-12.2 tok/s |

### Full context: 4 streams x 245K tokens

`bench.py long 4 250000 256`: four distinct 245,037-token prompts sent at once, 256 new tokens each, greedy,
thinking off. The prompts prefill one after another (~300 s each); the earlier ones decode while the later ones
prefill, so their decode rate is mostly the interleave rate.

| Request | TTFT | Decode tok/s (interval 16) | Decode tok/s (interval 0) |
|---|---:|---:|---:|
| 1st | 302.8 s | 3.4 | 0.3 |
| 2nd | 622.5 s | 12.3 | 0.8 |
| 3rd | 930.1 s | 12.9 | 0.8 |
| 4th (nothing left to prefill) | 1,235.1 s | 39.0 at 245K depth | 34.2 |

All four completed; minimum `MemAvailable` 3.9 GB with interval 16 (8.2 GB with interval 0). Swap use reached
5.3 GB over the session, which also ran a large HF download in parallel; watch it on a busy host.
Average prefill at this depth is ~800 tok/s.

## Boot

| Phase | Stock | Now | What changed |
|---|---:|---:|---|
| target weights (68 GB, 113,047 tensors) | 417-470 s | 32-33 s | `patches/0001` (below) |
| drafter weights | 12-15 s | ~1 s | same |
| FlashInfer autotune + kernel JIT | ~100 s | ~2 s | `.cache/sglang-{flashinfer,jit,nv}` persisted (first boot still tunes) |
| pools, CUDA graphs, warm-up | ~40 s | ~30 s | |
| **ready** | **~640 s** | **~70 s** | |

Cause of the slow load: on this box (GB10, CUDA 13, driver 580) a host-to-device copy from mmap'd safetensors pages the
process has not touched runs at ~0.15 GB/s, 10 ms per 1.5 MB expert tensor, even with the file in the page cache; from
touched or anonymous pages it runs at 37-44 GB/s (`tools/copy_bench.py`). SGLang's loader copies straight from untouched
mmap views. `patches/0001` makes the loader workers read their shard sequentially and touch its pages, hands large
tensors out as anonymous copies, and drops the shards' page cache once the model is loaded (this platform sizes the KV
pool from free memory that does not count cache). SGLang's own `--weight-loader-prefetch-checkpoints` only got to
371 s (the cache cannot hold the checkpoint beside the weights). Greedy output is byte-identical with and without the
patch (4 prompts x 300 tokens, `FAST_LOAD=0` vs `1`).

Cost: the scheduler keeps ~1-2 GB more host memory after the fast load, so the KV pool is 1.00-1.01M tokens instead of
1,048,600: four simultaneous ~252K-token contexts instead of 262K (requests beyond that queue). `FAST_LOAD=0` restores
the stock load (and the full pool).

The persisted autotune cache fixes the MoE tactic FlashInfer picked on the first boot. Tactics differ in rounding, so
greedy text (and with it acceptance on a given prompt set, +-5%) depends on that boot; round time does not
(75.9 vs 76.3 ms at 3.47 vs 3.64 accepted). Delete `.cache/sglang-flashinfer` to retune.

## Memory

At ready: `MemAvailable` ~8 GB (121.6 GiB total). Pools: target weights 68.1 GB, drafter 1.1 GB, target KV 12 GB,
draft KV 12 GB, GDN states 6.6 GB (90 slots) + 2.1 GB verify intermediates (6 per running request), CUDA graphs
~1 GB. `start.sh` sizes the GDN pool from what is left after `PARALLEL x CONTEXT` of KV (constants measured on
this box, at the top of its memory block) and refuses a `PARALLEL` that does not fit, printing the maximum.

A fifth full-context stream needs ~7.6 GB more than `MEM_FRACTION=0.91` leaves; the largest single item is the
DFlash draft KV, which SGLang allocates for every token even though the drafter only attends to a 2,048-token
sliding window (`--speculative-draft-window-size` narrows what the draft reads, not what it stores). Shrinking it
would need a per-request ring buffer for the draft KV, which loses the draft context on prefix-cache hits; not done.
For more streams at shorter contexts lower `CONTEXT` (e.g. `PARALLEL=8 CONTEXT=131072`).

## Checks

- `bench/tools_check.py`: streamed and non-streamed x thinking on/off, two parallel calls, typed/nested/multiline
  arguments (escaped quotes, Unicode, a dotted tool name, the string "3", a "null" string inside an array), a
  multi-turn loop with simulated tool results, and a no-tool chat with tools offered: all pass.
- Known model behaviour: for a nullable argument after Python code, the checkpoint usually writes Python's `None`
  instead of `null` (raw output checked with `/generate`: 5 of 6 samples). The parser maps a literal `null` to JSON
  null; `"None"` stays a string because it is a valid value for `["string","null"]`.
- Vision: an image of a red rectangle and a blue circle is described correctly.

## Why decode is not faster (profile, 2026-10-22)

One stream, code, `DF_BLOCK=6`: a round is ~80 ms = draft 9.8 ms + verify 70.6 ms, for 3.5-4.0 accepted tokens
(torch profiler, 20 rounds). The verify is close to the memory-bandwidth floor of this checkpoint on GB10:

| Verify part | ms/round | Bytes | Rate |
|---|---:|---|---|
| NVFP4 routed experts (cutlass grouped GEMM) | 37.5 | ~40 distinct experts per layer for 6 tokens x 5.3 MB x 48 layers | ~240 GB/s |
| FP8 GDN/attention projections (cuBLASLt) | ~22 | 4.1 GB | ~230 GB/s; CUTLASS and Triton FP8 measured no faster at M=6 (`tools/gemm_bench.py`) |
| shared experts + `lm_head` (NVFP4 via Marlin) | ~4-5 | 0.7 GB | ~150-200 GB/s |
| GDN state update, norms, router, small bf16 GEMMs | ~6 | intermediate GDN states for rollback (12 MB/layer) | small GEMMs run on a side stream; the norm's 45 us is mostly a PDL wait |

Each extra verified token routes to ~6-7 more experts per layer (~4.8 ms a round), and the drafter's acceptance does
not improve with a wider draft: drafting the trained 16 positions and verifying the first 6 accepts 3.66 tokens a round,
drafting 6 accepts 3.64 (per-round accept lengths logged from a `DF_BLOCK=16` run, 849 rounds). At ~65 ms (the floor
above plus the draft) and ~3.6 accepted tokens, the ceiling for code at one stream is ~55 tok/s with this checkpoint
and chain verification; 85 tok/s is not reachable on this path. The TensorFold port (`../ornith-1.5-122b-tensorfold`,
MLX 4-bit, tree verification, ~4.7 tokens a round) measures 71.8 tok/s code at one stream and 97.6 total at 4.

Block size, re-measured with `bench.py seq` (8 code prompts one after another, greedy) and `decode 4`:

| `DF_BLOCK` | code 1 stream | accept len | code 4 streams total |
|---:|---:|---:|---:|
| 5 | 46.0 | 3.26 | 82.5 |
| **6** | 45.9-47.7 | 3.47-3.64 | **90.3** |
| 7 | 49.2 | 3.93 | 81.2 |
| 8 | 50.9 | 4.31 | 76.6 |

`DF_BLOCK=8` is ~10% faster for a single user writing code and ~15% slower at 4 streams (and slower on prose, see
"Choosing the block size"); 6 stays the default.

## Tried and not kept

- `--moe-runner-backend marlin`: same decode speed, prefill 1,600 vs 2,600 tok/s, KV pool shrank to 833K tokens.
- `--speculative-draft-model-quantization fp8`: 49.9 vs 49.1 tok/s (`seq`, noise level), the draft loads at 1.48 GB
  and the KV pool shrank to 947K tokens.
- Drafting 16 positions and verifying fewer: no acceptance gain (above).
- `CHUNKED_PREFILL=16384`: a 32K prefill stalled with `MemAvailable` at 2 GB (killed).
- SGLang's adaptive speculation (`--speculative-adaptive`) is EAGLE-only; ragged verify is DSpark-only.

## Files

| File | Purpose |
|---|---|
| `start.sh` | launcher (knobs in its header), sizes pools, waits for readiness |
| `end.sh` | stops and removes the container, saves its log to `last.log` |
| `bench/bench.py` | `decode`, `seq`, `prefill`, `interleave`, `long` benchmarks (stdlib only) |
| `bench/tools_check.py` | functional tool-call check |
| `results/final-d6.md` | raw output of the final benchmark run |
| `patches/0001-weight-load-prefault.patch` | fast weight load (applied in the container at start; `PATCHES`, `EXTRA_PATCHES` dirs) |
| `tools/copy_bench.py`, `tools/gemm_bench.py` | host-to-device copy speed by memory kind; verify-shape GEMM backends |
