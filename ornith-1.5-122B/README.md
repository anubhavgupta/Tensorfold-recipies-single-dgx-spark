# Ornith-1.5-122B (NVFP4) + DFlash2 on SGLang, one DGX Spark

Serves [ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4) with
[SGLang](https://github.com/sgl-project/sglang), drafting with the DFlash2 block-diffusion drafter
[incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2), on one DGX Spark
(GB10, 128 GB unified memory). OpenAI-compatible API on port 8888, model name `Ornith-1.5-122B`, with reasoning
(`reasoning_content`), tool calling (`qwen3_coder`) and image input.

```bash
hf download ornith-ai/Ornith-1.5-122B-NVFP4      # 77.6 GB
hf download incoai/Ornith-1.5-122B-DFlash2       # 1.9 GB
./ornith-1.5-122B/start.sh                       # ~9-10 min to ready (weights load ~7 min)
python3 ornith-1.5-122B/bench/tools_check.py     # functional tool-call check
./ornith-1.5-122B/end.sh
```

Every knob is an environment variable (or a line in `ornith-1.5-122B/.env`), documented in the header of
`start.sh`; extra SGLang flags can be appended to `start.sh` and win over the defaults.

## Recipe

| Setting | Value | Why |
|---|---|---|
| Image | `lmsysorg/sglang:dev-cu13` @ `035f29e9` (main `65f759144`), stock | has `DFlash2DraftModel`, DFLASH for `Qwen3_5MoeForConditionalGeneration`, quantized `lm_head` |
| Speculation | DFLASH, `DF_BLOCK=6` verified tokens per round | measured optimum on GB10, see below (the drafter trained at 16) |
| Streams | `PARALLEL=4` x `CONTEXT=262144` | every stream can hold a full 262K context at once (verified, below) |
| KV cache | fp8 e4m3; target 12 KiB/token + draft 12 KiB/token | 1,048,600 tokens = 24 GiB |
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

## Tried and not kept

- `--moe-runner-backend marlin`: same decode speed, prefill 1,600 vs 2,600 tok/s, KV pool shrank to 833K tokens.
- `--speculative-draft-model-quantization fp8`: ~1.5% faster rounds (noise level), and the draft loads at 1.44 GB.
- `CHUNKED_PREFILL=16384`: a 32K prefill stalled with `MemAvailable` at 2 GB (killed).
- SGLang's adaptive speculation (`--speculative-adaptive`) is EAGLE-only; ragged verify is DSpark-only.

## Files

| File | Purpose |
|---|---|
| `start.sh` | launcher (knobs in its header), sizes pools, waits for readiness |
| `end.sh` | stops and removes the container, saves its log to `last.log` |
| `bench/bench.py` | `decode`, `prefill`, `interleave`, `long` benchmarks (stdlib only) |
| `bench/tools_check.py` | functional tool-call check |
| `results/final-d6.md` | raw output of the final benchmark run |
