# Ornith-1.5-122B (NVFP4) + DFlash2 on TensorFold 1.1.7, one DGX Spark

The `ornith-1.5-122B` recipe (SGLang) ported to the latest TensorFold release, **v1.1.7**, whose CUDA server is the
native Zig engine (`tensorfold-native`). Same checkpoint ([ornith-ai/Ornith-1.5-122B-NVFP4](https://huggingface.co/ornith-ai/Ornith-1.5-122B-NVFP4),
served as stored), same drafter ([incoai/Ornith-1.5-122B-DFlash2](https://huggingface.co/incoai/Ornith-1.5-122B-DFlash2)),
same port and API: OpenAI-compatible on port 8888, model name `Ornith-1.5-122B`, reasoning (`reasoning_content`),
`qwen3_coder`-style tool calls, image input (PNG, JPEG, WebP). Only one of the two recipes can run at a time.

```bash
./ornith-122b-2/start.sh                     # builds the image once (~8 min); ready in ~60 s (~90 s the first time)
python3 ornith-122b-2/bench/tools_check.py   # functional tool-call check
./ornith-122b-2/end.sh
```

It uses the checkpoints already in the Hugging Face cache (the SGLang recipe's); `scripts/prepare.sh` downloads only
when they are missing. Settings are environment variables or `.env` lines (`scripts/config.sh`): `CONTEXT` (262144),
`PARALLEL` (4), `DRAFTER` (`dflash`; `none`), `KV_CACHE` (`bf16`; `fp8`), `VISION` (1), `THINKING` (1),
`TEMPERATURE`/`TOP_P`/`TOP_K` (0.6/0.95/20), `TENSORFOLD_PREFILL_SHARE` (0.85), `PORT` (8888). Extra arguments to
`start.sh` go to `tensorfold-native serve` last.

## Why this port, and what it needed

The earlier TensorFold port (`ornith-1.5-122b-tensorfold`, 1.1.1, removed) had to use the MLX affine 4-bit export,
because TensorFold read Qwen3.5-MoE only as MLX or EXL3; that export (attention/GDN projections at 4-bit, no
calibration) gave clearly worse output than the NVFP4 checkpoint. 1.1.7 still refuses a ModelOpt config for
`qwen3_5_moe` ("the routed experts read MLX tables only"), but its dense Qwen path reads ModelOpt NVFP4/FP8 and its
core has NVFP4 grouped experts (GLM's, Flash Next's). `patches/0001` joins the two, so this recipe serves the very
checkpoint the SGLang recipe serves: NVFP4 experts and head, FP8 attention/GDN projections, bf16 router and vision.

| Patch | What and why |
|---|---|
| `0001-qwen35-moe-nvfp4-experts` | admits a `qwen3_5_moe` ModelOpt checkpoint; packs each layer's 256 routed experts + the shared one (NVFP4 codes, e4m3 block scales, fp32 global scales as stored) for `cuda.experts.Nvfp4Experts` on the host's cores, straight into the arena; the routed rows run them with bf16 activations (W4A16, the checkpoint's own recipe for its experts) |
| `0002-qwen35-moe-dflash2` | stock drafts `qwen3_5_moe` only with an MLX MTP file; `--drafter` now opens a DFlash2 model for it as the dense family does, trained block 8 or 16 (re-port of the 1.1.1 recipe's patch) |
| `0003-moe-router-warp` | decode-sized routing (verify windows, steps) on a warp per expert instead of a 16x16 tile with one 3072-long FMA chain a thread: ~63 us -> a few us a layer, a one-row forward 38.8 -> 35.6 ms. Row-invariant (fixed per-lane order + fixed butterfly). `TF_QWEN_ROUTER=tile` restores stock |
| `0004-think-close-at-line-start` | `</think>` is an ordinary token here, the same id quoted or real: the reasoning ends at the first one that starts a line outside a fence, so a reasoning that quotes the tag stays reasoning (and its quoted calls are not run). `TENSORFOLD_THINK_LINE_CLOSE=0` restores stock |
| `0005-nll-tool` | `tf-qwen35-cuda <model> nll <ids> <out> <skip> [fp8]`: teacher-forced logprobs through verify windows on one engine, for the quality comparison below (test tool only) |

The image (`scripts/prepare.sh`) is `tensorfold-native` from TensorFold v1.1.7 (`59044397`) plus the patches, built
with Zig 0.17.0 on `nvcr.io/nvidia/pytorch:26.07-py3`, with libtfgrammar (structured output, forced tool choice) and
the decoders TensorFold pins and checks for `--vision` (libjpeg-turbo 3.1.4.1, libwebp 1.6.0) built from source; it
is rebuilt when a patch changes. The window costs the engine times on its first start (~29 s) are kept in
`~/.local/state/tf-ornith-122b-2/tensorfold/` for the next starts.

At startup: `model 70.26 GiB; 4 streams at once, 6.87 GiB each at a 262144-token window, of 37.5 GiB left after a 8.0
GiB reserve; prompts in passes of up to 4096 rows`, loaded in ~50 s.

## Quality

**Same checkpoint, same or better numerics than SGLang.** Teacher-forced mean NLL of four fixed texts (the repo's
`README.md`, `readme-detailed.md`, `ornith-1.5-122B/bench/bench.py`, `tensorfold.sh`; 15,853 scored tokens), on the
same token ids: SGLang's `input_token_logprobs` (its prefill path; fp8 KV, FP8 activations into the FP8 layers) and
TensorFold's verify-window path (`tf-qwen35-cuda nll`, the path replies take; bf16 activations against the stored
weights). `bench/nll.py` does both sides.

| Engine | KV cache | Router | mean NLL | vs SGLang |
|---|---|---|---:|---:|
| SGLang (`ornith-1.5-122B`) | fp8 e4m3 | | 1.2848 | |
| TensorFold, this recipe | bf16 (default) | warp (0003) | 1.2822 | -0.20% |
| TensorFold | bf16 | tile (stock) | 1.2804 | -0.34% |
| TensorFold | fp8 (`KV_CACHE=fp8`) | tile | 1.2834 | -0.11% |

Per-token |dlogp| between the engines is 0.146 mean / 0.039 median; between TensorFold's own bf16 and fp8 caches it is
0.082: in a 256-expert MoE a last-bit difference in a router logit swaps an expert, so small numeric differences
spread, both ways. The tile/warp router difference (+0.0018) is of that kind (fp32 sums in another order), not a
precision change.

**Exact speculation.** Greedy replies (3 prompts, 400 tokens, thinking off) hash the same with drafting and with
`"draft": false`, each alone and the three at once (`ALL EQUAL`). `tf-qwen35-cuda` window (widths 1-64) and tree
logits equal one-row steps bit for bit with the warp router.

**Functional checks** on the served recipe: `bench/tools_check.py` all pass (streamed and not, thinking on/off,
parallel calls, typed/nested/multiline arguments, a multi-turn loop with simulated results, a no-tool chat);
`bench/think_check.py` 12/12 (added `quoted_tag`: a reasoning that quotes `</think>` 8 times keeps all of it, the
answer follows); a PNG, a JPEG and a WebP of a red square and a blue circle are each described correctly.

## Results

DGX Spark (GB10), 2026-10-27, `PARALLEL=4`, a full 262K window reserved for every stream, KV bf16, the recipe's image
(raw output in `results/`). Same `bench/bench.py` and method as the SGLang recipe's README: N simultaneous streaming
chat requests with distinct prompts, 512 new tokens with `ignore_eos`; accept length = generated tokens / verify
rounds per stream. Single runs, ~5-10% noise.

### Decode, tok/s (N=1 per stream; N=2, N=4 total)

| Workload | TF N=1 | SGLang N=1 | TF N=2 | SGLang N=2 | TF N=4 | SGLang N=4 |
|---|---:|---:|---:|---:|---:|---:|
| code, greedy, thinking off | **63.4** | 49.1 | **72.6** | 55.6 | 90.4 | 93.3 |
| mixed, greedy, thinking off | **49.8** | 39.9 | 52.2 | 50.1 | **68.6** | 61.4 |
| mixed, T=0.6, thinking on | **46.0** | 34.1 | 48.2 | 51.5 | 64.1 | 64.8 |
| code, T=0.6, thinking on | 45.5 | 46.3 | 54.1 | | 76.6 | 78.5 |
| no drafting, greedy | 28.4 | 30.1 | | | | |

One stream through all 8 prompts of a set (`bench.py seq`, greedy, thinking off): code 63.6 tok/s (4.54 tokens a
round), mixed 54.1 (3.69). TensorFold verifies a draft tree the scheduler sizes from measured window costs (5-9
rows a round alone); SGLang a 6-token chain. Alone, TensorFold is 25-35% faster; at four streams the two are level,
because four streams' windows route to most of the 256 experts and both engines then stream the expert weights.

### Prefill, one request, nothing else running (`bench.py prefill`)

| Prompt tokens | TensorFold | SGLang | old TF port (MLX, 1.1.1) |
|---:|---:|---:|---:|
| 1,997 | 2,362 tok/s (0.85 s) | 2,762 | 1,466 |
| 7,877 | 2,494 (3.2 s) | 3,243 | 1,525 |
| 31,397 | 2,156 (14.6 s) | 2,561 | 1,380 |
| 98,037 | 1,443 (68 s) | 1,535 | 1,063 |

1.1.4+'s prompt recipe (4,096-row passes, tensor-core attention, a chained delta rule) closes most of the old gap:
within 6-23% of SGLang.

### A long prompt beside a decoding stream (`bench.py interleave 100000 3000`)

TensorFold steps a long prompt 4,096 rows at a time between other streams' rounds; `TENSORFOLD_PREFILL_SHARE=p` gives
the prompt steps share p of the time while others decode (upstream knob; this recipe sets 0.85):

| Setting | 98K prompt TTFT | decoding stream during it |
|---|---:|---:|
| share 1 (stock) | 78.8 s | 2.1 tok/s |
| **share 0.85** (default) | 76.2-95.1 s (two runs) | 10.7 tok/s |
| share 0.7 | 93.3 s | 19.7 tok/s |
| SGLang, `PREFILL_DECODE_INTERVAL=16` | 78-79 s | 10-12 tok/s |

### Full context: 4 streams x 245K tokens

`bench.py long 4 250000 256`: four distinct 245,037-token prompts sent at once, 256 new tokens each, greedy, thinking
off. All four completed in 1,138 s; minimum `MemAvailable` 9.7 GB. The prompts prefill one after another (~300 s
each, ~800 tok/s at this depth); a stream whose prompt is in decodes while the next one prefills.

| Request | TTFT TF | TTFT SGLang | Decode TF | Decode SGLang (interval 16) |
|---|---:|---:|---:|---:|
| 1st | 318.5 s | 302.8 s | 4.7 tok/s | 3.4 |
| 2nd | 614.3 s | 622.5 s | 5.1 | 12.3 |
| 3rd | 919.2 s | 930.1 s | 5.1 | 12.9 |
| 4th (nothing left to prefill) | 1,133.0 s | 1,235.1 s | **51.8 at 245K depth** | 39.0 |

Decode at depth no longer collapses as it did on 1.1.1 (12.5 tok/s at 245K): 1.1.4+ runs decode and verify attention
on tensor cores. Beside a 245K prefill, share 0.85 leaves a stream ~5 tok/s (a 4,096-row step at that depth takes
~5 s); `TENSORFOLD_PREFILL_SHARE=0.7` buys more there at the prompt's expense.

## Where a decode round goes, and what was tried

Stream-synced section timers on an eager one-row forward (40.3 ms with the timers; 38.5 ms captured), stock router:

| Section (48 layers) | ms | | Section | ms |
|---|---:|---|---|---:|
| GDN projections (FP8 qkv, z; bf16 a, b) | 10.1 | | routed experts up (gate+up) | 7.7 |
| GDN out projection (FP8) | 5.3 | | routed experts down | 3.8 |
| GDN recurrence, conv, norms | 1.8 | | **router + top-k + plan** | **4.1** |
| attention q/k/v + o (FP8) | 4.2 | | combine | 0.3 |
| attention core (bf16 cache) | 0.3 | | head (W4A16 NVFP4) + norms | 2.6 |

Patch 0003 end to end (`results/tf-router-ab.md`, the recipe's server, `TF_QWEN_ROUTER=tile` then the default, back to
back; one stream through each set's 8 prompts, and 4 streams):

| Router | code, 1 stream | mixed, 1 stream | code, 4 streams | mixed, 4 streams |
|---|---:|---:|---:|---:|
| tile (stock) | 60.1 tok/s (4.73 a round) | 54.2 (4.00) | 83.2 | 66.6 |
| **warp (0003)** | **63.8** (4.73) | 54.6 (3.69) | **93.1** | 67.6 |

The same acceptance on code gives +6% from the shorter rounds alone; on the mixed set the greedy replies differ in
places (the router's fp32 sums in another order), so acceptance moved, and tokens a second held.

The weights a one-row step reads (~7 GB: 3.2 GB GDN FP8, 0.9 GB attention FP8, 2.3 GB of 9 experts, 0.43 GB head)
stream at ~190-230 GB/s in their kernels; the router stood out (0.08 GB in 3 ms), hence 0003. An 8-row window of
distinct tokens costs 65.7 ms, 34 of it in the experts: each extra row pulls in more of the 256 experts, so wide
windows are what a MoE round pays for, as on SGLang (~5 ms a verified token there).

Measured and not kept (within noise, or a bad trade):

| Change | Result |
|---|---|
| Calibrate window costs on distinct token ids (stock: token 0 in every row, so rows share experts and wide windows look cheap) and/or unpadded verify windows of 2-16 rows (the 1.1.1 port's 0004) | code/mixed one stream 57.4/50.1 (both), 56.6/51.4 (unpadded), 57.6/50.5 (stock); four streams 91.4/72.2, 93.1/68.4, 96.1/66.8: the lane scheduler already sizes drafts from measured round costs |
| Smaller prompt passes (2,048 / 1,024 rows) for interleaving | prefill -6..-13% / -15..-28%, decode beside a 98K prompt 3.0 / 7.7 tok/s: the prefill share above is the better knob |

## Other differences from the SGLang recipe

- KV cache bf16 by default (fp8 e4m3 rows with a scale each by `KV_CACHE=fp8`; SGLang uses fp8 with one scale).
- Tool parsing: a nullable argument written as `null` now comes back as JSON null, as from SGLang (1.1.1 returned the
  string `"null"`).
- Sampling defaults, thinking default and served name match.
- Not covered: TensorFold's MTP drafter path for this family reads only an MLX `mtp-4bit.safetensors`; this
  checkpoint's MTP head (bf16) is not used. `--drafter-bits 0` (bf16 drafter) is refused for this family; the drafter
  runs at affine 4-bit, which only changes acceptance, never output.

## Files

| File | Purpose |
|---|---|
| `start.sh`, `end.sh` | launch (prepares if needed, waits for `serving`, smoke request) and stop |
| `scripts/config.sh`, `scripts/prepare.sh` | settings and defaults; the image build (`--image-only`, `--rebuild`) |
| `patches/0001`-`0005` | see above; each has a header with what, why and how it was verified |
| `bench/bench.py` | the SGLang recipe's benchmark (decode, seq, prefill, long, interleave) |
| `bench/nll.py` | teacher-forced NLL on either server (`score`, `compare`) |
| `bench/tools_check.py`, `bench/think_check.py`, `bench/template_check.py` | tool calls; reasoning split; chat template |
| `results/` | raw output of the runs above |

Debugging that helped: `TF_TIMING=1` prints the window-cost sweep and each round's draft/verify time; nsys records no
kernels from `tensorfold-native` or `tf-qwen35-cuda`, so time sections with stream syncs, or cost one kernel exactly
by launching it twice inside the captured graph and timing `tf-qwen35-cuda <model> step <ids> <out>`.
