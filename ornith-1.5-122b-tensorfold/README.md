# Ornith-1.5-122B + DFlash2 on TensorFold 1.1.1 (native), one DGX Spark

The `ornith-1.5-122B` recipe (SGLang) ported to the latest TensorFold release, **v1.1.1**, whose CUDA server is the native
Zig engine (`tensorfold-native`). Same model, drafter, port and API: OpenAI-compatible on port 8888, model name
`Ornith-1.5-122B`, reasoning, `qwen3_coder`-style tool calls, image input. Only one of the two recipes can run at a time.

```bash
./ornith-1.5-122b-tensorfold/start.sh     # builds the image once (~6 min), downloads, serves (ready in ~100 s)
python3 ornith-1.5-122b-tensorfold/bench/tools_check.py
./ornith-1.5-122b-tensorfold/end.sh
```

Settings are environment variables or `.env` lines (`scripts/config.sh`): `CONTEXT` (262144), `PARALLEL` (4),
`DRAFTER` (`dflash`; `none`), `KV_CACHE` (`bf16`; `fp8`), `VISION` (1), `THINKING` (1), `TEMPERATURE`/`TOP_P`/`TOP_K`
(0.6/0.95/20), `PORT` (8888). Extra arguments to `start.sh` go to `tensorfold-native serve` last.

## What the port needed

| Item | Why |
|---|---|
| Checkpoint `ornith-ai/Ornith-1.5-122B-MLX-4bit` (68.7 GB) instead of the NVFP4 one | TensorFold's CUDA engine reads Qwen3.5-MoE only as MLX affine 4-bit, group 64 (router gates 8-bit) |
| `scripts/fix_vision.py` | that repo's index maps the 333 `vision_tower.*` tensors to `model-vision.safetensors`, which it never uploaded. The script fetches those tensors (0.9 GB, bf16, one HTTP range request each) from `BobClawblaw/Ornith-1.5-122B-MLX-4bit` at a pinned revision and writes the file into the snapshot. Three tensors are compared byte for byte with the NVFP4 checkpoint's own tower first: identical |
| `patches/0001-qwen35-moe-dflash2.patch` | stock 1.1.1 drafts `qwen3_5_moe` only with the checkpoint's MTP layer (`mtp-4bit.safetensors`, which this export does not ship). The dense `qwen3_5` family on the same engine already drafts with DFlash2, so the patch lets `--drafter` open a DFlash2 model for the MoE family too (the MTP path is unchanged), and accepts a drafter trained with a 16-token block as well as 8 |
| `patches/0002-prompt-attention.patch` | prefill: a tensor-core attention kernel for prompt passes (see [Prefill](#prefill-one-request-nothing-else-running)). `TENSORFOLD_PROMPT_ATTN=0` restores the stock kernel |
| `patches/0004-narrow-verify-windows.patch` | decode: verify windows of 2-16 rows are no longer padded to 16, so the scheduler drafts ~6 tokens a round instead of ~13 (see [Decode](#decode-toks-per-stream-at-n1-total-at-n2-and-n4)). Ported from MiaAI-Lab's Ornith recipe (MIT) |
| `patches/0003-prompt-rows-1024.patch` | prefill: prompt passes of 1,024 rows instead of 128, so the 66 GiB of routed experts are read an eighth as often a prompt token |

The image is `tensorfold-native` built from TensorFold v1.1.1 (`b10231b9`) plus the patches with Zig 0.17.0 on
`nvcr.io/nvidia/pytorch:26.07-py3` (`scripts/prepare.sh`, rebuilt when a patch changes).

Checks: drafted greedy replies equal `"draft": false` (same digest); `bench/tools_check.py` passes (streamed and not,
thinking on/off, parallel calls, typed/nested/multiline arguments, a multi-turn loop with simulated results, a no-tool
chat); an image of a red rectangle and a blue circle is described correctly. At startup the engine reports
`model 68.47 GiB; 4 streams at once, 6.79 GiB each at a 262144-token window, of 39.5 GiB left after a 8.0 GiB reserve;
prompts in passes of up to 1024 rows` (stock: 66.39 GiB, 6.69 GiB, 41.2 GiB, 128 rows; the difference is the larger
prompt scratch).

## TensorFold 1.1.1 vs SGLang (`ornith-1.5-122B`)

Same machine, same `bench/bench.py` and method: N simultaneous streaming chat requests with distinct prompts, 512 new
tokens with `ignore_eos`, 4 streams each able to hold a full 262K context. Accept length = generated tokens / verify
rounds per stream from each server's counters. Note the weights differ: MLX affine 4-bit here, NVFP4 (ModelOpt,
FP8 attention/GDN projections) under SGLang. TensorFold verifies a draft tree (~14 drafted tokens a round with one
stream), SGLang a 6-token chain (`DF_BLOCK=6`, its best setting on this box).

### Decode, tok/s (per stream at N=1; total at N=2 and N=4)

With all four patches (0004 included; same method, same model and drafter):

| Workload | N=1 | N=2 total | N=4 total | SGLang N=1 / N=4 |
|---|---:|---:|---:|---:|
| code, greedy, thinking off | **71.8** | 84.5 | 97.6 | 49.1 / 93.3 |
| mixed, greedy, thinking off | **46.2** | 54.5 | 73.9 | 39.9 / 61.4 |
| mixed, T=0.6, thinking on | **50.6** | | 73.8 | 34.1 / 64.8 |

**Why 0004 matters.** The lane scheduler picks how many tokens to draft from window costs it measures at startup.
Stock pads every verify window of 2-16 rows to 16, so those widths all measured 57-60 ms and wide drafts looked free:
it drafted ~13 tokens a round, and a round took ~92 ms (~60 ms verify plus the drafter and host). On this MoE each
extra real row routes to more experts, so narrow windows really are cheaper: unpadded, 2 rows cost 39.8 ms, 6 rows
49.5, 16 rows 53.7; the scheduler drafts 5-6 tokens and a round takes ~65 ms. Fewer tokens are accepted a round (3.0
vs 3.7 on mixed text) but rounds are ~30% shorter. MiaAI-Lab's engine (same model and drafter) showed the same 5-6
drafted tokens and 61-67 ms rounds, which is how the cause was found. Drafted replies equal `"draft": false` with one
stream and with three concurrent streams; the tool-call check passes; one stream at 117K tokens decodes at 23.1 tok/s.

The table below is the build before 0004 (0001 only), kept for the comparison:

| Workload | TF N=1 | SGLang N=1 | TF N=2 | SGLang N=2 | TF N=4 | SGLang N=4 |
|---|---:|---:|---:|---:|---:|---:|
| code, greedy, thinking off | **74.0** | 49.1 | **77.0** | 55.6 | **96.1** | 93.3 |
| mixed, greedy, thinking off | **42.0** | 39.9 | **52.1** | 50.1 | **69.9** | 61.4 |
| mixed, T=0.6, thinking on | **42.1** | 34.1 | 51.3 | 51.5 | **74.5** | 64.8 |
| code, T=0.6, thinking on | 43.1 | **46.3** | | | **94.4** | 78.5 |
| accept length, code greedy N=1 | 7.01 | 3.71 | | | | |
| accept length, mixed greedy N=1 | 3.74 | 3.07 | | | | |

TensorFold's tree verify accepts more per round (7.0 vs 3.7 tokens on code) at a similar round time (~88 ms here,
~75 ms for SGLang's 6-token chain), so single-stream decode is up to 1.5x faster; at 4 streams the two are within
~15%. Single runs, ~5-10% noise (code thinking-on N=1 has one prompt and is the noisiest row).

### Prefill (one request, nothing else running)

`bench.py prefill`: unique word-salad prompts, `max_tokens=1`, thinking off.

| Prompt tokens | TF stock 1.1.1 | TF + 0002 (attention) | TF + 0002 + 0003 (1,024-row passes) | SGLang |
|---:|---:|---:|---:|---:|
| 2K | 597 tok/s (3.3 s) | 633 | **1,466** (1.4 s) | 2,762 |
| 8K | 497 (15.9 s) | 639 | **1,525** (5.2 s) | 3,243 |
| 31K | 271 (116 s) | 593 | **1,380** (22.8 s) | 2,561 |
| 98K | 107 (919 s) | 501 | **1,063** (92 s) | 1,535 |
| 212K (needle at 30%) | | | 735 incl. reply (289 s), retrieved | |

Where stock 1.1.1's prompt time went (stream-synced timers around each part of a pass, 8-32K tokens; a 128-row pass
~205 ms): the routed experts ~144 ms, because 128 rows route to nearly all 256 experts of a layer, so every pass
streams most of the 66 GiB of expert weights; and attention, ~2 ms a pass at 2K growing to ~1.7 s at 100K, because
the prompt rows went through the decode kernel, which gives every row its own blocks, reads the whole cache once a
row and runs on CUDA cores (~1.5 TFLOPS).

- **0002** adds `tf_q35_attn_prompt` (`qwen35_prompt_attn.cuh`): a block takes 8 consecutive prompt rows (a warp
  each; a row's 16 query heads of one key head are the 16 rows of an mma tile), loads each 16-key tile of keys and
  values into shared memory once for all 128 queries (double-buffered), and computes Q K^T and P V with `mma.sync`
  bf16 tensor-core instructions (fp32 sums) and an online softmax. On one layer against the stock kernels
  (`tools/pa_test.cu`, 128 and 512 rows): outputs equal up to bf16 rounding (max |d| 0.0078), 10x faster at 2K keys
  and 24x at 100K (~35 TFLOPS). Only prompt passes use it; draft verify windows and decode rows keep the stock
  kernels, so drafted replies still equal `"draft": false` (checked after every change).
- **0003** raises the forward's row limit to 1,024 (`max_rows`, its round tables, the kernels' `MAXR`, and
  `Shape.total` from u8 to u16). The experts then cost 0.30 ms a prompt row instead of 1.12. A 1,024-row pass at
  8-32K: experts ~312 ms, attention ~195 ms (averaged over depths), GDN scan ~132 ms, the rest ~145 ms. 2,048 rows
  would buy ~10% for another ~2 GiB of scratch.

Remaining gap to SGLang (~1.5-2x): the GDN recurrence runs a row at a time (~0.13 ms a prompt row), attention is at
~35 TFLOPS, and the experts still stream at memory bandwidth. Prompt rows now round differently than before
(attention), so greedy replies differ from stock 1.1.1's in places; drafted and undrafted replies stay equal.

### A long prompt beside a decoding stream (`bench.py interleave 100000 3000`)

| | 98K prompt TTFT | decoding stream during it |
|---|---:|---:|
| TensorFold stock | 899 s | ~0 tok/s (32 tokens in 899 s) |
| TensorFold + 0002 + 0003 | 92 s | ~0 tok/s (14 tokens) |
| SGLang (`PREFILL_DECODE_INTERVAL=16`) | 78 s | 10-12 tok/s |

The Qwen3.5 CUDA backend runs a prompt whole (the lane core can take a prompt in pieces between rounds, but this backend
does not implement it), so every other stream waits for the whole prompt; the patches make that wait ~10x shorter.

### Full context: 4 streams x 245K tokens

`bench.py long 4 250000 256` (with 0002 + 0003): four distinct 245,037-token prompts at once, 256 new tokens each,
greedy, thinking off. All four completed; minimum `MemAvailable` 12 GB. Each prompt prefilled in ~340 s (~720 tok/s at
this depth; stock 1.1.1 would take well over an hour each). The streams decode only between prompt passes, so the
first three made almost no progress until the fourth prompt was in, then the four decoded together at ~7 tok/s
combined (all done in 1,533 s).

**Decode at depth is the next limit, and it is stock TensorFold's, not the patches'.** Draft verify windows and decode
rows still use the stock attention (0002 leaves them alone so drafting stays exact), which re-reads the whole cache
for every row on CUDA cores. Per round: ~98 ms at 31K, ~238 ms at 245K with one stream (12.5 tok/s, 512 tokens);
two streams at 117K decoded ~6 tok/s each. SGLang decodes one stream at 245K at 39 tok/s. Using the tensor-core kernel
for verify windows would need it to be row-invariant (fixed key spans instead of splits sized by the block's longest
row, tree masks from the ancestor bits, graph-captured grids), so that a drafted reply's bits still equal
`"draft": false`; not done here.

Decode at depth: after a 31,397-token prompt, one stream decoded 512 tokens at 30.0 tok/s (2.9 tokens a round).
Decode after the prefill patches (same method as the table above): code 66.6 / 41.7 x 2 / 29.8 x 4 tok/s per stream,
mixed 38.9 / 25.4 / 20.0, mixed thinking on 42.5 / 20.5; the round time is unchanged (~95 ms), the spread is the
prompts' acceptance.

### Which to use

- **TensorFold** (with 0002 and 0003) for agent/chat traffic: faster decode (especially code), prefill within ~1.5-2x
  of SGLang, ~90 s startup instead of ~9 min.
- **SGLang** when long prompts arrive while other streams decode (they keep decoding during a prefill), or for the
  fastest prefill.

## Other differences

- Tool parsing: for a nullable argument that the model writes as `null` (`"note": {"type": ["string", "null"]}`),
  TensorFold returns the string `"null"`, SGLang JSON null. Both stay valid against the schema.
- KV cache: bf16 by default here (fp8 with `KV_CACHE=fp8`, not measured); SGLang uses fp8.
- Sampling defaults, thinking default and served name match the SGLang recipe.

## Files

| File | Purpose |
|---|---|
| `start.sh`, `end.sh` | launch (prepares if needed, waits for `serving`, smoke request) and stop |
| `scripts/config.sh` | settings and defaults |
| `scripts/prepare.sh` | builds the image (`--image-only` skips downloads), downloads, fixes the vision shard |
| `scripts/fix_vision.py` | supplies `model-vision.safetensors` (see above) |
| `patches/0001-qwen35-moe-dflash2.patch` | DFlash2 drafting for `qwen3_5_moe` |
| `patches/0002-prompt-attention.patch`, `patches/0003-prompt-rows-1024.patch` | prefill (see above) |
| `tools/pa_test.cu` | the prompt attention kernel against the stock one on one layer (accuracy, time) |
| `bench/bench.py`, `bench/tools_check.py` | the SGLang recipe's benchmarks and tool check (read either server's counters) |
| `results/tf-dflash.md`, `results/tf-dflash-prefill.md` | raw output of the runs above (stock 1.1.1 + 0001; with 0002 + 0003) |
