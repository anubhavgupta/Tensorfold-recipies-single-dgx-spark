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

The image is `tensorfold-native` built from TensorFold v1.1.1 (`b10231b9`) plus the patch with Zig 0.17.0 on
`nvcr.io/nvidia/pytorch:26.07-py3` (`scripts/prepare.sh`, rebuilt when the patch changes).

Checks: drafted greedy replies equal `"draft": false` (same digest); `bench/tools_check.py` passes (streamed and not,
thinking on/off, parallel calls, typed/nested/multiline arguments, a multi-turn loop with simulated results, a no-tool
chat); an image of a red rectangle and a blue circle is described correctly. At startup the engine reports
`model 66.39 GiB; 4 streams at once, 6.69 GiB each at a 262144-token window, of 41.2 GiB left after a 8.0 GiB reserve`;
minimum `MemAvailable` over the whole run below was 12 GB.

## TensorFold 1.1.1 vs SGLang (`ornith-1.5-122B`)

Same machine, same `bench/bench.py` and method: N simultaneous streaming chat requests with distinct prompts, 512 new
tokens with `ignore_eos`, 4 streams each able to hold a full 262K context. Accept length = generated tokens / verify
rounds per stream from each server's counters. Note the weights differ: MLX affine 4-bit here, NVFP4 (ModelOpt,
FP8 attention/GDN projections) under SGLang. TensorFold verifies a draft tree (~14 drafted tokens a round with one
stream), SGLang a 6-token chain (`DF_BLOCK=6`, its best setting on this box).

### Decode, tok/s (per stream at N=1; total at N=2 and N=4)

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

| Prompt tokens | TensorFold TTFT | TensorFold tok/s | SGLang tok/s |
|---:|---:|---:|---:|
| 1,997 | 3.3 s | 597 | 2,762 |
| 7,877 | 15.9 s | 497 | 3,243 |
| 31,397 | 116 s | 271 | 2,561 |
| 98,037 | 919 s | 107 | 1,535 |

TensorFold reads prompts in passes of at most 128 rows (`max_rows` in `cuda_engine.zig`). A pass costs ~0.2 s at the
start of a prompt (128 rows route to nearly all 256 experts, so each pass reads most of the 66 GiB of weights) plus
~20 us per cached token: its prompt attention runs at ~2.5 TFLOPS, i.e. not on tensor cores. Long prompts therefore
cost 5-15x more than under SGLang. A 245K prompt would take well over an hour; the 4 x 245K test of the SGLang recipe
was not repeated here (the engine's admission sizes the 4 full windows up front, see above).

### A long prompt beside a decoding stream (`bench.py interleave 100000 3000`)

| | 98K prompt TTFT | decoding stream during it |
|---|---:|---:|
| TensorFold | 899 s | ~0 tok/s (32 tokens in 899 s) |
| SGLang (`PREFILL_DECODE_INTERVAL=16`) | 78 s | 10-12 tok/s |

The Qwen3.5 CUDA backend runs a prompt whole (the lane core can take a prompt in pieces between rounds, but this backend
does not implement it), so every other stream waits for the whole prompt.

Decode at depth: after a 31,397-token prompt, one stream decoded 512 tokens at 30.0 tok/s (2.9 tokens a round).

### Which to use

- **TensorFold** for agent/chat traffic with short prompts or good prefix reuse: faster decode (especially code),
  ~100 s startup instead of ~9 min.
- **SGLang** when prompts are long or arrive while others decode: 5-15x faster prefill, and other streams keep
  decoding during it.

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
| `bench/bench.py`, `bench/tools_check.py` | the SGLang recipe's benchmarks and tool check (read either server's counters) |
| `results/tf-dflash.md` | raw output of the runs above |
