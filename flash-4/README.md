# Qwen3.8-Flash-Next EXL3 + DFlash2 // TensorFold 1.3.5 (native) // one DGX Spark

`flash-3` ported to TensorFold's latest release, **v1.3.5, the native Zig engine**. Same model files
(`turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, drafter `incoai/Qwen3.8-Flash-Next-DFlash2`, all from the Hugging Face
cache: nothing is downloaded), OpenAI API on port 8888, model id `Qwen3.8-Flash-Next`.

v1.3.5's Flash Next path on CUDA has no EXL3 loader and no DFlash2 drafter (it reads MLX 4-bit / NVFP4 / FP8 packs and the
checkpoint's MTP head), so both are written here as patches in Zig/CUDA against the pristine tag. flash-3's Python patches do not
apply to the native engine.

```bash
./flash-4/start.sh          # first start: builds the model folder and the image (~10 min, offline), then ~1 min to load
docker logs -f tf-flash-4
./flash-4/end.sh
DRAFTER=mtp ./flash-4/start.sh      # the checkpoint's MTP head instead of DFlash2 (DRAFTER=none: no drafts)
```

Settings are environment variables or `.env` lines, listed at the top of `start.sh` (defaults in `scripts/config.sh`).
Autostart at boot: `docs/autostart.md` and `systemd/`.

## Results (same GB10, flash-3 numbers measured the same way)

Method: `bench/suite.py content`, steady-state aggregate decode tok/s over N simultaneous clients, **distinct prompts per
client, greedy, thinking off, 256-token replies**, server with `--context 131072 --parallel 9`. Noise is 3-5 % between runs,
and one client's number depends on the text it happens to generate (draft acceptance), so N=1 moves by up to 15 % between builds.
"bf16 KV" is flash-4 through patch 0006, "fp8 KV" (default) adds patch 0007.

| class | flash-3 N=1 / 4 / 9 | flash-4 bf16 KV | **flash-4 fp8 KV** (default) | MTP head, bf16 KV (`DRAFTER=mtp`) |
|---|---|---|---|---|
| code | 114.9 / 202.3 / 269.2 | 115.3 / 193.5 / 280.5 | 96.9 / 184.4 / 273.3 | 100.2 / 186.2 / 265.3 |
| prose | 58.0 / 115.6 / 167.1 | 52.7 / 109.1 / 163.9 | 57.1 / 106.7 / 164.1 | 51.2 / 109.5 / 166.6 |
| devops | 90.5 / 182.7 / 254.6 | 87.7 / 177.5 / 248.1 | 88.8 / 174.0 / 248.5 | 103.4 / 176.0 / 244.1 |
| json | 106.0 / 223.7 / 306.8 | 107.7 / 208.6 / 312.9 | 107.7 / 209.9 / 306.7 | 103.4 / 193.0 / 282.1 |

Sampled (T=0.7, 4 / 9 clients), fp8 KV: code 171.9 / 256.0, prose 103.9 / 154.5 (bf16 KV: code 179.5 / 269.1, prose 105.6 / 152.5).
Not measured on flash-3.

**Prefill** (one prompt, cold, tok/s at 4K / 16K / 65K): flash-3 1054 / 1037 / 1020; flash-4 bf16 KV 1314 / 1294 / 1261 (+25 %);
**fp8 KV 1310 / 1297 / 1265 (+25 %)**, level with bf16 KV. A 3072-row prefill chunk went 2.5 s -> 2.3 s from multi-tile expert
kernels (patch 0002); experts are ~60 % of it, attention only ~7.5 %.

**Decode** is level with flash-3 on json and code at 9 clients and 0-8 % lower elsewhere; fp8 KV is within the run-to-run noise
of bf16 KV, apart from one-client code, whose number follows the text it generates (acceptance 89 % vs 76 % on the same prompt
with the two caches). The same build with and without fp8, one client, ms per round of a verify window: 4K context 44 / 48 / 51
(bf16) vs 44 / 47 / 52 (fp8); 100K context 46 / 51 / 53 vs 45 / 49 / 50, i.e. fp8 is 1-5 % faster per round at long context.
**fp8 KV does not make this model much faster, by construction:** decode attention reads only the indexer's top ~2K keys (at most
~8K), a few tens of MB a round against ~8 GB of expert weights, and prefill attention is a compute-bound ~7.5 % of a chunk.
Its gain is memory: 8-9 full windows instead of 5. Decode is bound by expert-weight bandwidth (a verify window costs ~33 ms
plus ~2.9 ms a row), so gains can only come from acceptance. Acceptance (server `done` lines, bf16 KV): code 81 %, prose 57 %,
devops 72 %, json 75 % (flash-3: json 84 %).

**Load time** ~60 s (flash-3 ~3 min). **Exactness:** greedy and sampled replies are byte-identical with drafts on and off
(17 checks on the bf16 build: 4 clients, a resumed turn, 6K and 25K prompts, vision, `--parallel 1`; `bench/exact_draft.py`
re-checks 5 prompts on the fp8 build).

### fp8 KV cache (patch 0007)

TensorFold 1.3.5 keeps Flash Next's attention K/V in bf16 and has no int8/fp8 option for it (`--kv-cache fp8` is refused for this
family). Patch 0007 stores them as E4M3 codes with an fp32 scale per position and KV head (528 bytes a position for keys, same for
values, against 1,024), converts back to bf16 in shared memory (code x scale, rounded once) and leaves the attention tiles, the
indexer keys and everything else as they were. The first version cost +13 ms of a chunk's 180 ms attention (prefill -1 %); the
prompt kernel now prefetches keys and values together and the next tile during the value products, which brought prefill back to bf16's level. A full 262,144-token window takes **4.9-5.1 GiB instead of 8.1 GiB**.

I chose fp8 (E4M3) over int8: no scale search or outlier handling beyond one amax per row, a hardware conversion, and it is the
format the engine's other families use for their KV. Quality, bf16 vs fp8 on the same server, greedy
(`bench/kvq.py`): 12 prompts of 300 tokens: 6 replies byte-identical, the rest diverge after 113-980 characters into equally good
text (a different synonym or ordering; greedy near-ties); needle retrieval of an 8-character passphrase at 10 / 50 / 90 % of
**110K tokens**: 3 of 3 for both. That is a small probe, not a benchmark: run your own long-context evaluations before relying on it.
`wt.kv_fp8` (`cuda_prompt_wt.zig`) and `KV8` (`flashnext_pages.cuh`) switch it off at compile time.

### Parallel full 262K streams

Full 262,144-token window, fp8 KV: the model takes 67.3 GiB and each stream 4.87 GiB (16K-token vision budget) to 5.10 GiB
(65,536-token vision budget; every stream reserves that many embedding rows).

| setting | streams | result |
|---|---|---|
| `PARALLEL=8` (default), `VISION_IMAGE_TOKENS=65536` | **8** | admitted (the engine's own window + 7 in a 33 GiB pool of 256-row pages, 7,798 pages = 7.6 windows) |
| `PARALLEL=9 KV_PAGES=0 VISION_IMAGE_TOKENS=16384` | **9** | 9 simultaneous ~249K-token prompts all completed (below) |
| `PARALLEL=9`, 65,536 vision tokens | - | refused at start: needs 45.9 GiB, 43.9 GiB left after the 2 GiB reserve (the minimum) |

The 9-stream run: 9 prompts of 248.9-249.5K tokens at once, all 9 completed in 2,514 s (wall); prompts are interleaved,
so the first token of every stream came at 2,331-2,510 s, aggregate prefill 893 tok/s (deep prompts prefill slower than the
1,230-1,310 tok/s of short ones), then 7-41 tok/s decode while the others still prefilled. **Memory peaked at 117 GiB used,
4 GiB available of 121 GiB** and the kernel's reclaim (kswapd) ran flat out for the second half, so 9 is the limit of this
machine, not a comfortable setting; 8 leaves ~5 GiB more. With bf16 KV the limit was 5 streams (7.9-8.1 GiB each); flash-3 reached
9 with int8 KV (4.1 GiB each). `KV_PAGES` > 0 (default 256) lets streams share a page pool, so more streams than whole windows can be
admitted, which only works while they are shorter on average; the prompt cache is what is left (0.1 GiB at 9 full streams).

## Features kept from flash-3

| flash-3 feature | flash-4 |
|---|---|
| EXL3 4.05 bpw pack | native loader: dense, routed + shared experts, router, n-gram tables, MTP head (patch 0001) |
| DFlash2 drafter | native port (patch 0006), 4-bit drafter; `DRAFTER=mtp` or `none` available |
| 262,144-token window | yes (`CONTEXT`) |
| Vision (images, 50 per request, 36,100-token budget) | yes; tower converted to bf16 by `build_model_dir.sh`; patch 0005 lifts the engine's 4096 cap on the shared budget; image prompts are prefix-cached by content (native) |
| Tool-call fixes | native parser plus structure-aware closer (patch 0003): flash-3's 44-case repro matrix 12 -> 4 mismatches, the 4 being token-limit cuts (a truncated call is still emitted with its partial arguments) |
| Media limits | patch 0004 (`TENSORFOLD_MAX_BODY_MIB`, default 192 here) |
| Client cancel | native (a dropped long prefill frees the server; resend resumes from the kept state; `bench/test_cancel.py`) |
| Thinking, sampling defaults | checkpoint's own chat template; 1.0/0.95/20 thinking, 0.7/0.8 otherwise |
| Autostart | `systemd/`, `docs/autostart.md` |

Not carried over: int8 KV (fp8 instead, patch 0007), `TENSORFOLD_KEEP_MAX` elastic kept states (the native prompt cache replaces it).

## Patches (`patches/`, applied to the v1.3.5 tag by `scripts/prepare.sh`)

| Patch | What |
|---|---|
| `0001-flashnext-exl3` | EXL3 packs on native CUDA: loader, dense/expert/shared GEMMs, n-gram table codec, MTP head rows, `TF_FLASHNEXT_PROFILE=1` stage profiler |
| `0002-exl3-multitile-experts` | grouped expert kernel decodes each weight tile once for several row tiles (`TF_EXL3_MTL`, default 3); per-row summation order unchanged |
| `0003-tool-call-structure` | tool-call closer that respects JSON strings / XML parameter boundaries |
| `0004-media-limits` | request/image/video limits |
| `0005-vision-budget` | shared image budget up to 65,536 rows |
| `0006-flashnext-dflash2` | DFlash2 drafter: taps after layers 5/15/25/35/45, 5-layer drafter at 4 bits, block of 7, selector. Knobs `TF_FLASHNEXT_DFLASH_{DRAFTS,CONFIDENCE,COPIES,TIMING}` |
| `0007-flashnext-kv-fp8` | fp8 (E4M3) attention K/V caches with an fp32 scale per position and KV head: ~43 % fewer cache bytes (see above) |

Verified: applying them in order to the pristine tag gives the tree that was built and tested.

## Decisions made while building (user was away)

- Chose native EXL3 + DFlash2 implementations (rather than staying on 0.6.6).
- **DFlash2 is the default drafter**, with `TF_FLASHNEXT_DFLASH_COPIES=0` (best on the suite; `1` is 2.4x faster on echo-style
  replies but 6-8 % slower on json). The MTP head is one env var away.
- `PARALLEL=8` default (8 full windows fit with fp8 KV at the 65,536-token vision budget); `KV_PAGES=256` and
  `VISION_IMAGE_TOKENS=65536` follow your instructions (flash-3's `.env` has 65536 too).
- Added the fp8 KV cache (patch 0007) on your request: fp8 rather than int8 (reasons above); quality was checked with a small probe only.
- Model folder `~/projects/models/Qwen3.8-Flash-Next-exl3-4.05bpw` is built from hard links to the cached files; the f16 vision
  tower and a few f16 plain tensors are converted to bf16 at load/build.
- Raised the engine's vision budget cap (patch 0005) and the body limit to 192 MiB to keep flash-3's limits.
- Known gaps: 9 full streams only with a 16K vision budget and at the memory limit; a token-limit-truncated tool call is still emitted; DFlash2 drafts only on one GPU,
  4-bit drafter, EXL3 pack only; JPEG/WebP need the libraries in the image (the recipe image installs them; PNG was what I exercised).
