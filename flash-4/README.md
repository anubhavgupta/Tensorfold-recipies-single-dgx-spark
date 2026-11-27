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
client, greedy, thinking off, 256-token replies**, server with `--context 131072 --parallel 9`. Noise is 3-5 % between runs.

| class | flash-3 N=1 / 4 / 9 | **flash-4** N=1 / 4 / 9 | MTP head (flash-4, `DRAFTER=mtp`) |
|---|---|---|---|
| code | 114.9 / 202.3 / 269.2 | **115.3 / 193.5 / 280.5** | 100.2 / 186.2 / 265.3 |
| prose | 58.0 / 115.6 / 167.1 | 52.7 / 109.1 / 163.9 | 51.2 / 109.5 / 166.6 |
| devops | 90.5 / 182.7 / 254.6 | 87.7 / 177.5 / 248.1 | 103.4 / 176.0 / 244.1 |
| json | 106.0 / 223.7 / 306.8 | 107.7 / 208.6 / **312.9** | 103.4 / 193.0 / 282.1 |

Sampled (T=0.7, 4 / 9 clients): flash-4 code 179.5 / 269.1, prose 105.6 / 152.5 (`DRAFTER=mtp` earlier build: code 173.6 / 230.3,
prose 96.4 / 144.4). Not measured on flash-3.

**Prefill** (one prompt, cold, tok/s): flash-3 1054 / 1037 / 1020 at 4K / 16K / 65K; **flash-4 2009 / 2068 / 2054 (+91–101 % over flash-3)** and **1842 tok/s at 248K (134.4 s vs 229.7 s before patches 0007–0010, -95.3 s / +71 % faster)**.
Patch 0007 enables TensorFold's 3-stage `cp.async` transposed EXL3 prompt expert kernels (`exl3_prompt.cuh`) for `mul1`, running all 512 4-bit routed experts across the full 3,072-row chunk in one pass with no split-K fp32 partials; patch 0010 adds 6-bit `mul1` transposed prompt kernels (`k12_mul1`, 48 KiB dynamic SMEM) so shared expert 512 also runs in one 3,072-row pass reusing the same router output and rotated inputs (+4–6 % at 4K–65K, -11.8 s / +8.7 % at 248K); patch 0009 switches long-context QSA indexer scoring (`qsa_score`) to BF16 Tensor Cores (`mma16816` with FP32 accumulators, cutting 248K TTFT by 5.8 s with byte-identical 120K needle retrieval); and `start.sh` enables `TF_FLASHNEXT_READOUT_FUSED=1` (fused BF16 hyper-connection read-out) and `TF_QMMF_HALF_BELOW=0` (64-col BF16 decode tiles: 1 exact wave of 48 SMs on GB10 for HC `[324, 10240]`, +0.8–1.5 % 1-stream decode, bit-exact). Patch 0008 extends EXL3 `decodeGroup` to 6-bit `mul1` stitched projections (`in_proj_qkv+z`, `q+k+v`), adding +2.3–3.4 % 1-stream decode and +1.5 % 4-stream prose with 100 % identical greedy and sampled token hashes. Earlier flash-4 (before patch 0007, 1,024-row `grouped_body_mtl`) ran 1314 / 1294 / 1261 tok/s.

**Decode** is level with flash-3: better on code and json at 9 clients, 5-12 % lower on prose with one client and 3-5 % lower at
4 clients. It is bound by expert-weight bandwidth (a verify window costs ~33 ms plus ~2.9 ms a row), so gains can only come from
acceptance. Acceptance (server `done` lines): code 81 %, prose 57 %, devops 72 %, json 75 % (flash-3: json 84 %).

**Load time** ~60 s (flash-3 ~3 min). **Exactness:** greedy and sampled replies are byte-identical with drafts on and off
(17 checks: 4 clients, a resumed turn, 6K and 25K prompts, vision, `--parallel 1`).

### BF16 performance follow-up

Tests used the same cached EXL3 target and DFlash2 drafter on one Spark, with the 262,144-token window and vision enabled.

For a machine dedicated to four full windows and long follow-ups that are often compacted, use:

```bash
PARALLEL=4 KV_PAGES=0 PROMPT_CACHE_GIB=8 TF_PROMPT_INTERMEDIATE=1 ./flash-4/start.sh restart
```

This trades elastic pages for four full windows and reserves room for intermediate prompt checkpoints. The actual
recipe image admitted all four windows and kept an 8 GiB cache; if other workloads leave insufficient memory,
reduce `PARALLEL` rather than reducing the safety reserve. Keep the regular launcher defaults for elastic paging.
At seed 42, the stock and warm-cache profiles matched generated-token hashes in the tool and PNG vision checks.
The same two typed-tool fixtures failed in both profiles.

- **Compacted multi-turn prompts:** with three full windows and a 10 GiB cache
  (`PARALLEL=3 KV_PAGES=0 PROMPT_CACHE_GIB=10`), `TF_PROMPT_INTERMEDIATE=1` cut time to first token on a 43,504-token
  follow-up from **35.71 s to 8.90 s** by restoring 32,768 cached tokens instead of two. The cold 69,078-token request,
  its identical repeat, and the compacted request all produced the same token hashes in both arms. The four-window
  recipe above repeated the hit in **9.07 s**, with matching token hashes. The 5.0 GiB effective cache in the
  four-stream paged A/B did not retain a usable intermediate checkpoint, so the setting is not enabled by default.
  More checkpoints use more cache memory; this does not speed up cold prompts or decode.
- **Cold prefill:** keep the 1,024-row EXL3 expert window. In one matched sweep at 4K / 16K / 65K, it ran at
  1,290 / 1,277 / 1,257 tok/s. The 2,048-row window ran at 1,254 / 1,232 / 1,195 and the 3,072-row window at
  931 / 864 / 836. Widening the expert window is not the same as changing the outer prompt chunk size.
- **Long-context indexer:** in an instrumented 249K prompt, the 3,072-row chunks nearest 128K / 192K / 249K
  spent about 176 / 265 / 331 ms scoring candidate blocks and 36 / 57 / 71 ms selecting them, out of
  2,804 / 2,908 / 3,007 ms total (median of the three closest full chunks). Scoring, not selection, is the
  larger opportunity. A warp-aggregated selector produced identical results but did not improve measured selection
  time. A four-row score tile was bit-exact but slower than the existing eight-row tile in a component A/B
  (3.54 vs 2.57 ms at 262K, 256 queries); neither change was kept.
- **Decode:** the distinct packed expert-weight footprint tracked kernel time at fixed verification widths.
  Across 24 profiled eight-row passes at short context, the distinct footprint ranged from 3.93 to 7.20 GB,
  with Pearson r=0.997 against expert stage time. This counts distinct selected weight bytes, not hardware DRAM
  transactions; it does not establish that choosing different drafts would increase end-to-end throughput.
  In a separate greedy sweep (192-token cap, thinking off, 1 / 4 clients), the existing seven-draft cap and 0.5
  confidence were the best all-round settings. Four drafts reduced single-client code from 107-109 to 88 tok/s.
  Lowering confidence to 0.35 moved four-client JSON from 205-207 to 209 tok/s, within noise, without improving code or prose.

To reproduce the prefix-cache A/B, restart between configurations and run
`python3 flash-4/bench/suite.py compact TAG 42000`. It reports the server's cold, repeated, and compacted
prompt latency, cached tokens, and exact generated-token hashes. Compare hashes across configurations before
comparing performance.

#### EXL3 kernel experiments

Same BF16 recipe, cached checkpoints, 262K server window, four admitted streams, greedy output for throughput.
Prefill is cold, one stream, and reports prompt tokens per second at 4K / 16K / 65K:

| Optimization | Stock tok/s | Experimental tok/s | Decision |
|---|---:|---:|---|
| Four-warps-per-item split-K finish | 1,302 / 1,292 / 1,258 | 1,300 / 1,286 / 1,263 | No repeatable improvement |
| Fuse gate/up finish and down input rotation | 1,309 / 1,302 / 1,268 | 1,307 / 1,296 / 1,265 | Removed a launch and intermediate write, but no request speedup |

Component tests matched output bits for both finishes and for the fused down rotation. The fusion retains the
original BF16 rounding between SwiGLU and the down transform; the cold, repeated, and compacted request hashes
also matched. Neither prefill experiment is enabled.

Narrowing the decode's output tiles from 128 to 64 columns matched greedy and seeded sampled token IDs
with DFlash2 on and off. It reduced four-client JSON from 200-206 to 147 aggregate tok/s, so it was rejected.

Grouping up to 32 verification members per expert instead of 16 let the grouped kernel reuse a decoded weight
tile across both 16-row tiles. In a separate greedy A/B/A at 98,304 context with nine admitted streams and a
192-token cap, all 78 generated-token hashes matched. At nine clients, code / prose / JSON throughput was
268 / 169 / 307 tok/s, within or below the two stock runs' ranges of 268-270 / 165-172 / 301-309. At four clients,
the change did not beat both stock runs on any workload either. The original grouping remains the default.

### Parallel full 262K streams

Full 262,144-token window, bf16 KV (the native engine has no int8/fp8 KV): **7.9-8.1 GiB per stream**; the model takes 67.3 GiB, so
**5 streams fit** (`PARALLEL=6` is refused at start: it needs more than the 43 GiB left after the 2 GiB reserve).

Measured: 5 simultaneous ~249K-token prompts, all 5 completed (ttft 459 / 732 / 927 / 1078 / 1196 s, prompts prefill one after the
other, 1,041 tok/s aggregate at that depth; decode overlaps). Memory peaked at 97 GiB used, 24 GiB available.
flash-3 admitted 9 (int8 KV, 4.1 GiB a stream), so on this metric flash-4 is lower. `KV_PAGES=256` makes the engine admit more
streams than whole windows (they share a page pool), which is oversubscription and works only while the streams are shorter on
average; it is not 9 truly full streams. An int8 KV port to the native attention would be the way to 9 (not done).
`PARALLEL` defaults to 4.

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

Not carried over: int8 KV, the 9-stream default, `TENSORFOLD_KEEP_MAX` elastic kept states (the native prompt cache replaces it).

## Patches (`patches/`, applied to the v1.3.5 tag by `scripts/prepare.sh`)

| Patch | What |
|---|---|
| `0001-flashnext-exl3` | EXL3 packs on native CUDA: loader, dense/expert/shared GEMMs, n-gram table codec, MTP head rows, `TF_FLASHNEXT_PROFILE=1` stage profiler |
| `0002-exl3-multitile-experts` | grouped expert kernel decodes each weight tile once for several row tiles (`TF_EXL3_MTL`, default 3); per-row summation order unchanged |
| `0003-tool-call-structure` | tool-call closer that respects JSON strings / XML parameter boundaries |
| `0004-media-limits` | request/image/video limits |
| `0005-vision-budget` | shared image budget up to 65,536 rows |
| `0006-flashnext-dflash2` | DFlash2 drafter: taps after layers 5/15/25/35/45, 5-layer drafter at 4 bits, block of 7, selector. Knobs `TF_FLASHNEXT_DFLASH_{DRAFTS,CONFIDENCE,COPIES,TIMING}` |
| `0007-exl3-prompt-experts` | Transposed 3-stage `cp.async` EXL3 prompt experts (`exl3_prompt.cuh`) for `mul1` 4-bit routed experts in one 3,072-row pass (no split-K `z` partials); shared 6-bit expert (512) via `skip = -514`. Knob `TF_EXL3_PROMPT_FAST` (default 1) |
| `0008-exl3-decode-group6` | Extends EXL3 `decodeGroup` (`tf_exl3_group_12_2_{4,8}`) to 6-bit `mul1` stitched projections (`in_proj_qkv+z`, `q+k+v`), fusing rotation + GEMM into one launch (+2.3–3.4 % 1-stream decode, bit-exact). Knob `TF_EXL3_DECODE_GROUP` (default 1) |
| `0009-qsa-score-tc` | BF16 Tensor-Core (`mma16816` + FP32 accumulators) QSA indexer scores (`fn_qsa_scores_tc` / `_paged`) for prompt chunks (`rows >= 16`), cutting 248K prefill by 5.8 s (147.4 → 141.6 s, 1,748 tok/s, byte-identical 120K needle). Knob `TF_FLASHNEXT_QSA_TC` (default 1) |
| `0010-exl3-prompt-shared6` | 6-bit `mul1` transposed EXL3 prompt kernels (`k12_mul1`, 48 KiB dynamic SMEM) for shared expert 512 in one 3,072-row pass reusing routed scratch (4K/16K/65K/248K: `2,009 / 2,068 / 2,054 / 1,842` tok/s; 248K TTFT `146.2 → 134.4 s`, 100% byte-identical). Knob `TF_EXL3_PROMPT_SHARED` (default 1) |

Verified: applying them in order to the pristine tag gives the tree that was built and tested.

## Decisions made while building (user was away)

- Chose native EXL3 + DFlash2 implementations (rather than staying on 0.6.6).
- **DFlash2 is the default drafter**, with `TF_FLASHNEXT_DFLASH_COPIES=0` (best on the suite; `1` is 2.4x faster on echo-style
  replies but 6-8 % slower on json). The MTP head is one env var away.
- Default `PARALLEL=4` (5 is the maximum for full windows); `KV_PAGES` is opt-in.
- Model folder `~/projects/models/Qwen3.8-Flash-Next-exl3-4.05bpw` is built from hard links to the cached files; the f16 vision
  tower and a few f16 plain tensors are converted to bf16 at load/build.
- Raised the engine's vision budget cap (patch 0005) and the body limit to 192 MiB to keep flash-3's limits.
- Known gaps: no int8 KV (5 full streams, not 9); a token-limit-truncated tool call is still emitted; DFlash2 drafts only on one GPU,
  4-bit drafter, EXL3 pack only; JPEG/WebP need the libraries in the image (the recipe image installs them; PNG was what I exercised).
