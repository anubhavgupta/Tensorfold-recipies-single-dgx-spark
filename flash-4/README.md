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

**Prefill** (one prompt, cold, tok/s): flash-3 1054 / 1037 / 1020 at 4K / 16K / 65K; **flash-4 1314 / 1294 / 1261 (+25 %)**.
A 3072-row prefill chunk went 2.5 s -> 2.3 s from multi-tile expert kernels (patch 0002); experts are ~60 % of it.

**Decode** is level with flash-3: better on code and json at 9 clients, 5-12 % lower on prose with one client and 3-5 % lower at
4 clients. It is bound by expert-weight bandwidth (a verify window costs ~33 ms plus ~2.9 ms a row), so gains can only come from
acceptance. Acceptance (server `done` lines): code 81 %, prose 57 %, devops 72 %, json 75 % (flash-3: json 84 %).

**Load time** ~60 s (flash-3 ~3 min). **Exactness:** greedy and sampled replies are byte-identical with drafts on and off
(17 checks: 4 clients, a resumed turn, 6K and 25K prompts, vision, `--parallel 1`).

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
