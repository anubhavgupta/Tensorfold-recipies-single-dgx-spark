# Qwen3.8-Flash-Next EXL3 + DFlash2 // TensorFold 0.6.6 // one DGX Spark

A copy of `flash-2` (turboderp's EXL3 pack `turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, 9 streams, 262,144-token
window, vision, tool-call fixes), moved to TensorFold **0.6.6** and drafting with the DFlash2 model
[`incoai/Qwen3.8-Flash-Next-DFlash2`](https://huggingface.co/incoai/Qwen3.8-Flash-Next-DFlash2) instead of the
checkpoint's MTP head. OpenAI-compatible API on port 8888, model id `Qwen3.8-Flash-Next`.

Why 0.6.6: it is the last release of TensorFold's Python engine. From 1.0 on TensorFold ships as a native (Zig) binary,
whose Flash Next path on CUDA serves only the MLX 4-bit group-32 checkpoint with its MTP head. It has no EXL3 loader
and no DFlash drafter for this family, and flash-2's Python patches do not apply to it. 0.6.6 differs from 0.6.5 only
by `--name-priority` (`cli*.py`, `serve_options.py`, `cuda/server.py`), so flash-2's patches carry over unchanged.

```bash
hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6
hf download incoai/Qwen3.8-Flash-Next-DFlash2
./flash-3/start.sh          # first start: builds tensorfold:v0.6.6 + patches, compiles kernels (~3 min load)
docker logs -f tf-flash-3-exl3
./flash-3/end.sh
DRAFTER=none ./flash-3/start.sh     # the MTP head instead (flash-2's drafting, on 0.6.6)
```

The vision tower needs the one-time conversion described in `start.sh` (the same file flash-2 uses). Settings are
environment variables or `.env` lines, listed at the top of `start.sh`. The main ones are `DRAFTS` (7),
`DRAFT_CONFIDENCE` (0.5), `PARALLEL` (9; DFlash2 needs at least 2), `CONTEXT` (262144) and `KV_DTYPE` (int8).

## Patches (TensorFold 0.6.6)

`start.sh` copies the enabled patches into `runtime/patches` (Git-ignored). `../tensorfold.sh --tf-patches` then
builds one derived image for each distinct set. flash-2 mounted its changes as whole-file overlays. Those are patches
here, so the DFlash2 patch can change the same files.

| Patch | What | Off |
|---|---|---|
| `0001-tool-parser.patch` | flash-2's tool-call parser fix (boundary-aware envelopes, union types; with `tools/tool_calls.py`) | `TOOL_PARSER_FIX=0` |
| `0002-exl3-prompt-kernels.patch` | flash-2 `overlay/exl3`: grouped expert kernel decodes each weight tile once for several row tiles (`TENSORFOLD_MTL`), faster routing groups; bit-identical | `OVERLAY=0` |
| `0003-media-limits.patch` | flash-2 `overlay/limits`: larger request/image/video limits | `VISION_LIMITS=0` |
| `0004-ngram-unpinned.patch` | flash-2 `overlay/ngram`: n-gram tables stay reclaimable page cache | `TENSORFOLD_NGRAM_LOCK_GIB=stock` |
| `0005-vision-prefix-cache.patch` | flash-2 `overlay/cache`: prefix cache for image/video prompts, elastic kept states (`TENSORFOLD_KEEP_MAX`) | `VISION_CACHE=0` (needs `DRAFTER=none`) |
| `0006-dflash2-drafter.patch` | DFlash2 drafting for Flash Next (below) | `DRAFTER=none` |

`tools/patches/0001-tool-trace.patch` (opt-in tracing without the parser fix) and the diagnostics in `tools/` are
flash-2's. Run the CPU tests in the image with all patches applied:

```bash
cd flash-3 && docker run --rm --network none --entrypoint bash -v "$PWD:/recipe:ro" tensorfold:v0.6.6 -c \
  'cd /usr/local/lib/python3.12/dist-packages && for p in /recipe/patches/*.patch; do patch -p0 -s < $p || exit 1; done &&
   cp /recipe/tools/tool_trace.py /recipe/tools/tool_calls.py tensorfold/cuda/ &&
   PYTHONDONTWRITEBYTECODE=1 python3 /recipe/tools/test_tool_calls.py'
```

See `../flash-2/overlay/README.md` for how the tool-call defects were found and what the parser fix does. That
covers the intentional streaming change (a tool event carries its complete arguments), opt-in tracing (`TOOL_TRACE=1`,
sensitive) and the remaining model-generation limits.

### How DFlash2 runs here (`0006`)

- **Taps.** The drafter was trained on the target's outputs after layers 5, 15, 25, 35 and 45, each the mean of the
  four hyper-connection streams (the checkpoint's SGLang capture patch). TensorFold writes a layer's MoE output back
  lazily, at the start of the next layer, so the tap is read right after that write-back. The target's arithmetic does
  not change.
- **Drafter.** It has 5 sliding-window (2,048) layers with two-tap dynamic convolutions, drafts a block of 8 (the
  pending token and 7 masks), keeps the top 16 candidates a position, and picks a chain with the selector. The kernels
  and chain rule are TensorFold 0.6.6's GLM DFlash2 port (`glm5_next/cuda/dflash2.py`) with 4-bit linears. Logits use
  Flash Next's draft vocabulary (79,591 ids; the EXL3 head's rows requantized to 4 bits);
  `TENSORFOLD_DFLASH_VOCAB=full` uses the whole head. Drafter memory is 0.80 GiB, counted in the startup estimate.
- **Contexts.** Each stream slot has a 2,176-row ring. A round's committed rows and a prompt's last 2,048 rows feed it.
  A resumed prompt (prefix cache) starts its context at the resume point, and the attention masks keys before it.
- **Batching.** All drafting streams share one context pass and one block pass a round, captured as CUDA graphs.
  Drafting for 9 streams fell from 21.5 ms to 5.4 ms a round (verify: ~120 ms).
- **Chain length.** `DRAFTS` (7) caps drafts a round. A chain stops where the product of its picks' probabilities
  falls under `DRAFT_CONFIDENCE`. Greedy runs, aggregate tok/s at 9 clients (code / prose / devops / JSON):
  0.15: 244 / 144 / 222 / 275; 0.3: 264 / 157 / 240 / 282; **0.5: 259 / 167 / 247 / 287**; 0.7: 268 / 163 / 248 / 291.
- **One stream.** The engine's one-stream CUDA-graph slot stays MTP-only. With DFlash2 a lone stream decodes in the
  shared (eager) rounds. EXL3 verification is GPU-bound: graphs did not make one stream faster (prose ~43 ms a round
  either way). Moving streams into the graph slot made later lone requests slower; flash-2 still shows this (below).

Exactness checks on the GB10 (all greedy):

- Drafted replies equal `"draft": false` byte for byte (same reply sha). This held for a 16,993-token prompt, a resumed
  chat turn and an image prompt.
- 4 concurrent drafted replies each equal that prompt alone with drafts off.

## Results

The same method as flash-2's README entry, back to back on the same day: `bench/suite.py content TAG 1,4,9 256`. It
sends chat prompts per content class (4 prompts a class, a distinct variant per client), 256-token replies, thinking
off, greedy, and reports the second of two rounds. The figure is steady-state aggregate decode tok/s (`aggregate_steady`).
flash-2 is `../flash-2/start.sh` (TensorFold 0.6.5, MTP 6 drafts / 0.60). flash-3 has the defaults above. Both use
`PARALLEL=9`, int8 KV and the 262,144-token window.

| Content | Clients | flash-2 (MTP) | flash-3 (DFlash2) | Change |
|---|---:|---:|---:|---:|
| Code | 1 | 95.4 | 110.5 | +16% |
| Code | 4 | 177.6 | 190.4 | +7% |
| Code | 9 | 254.4 | 261.8 | +3% |
| Prose | 1 | 55.1 | 56.9 | +3% |
| Prose | 4 | 104.4 | 114.2 | +9% |
| Prose | 9 | 156.5 | 166.6 | +6% |
| DevOps | 1 | 100.6 | 88.9 | −12% |
| DevOps | 4 | 167.3 | 179.6 | +7% |
| DevOps | 9 | 249.4 | 260.5 | +4% |
| JSON | 1 | 99.0 | 105.0 | +6% |
| JSON | 4 | 186.4 | 210.9 | +13% |
| JSON | 9 | 294.3 | 284.4 | −3% |

Sampled (`TEMPERATURE=0.7 bench/suite.py content TAG 4,9 256`, top_p 0.8, top_k 20, Qwen's non-thinking setting):

| Content | 4 clients: flash-2 / flash-3 | 9 clients: flash-2 / flash-3 |
|---|---:|---:|
| Code | 157.0 / 180.5 (+15%) | 225.3 / 239.7 (+6%) |
| Prose | 102.4 / 111.7 (+9%) | 149.0 / 157.4 (+6%) |
| DevOps | 140.9 / 155.1 (+10%) | 223.3 / 219.5 (−2%) |
| JSON | 181.1 / 204.2 (+13%) | 269.6 / 272.8 (+1%) |

- One run per cell, and run-to-run noise is about 5–10%. The clearest gains are at 4 clients (+7% to +15%), at 1
  client for code, and when sampling. At 9 clients the two are within noise, except prose (+6%). DevOps at 1 client is
  slower: there DFlash2's chains average fewer tokens a round than MTP's.
- flash-2's 1-client rows come from a freshly started server. In the full run, after the 4- and 9-client cells,
  flash-2's lone requests fell to 32.2 (prose), 47.2 (DevOps) and 52.4 (JSON) tok/s. That is its graph slot (see
  above). flash-3's lone requests stayed at the values in the table.
- Acceptance in the flash-3 greedy run, averaged over its 28 requests a class from the server's `done req` lines:

  | Class | Drafts accepted | Tokens a round |
  |---|---:|---:|
  | Code | 79% | 4.7 |
  | DevOps | 73% | 4.1 |
  | JSON | 84% | 6.0 |
  | Prose | 57% | 2.1 |
- Memory: the startup estimate is 74.98 GiB of 115. 9 streams fit at the full window (45.0 GiB for caches, 4.13 GiB
  each), and 42 GiB stays available while serving. The 9 x 249K-token concurrency run from flash-2 was not repeated.

## Limits

- DFlash2 needs `PARALLEL >= 2` and one GPU (no `--tp 2`), as does vision on an EXL3 pack.
- A resumed prompt's drafter context holds only the rows prefilled after the resume point. The first rounds of a
  long chat turn whose prefix came from the cache draft from less context.
- `VISION_CACHE=0` needs `DRAFTER=none`, because 0006 builds on 0005.
- Tool calls were checked live: streamed and non-streamed, thinking on and off, a nullable integer argument, a
  follow-up turn after a simulated tool result, and ordinary chat. The CPU suite passes (27 tests).
