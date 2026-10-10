---
name: model-port
description: "Port, serve, patch, benchmark and optimise an LLM (target + speculative drafter, optional vision) on TensorFold on an NVIDIA DGX Spark (GB10, 128 GB unified memory), as a recipe directory in ~/projects/tensorfold (start.sh/end.sh/patches/bench/README). Use when the user asks to set up a new model or quant (EXL3, MLX 4-bit, GGUF-derived, NVFP4), add a drafter (DFlash2, MTP), make TensorFold patches, compare recipes, measure decode/prefill/throughput/max concurrency at full context, or investigate why one recipe is faster or lower quality than another."
---

# Porting a model to TensorFold on one DGX Spark

Workspace: `~/projects/tensorfold` (git repo; recipes are sibling dirs: `flash-2/`, `qwen38-27b/`, `27B-2/`, `glm5.3/` ...).
`./tensorfold.sh [--tf-* opts] serve ...` runs TensorFold in Docker; `--tf-patches DIR` builds a derived image
(`tensorfold:vX-<hash>-p<hash>`) that applies `DIR/*.patch` in name order with `patch -p0` and rebuilds automatically
when the patches change. Never commit `dashboard/cost.json` or other files that are not yours (check `git status`).

Helper scripts (in `scripts/` beside this file):

| Script | Use |
|---|---|
| `tf-tree.sh IMAGE DIR [PATCHES]` | copy the image's `tensorfold` package to `DIR` (pristine), optionally apply patches |
| `tf-run.sh IMAGE TREE script.py [-e K=V]` | run a test/profiling script in the image with `TREE` mounted over the package |
| `mkpatch.sh BASE NEW out.patch "header"` | `patch -p0` file (paths `tensorfold/...`) from tree diff, with a header paragraph |
| `verify-patches.sh PRISTINE PATCHES WORK` | apply all patches in order to a pristine copy and diff against the working tree |

## 1. Recipe layout (copy the closest existing recipe)

- `start.sh`: `.env` loader (environment wins), resolve snapshots from `~/.cache/huggingface/hub/models--X/refs/<rev>`,
  knobs as env vars with defaults documented in the header comment, `--tf-patches "$SCRIPT_DIR/patches"`, container name
  `tf-<recipe>`, port 8888, export `TENSORFOLD_*` vars (forwarded by `tensorfold.sh`). Die with a hint when a required
  converted file is missing.
- `end.sh`, `patches/NNNN-name.patch` (each with a header paragraph saying what and why, knobs, and how verified),
  `bench/` scripts, `README.md` (recipe, patches table, results tables, limits, how measured).
- Defaults that matter: `PARALLEL`, `CONTEXT` (262144), `KV_DTYPE` (fp8 = 32 KiB/token = 8 GiB per 262K window for a
  27B), `MEMORY_RESERVE_GIB`, `CHECKPOINT_SLOTS`, vision limits, `THINKING`, sampling per Qwen's recommendation.
- Only one big server at a time; stop the other (`./<recipe>/end.sh`) and check `free -g` before starting.

## 2. Downloads and model facts

- `hf download REPO --revision BRANCH` (EXL3 packs keep bpw variants on branches). Delete stale 0-byte
  `blobs/*.incomplete` after interrupted downloads. Snapshot files are symlinks into `blobs/`: mount the whole
  `~/.cache/huggingface` into containers, not a snapshot dir.
- Read `config.json` first: layers/hidden/intermediate, `layer_types` (GDN vs attention), `quantization_config`
  (MLX: `bits`, `group_size`, per-layer overrides), drafter `dflash_config.block_size`, `is_causal`, `target_layer_ids`.

## 3. Things that needed patches before (check whether the new model needs the same)

- **EXL3 drafter** (TensorFold only loads bf16/affine drafters): decode trellis to bf16 (`W = diag(suh) H W_q H diag(svh)`,
  W_q from the GPU unpack), then the normal affine 4-bit packing; count trellis tensors at packed size in admission.
  Verify against the reference dequantizer (correlation > 0.99999). Drafter precision only changes acceptance, never
  output: double quantization (EXL3 -> bf16 -> affine 4-bit) measured the same acceptance as bf16 (6.06 vs 6.00 tok/round).
- **EXL3 vision tower inside the shards**: extract `model.visual.*` and run `python -m tensorfold.vision.exl3_convert`
  once (`convert_vision.sh`), point `TENSORFOLD_VISION_WEIGHTS` at the result.
- **Vision/video prefix cache**: stock never caches image prompts (placeholder ids are identical across images). Key the
  cache by the prompt with each image/video placeholder run replaced by ids hashed from the media's hash; drop the
  "vision" guards on stops/ends/resume; drop the "fresh state" rope check.
- **Elastic kept states**: floor = stock `KEEP`, ceiling `TENSORFOLD_KEEP_MAX`, shed oldest-first when `MemAvailable`
  < `TENSORFOLD_KEEP_HEADROOM_GIB`. A conversation leaves ~1 state per message start + 1 at the prompt end.
- **Media limits** (request body, image/video bytes) and **fp8 KV**, **memory reserve**, **KV pool**: see `qwen38-27b/patches`.

## 4. Patch workflow

1. `tf-tree.sh tensorfold:vX /tmp/tfbase <recipe>/patches` (base = existing patches), `cp -r /tmp/tfbase /tmp/tfwork`.
2. Edit `/tmp/tfwork`; test with `tf-run.sh` (no image rebuild needed). Bump a CUDA extension's `load(name=...)` suffix
   when you change its sources, or the cached build is reused.
3. `mkpatch.sh /tmp/tfbase /tmp/tfwork <recipe>/patches/00NN-x.patch "header"`; split by file when changes are
   independent. Then `tf-tree.sh tensorfold:vX /tmp/pristine` and `verify-patches.sh /tmp/pristine <recipe>/patches /tmp/tfwork`.
4. Start the server (first start compiles kernels: allow several minutes), check the startup lines:
   "startup estimate", "up to N streams ... K prompt states kept", "verify ms by rows".
5. Keep a per-feature env switch that restores stock behaviour (e.g. `TENSORFOLD_EXL3_MP=1`, `TENSORFOLD_OVERLAP_REPLAY=0`).

## 5. Exactness rules (do not break)

- TensorFold guarantees a row's bits depend only on that row (row invariance), so speculative decoding is exact. Any
  kernel change must keep each output element's summation order (same K slices, k order, mma fragments, warp-order
  reduction). Test: `y = L(X[:m])` must `torch.equal` the concatenation of `L(X[r:r+1])` for every m in 1..128.
- End-to-end check: greedy (`top_k=1`) multi-stream generation, compare output hashes with the change on/off and with
  the stock kernel. Prefill and decode bits legitimately differ from each other.
- Block/tile choices that keep per-element order (e.g. qmm tiles, Triton BK 32->64 when K order is kept) are safe; check `same=True`.

## 6. Benchmarking (methodology decides the numbers)

- State the method in every table: prompts (same vs distinct), temperature (greedy accepts far more drafts than T=1),
  reply length, thinking on/off, PARALLEL, warm or cold cache. The repo README's 27B table uses the same prompt for
  every client, greedy, thinking off, 200-token replies; a distinct-prompt T=1 test gives much lower numbers.
- Tools used: `27B-2/bench/bench.py` (decode, `prefill N...` with unique prompts, `conc N TOKENS NEW`),
  `27B-2/bench/tput.py [--code] N...` (aggregate tok/s, distinct prompts, T=1), `27B-2/bench/same.py N...`
  (README method). `prompt_of` undercounts: report the server's `prompt_tokens`.
- Noise is ~5-10% between runs; GPU temperature moves the 1-row verify 64->72 ms. Do A/B back to back, warm up first,
  and repeat. Do not claim a gain inside the noise.
- Per-request truth is in the server log: `done req ... tokens= tok/s= ttft= prefill= rounds= accepted=A/D`
  (tokens/round = tokens/rounds; drafted per round = D/rounds).
- Max concurrency at full context: N simultaneous ~250K-token prompts (`bench.py conc N 155000 NEW`; ask for a long
  reply, or a "Say OK" prompt ends after 2 tokens). Prompts prefill one at a time (~225-390 s each for a 27B), so
  report completion and memory (`free -g`), and say that decode overlapped prefill. Use `prompt_tokens < context`.

## 7. Profiling recipe

- Decode is memory-bound: 1-row verify time x bandwidth ~= weight bytes (27B 4-bit: ~15.3 GB, ~62-70 ms, ~245 GB/s).
  Gains come from tokens per round (acceptance, tree width) or cheaper wide verifies.
- Scheduler round = draft (batched DFlash2) + verify forward + sample + commit (GDN fp32 state replay ~150 MB per stream
  for a 27B) + Python. Time each with wrapped functions + `torch.cuda.synchronize()` (this hides any stream overlap;
  measure overlap end to end instead), and `torch.profiler` for kernel shares.
- The verify-cost curve ("verify ms by rows") drives tree widths; a cliff in it caps multi-stream throughput. Compare
  curves between formats: on GB10, MLX affine 4-bit 64 rows = 1.28x one row; EXL3 stock 3.4x, patched 1.7x.
- For a slow kernel, ablate (remove decode, remove mma, make loads trivially cached) and read register use with
  `cuobjdump --dump-resource-usage` on the built `.so` (in `/cache/torch_extensions/...` inside the container).
  255 registers x 256 threads = one block per SM: latency-bound.

## 8. Format facts learned on GB10

- EXL3 (trellis + Hadamard): best quality per bit (turboderp: Qwen3.8 Flash Next KLD 4.05 bpw 0.0067 vs GGUF
  UD-IQ4_XS 0.0165, UD-Q4_K_XL ~5.2 bpw 0.0084), but under TensorFold: wide verifies cost more (decode ALU, registers),
  prompts unpack W to fp16 per 4096-row chunk (~0.3 s fixed for a 27B, ~5% of long prefill), no FP8 prompt activations.
  Measured 27B: prefill 581-931 tok/s vs MLX 1151-1376 (bf16 acts) / 1575-1898 (FP8 acts); 1-client decode 46/66
  (prose/code) vs MLX 50/73.
- MLX affine 4-bit gs64 (~4.5 bpw, no calibration): fast and flat verify curve, lower quality than EXL3/K-quants.
- Same DFlash2 drafter (trained block 8; TensorFold drafts 8-12 positions with PARALLEL>1, 16 single-stream; top-16
  candidates per position; ~15 tree nodes verified per stream per round with one client).

## 9. Pitfalls

- Never `pkill`/`killall`; kill by PID. A `while pgrep -f "x"` loop matches itself.
- `open(p,'w').write(open(p).read()...)` truncates first: read, then write.
- `diff` exits 1 when files differ: guard it under `set -e -o pipefail`.
- A server is ready when the log says `serving <model>`, not when `/v1/models` answers; poll the log.
- `docker rm` loses logs: grab stats before `end.sh`.
- `/tmp` trees can vanish: rebuild with `tf-tree.sh` + patches.
- Commit with the trailer `Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>`; `git status`
  before `git add`; leave the machine in a stated state (which server runs, on which port).
