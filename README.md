# TensorFold v0.6.5 on DGX Spark

This repo contains recipes for running [TensorFold](https://github.com/ashhart/TensorFold) on a single DGX Spark
(GB10, 128 GB) for several models (v0.6.5 Python engine, and TensorFold 1.0.12 native for `dory-4b` and `3.5-glm-2`). Key measured decode speeds (200-token replies, thinking off): "per stream" is one
client; "peak" is the aggregate over the preset's max tested concurrency (8, 8 and 5 clients):

| Model | Per stream (prose / code) | Peak aggregate (prose / code) |
|---|---:|---:|
| Qwen3.8-27B | 49.0 / 85.2 tok/s | 236.6 / 334.2 tok/s |
| Ternary-Bonsai-2-27B | 44.2 / 87.6 tok/s | 195.8 / 308.9 tok/s |
| Qwen3.8-Flash-Next | 46.8 / 63.7 tok/s | 186.4 / 260.3 tok/s |
| Qwen3.8-Flash-Next EXL3 (`flash-2`)* | 54 / 93 tok/s | 158 / 254 tok/s |
| Qwen3.8-Flash-Next EXL3 + DFlash2 (`flash-3`)*** | 57 / 111 tok/s | 167 / 262 tok/s |
| Nemotron-Labs Dory1.0-4B NVFP4 (`dory-4b`)** | 52 / 52 tok/s | 511 / 515 tok/s |
| GLM-5.3-Flash EXL3 2.05bpw (`3.5-glm-2`)** | 31 / 32 tok/s | 40 / 45 tok/s |
| Ornith-1.5-122B NVFP4 + DFlash2 (`ornith-1.5-122B`)**** | 40 / 49 tok/s | 61 / 93 tok/s |
| Ornith-1.5-122B MLX 4-bit + DFlash2, TensorFold 1.1.1 (`ornith-1.5-122b-tensorfold`)**** | 42 / 74 tok/s | 70 / 96 tok/s |

\* Different method: 256-token chat replies, 9 clients (see its section).
\** TensorFold 1.0.12 native (Zig) engine with this repo's patches; peak at 16 (Dory) and 4 (GLM) clients.
\*** flash-2 moved to TensorFold 0.6.6 (the last Python engine) with the DFlash2 drafter instead of MTP; flash-2's method.
\**** Served with SGLang, not TensorFold; distinct prompts, 512-token replies, greedy, 4 clients each holding a full
262K context (see its section).

Full results in [Benchmarks](#benchmarks).

## Benchmarks

Measured on one DGX Spark (GB10, 128 GB), TensorFold v0.6.5, October 2026, with the presets' default settings unless
noted. Throughput is aggregate decode tok/s across N clients sending the same prompt at once, thinking off
(`chat_template_kwargs: {"enable_thinking": false}`), greedy. "Old" is the earlier patched recipe on this Spark
(`Qwen3.8-27B-DGX-Spark-TensorFold` on v0.6.0, `Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` on v0.6.1).

### Qwen3.8-27B (`qwen38-27b/start.sh`)

Throughput, 200-token replies, `--parallel 8`, no KV pool:

| Clients | Prose: fp8 KV | Prose: bf16 KV | Prose: old | Code: fp8 KV | Code: bf16 KV | Code: old |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 49.0 | 48.2 | 48.3 | 85.2 | 71.7 | 78.6 |
| 2 | 85.1 | 82.4 | 83.2 | 145.6 | 126.5 | 134.9 |
| 4 | 144.0 | 144.4 | 134.7 | 243.4 | 215.8 | 195.7 |
| 8 | 236.6 | 222.8 | 204.7 | 334.2 | 332.5 | 262.1 |

Pinned KV pool vs none (fp8 KV, 512-token replies including prefill, 1/2/4/8 clients): the pool costs nothing.

| | 1 | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| Prose, 78 GiB pool | 40.7 | 70.4 | 120.4 | 194.8 |
| Prose, no pool | 40.7 | 70.9 | 121.1 | 192.3 |
| Code, 78 GiB pool | 72.3 | 126.5 | 212.8 | 306.9 |
| Code, no pool | 70.8 | 126.6 | 206.7 | 308.9 |

Memory and long context (fp8 KV, 78 GiB pool):

| Test | Result |
|---|---|
| Startup estimate | 107.7 GiB of 115 (pool counted in full); no pool: 37.8 GiB (bf16 KV: 45.4) |
| KV per token / full 262,144 window | 32 KiB / 8 GiB (bf16: 64 KiB / 16 GiB) |
| 8 concurrent 44K-token needle prompts | all 8 correct, 4m10s total, host memory flat at 107.8-109.8 GB |
| **Concurrent full-262K streams** | **8 measured**: a 258,537-token prompt was prefilled once (333 s), then sent 10 times at once with `--parallel 12`; 8 started within 5 s, each ran to 261,733 tokens (3,196-token replies, ~7.4 tok/s per stream, identical output), and the other 2 waited for memory. The pool holds 9 full windows; here the cached copy of the shared prompt took the 9th, so 9 should fit with different prompts (not measured: separate 259K prompts prefill one after another and never overlap). 10 would need a pool of ~82 GiB. |
| Prefill of one 259K-token prompt | ~330 s (~780 tok/s); prompts this long prefill one at a time, since batched prefill shares 1,024 tokens a step (4,096 when nothing decodes) |
| Video, 6 s 1280x720 clip | described correctly (4,399 prompt tokens) |
| Images | 44 MiB body with 33 MiB of images answered; 50 images accepted, a 51st refused |

### Qwen3.8-Flash-Next (`qwen38-flash-next/start.sh`)

Throughput, 200-token replies, `--parallel 5`, int8 KV, `--ple-on-ssd`, MTP drafts 6 / confidence 0.60; new (stock
v0.6.5) vs old (patched v0.6.1):

| Clients | Prose: new | Prose: old | Code: new | Code: old |
|---:|---:|---:|---:|---:|
| 1 | 46.8 | 47.5 | 63.7 | 70.0 |
| 2 | 93.7 | 92.4 | 138.4 | 139.5 |
| 3 | 130.3 | 128.7 | 188.5 | 189.1 |
| 4 | 161.3 | 160.3 | 232.3 | 224.1 |
| 5 | 186.4 | 181.1 | 260.3 | 245.6 |

MTP accept rate on code: 73.9%. With stock TensorFold defaults (MTP confidence 0.70, default prefill rows, 4 GiB vision
workspace), prose was 45.9 / 65.4 / 112.5 / 114.4 / 165.8 tok/s at 1-5 clients; the preset's three settings close that
gap. Concurrent full-context streams were not measured for Flash Next.

### Qwen3.8-Flash-Next EXL3 (`flash-2/start.sh`)

`turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6` on TensorFold v0.6.5 with the `flash-2/overlay` patches (faster EXL3
prompt kernels, image prefix cache, elastic kept states, bigger media limits, n-gram table left unpinned), MTP drafts,
`--parallel 9`, 262,144-token context. A different method from the tables above: chat prompts per content class,
greedy, 256-token replies, thinking off, aggregate decode tok/s (`flash-3/bench/suite.py`).

| Content | 1 client | 4 clients | 9 clients |
|---|---:|---:|---:|
| Code | 93 | 177 | 254 |
| Prose | 54 | 106 | 158 |
| DevOps | 98 | 172 | 254 |
| JSON | 97 | 188 | 274 |

These were measured before the prompt-kernel overlay; with it decode was unchanged (code 66 / 176 / 249 steady and prose
151 at 9 clients in a rerun). A single short request alone decodes at about 47 tok/s. For comparison, TabbyAPI + the
patched exllamav3 fork on the same pack: code 76 / 131 / 133, prose 50 / 91 / 117 at 1 / 4 / 9 clients.

Prefill, one cold stream (tok/s):

| Prompt | Stock v0.6.5 | flash-2 overlay |
|---|---:|---:|
| 4K | 859 | 1026 (+19%) |
| 32K | 846 | 1032 (+22%) |
| 131K | 790 | 934 (+18%) |
| 249K | 748 | 880 (+18%) |

| Test | Result |
|---|---|
| **Concurrent full-262K streams** | 9 x ~249K-token prompts + 256 new tokens: **9/9 completed** (3034 s wall). Prefills run one at a time (~740 tok/s); per-stream decode 8-49 tok/s depending on start time. Measured on stock v0.6.5; startup reports 9 full windows (4.47 GiB of KV each) with the overlay too. |
| n-gram table unpinned (`TENSORFOLD_NGRAM_LOCK_GIB=0`, default) | 45 GiB `MemAvailable` after start (stock pins it: 7.5 GiB); decode 46.9 tok/s and cold 35K prefill 33.5 s, the same as stock; 6 x 80K and 6 x 100K concurrent runs had no errors |
| Elastic kept states (`TENSORFOLD_KEEP_MAX`) | 9 parallel ~15K-token chats: all 9 follow-ups warm (~1 s each); stock re-prefilled 5 of 9 (71 s) |
| Image prefix cache | a repeated image prompt resumed 1,767 cached tokens; greedy output identical cached or not |
| `TENSORFOLD_PREFILL_ROWS` 4096 / 8192 | +3% / +4% prefill at 32K, but one fewer full-window stream; the default 2048 stays |

Details and diffs: `flash-2/overlay/README.md`; raw results: `instructions.md` (rounds 2-4).

### Qwen3.8-Flash-Next EXL3 + DFlash2 (`flash-3/start.sh`)

flash-2 on TensorFold 0.6.6 with its overlays as patches, drafting with `incoai/Qwen3.8-Flash-Next-DFlash2` instead of
the MTP head (`flash-3/patches/0006`: target taps, a batched drafter, exact verification). TensorFold 1.0 and later are
native binaries without EXL3 or DFlash for Flash Next, so 0.6.6 is the latest engine this pack runs on. Same method as
flash-2 above (greedy, 256 tokens, thinking off, steady aggregate tok/s), back to back with flash-2:

| Content | 1 client: flash-2 / flash-3 | 4 clients | 9 clients |
|---|---:|---:|---:|
| Code | 95 / 111 | 178 / 190 | 254 / 262 |
| Prose | 55 / 57 | 104 / 114 | 157 / 167 |
| DevOps | 101 / 89 | 167 / 180 | 249 / 261 |
| JSON | 99 / 105 | 186 / 211 | 294 / 284 |

Sampled at T=0.7, 4 clients: +9% to +15%; 9 clients: −2% to +6%. Greedy replies equal `"draft": false`.
Details, sweeps and limits: `flash-3/README.md`.

### Ternary-Bonsai-2-27B (`bonsai-27b/start.sh`)

Throughput, 200-token replies (`ignore_eos`), `--parallel 8`, fp8 KV, default pool; 0007 and the 0006 path
(`TENSORFOLD_BONSAI_PQ2=0`) measured back to back with the same script:

| Clients | Prose: 0007 + DFlash2 | Prose: 0006 + DFlash2 | Prose: 0007, no drafts | Code: 0007 + DFlash2 | Code: 0006 + DFlash2 | Code: 0007, no drafts |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 44.2 | 48.4 | 26.3 | 87.6 | 79.8 | 26.3 |
| 2 | 86.3 | 83.4 | 47.8 | 148.3 | 133.4 | 48.0 |
| 4 | 134.1 | 137.1 | 88.0 | 234.9 | 219.7 | 87.6 |
| 8 | 195.8 | 204.4 | 142.3 | 308.9 | 294.9 | 143.0 |

Host memory in use while serving: 112 GB with 0007 vs 119 GB with 0006 (same 90 GiB pool). Without drafts 0007
decodes at 26.3 tok/s single-stream (0006: ~21.7, 0005: ~1.3). Drafted prose varies by a few percent between runs
because PQ2's exact fp16 scales change a few tokens, and with them the acceptance rate.

Earlier comparison against Qwen3.8-27B + DFlash2 with 0006 (a different script, without `ignore_eos`): Bonsai beat
the 27B on prose at 1-4 clients and tied at 8; the 27B led on code at higher concurrency (DFlash2 is trained on the
27B, and large verification batches are compute-bound).

### Nemotron-Labs Dory1.0-4B NVFP4 (`dory-4b/start.sh`)

`nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4` (a looped nGPT, 2 recurrent loops by default) on TensorFold 1.0.12's
native engine. TensorFold has no Dory family, so `dory-4b` adds one: `libdory.so` (NVFP4 tensor-core GEMMs, head-dim-256
flash attention with sliding windows, slerp residuals, the recurrent loops, CUDA graphs) behind a lane backend registered
by `dory-4b/patches/0001-register-dory.patch`. Logits match an fp32 PyTorch reference (top-1 99-100 %).

Same-prompt greedy, 200-token replies, thinking off, 16 streams x 262,144-token window:

| Clients | Prose | Code |
|---:|---:|---:|
| 1 | 52 | 52 |
| 4 | 179 | 182 |
| 16 | 511 | 515 |

Prefill 3,000-3,450 tok/s up to 26K tokens, 1,150 tok/s at 249K; an edit-style request (copy drafts) 242 tok/s.
`LOOPS=1` / `LOOPS=4`: 72 / 32 tok/s single stream. Details: `dory-4b/README.md`.

### GLM-5.3-Flash EXL3 2.05bpw (`3.5-glm-2/start.sh`)

`turboderp/GLM-5.3-Flash-exl3 @ 2.05bpw` (80 GiB) on one Spark with TensorFold 1.0.12's native engine. Upstream serves
this model on CUDA only from NVIDIA's 190 GiB NVFP4 checkpoint over several hosts; `3.5-glm-2/patches` add EXL3 to the
CUDA engine (upstream PR #837, rebased) and load the pack in the GLM family at one rank, with a faster EXL3 prompt-experts
kernel. MTP drafts on, 4 streams x 131,072-token window, port 8890.

| Clients | Prose (same prompt, greedy) | Code (same prompt, greedy) | Distinct prompts, T=1 |
|---:|---:|---:|---:|
| 1 | 30.9 | 32.4 | 28.7 |
| 4 | 39.6 | 44.7 | 30.3 |

Prefill 433-456 tok/s up to 78K tokens. Drafted replies equal undrafted ones, concurrent equal solo. Details:
`3.5-glm-2/README.md`.

## How to run

There is nothing to install on the host: everything runs in a Docker container. It needs:

- A GB10 (aarch64) DGX Spark with 128 GB of unified memory.
- Docker with the NVIDIA Container Toolkit, so the container can see the GPU.
- `git` and `patch` on the host.
- A free [NGC](https://developer.nvidia.com/ngc) account: `docker login nvcr.io`, once, for the base image.
- Internet the first time each version is used (the script builds its image then); afterwards it runs offline.

Then:

```bash
git clone https://github.com/anubhavgupta/Tensorfold-recipies-single-dgx-spark
cd Tensorfold-recipies-single-dgx-spark

./tensorfold.sh --version            # first run: pulls the base image, builds the tensorfold image

# one line per model - pull only what you will run (the 27B and Bonsai lines include their DFlash2 drafter)
./tensorfold.sh pull Vontra/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2            # Qwen3.8-27B preset
./tensorfold.sh pull prism-ml/Ternary-Bonsai-2-27B-mlx-2bit z-lab/Qwen3.8-27B-DFlash2  # Ternary-Bonsai-2-27B preset
./tensorfold.sh pull Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP                             # Qwen3.8-Flash-Next preset (drafts with its built-in MTP head)
```

Then start and stop the model you want with its preset's `start.sh` and `end.sh` (one pair per model):

```bash
# Qwen3.8-27B preset
./qwen38-27b/start.sh            # background container tf-qwen38-27b; server on port 8888
./qwen38-27b/end.sh              # stop and remove the container

# Ternary-Bonsai-2-27B preset
./bonsai-27b/start.sh            # background container tf-bonsai-27b; server on port 8888
./bonsai-27b/end.sh

# Qwen3.8-Flash-Next preset
./qwen38-flash-next/start.sh     # background container tf-qwen38-flash-next; server on port 8888
./qwen38-flash-next/end.sh
```

The Dory and GLM recipes build their own TensorFold 1.0.12 image on the first `start.sh` (no `tensorfold.sh pull`
needed; they download their checkpoint if it is missing):

```bash
./dory-4b/start.sh               # container tf-dory-4b; server on port 8888
./dory-4b/stop.sh
./3.5-glm-2/start.sh             # container tf-glm53-flash; server on port 8890 (~4 min load, needs ~100 GiB free)
./3.5-glm-2/stop.sh
```

`start.sh` runs in the background by default; `FOREGROUND=1` attaches to it instead (Ctrl+C stops it).
Track output with `docker logs -f <container name>`. Full settings for each preset are in [readme-detailed.md](readme-detailed.md).

The first request after a build takes about a minute while the CUDA/Triton kernels compile; restarts after
that are fast. For day-to-day use, see [readme-detailed.md](readme-detailed.md).

## Credits

The Qwen3.8-27B preset is based on [MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold](https://github.com/MiaAI-Lab/Qwen3.8-27B-DGX-Spark-TensorFold): its DGX Spark recipe and the Qwen3.8-27B patches it ships are the starting point for this repository (ported here to newer TensorFold releases).

## Full guide

Everything else - files, wrapper options, examples, per-preset settings, patches, the chat
template, updating, offline use and troubleshooting - is in
[readme-detailed.md](readme-detailed.md).

### Ornith-1.5-122B NVFP4 + DFlash2 on SGLang (`ornith-1.5-122B/start.sh`)

`ornith-ai/Ornith-1.5-122B-NVFP4` (a Qwen3.5-122B-A10B-shaped hybrid GDN MoE, vision on) with the DFlash2 drafter
`incoai/Ornith-1.5-122B-DFlash2`, served by stock SGLang (`lmsysorg/sglang:dev-cu13`) rather than TensorFold.
4 streams x 262,144 tokens fit at once (checked with four 245K prompts). Distinct prompts, 512-token replies, greedy,
thinking off; 6 verified tokens per round (each verified token pulls in more experts, so the drafter's trained block
of 16 is slower here):

| Clients | Mixed (prose/code/math) | Code |
|---:|---:|---:|
| 1 | 39.9 | 49.1 |
| 2 | 50.1 | 55.6 |
| 4 | 61.4 | 93.3 |

Without speculation: 30 tok/s single stream. Prefill 2,600-3,200 tok/s up to 32K, 1,535 tok/s at 98K, ~800 tok/s
averaged over a 245K prompt. Details: `ornith-1.5-122B/README.md`.

The same model on TensorFold 1.1.1's native engine (`ornith-1.5-122b-tensorfold`, MLX 4-bit export, a patch that lets
the qwen3_5_moe family draft with DFlash2): decode 42 / 74 tok/s single stream (mixed / code) and 70 / 96 at 4 streams,
ready in ~100 s, but prefill is 600 tok/s at 2K falling to 107 at 98K (128-row prompt passes, prompt attention off the
tensor cores) and other streams stall during a prompt. Comparison: `ornith-1.5-122b-tensorfold/README.md`.

### DeepSeek-V4.1-Flash-1017 SAGE EXL3 1.62bpw on vLLM (`ds4.1-flash-1017/scripts/serve.sh`)

A copy of [vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe](https://github.com/vcruz305/DSV4.1-Flash-1017-EXL3-Spark-recipe)
(vLLM with EXL3 experts, a cold-expert alias side file read from NVMe, DSpark k=5), set up on this Spark for one stream
at a 229,376-token window: OpenAI API on port 8888, model `dsv41`. Short-prompt decode 41.3 tok/s (128 ms/step,
4.9 tok/step); a 225K prompt prefills at 725 tok/s and decodes at 28.7 tok/s with the needle found. Local changes
(base image rebuilt from spark-ds41f, no-sudo helpers, a 0.5 GiB KV pool for this box's memory) are listed at the top
of `ds4.1-flash-1017/README.md`.

