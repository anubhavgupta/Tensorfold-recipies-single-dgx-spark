# TensorFold v0.6.5 on DGX Spark

This repo contains a recipe for running [TensorFold](https://github.com/ashhart/TensorFold) on a single DGX Spark
(GB10, 128 GB) for three models. Key measured decode speeds (200-token replies, thinking off): "per stream" is one
client; "peak" is the aggregate over the preset's max tested concurrency (8, 8 and 5 clients):

| Model | Per stream (prose / code) | Peak aggregate (prose / code) |
|---|---:|---:|
| Qwen3.8-27B | 49.0 / 85.2 tok/s | 236.6 / 334.2 tok/s |
| Ternary-Bonsai-2-27B | 44.2 / 87.6 tok/s | 195.8 / 308.9 tok/s |
| Qwen3.8-Flash-Next | 46.8 / 63.7 tok/s | 186.4 / 260.3 tok/s |

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
