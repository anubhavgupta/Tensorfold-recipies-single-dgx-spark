# TensorFold on DGX Spark

`tensorfold.sh` runs [TensorFold](https://github.com/ashhart/TensorFold) inside NVIDIA's PyTorch
container (`nvcr.io/nvidia/pytorch:26.07-py3`) on this machine (GB10, aarch64). Each TensorFold
version gets its own Docker image, built automatically the first time it is used.

```
./tensorfold.sh [--tf-* options] <tensorfold arguments...>
```

- `--tf-*` options are read by the script and must come **first**.
- Everything else is passed to `tensorfold` unchanged, so every current and future TensorFold flag
  (`serve`, `--host`, `--port`, `--context`, ...) works without changing the script.

## Files

| Path | Purpose |
|---|---|
| `tensorfold.sh` | The script |
| `start-qwen38-27b.sh` / `stop-qwen38-27b.sh` | Start/stop Qwen3.8-27B (see [Qwen3.8-27B preset](#qwen38-27b-preset)) |
| `.env.qwen3.8-27b` | Optional config file `start-qwen38-27b.sh` reads for its settings |
| `patches/qwen38-27b/` | Patches for the 27B on v0.6.5 (FP8 KV cache, video and media limits, memory reserve 0, pinned KV pool), applied only in the image `start-qwen38-27b.sh` runs |
| `start-qwen38-flash-next.sh` / `stop-qwen38-flash-next.sh` | Start/stop Qwen3.8-Flash-Next (see [Qwen3.8-Flash-Next preset](#qwen38-flash-next-preset)) |
| `.env.flash-next` | Optional config file `start-qwen38-flash-next.sh` reads for its settings |
| `Dockerfile` | Image recipe: base image + `pip install tensorfold[vision]` at a given commit |
| `~/.cache/huggingface` | Model cache, mounted at `/root/.cache/huggingface` in the container |
| `~/.cache/tensorfold-docker/kernels/<commit>` | Compiled CUDA/Triton kernels, one folder per TensorFold commit |
| `~/.cache/tensorfold-docker/state` | TensorFold's own cache (`~/.cache/tensorfold` in the container) |

The container runs as root, so files it writes to these folders are owned by root.

## Wrapper options

| Flag | What it does |
|---|---|
| `--tf-version REF` | Picks the TensorFold version: a tag (`v0.6.4` or `0.6.4`), a branch (`main`), a commit, or `latest` (the default). Builds that version's image on first use. You can also set it with `TF_VERSION`. |
| `--tf-rebuild` | Rebuilds the image for that version from scratch. Needs internet. |
| `--tf-offline` | Skips the GitHub check and uses an image already built here. Also sets `HF_HUB_OFFLINE=1`, so only cached models work. You can also set it with `TF_OFFLINE=1`. Turns on automatically when GitHub can't be reached. |
| `--tf-name NAME` | Names the container, for use with `docker logs`, `docker stop` and `docker exec`. |
| `--tf-detach` | Runs the container in the background. Follow its output with `docker logs -f NAME`. |
| `--tf-docker-arg ARG` | Adds one extra argument to `docker run`. Repeatable. |
| `--tf-shell` | Opens a bash shell in the container instead of running `tensorfold`. |
| `--tf-patches DIR` | Runs a derived image with `DIR/*.patch` (diffs against site-packages, `patch -p0`) applied on top of the version's image, tagged `<image>-p<hash>`; built once, rebuilt when the patches change. |
| `--tf-help` | Shows the help text. |

Script settings, read from the environment:

| Variable | Default | Purpose |
|---|---|---|
| `TF_REPO` | `https://github.com/ashhart/TensorFold.git` | Git repository to build from |
| `BASE_IMAGE` | `nvcr.io/nvidia/pytorch:26.07-py3` | Base container image |
| `HF_CACHE` | `~/.cache/huggingface` | Host model cache |
| `TF_CACHE` | `~/.cache/tensorfold-docker` | Host folder for compiled kernels and TensorFold state |

Host variables named `TENSORFOLD_*`, `NCCL_*`, `HF_*` and `CUDA_VISIBLE_DEVICES` are passed into the
container. `TENSORFOLD_NO_UPDATE_CHECK` defaults to `1`.

## Examples

### Basics

```bash
./tensorfold.sh --version                 # latest release
./tensorfold.sh --help                    # tensorfold's own help
./tensorfold.sh serve --help              # every serve flag for this version
./tensorfold.sh models                    # supported model families
./tensorfold.sh --tf-help                 # this script's options
```

### Serve Qwen3.8-27B with DFlash2 drafting

The draft model `z-lab/Qwen3.8-27B-DFlash2` is picked up automatically when it is in the cache
(`--drafter auto`).

```bash
./tensorfold.sh serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model --host 0.0.0.0 --port 8080
```

Name the draft model explicitly, or turn drafting off:

```bash
./tensorfold.sh serve Vontra/Qwen3.8-27B-MLX-4bit --drafter z-lab/Qwen3.8-27B-DFlash2 --name local-model
./tensorfold.sh serve Vontra/Qwen3.8-27B-MLX-4bit --no-drafts --name local-model
```

`--host 0.0.0.0` makes the server reachable from other machines on your network; use
`--host 127.0.0.1` to keep it local. The container uses the host's network, so `--port` needs no
Docker port mapping.

### Run in the background

```bash
./tensorfold.sh --tf-detach --tf-name tf-qwen \
  serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model --host 0.0.0.0 --port 8080

docker logs -f tf-qwen        # follow output
docker stop tf-qwen           # stop; the container is removed when it stops
```

### Test the server

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/v1/models
curl -fsS http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"local-model","messages":[{"role":"user","content":"Say hello in one sentence."}],"max_tokens":256}'
```

OpenAI-compatible clients use the base URL `http://127.0.0.1:8080/v1`.

### Pick a version

```bash
./tensorfold.sh --tf-version v0.6.3 serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model
./tensorfold.sh --tf-version 0.6.3 --version          # "v" prefix optional
./tensorfold.sh --tf-version main --version           # development branch
./tensorfold.sh --tf-version 6ea5ade --version        # a commit
TF_VERSION=v0.6.3 ./tensorfold.sh --version           # via environment
./tensorfold.sh --tf-version v0.6.4 --tf-rebuild --version   # rebuild from scratch
```

### Memory and context settings

TensorFold reads these from environment variables; the script passes them in.

```bash
TENSORFOLD_CUDA_MEMORY_LIMIT_GB=80 ./tensorfold.sh serve Vontra/Qwen3.8-27B-MLX-4bit --context 65536
TENSORFOLD_MEMORY_RESERVE_GIB=16 ./tensorfold.sh serve Vontra/Qwen3.8-27B-MLX-4bit
```

### Download models

```bash
./tensorfold.sh pull TensorFold/Qwen3.8-27B-MLX-4bit z-lab/Qwen3.8-27B-DFlash2
HF_TOKEN=hf_xxx ./tensorfold.sh pull <gated-repo>      # HF_TOKEN is forwarded
./tensorfold.sh info Vontra/Qwen3.8-27B-MLX-4bit       # reads configuration only
```

### Extra Docker arguments and debugging

```bash
./tensorfold.sh --tf-docker-arg=-v/data/models:/models serve /models/my-checkpoint
./tensorfold.sh --tf-docker-arg=--memory=100g serve Vontra/Qwen3.8-27B-MLX-4bit
./tensorfold.sh --tf-shell                              # bash inside the container
./tensorfold.sh --tf-version v0.6.3 --tf-shell
docker exec -it tf-qwen bash                            # into a running named container
```

### Two DGX Sparks (tensor parallel)

`/dev/infiniband`, `--ulimit memlock=-1` and `IPC_LOCK` are added automatically. Run the script on each
machine with the same version and settings: start rank 1 first, then rank 0.

```bash
# rank 1
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 ./tensorfold.sh \
  serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 1 --master <rank0-ip>
# rank 0
NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 ./tensorfold.sh \
  serve Vontra/Qwen3.8-27B-MLX-4bit --tp 2 --rank 0 --master <rank0-ip> --name local-model --host 0.0.0.0
```

## Qwen3.8-27B preset

`start-qwen38-27b.sh` serves `Vontra/Qwen3.8-27B-MLX-4bit` with `--drafter z-lab/Qwen3.8-27B-DFlash2`, using the settings of
the `Qwen3.8-27B-DGX-Spark-TensorFold` recipe that stock TensorFold supports: port 8888, model name
`Qwen3.8-27B`, `--parallel 8 --context 262144`, FP8 KV cache (below), `--prefill-fp8 --vision --thinking`, `--max-tokens 163840`,
`--vision-max-images 50 --vision-image-tokens 16384`, `TENSORFOLD_VIDEO_TOKENS=16384`, video input, 96 MiB request
bodies, a 0 GiB memory reserve, a pinned KV pool and a 64 MiB stack limit. Sampling follows Qwen's recommendation and
switches with `THINKING`: 1.0 / 0.95 in thinking mode, 0.7 / 0.80 with `THINKING=0`; top_k 20 and min_p 0.0 either way
(TensorFold has no presence or repetition penalty).

**Patches (`PATCHES=1`, the default).** Four of the old recipe's v0.6.0 patches are ported to v0.6.5 in
`patches/qwen38-27b/`. The script runs them as `./tensorfold.sh --tf-patches patches/qwen38-27b`, which builds a derived
image `tensorfold:<version>-<commit>-p<patch hash>` on top of the stock one (rebuilt when a patch changes). The stock
image, Flash Next and every other command stay unpatched, and each patch only acts when its variable is set, which only
this script does:

| Patch | What it adds | Setting (default) | Stock v0.6.5 |
|---|---|---|---|
| `0001-qwen27-kv-fp8` | FP8 KV cache for the 27B | `KV_DTYPE=fp8` → `TENSORFOLD_KV_DTYPE` | `--kv-dtype` is Flash Next only |
| `0002-qwen27-video-media-limits` | video input for the 27B; size limits as variables | `REQUEST_BODY_MIB` (96), `IMAGE_TOTAL_MIB` (64), `VIDEO_MIB` (64), `VIDEO_TOTAL_MIB` (96) | video for Flash Next only; 32 / 20 / 16 / 20 MiB |
| `0003-qwen27-memory-reserve` | memory reserve down to 0 | `MEMORY_RESERVE_GIB` (0) | floor 2 GiB |
| `0004-qwen27-kv-pool` | pinned KV cache pool | `KV_POOL_GB` (auto) → `TENSORFOLD_KV_POOL_GIB` | caches grow and shrink on demand |

The many-images part of the old recipe's image patch is stock now (`--vision-max-images`, `--vision-image-tokens`,
v0.6.3+). Not ported: YaRN (1M-token windows). `PATCHES=0` runs the plain stock image; it needs `KV_DTYPE=bf16` and
defaults `KV_POOL_GB=0`, `MEMORY_RESERVE_GIB=2`.

**Pinned KV pool (`KV_POOL_GB=auto`).** Once loaded, the server takes one block of cache memory, touches its pages and
keeps it inside torch's allocator for good; every stream's cache and the kept prompt states grow inside it, and the
memory gate counts against the pool instead of the host's free memory. Memory use therefore stays flat instead of
rising and falling with each request, and other programs can't take what the caches need. `auto` is the memory free at
start minus 31 GiB (minus 1 GiB a stream over 8), at most 78 GiB - ~2.5M fp8 tokens, enough for 8 full 262k windows;
the startup line reads `inside a pinned 78.0 GiB cache pool (2,555,904 tokens)`. `KV_POOL_GB=<n>` sets it,
`KV_POOL_GB=0` turns it off. With `--vision-offload`, the vision tower's move to the CPU no longer empties torch's
cache while a pool is set (it would hand the pool back). Measured: startup estimate 107.7 GiB of 115 (the pool
counted in full), 8 concurrent 44k-token needle prompts all answered, host memory flat at 107.8-109.8 GB throughout,
and decode speed the same with or without the pool (512-token replies including prefill, 1/2/4/8 clients: prose
40.7/70.4/120.4/194.8 vs 40.7/70.9/121.1/192.3 tok/s, code 72.3/126.5/212.8/306.9 vs 70.8/126.6/206.7/308.9).

**Video and media limits.** Stock v0.6.5 has the video code (frame groups, `TENSORFOLD_VIDEO_TOKENS`) but turns it
on only for Flash Next; patch 0002 turns it on for the 27B too (`video_url` parts with an mp4 data URL, or an
https URL with `VISION_URLS=1`). Checked: a 6 s 1280x720 clip (4,399 prompt tokens) described correctly; a 44 MiB
body with 33 MiB of images (four 1700x1700 PNGs) answered; 50 images accepted and a 51st refused.

**FP8 KV cache (`KV_DTYPE=fp8`, the default).** Stock `--kv-dtype` is Flash Next only, so the 27B would keep a bf16
cache. Patch 0001 stores keys and values as e4m3, one byte, rounded the same way before use so drafted replies still
equal serial ones. Measured here (v0.6.5, `--parallel 8 --context 262144`, no pool):

| | bf16 | fp8 |
|---|---:|---:|
| KV per token / full 262,144 window | 64 KiB / 16 GiB | 32 KiB / 8 GiB |
| Startup estimate | 45.36 GiB | 37.77 GiB |
| GPU memory added by an 80k-token prompt | 6.2 GiB | 3.8 GiB |
| Full 262k windows that fit at once (~90 GiB free) | 4–5 | 8+ |
| Decode, prose, 1/2/4/8 clients (tok/s) | 48.2 / 82.4 / 144.4 / 222.8 | 49.0 / 85.1 / 144.0 / 236.6 |
| Decode, code, 1/2/4/8 clients (tok/s) | 71.7 / 126.5 / 215.8 / 332.5 | 85.2 / 145.6 / 243.4 / 334.2 |

Checks on the fp8 image: the 80k needle is found, 4 concurrent prompts (the batched prefill path) answer correctly,
and repeated greedy replies are identical. The old recipe measured the quality cost at 4k context: perplexity 3.315
vs 3.316, KL 0.0031 (FP8 prompts, `PREFILL_FP8=1`, cost ~14x more).

```bash
./start-qwen38-27b.sh                                   # background container tf-qwen38-27b
./start-qwen38-27b.sh --parallel 4 --context 131072     # extra args override the defaults
KV_DTYPE=bf16 PARALLEL=4 THINKING=0 ./start-qwen38-27b.sh  # bf16 cache
KV_POOL_GB=0 ./start-qwen38-27b.sh                      # no pinned pool: caches grow on demand
PATCHES=0 KV_DTYPE=bf16 ./start-qwen38-27b.sh           # plain stock image
FOREGROUND=1 ./start-qwen38-27b.sh                      # attached; Ctrl+C stops it
docker logs -f tf-qwen38-27b
./stop-qwen38-27b.sh                                    # stop and remove the container
```

Settings: `TF_VERSION`, `MODEL_ID`, `DRAFT_ID` (empty: `--no-drafts`), `SERVED_NAME`, `HOST`, `PORT`, `NAME`,
`FOREGROUND`, `PATCHES`, `KV_DTYPE`, `KV_POOL_GB`, `MEMORY_RESERVE_GIB`, `REQUEST_BODY_MIB`, `IMAGE_TOTAL_MIB`,
`VIDEO_MIB`, `VIDEO_TOTAL_MIB`, `PARALLEL`, `CONTEXT`, `PREFILL_FP8`, `CHECKPOINT_SLOTS`, `VISION`, `VISION_URLS`,
`VISION_MAX_IMAGES`, `VISION_IMAGE_TOKENS`, `THINKING`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`,
plus any `TENSORFOLD_*` variable. They can also go in `.env.qwen3.8-27b` (every line commented out at its default;
`ENV_FILE` picks another file); a variable already in the environment wins over the file.

Compared on this Spark with the old patched recipe (thinking off, 200-token replies, aggregate tok/s, both at
`--parallel 8`), new vs old: prose 48.2/48.3, 82.4/83.2, 144.4/134.7, 222.8/204.7 at 1/2/4/8 clients; code
71.7/78.6, 126.5/134.9, 215.8/195.7, 332.5/262.1.

How many parallel requests fit: the default 78 GiB pool holds ~2.5M fp8 tokens (32 KiB a token, 8 GiB per full
262,144-token request), so the default 8 at 262,144 all fit (~9 at full length). With `KV_DTYPE=bf16` (64 KiB a
token) the same pool is ~1.25M tokens (without a pool, ~90 GiB is left for caches, ~1.4M):

| `--parallel` (bf16 cache, `KV_POOL_GB=0`) | Longest `--context` all requests can use at once |
|---|---|
| 1–5 | 262,144 (full) |
| 6 | ~220K |
| 7 | ~190K |
| 8 | ~160K (`--context 163840`) |

`--context` only caps each request; caches grow as requests need them (inside the pool, if any). Past that total, new requests wait for
memory and, in the worst case, the newest running request is stopped with "ran out of memory". Other workloads on
the machine lower these numbers.

## Qwen3.8-Flash-Next preset

`start-qwen38-flash-next.sh` / `stop-qwen38-flash-next.sh` serve `Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP`,
using the settings of the `Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold` recipe - but with **stock**
TensorFold (no image patches): by v0.6.5, everything that recipe's patches added for v0.6.1 is upstream
(`--vision-max-images`, `--vision-image-tokens`, `--kv-dtype`, `--mtp-drafts`/`--mtp-confidence`,
`--ple-on-ssd`, `TENSORFOLD_PREFILL_ROWS`, `TENSORFOLD_VIDEO_TOKENS`,
`TENSORFOLD_MEMORY_RESERVE_GIB`). Not carried over, because stock TensorFold has no equivalent: the
`DRAFT_LANGUAGE` MTP-vocabulary patch and `TENSORFOLD_MTP_COPY` (prompt-lookup drafts ahead of MTP).

Defaults: port 8888, model name `Qwen3.8-Flash-Next`, `--parallel 5 --context 262144 --kv-dtype int8
--ple-on-ssd`, Qwen's recommended sampling, switching with `--thinking`/`--no-thinking`: thinking mode
`--temperature 1.0 --top-p 0.95`, instruct/non-thinking mode `--temperature 0.7 --top-p 0.80`; `--top-k 20`
and `--min-p 0.0` are the same either way (Qwen's instruct-mode `presence_penalty 1.5` and a
`repetition_penalty` aren't set - TensorFold has no such settings, always decoding as if both were off).
Also `--vision --vision-max-images 50 --vision-image-tokens 16384 --thinking`, `--mtp-drafts 6
--mtp-confidence 0.60` (matches the old recipe's swept value; TensorFold's own Flash Next default is
6 / 0.70, measured ~3-4% slower, same output), `TENSORFOLD_VIDEO_TOKENS=16384`,
`TENSORFOLD_MEMORY_RESERVE_GIB=2`, `TENSORFOLD_PREFILL_ROWS=2048` (while `PLE_ON_SSD=1`),
`TENSORFOLD_VISION_WORKSPACE_MIB=0` (scratch borrowed from the system reserve only while encoding,
instead of a standing 4 GiB reservation) and a 64 MiB stack limit. A concurrency 1-5 benchmark on
prose and code (2026-10-05) found these three - MTP confidence, prefill rows and vision workspace -
fully closed the throughput gap against the old patched recipe; with them matched the two are on par.

```bash
./start-qwen38-flash-next.sh                                 # background container tf-qwen38-flash-next
./start-qwen38-flash-next.sh --parallel 3 --kv-dtype bf16     # extra args override the defaults
PORT=9000 TF_VERSION=v0.6.5 ./start-qwen38-flash-next.sh
FOREGROUND=1 ./start-qwen38-flash-next.sh                     # attached; Ctrl+C stops it
docker logs -f tf-qwen38-flash-next
./stop-qwen38-flash-next.sh
```

Settings read from the environment: `TF_VERSION`, `MODEL_ID`, `SERVED_NAME`, `HOST`, `PORT`, `NAME`,
`FOREGROUND`, `PARALLEL`, `CONTEXT`, `KV_DTYPE`, `PLE_ON_SSD`, `VISION`, `VISION_MAX_IMAGES`,
`VISION_IMAGE_TOKENS`, `THINKING`, `MAX_TOKENS`, `TEMPERATURE`, `TOP_P`, `TOP_K`, `MIN_P`, `MTP_DRAFTS`,
`MTP_CONFIDENCE`, plus any `TENSORFOLD_*` variable. `stop-qwen38-flash-next.sh` reads `NAME`, `HOST`,
`PORT` (must match the start script's) and `STOP_TIMEOUT`.

**Config file:** `.env.flash-next`, beside the script, is read before the defaults above (so it only
changes what it sets) - `KEY=value` lines, `#` comments, quotes optional, never executed. A variable
already in the environment wins over the file either way. Ships with every setting commented out at
its current default; uncomment and edit a line to persist an override without passing env vars on
every start. `ENV_FILE=/path/to/other.env` points at a different file; `ENV_FILE=/dev/null` (or
deleting `.env.flash-next`) runs on pure script defaults.

## Updating

Without `--tf-version`, the script looks up the newest release on every run, so a new release is
picked up automatically. The image builds once, which takes a few minutes, and the first request then
compiles kernels (about a minute). If you pin a version with `--tf-version`, change it to the new tag.

Don't run `tensorfold update` in the container: the change is lost when the container stops.

To restart a running server on the new version, stop it and run the same command again.

Remove an old version:

```bash
docker images tensorfold
docker rmi tensorfold:v0.6.4 tensorfold:v0.6.4-6ea5ade26c43
sudo rm -rf ~/.cache/tensorfold-docker/kernels/6ea5ade26c43   # root-owned
```

## Offline use

- If GitHub can't be reached within 15 seconds, the script switches to offline mode by itself.
  `--tf-offline` (or `TF_OFFLINE=1`) skips the check.
- Without `--tf-version`, it runs the newest version already built here. With `--tf-version`, that
  version must already be built; otherwise the script lists the versions you have.
- `HF_HUB_OFFLINE=1` is set, so only models already in `~/.cache/huggingface` can be used.
- Building a new version, `--tf-rebuild` and downloading models need internet. Run a version once
  while online if you'll need it offline later.

```bash
./tensorfold.sh --tf-offline serve Vontra/Qwen3.8-27B-MLX-4bit --name local-model --port 8080
./tensorfold.sh --tf-offline --tf-version v0.6.4 --version
```

## Troubleshooting

| Symptom | Check |
|---|---|
| First request takes about a minute | Kernels are compiling; later requests and restarts reuse them |
| `unknown TensorFold version/ref` | The tag, branch or commit doesn't exist on GitHub |
| `offline and no local image for ...` | Build that version while online |
| Server start stops after `loading ...` with the GPU idle | A stale kernel build lock: stop the server, delete the lock file named in the log under `~/.cache/tensorfold-docker/kernels/<commit>`, and start again |
| `not a checkpoint TensorFold is tested with` | A warning for checkpoints other than the official ones; the model still runs |
| Can't reach the server from another machine | Use `--host 0.0.0.0` and check the firewall |
