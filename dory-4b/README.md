# NVIDIA Nemotron-Labs Dory1.0-4B (NVFP4) // TensorFold // DGX Spark

Serve **[nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4)**
on one DGX Spark (GB10) with [TensorFold](https://github.com/ashhart/TensorFold) 1.0.12 (`46a08c1`, the latest release),
native Zig engine `tensorfold-native`. Up to 16 streams, each with the model's full 262,144-token window, OpenAI-compatible
API with reasoning split, tool calls, structured output and TensorFold's prompt cache.

```bash
./start.sh                 # builds the image the first time (~4 min), downloads the checkpoint (3.2 GB), serves on :8888
./start.sh restart         # after changing settings
./stop.sh                  # (end.sh is the same script)
LOOPS=4 ./start.sh restart # more thinking per token, see below
```

Settings come from the environment or `./.env` (defaults and comments in `scripts/config.sh`): `PARALLEL` (16), `CONTEXT`
(262144), `LOOPS` (2), `KV_MODE` (`per_loop`), `DRAFTS` (1, speculation by copying), `PROMPT_CACHE_GIB` (8), `THINKING`
(1), `TEMPERATURE`/`TOP_P` (1.0/0.95, NVIDIA's recommendation), `MAX_TOKENS`, `PORT`, `SERVED_NAME` (`Dory1.0-4B`).
Extra arguments to `start.sh` go to `tensorfold-native serve`. Logs: `~/.cache/tf-dory-4b/logs/latest.log`.

## Quick start

Requirements: a DGX Spark (GB10, 128 GB unified memory), Docker with the NVIDIA Container Toolkit, an NGC login for the
base image (`docker login nvcr.io`, once), internet on the first run, and the `hf` CLI on the host (else the download
runs in the container). About 10 GB free memory is enough to start; more for long contexts and many streams.

```bash
git clone https://github.com/anubhavgupta/Tensorfold-recipies-single-dgx-spark
cd Tensorfold-recipies-single-dgx-spark/dory-4b
./start.sh                       # first run: builds the image, downloads the checkpoint (3.2 GB), serves in ~10 s once built
docker logs -f tf-dory-4b       # follow the server (or the log under ~/.cache/tf-dory-4b/logs)
```

Send a request (OpenAI API at `http://localhost:8888/v1`, model `Dory1.0-4B`):

```bash
# thinking on (the default): the reasoning comes back in message.reasoning_content
curl -s localhost:8888/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Dory1.0-4B","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":1000}'

# thinking off for this request
curl -s localhost:8888/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"Dory1.0-4B","messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}'

# tool calls: pass "tools" as with any OpenAI-compatible server; calls come back in message.tool_calls
```

Change settings with environment variables or a `.env` file in this folder (see below), then `./start.sh restart`.
Stop with `./stop.sh`. Only one large server fits in memory: stop other recipes first.

### Choosing `LOOPS` and `KV_MODE`

`LOOPS` is how many times the recurrent block runs per token (the model card's "loop count"): more loops think more per
token, so accuracy goes up and speed goes down. `KV_MODE` decides whether each loop keeps its own attention cache.

| Setting | Use it when | 1 client | 16 clients | Full-attention KV a token |
|---|---|---:|---:|---:|
| `LOOPS=1` | speed matters most, simple chat | 72 tok/s | 731 tok/s | 36 KiB |
| `LOOPS=2` (default) | NVIDIA's recommended balance | 52 | 511 | 56 KiB (`per_loop`) |
| `LOOPS=4` | hardest tasks (agentic, SWE-bench-like); fewer reply tokens | 32 | 316 | 96 KiB (`per_loop`) |
| `KV_MODE=last_loop` | long contexts or many long streams: KV at the one-loop size whatever `LOOPS` is | same speed | same | 36 KiB |

- `per_loop` (default) is the reference mode: each loop of each recurrent attention layer has its own cache.
- `last_loop` keeps one cache per layer, which each loop overwrites; NVIDIA reports accuracy close to `per_loop`. At
  262,144 tokens it fits ~10 streams instead of ~6, and long-context decode reads 36 % less.
- Short-context speed depends on `LOOPS` only; `KV_MODE` changes memory and long-context speed.

```bash
LOOPS=4 ./start.sh restart                     # most accurate
LOOPS=2 KV_MODE=last_loop ./start.sh restart   # default quality, less KV memory
echo "LOOPS=1" >> .env && ./start.sh restart   # keep a setting across restarts
```

Re-run the benchmarks against a running server (`bench/`, copied from the former `27B-2/bench`, now `27B-3/bench`):

```bash
cd bench
python3 same.py 1 2 4          # same prompt for every client, greedy, 200-token replies
python3 tput.py 1 2 4          # distinct prompts, T=1, 800 tokens each
python3 bench.py prefill 4000 16000 64000
```


## The model

Dory is a *looped* normalized GPT (nGPT): 72 physical sublayers, each attention **or** SwiGLU MLP, with a slerp residual
on the unit sphere (`h = justnorm(justnorm(h) + |alpha| (justnorm(f(h)) - justnorm(h)))`). They form three groups:
input (16), recurrent (40) and output (16). The recurrent group runs `n_recurrent_loops` times (2 by default; 1 to 4
work), its input re-injected every loop (`h = justnorm(state + input)`), so 2 loops run 112 sublayers per token from 4B
parameters. Attention is GQA 16/4 with head dimension 256, nGPT q/k normalisation after RoPE (`q *= sqk^2`), a sigmoid
output gate and a normalised output; 3 of every 4 attention layers are sliding-window (1024 tokens, RoPE theta 1e4), the
rest full (theta 1e7). Linear layers are NVFP4 (e2m1, e4m3 block scale per 16, fp32 tensor scale; W4A16), embeddings and
head bf16, logits scaled per token.

`KV_MODE=per_loop` (default, the reference mode, NVIDIA's vLLM command) keeps a cache per loop for each recurrent
attention layer: 56 caches, 14 of them full-attention, 56 KiB per token. `last_loop` keeps one per layer, overwritten by
every loop (36 caches, 9 full, 36 KiB per token); NVIDIA reports accuracy close to `per_loop`.

## How it is built

TensorFold has no Dory family, so this recipe adds one, the same way the Nemotron 3.5 Bolt recipe (`../3.5-bolt`) does:

| piece | what |
|---|---|
| `engine/dory_host.cu` | `libdory.so`: loads the checkpoint (NVFP4 tile-repacked for tensor cores), the block forward with the loops, KV caches, CUDA graphs, calibration, prompt-cache snapshots, the C ABI |
| `engine/dory_mma.cu` | NVFP4 and bf16 tensor-core GEMMs (from libbolt), one launch over q/k/v/g or gate/up; decode rows sum K in a fixed order (row invariance) |
| `engine/dory_attn.cu` | new flash attention for head dim 256: a warp's m16 tile is 4 rows x the 4 query heads of one KV head, ldmatrix K/V tiles, sliding window, keys split at absolute chunk boundaries that depend only on the stream's own length |
| `engine/dory_kernels.cu` | RoPE + q/k justnorm + KV store, gated output norm, scaled SwiGLU + norm, slerp, loop input, logit scale, top-64 candidates |
| `engine/dory_native.zig`, `dory_api.zig` | the TensorFold lane backend (prompts, shared verify windows, rollback, keyed sampling, structured-output masks, prompt cache) |
| `patches/0001-register-dory.patch` | registers `model_type: dory` in the CUDA registry and adds `zig build dory` |
| `scripts/prepare.sh` | builds `tensorfold-dory:zig-46a08c1` (TensorFold at `TF_REF` + patches + engine, Zig 0.17, on `nvcr.io/nvidia/pytorch:26.07-py3`), downloads the checkpoint |

Memory: weights 3.0 GB on the GPU. Each stream holds 0.33 GiB of sliding-window rings (2,048 rows per sliding layer)
from the start; full-attention rows are mapped 1,024 tokens at a time as a stream grows and returned when it ends.

Speculation: Dory ships no drafter. The backend reports exact verify windows of 8 rows, so TensorFold's round loop
proposes **copies** (suffix matches from the prompt and reply) and verifies them in shared windows. It costs nothing when
nothing matches and pays off on edits and repeated code (below). Rolling back is free: there is no recurrent state.

## Correctness

- `tests/ref.py` is an fp32 PyTorch reference of the forward (after vLLM's `dory.py`, PR 60149). `tests/compare.py`
  checks libdory against it: a 100-token prompt block (max |logit diff| 0.34 on logits up to 31, top-1 99 %), prompt +
  one-row decode steps + an 8-row verify window (top-1 100 %), a rolled-back window redone (100 %).
  `tests/compare_long.py`: 5,000 tokens of code, chunked prefill with the sliding rings wrapping, then decode: top-1 100 %.
- Row invariance: a window's candidates are bit-identical alone and among other streams (`compare.py`), and in the server
  all 31 greedy replies of `bench/same.py` (1 to 16 concurrent streams) hash identically per prompt.
- Served: chat with thinking (reasoning in `reasoning_content`), without, tool calls (`get_weather` parsed into
  `tool_calls`), prompt-cache hits, a 249,375-token prompt.

## Results (GB10, LOOPS=2, per_loop, PARALLEL=16, 262,144-token window)

Decode, `bench/same.py` (the repo's README method: the same prompt for every client, greedy, thinking off, 200-token
replies, warm):

| clients | prose aggregate | code aggregate |
|---:|---:|---:|
| 1 | 52 tok/s | 52 tok/s |
| 2 | 96 | 96 |
| 4 | 179 | 182 |
| 8 | 330 | 334 |
| 16 | 511 | 515 |

`bench/tput.py` (distinct essay prompts, T=1, 800 tokens each): 1 client 49.5 tok/s, 4: 169, 8: 304, 16: 458 aggregate
(29.5 per stream).

Copy speculation, one client, greedy, "rewrite this 1,280-token code block renaming one variable": 1,237 tokens at
**242 tok/s** including prefill; 1,019 of 1,099 copied tokens accepted (5.6 tokens a round).

Prefill (`bench/bench.py prefill`, cold, distinct prompts; server `prompt_tokens`): 6,455 tok 3,455 tok/s; 25,831 tok
3,037 tok/s; 103,053 tok 1,943 tok/s; 249,375 tok 1,150 tok/s (216 s). Decode after a 249K-token prompt: 11.2 tok/s
(each token reads 14 full-attention caches of 249K rows in bf16, ~14 GB).

Loop count and KV mode (same.py, prose; library 1-row forward ms):

| setting | 1-row forward | 1 client | 8 clients | 16 clients |
|---|---:|---:|---:|---:|
| LOOPS=1 | 13.6 ms | 72 tok/s | 466 | 731 |
| LOOPS=2 (default) | 19.4 ms | 52 | 330 | 511 |
| LOOPS=2 KV_MODE=last_loop | 19.3 ms | 51 | 329 | 512 |
| LOOPS=4 | 30.8 ms | 32 | 204 | 316 |

`last_loop` does not change short-context speed; it cuts KV memory and long-context attention by 36 %.

Concurrency at full context: a stream at 262,144 tokens holds ~14 GiB (`per_loop`, 2 loops) or ~9 GiB (`last_loop`)
plus its 0.33 GiB of rings; with ~95 GiB left after the 10 GiB reserve that is about 6 (per_loop) or 10 (last_loop)
streams at the full window at once. Streams that would grow past free memory wait (`LaneMemory`) instead of failing.

Where a 1-row decode forward goes (`DORY_PROF=1 tests/prof_decode.py`, eager): NVFP4 GEMMs 63 % (at ~205 GB/s), bf16
head 14 % (671 MB a token), attention 8 %, elementwise kernels the rest.

## Limits and notes

- Text only. libdory serves Dory1.0-4B's shape and the NVFP4 export (one `model.safetensors`); the BF16/MXFP8 exports
  would need other weight loaders.
- KV cache is bf16 (no fp8 cache yet): long-context decode is bound by reading it.
- The startup line reports "model 0.00 GiB": libdory allocates outside TensorFold's accounting (as libbolt does); the
  admission still uses `dory_seq_bytes` per stream.
- `DORY_PROF=1` (stage timings), `DORY_NO_GRAPHS=1`, `DORY_STATS=1` (round loop timings in the log), `DORY_KV_EAGER=1`
  (no lazy mapping), `DORY_KV_PIN=1` (map every window at load) are debugging switches; any `DORY_*` variable is passed in.

## Tests (inside the pytorch container, e.g. `docker run --gpus all -v ~/.cache/huggingface:/root/.cache/huggingface -v $PWD:/work/dory-4b -v ~/.cache/dory-dev:/cache nvcr.io/nvidia/pytorch:26.07-py3`)

```bash
tests/build.sh                                   # /cache/build/libdory.so
python tests/compare.py MODEL_SNAPSHOT_DIR       # logits vs the fp32 reference, rollback, row invariance
python tests/compare_long.py MODEL_SNAPSHOT_DIR 5000
python tests/bench_lib.py MODEL_SNAPSHOT_DIR     # calibration, prefill, decode without the server
DORY_PROF=1 python tests/prof_decode.py MODEL_SNAPSHOT_DIR
```
