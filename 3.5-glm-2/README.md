# GLM-5.3-Flash EXL3 2.05bpw // TensorFold 1.0.12 native // one DGX Spark

Serve **GLM-5.3-Flash** from turboderp's EXL3 pack
([`turboderp/GLM-5.3-Flash-exl3`](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3), branch `2.05bpw`, 80 GiB) on
**one** DGX Spark (GB10, 128 GB) with [TensorFold](https://github.com/ashhart/TensorFold)'s native Zig engine at the
latest release, 1.0.12 (`46a08c1`). OpenAI-compatible API on port 8890, model id `GLM-5.3-Flash`: thinking
(`reasoning_content`), tool calls, structured output, MTP drafts, 4 concurrent streams with a 131,072-token window each,
and TensorFold's prompt cache.

```bash
./start.sh                 # builds the image the first time (~5 min), downloads the pack if missing, loads it (~4 min)
./start.sh restart         # after changing settings
./stop.sh                  # (end.sh is the same script)
CONTEXT=65536 PARALLEL=8 ./start.sh restart
```

Settings (environment or `./.env`, defaults in `scripts/config.sh`): `CONTEXT` (131072), `PARALLEL` (4), `DRAFTS` (1: the
checkpoint's MTP head), `PROMPT_CACHE_GIB` (0: sized by the server from what is left), `THINKING` (1),
`TEMPERATURE`/`TOP_P` (1.0/0.95), `MAX_TOKENS`, `PORT` (8890), `SERVED_NAME`. Extra arguments go to
`tensorfold-native serve`; `TENSORFOLD_*` and `TF_GLM_*` variables are passed into the container. Logs:
`~/.cache/tf-glm53-flash/logs/latest.log`. Only one big server fits: stop others first (the start warns below 100 GiB free).

## Quick start

Requirements: a DGX Spark (GB10, 128 GB unified memory), Docker with the NVIDIA Container Toolkit, an NGC login for the
base image (`docker login nvcr.io`, once), internet on the first run, and the `hf` CLI on the host (else the download
runs in the container). About 100 GiB of free memory when it starts (the start script warns below that).

```bash
git clone https://github.com/anubhavgupta/Tensorfold-recipies-single-dgx-spark
cd Tensorfold-recipies-single-dgx-spark/3.5-glm-2
./start.sh                       # first run: builds the image, downloads the 2.05bpw pack (80 GiB), loads it in ~4 min
docker logs -f tf-glm53-flash       # follow the server (or the log under ~/.cache/tf-glm53-flash/logs)
```

Send a request (OpenAI API at `http://localhost:8890/v1`, model `GLM-5.3-Flash`):

```bash
# thinking on (the default): the reasoning comes back in message.reasoning_content
curl -s localhost:8890/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"What is 17*23?"}],"max_tokens":1000}'

# thinking off for this request
curl -s localhost:8890/v1/chat/completions -H 'content-type: application/json' \
  -d '{"model":"GLM-5.3-Flash","messages":[{"role":"user","content":"Write a haiku about GPUs."}],"max_tokens":200,"chat_template_kwargs":{"enable_thinking":false}}'

# tool calls: pass "tools" as with any OpenAI-compatible server; calls come back in message.tool_calls
```

Change settings with environment variables or a `.env` file in this folder (see below), then `./start.sh restart`.
Stop with `./stop.sh`. Only one large server fits in memory: stop other recipes first.

Re-run the benchmarks against a running server (`bench/`, copied from `27B-2/bench`):

```bash
cd bench
python3 same.py 1 2 4          # same prompt for every client, greedy, 200-token replies
python3 tput.py 1 2 4          # distinct prompts, T=1, 800 tokens each
python3 bench.py prefill 4000 16000 64000
```

The benchmark scripts here point at port 8890.


## Why patches were needed

TensorFold 1.0.12 serves GLM-5.3-Flash on CUDA only from NVIDIA's NVFP4 checkpoint, at tensor parallel over several
hosts: that checkpoint is 190 GiB (NVFP4 experts, BF16 attention), more than one Spark holds, and the engine had no EXL3
path on CUDA. The EXL3 2.05 bpw pack fits (2-bit routed experts, 3-5 bits elsewhere), so the recipe adds EXL3 to the
native CUDA engine and teaches the GLM family to load it on one rank:

| patch | what |
|---|---|
| `0001-exl3-cuda-kernels.patch` | upstream PR [#837](https://github.com/ashhart/TensorFold/pull/837) rebased onto 1.0.12: EXL3 linear and grouped routed-expert kernels for every codebook and width, `core/exl3_load`, the uploader, `qlinear`'s EXL3 arm, `tf-cuda-test exl3` |
| `0002-glm-exl3-one-gpu.patch` | the GLM family on an EXL3 pack at `TF_GLM_WORLD=1` (below), and a faster prompt path for the EXL3 experts |

`0002` in detail:
- **Loader** (`cuda_weights.zig`): KDA `[q|k|v]` and `o`, DSA `q_a`, `kv_a`, `q_b`, `o` and the indexer's `wq_b`, the three
  dense MLPs, all 288 routed experts plus the shared expert (stacked as expert 288, which the router's shared slot already
  picks with weight 1), MTP's `eh_proj` and the head as EXL3 groups; the small fp16 matrices (`f_a`, `g_a`, `b`, `f_b`,
  `g_b`, `kv_b`, router, indexer `wk`/`weights_proj`/gate/norm) as bf16; the fused `conv1d`. Pages are released per layer.
- **Projections** (`cuda_mat.zig`): a projection is bf16 or EXL3, one `matmul` for both. EXL3 matrices cannot be fused
  (each has its own input Hadamard signs), so KDA's `[q|k|v]` and `[f_a|g_a|b]` rows are computed apart and copied into
  place, and DSA's `q_a`/`kv_a` rows are normed at their own stride.
- **MoE**: the EXL3 grouped experts compute every slot and the weighted sum in one call (2,048 rows a call).
- **MTP** drafts through the EXL3 head (there is no BF16 head to derive the 4-bit draft head from).
- One rank skips the multi-rank piece exchanges; EXL3 is refused at more than one rank; `qlinear` gains fp32 EXL3 outputs.
- **Prompt experts kernel**: at 2,048 rows the grouped GEMM launched ~2.4 M programs a projection, ~94 % of them empty
  (one per expert x member tile x column block x split, while an expert has only ~64 members), and each program decoded
  its weights for 16 rows. Prompt rows (256+) now run over a device work list of the member groups that exist, with
  programs of 64 columns x 64 members that decode each weight tile once for four 16-row tiles. Each row's arithmetic and
  summation order are unchanged (same reply hashes; `tf-cuda-test exl3` passes), and decode rows keep the old shape.
  Prefill on varied text: 354 -> 453 tok/s.

The image `tensorfold-glm-exl3:zig-46a08c1` is built by `scripts/prepare.sh` from TensorFold at `TF_REF` plus the
patches, with Zig 0.17.0 on `nvcr.io/nvidia/pytorch:26.07-py3` (the server, `tf-cuda-test`, and libtfgrammar for
structured output). The server runs with `TF_GLM_RANK=0 TF_GLM_WORLD=1`.

## Results (GB10, defaults: 4 streams, 131,072-token window, MTP drafts on)

Memory: 86.4 GiB for the model and the streams' buffers, 2.65 GiB a stream at the full window, a 6.8 GiB prompt cache.
Load: ~225 s.

Decode, `bench/same.py` (the same prompt for every client, greedy, thinking off, 200-token replies):

| clients | prose aggregate | code aggregate |
|---:|---:|---:|
| 1 | 30.9 tok/s | 32.4 tok/s |
| 2 | 32.8 | 37.0 |
| 4 | 39.6 | 44.7 |

`bench/tput.py` (distinct essay prompts, T=1, 800 tokens each): 1 client 28.7 tok/s, 2: 25.3, 4: 30.3 aggregate. With
`TF_GLM_SHARED=0` (each stream's window alone) it is flat, 29.0 / 28.7 / 28.5, and greedy same-prompt drops to ~30 at 4
clients, so shared windows stay the default. Without drafts (`DRAFTS=0`) one stream decodes at 24-27 tok/s and streams do
not share rounds (the family shares windows only with its MTP head).

Exactness: a drafted reply equals the same request with `"draft": false`, and concurrent replies equal solo ones (all 7
of 7 per prompt in `same.py`), at greedy.

Prefill (`bench/bench.py prefill`, cold, distinct varied prompts; server `prompt_tokens`): 4,945 tok 433 tok/s; 19,687 tok
453 tok/s; 78,471 tok 456 tok/s. Repetitive text routes to fewer experts and runs a little faster.

For reference, the previous setup on this Spark (`../glm5.3`, exllamav3 + TabbyAPI, same pack): ~24 tok/s decode
without drafting, 32-33 with MTP, prefill ~450-500 tok/s, one request at a time.

Where a 2,048-row prompt chunk goes (before the prompt-experts change; timed with stream syncs): routed experts 56 %, KDA
blocks 26 % (their EXL3 `[q|k|v]`/`o` projections about half), DSA blocks 17 %, dense MLPs 2 %. The EXL3 dense prompt path
still decodes its weights per 128-row block; a dequantize-then-GEMM prompt path is the next prefill step.

## Limits

- Concurrent distinct sampled streams scale poorly: the GLM family gives TensorFold's round loop no window costs, so shared
  rounds draft deeply, and on this MoE each extra row routes to more experts. Measured costs in the family's facts are the
  fix.
- Text only (the pack's vision tower is not loaded on CUDA).
- One rank only for EXL3 (`TF_GLM_WORLD=1`).
- The prompt cache keeps states on the family's 2,048-row grid (upstream behaviour).

## Checks

```bash
# inside the image: the EXL3 kernels' float64, row- and chunk-invariance tests
docker run --rm --gpus all --entrypoint /opt/tensorfold/bin/tf-cuda-test tensorfold-glm-exl3:zig-46a08c1 exl3
```
