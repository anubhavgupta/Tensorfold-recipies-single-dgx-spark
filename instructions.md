# Qwen3.8-Flash-Next EXL3 on one DGX Spark: TensorFold vs TabbyAPI

Model for every result below: `turboderp/Qwen3.8-Flash-Next-exl3`, revision `4.05bpw_h6_ng6`
(EXL3 4.05 bits per weight, about 101 GB, MTP drafting), downloaded with
`hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6`
(snapshot in `~/.cache/huggingface/hub/models--turboderp--Qwen3.8-Flash-Next-exl3/snapshots/55a732e0c4c3d4614bc42b68493bb930d9b02c0a`).
Only one server can run at a time (each takes nearly all of the 128 GB unified memory); stop one before starting the other.

## Start / stop

### TensorFold v0.6.5 (`flash-2/`, port 8888, model id `Qwen3.8-Flash-Next`)

```bash
cd ~/projects/tensorfold
./flash-2/start.sh                 # background container tf-flash-2-exl3; defaults: PARALLEL=9, CONTEXT=262144, int8 KV, vision on
PARALLEL=4 KV_DTYPE=bf16 ./flash-2/start.sh   # example override
FOREGROUND=1 ./flash-2/start.sh    # attached, logs on the terminal (a detached container is removed on exit, so its logs are lost)
docker logs -f tf-flash-2-exl3     # startup lines: "startup estimate", "up to N streams", "free for their caches"
curl -s localhost:8888/health
./flash-2/end.sh                   # stop and remove the container
```

- Needs the one-time vision conversion (the pack keeps its vision tower in the unindexed `vision_k6.safetensors`);
  the converted file is `~/.cache/huggingface/vision-f16-4.05.safetensors` and `start.sh` picks it up. To redo it:
  ```bash
  SNAP=/root/.cache/huggingface/hub/models--turboderp--Qwen3.8-Flash-Next-exl3/snapshots/55a732e0c4c3d4614bc42b68493bb930d9b02c0a
  docker run --rm --gpus all --entrypoint python3 -v $HOME/.cache/huggingface:/root/.cache/huggingface tensorfold:v0.6.5 \
    -m tensorfold.vision.exl3_convert $SNAP/vision_k6.safetensors /root/.cache/huggingface/vision-f16-4.05.safetensors
  ```
- EXL3 limits: no `--ple-on-ssd`, no `--tp 2`, and `--vision` needs `--parallel 2` or more.

### TabbyAPI + vcruz305/exllamav3 fork (`flash-3/`, port 8899, model id `Qwen3.8-Flash-Next-EXL3`)

```bash
cd ~/projects/tensorfold/flash-3
./run.sh check                                     # verifies fork runtime and TabbyAPI
PROFILE=single ./run.sh serve                      # 1 stream, 262144 context, n-gram table in RAM (fastest single stream)
PROFILE=concurrent MAX_BATCH_SIZE=9 CACHE_SIZE=$((9*262144)) ./run.sh serve   # 9 streams at full 262144 each
curl -s localhost:8899/v1/models
./run.sh stop                                      # then wait until `free -g` shows ~115 GiB available before the next start
```

- Everything lives in `flash-3/runtime` (venv, fork build, TabbyAPI). `serve` runs detached; log it with
  `setsid nohup ./run.sh serve > /tmp/tabby_serve.log 2>&1 < /dev/null &`.
- `CACHE_SIZE` is one KV pool shared by all streams (8-bit KV, ~13 KB per token), so full context for N streams needs `N*262144`.
- TabbyAPI is pinned to commit `7a524a9`: current `main` needs exllamav3 1.5.4 but the fork is 1.5.1. Do not re-run `setup.sh`
  (it moves TabbyAPI back to `main`). `Python.h` comes from `flash-3/runtime/pyinc` (no root on this host); `run.sh` adds it to `CPATH`.

## Benchmarks

Clients: `flash-3/bench/suite.py` (content classes, prefill, full-capacity) and the recipe's `bench_v1.py` / `concurrency.py`.
```bash
cd ~/projects/tensorfold/flash-3
BASE_URL=http://127.0.0.1:8888/v1 python3 bench/suite.py content tf 1,2,4,9      # TensorFold
BASE_URL=http://127.0.0.1:8899/v1 python3 bench/suite.py content tabby 1,2,4,9   # TabbyAPI
python3 bench/suite.py prefill <tag> 4096,32768,131072,250000
python3 bench/suite.py full <tag> <N> 250000 256
```
Raw JSON is in `flash-3/results/`.

## Results, first round (3,000-token prompt, 400 generated tokens, greedy, MTP on)

| Test | TensorFold | TabbyAPI |
|---|---|---|
| Single stream, code prompt (`bench_v1.py`), tok/s | **88.8** | 76-82 (median 75.8) |
| 1 stream, repetitive filler prompt, tok/s | 75.5 | 61-67 |
| 2 streams, aggregate tok/s | **~94** | ~41 |
| 4 streams, aggregate | **~97** | ~51 |
| 9 streams, aggregate | **~102** | ~46 |
| Streams that load at 262K each | 9 (42.7 GiB free for caches, 4.47 GiB per stream) | 9 (pool 9 x 262144; ~1 GiB free) |

Later rounds (per content class, prefill rate, full-context concurrency) are appended below.
