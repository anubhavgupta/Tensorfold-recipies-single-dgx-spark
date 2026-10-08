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

## Round 2: content classes, prefill, full 262K concurrency
Model: `turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, MTP on, greedy, 256 generated tokens, chat prompts. Aggregate decode tok/s.

| Class | N | TensorFold | Tabby (stock) | Tabby (patched) |
|---|---|---|---|---|
| code | 1 / 4 / 9 | 93 / 177 / 254 | 79 / 49 / 64 | 76 / 131 / 133 |
| prose | 1 / 4 / 9 | 54 / 106 / 158 | 50 / 36 / 44 | 50 / 91 / 117 |
| devops | 1 / 4 / 9 | 98 / 172 / 254 | 80 / 47 / 64 | not rerun |
| json | 1 / 4 / 9 | 97 / 188 / 274 | 88 / 55 / 74 | not rerun |

Prefill (one stream, cold): TensorFold 859 / 846 / 790 / 748 tok/s at 4K / 33K / 131K / 249K; Tabby ~534 at 4K, 821 / 800 / 781 at 33K / 131K / 249K.

Full capacity, 9 streams x ~249K prompt + 256 generated, all engines: 9/9 succeeded.
- Tabby: wall 2818 s, prefill 798 tok/s aggregate, steady decode ~55 tok/s aggregate (prefill overlaps decode).
- TensorFold: wall 3034 s, prefill 740 tok/s aggregate (prefills are serialised), per-stream decode 8-49 tok/s depending on when each stream started.

### Why concurrency did not scale
- TensorFold: it does scale. The earlier flat result came from `concurrency.py` (identical repetitive prompts, prefill inside the timing window). Use `bench/suite.py`.
- Tabby: the fused MoE decode kernel handles at most `MAX_BSZN=8` rows per forward. With MTP, rows = streams x (1+drafts), so 2+ streams fell onto the slow path. Patch `flash-3/patches/exllamav3-fused-decode-rows.patch` raises `MAX_BSZN` to 25 (the kernel limit is 256 slots / 10 experts per token) and shortens the draft window so rows stay <= 25 (`EXL3_FUSED_DECODE_ROWS`, 0 disables). Apply it in `flash-3/runtime/exllamav3` and rebuild (`pip install --no-build-isolation -e .`, ~45 min without ninja). It is already applied and built in the local runtime.
- Remaining gap: TensorFold still wins at N>=4 because Tabby's verify path above 8 rows is less efficient.
