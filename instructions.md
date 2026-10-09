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

## Round 3: TENSORFOLD_PREFILL_ROWS sweep (flash-2, TensorFold 0.6.5, EXL3 4.05bpw_h6_ng6)
Cold 32K-token prompt, one stream. Larger rows barely help and cost KV memory, so the default (2048) stays.

| Rows | Prefill tok/s | Free for KV caches | Full-window (262K) streams |
|---|---|---|---|
| 2048 (default) | 792 | 41.8 GiB | 9 |
| 4096 | 818 (+3%) | 39.6 GiB | 8 (+ n-gram pages spill to disk warning) |
| 8192 | 828 (+4%) | 39.2 GiB | 8 (same warning) |

## Round 4: faster EXL3 prompt processing (flash-2 overlay)

Model: `turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, TensorFold 0.6.5. Start/stop: `./flash-2/start.sh` / `./flash-2/stop.sh`.
The overlay is on by default; `OVERLAY=0 ./flash-2/start.sh` runs stock. Details and the diff are in `flash-2/overlay/`.

Profile (torch.profiler): about 48% of prefill is the grouped expert GEMM (`grouped_kernel`), about 10% is `group_kernel`.
Two fixes, with greedy output identical to stock:
- the expert kernel decodes each weight tile once for up to 5 row tiles (it was once per 16-row tile);
- `group_kernel` was rewritten with a shared-memory histogram and ballot ordering (1.4 s -> 0.15 s per 8K prompt).

| Cold prompt | Stock tok/s | Overlay tok/s | Gain |
|---|---|---|---|
| 4K | 859 | 1026 | +19% |
| 32K | 846 | 1032 | +22% |
| 131K | 790 | 934 | +18% |
| 249K | 748 | 880 | +18% |

`TENSORFOLD_MTL` sweep, 8K prompt: stock 842, MTL 4 1038, MTL 5 1064 (default), MTL 6 984, MTL 8 956.
Decode is unchanged (code N=1 66.4, N=4 176 steady, N=9 249 steady tok/s; prose N=9 151 steady). Startup still reports
9 full-window streams (42.4 GiB for KV, 4.47 GiB per full window). A full 9 x 250K run was not repeated with the overlay.
Further gains need a different expert design (for example dequantising to fp16 tiles).

## flash-4 (MiaAI-Lab Zig recipe, INT4-AutoRound checkpoint)

Set up in `flash-4/` (`./flash-4/start.sh`, `./flash-4/stop.sh`, port 8888). The smoke test passed (8 streams, 262K, MTP). Benchmarked with `flash-3/bench/suite.py` (checkpoint `azampatti/Qwen3.8-Flash-Next-125B-A5B-INT4-AutoRound`, a different quant, 8 streams max):

| Test | Result |
|---|---|
| Prefill 4K / 32K (first requests, likely warm-up) | 350 / 600 tok/s |
| Prefill 131K / 249K | 1336 / 1286 tok/s |
| Code N=1 / 4 / 8 (steady aggregate) | 90 / 204 / 304 tok/s |
| Prose N=1 / 4 / 8 (steady aggregate) | 53 / 125 / 186 tok/s |

Full-window concurrency was not run.
