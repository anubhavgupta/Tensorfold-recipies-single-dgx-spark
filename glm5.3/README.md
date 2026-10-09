# GLM-5.3-Flash EXL3 2.05bpw on one DGX Spark

`turboderp/GLM-5.3-Flash-exl3` @ `2.05bpw` (80 GiB). TensorFold's GLM engine needs `--tp 2` (two machines), so this runs on
the exllamav3 fork + TabbyAPI runtime from `../flash-3`.

```
../flash-3/run.sh setup                                   # once
hf download turboderp/GLM-5.3-Flash-exl3 --revision 2.05bpw
./serve.sh        # port 8890, MTP drafting, glm4_5 tool calls, reasoning split
./stop.sh
./tune.sh <tag> VAR=val ...                               # restart + benchmark
```

Defaults: chunk 4096, MTP x2, KV 8/8 bit, 128K context, batch 1. Knobs are listed in `serve.sh`.

## Results (GB10)
| | |
|---|---|
| Prefill | ~450 tok/s at 8K, ~500 at 32K (chunk 2048: ~360) |
| Decode | ~24 tok/s without drafting, ~32-33 with MTP x2 |

Chunk 8192 fails to load (not enough memory); 4096/6144 work. MTP depth 1-5 and confidence 0.4-0.8 are within noise.
The `EXL3_MOE_*` prefill knobs had no effect. A torch profile of a 4K chunk shows no single hotspot (Hadamard weight
reconstruction 16%, dtype copies 10%, DtoH syncs 7%), so further gains need kernel work.

Gotcha: after a download or earlier load the page cache counts against free memory and the load fails with
"Insufficient VRAM". `serve.sh` syncs and drops the pack's cache first.
