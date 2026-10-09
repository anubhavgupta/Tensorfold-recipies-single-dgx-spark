# EXL3 prompt-speed overlay for TensorFold 0.6.5

Three files from `tensorfold/cuda/exl3/` of the 0.6.5 image, mounted read-only over the originals by `start.sh`
(`OVERLAY=0` turns it off). The output is bit-identical to stock (greedy text compared; every row's sum keeps the same
order). Measured on `turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6`, cold 8K prompt: 842 -> 1064 tok/s.

1. `experts_grouped.cuh`: the grouped expert kernel decoded each expert's weights once per 16-row tile, about five times
   per 2048-row chunk. With 64+ rows in the window one program now takes up to 5 row tiles (`TENSORFOLD_MTL`: 4, 5, 6 or 8)
   and decodes each weight tile once for all of them. Decode (few rows) uses the old kernel.
2. `experts.cu`: `group_kernel` (routing groups) was one block scanning every pick per expert; it now uses a shared-memory
   histogram and ballot-ordered member lists (1.4 s -> 0.15 s in the 8K profile).
3. `experts.py`: extension name `_v2`, so the JIT cache rebuilds.

The first start compiles the kernels (a few extra minutes). `patch.diff` is the change against stock.

## Prefix cache for image requests (`overlay/cache`)

Stock 0.6.5 never keeps or reuses a prompt prefix when the request has an image, so every turn of an image chat
re-prefills the whole history. `multi.py`, `multi_fill.py` and `prefixes.py` (mounted over the image's copies;
`VISION_CACHE=0` turns it off, `cache.diff` is the change) now:

- key kept prefixes by the prompt with each image's placeholder run replaced by a token derived from that image's
  content hash, so the same image matches and a different one cannot;
- drop the "image stream" guards on kept prompt points and reuse.

On a hit the images are still encoded (the full rotary positions are needed), but only the uncached rows are prefilled.
Tested on `Qwen3.8-Flash-Next` with a 1.7K-token image prompt: first 0 cached, repeat 1767, next turn 1767, a different
image after the same system text 1551 (the text before the image only); greedy output identical cached and uncached.
