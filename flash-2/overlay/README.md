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
