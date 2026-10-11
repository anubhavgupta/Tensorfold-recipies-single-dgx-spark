#!/usr/bin/env bash
# One-time: the 27B EXL3 pack keeps its vision tower EXL3-quantized inside the model shards (model.visual.*), which
# TensorFold cannot load directly. This pulls those tensors into a sidecar and converts it to a floating tower
# (tensorfold.vision.exl3_convert) at $HF_CACHE/vision-f16-27b.safetensors, which start.sh then picks up.
# Usage: ./27B-2/convert_vision.sh [bf16|exl3]        Env: HF_CACHE, TF_VERSION (0.6.6), REPO, REVISION
#   bf16 (default): the same model's full-precision tower, copied unchanged from Vontra/Qwen3.8-27B-MLX-4bit
#     (downloaded if missing, ~15 GB; only its 333 vision_tower.* tensors are kept) -> vision-bf16-27b.safetensors
#   exl3: the pack's own 6-bit EXL3 tower converted to fp16 -> vision-f16-27b.safetensors
set -euo pipefail
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
if [[ "${1:-bf16}" == bf16 ]]; then
  out="$HF_CACHE/vision-bf16-27b.safetensors"
  [[ ! -f "$out" ]] || { echo "$out exists"; exit 0; }
  vdir="$HF_CACHE/hub/models--Vontra--Qwen3.8-27B-MLX-4bit"
  [[ -d "$vdir/snapshots" ]] || hf download Vontra/Qwen3.8-27B-MLX-4bit
  docker run --rm -i -u "$(id -u):$(id -g)" --entrypoint python3 -v "$HF_CACHE:/hf" "tensorfold:v${TF_VERSION:-0.6.6}" - <<'EOF'
import glob
from safetensors import safe_open
from safetensors.torch import save_file
src = glob.glob("/hf/hub/models--Vontra--Qwen3.8-27B-MLX-4bit/snapshots/*/")[0]
out = {}
for shard in sorted(glob.glob(src + "*.safetensors")):
    with safe_open(shard, "pt") as f:
        for k in f.keys():
            if k.startswith("vision_tower."):
                out[k] = f.get_tensor(k).contiguous()
assert len(out) == 333, len(out)
save_file(out, "/hf/vision-bf16-27b.safetensors",
          metadata={"source": "Vontra/Qwen3.8-27B-MLX-4bit vision_tower (bf16, unchanged)"})
print(len(out), "bf16 vision tensors")
EOF
  ls -la "$out"; exit 0
fi
REPO="${REPO:-turboderp/Qwen3.8-27B-exl3}"
REVISION="${REVISION:-SC_4.00bpw_H5_V6}"
dir="$HF_CACHE/hub/models--${REPO//\//--}"
snap="$(<"$dir/refs/$REVISION")"
out="$HF_CACHE/vision-f16-27b.safetensors"
[[ ! -f "$out" ]] || { echo "$out exists"; exit 0; }
work="$(mktemp -d "$HF_CACHE/.vision-XXXXXX")"
trap 'rm -rf "$work"' EXIT
cp -L "$dir/snapshots/$snap/config.json" "$work/config.json"
docker run --rm -i --gpus all --entrypoint python3 -v "$HF_CACHE:/root/.cache/huggingface" "tensorfold:v${TF_VERSION:-0.6.6}" - \
  "/root/.cache/huggingface/hub/models--${REPO//\//--}/snapshots/$snap" "/root/.cache/huggingface/$(basename "$work")" <<'EOF'
import sys, glob
from safetensors import safe_open
from safetensors.numpy import save_file
from tensorfold.vision import exl3_convert
src, work = sys.argv[1:3]
tensors = {}
for shard in sorted(glob.glob(src + "/*.safetensors")):
    with safe_open(shard, framework="np") as f:
        for k in f.keys():
            if k.startswith("model.visual."):
                tensors[k] = f.get_tensor(k)
print(len(tensors), "vision tensors")
save_file(tensors, work + "/vision_k6.safetensors")
print(exl3_convert.convert(__import__("pathlib").Path(work + "/vision_k6.safetensors"),
                           __import__("pathlib").Path("/root/.cache/huggingface/vision-f16-27b.safetensors")))
EOF
ls -la "$out"
