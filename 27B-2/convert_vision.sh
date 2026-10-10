#!/usr/bin/env bash
# One-time: the 27B EXL3 pack keeps its vision tower EXL3-quantized inside the model shards (model.visual.*), which
# TensorFold cannot load directly. This pulls those tensors into a sidecar and converts it to a floating tower
# (tensorfold.vision.exl3_convert) at $HF_CACHE/vision-f16-27b.safetensors, which start.sh then picks up.
# Usage: ./27B-2/convert_vision.sh        Env: HF_CACHE, TF_VERSION (0.6.6), REPO, REVISION
set -euo pipefail
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
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
