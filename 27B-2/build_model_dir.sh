#!/usr/bin/env bash
# One-time: build the model folder 27B-2 serves (and that can be published as one Hugging Face repo):
#   the EXL3 pack (turboderp/Qwen3.8-27B-exl3 @ SC_4.00bpw_H5_V6) copied from the HF cache, plus
#   vision_bf16.safetensors: the model's official full-precision vision tower (the 333 model.visual.* bf16 tensors of
#   Qwen/Qwen3.8-27B, unchanged; only the one shard holding them is downloaded, ~4 GB). The pack's own vision tower is EXL3-quantized (6 bits) inside the
#   shards, which TensorFold cannot load; start.sh points TENSORFOLD_VISION_WEIGHTS at vision_bf16.safetensors.
# Usage: ./27B-2/build_model_dir.sh     Env: MODEL_PATH (/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision),
#   HF_CACHE (~/.cache/huggingface), REPO, REVISION, TF_VERSION (0.6.6). Downloads what is missing (~16 + ~4 GB).
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO="${REPO:-turboderp/Qwen3.8-27B-exl3}"
REVISION="${REVISION:-SC_4.00bpw_H5_V6}"
mkdir -p "$MODEL_PATH"

if [[ ! -f "$MODEL_PATH/model.safetensors.index.json" ]]; then
  dir="$HF_CACHE/hub/models--${REPO//\//--}"
  [[ -f "$dir/refs/$REVISION" ]] || hf download "$REPO" --revision "$REVISION"
  snap="$dir/snapshots/$(<"$dir/refs/$REVISION")"
  for f in "$snap"/* "$snap"/.gitattributes; do
    [[ "$(basename "$f")" == *-back || "$(basename "$f")" == *.orig ]] && continue
    cp -L "$f" "$MODEL_PATH/"
  done
  echo "copied $REPO @ $REVISION"
fi

if [[ ! -f "$MODEL_PATH/vision_bf16.safetensors" ]]; then
  # the official tower: Qwen/Qwen3.8-27B keeps all 333 model.visual.* tensors (bf16) in its first shard
  qdir="$(mktemp -d "$HF_CACHE/.qwen-vision-XXXXXX")"
  trap 'rm -rf "$qdir"' EXIT
  hf download Qwen/Qwen3.8-27B model.safetensors.index.json --local-dir "$qdir" >/dev/null
  shard="$(python3 -c "import json,sys; m=json.load(open(sys.argv[1]))['weight_map']; s={f for k,f in m.items() if k.startswith('model.visual.')}; assert len(s)==1, s; print(s.pop())" "$qdir/model.safetensors.index.json")"
  hf download Qwen/Qwen3.8-27B "$shard" --local-dir "$qdir" >/dev/null
  docker run --rm -i -u "$(id -u):$(id -g)" --entrypoint python3 -v "$qdir:/q:ro" -v "$MODEL_PATH:/out" \
    "tensorfold:v${TF_VERSION:-0.6.6}" - "/q/$shard" <<'PY'
import sys
from safetensors import safe_open
from safetensors.torch import save_file
with safe_open(sys.argv[1], "pt") as f:
    out = {k: f.get_tensor(k).contiguous() for k in f.keys() if k.startswith("model.visual.")}
assert len(out) == 333, len(out)
save_file(out, "/out/vision_bf16.safetensors",
          metadata={"source": "Qwen/Qwen3.8-27B model.visual.* (bf16, unchanged)"})
print(len(out), "bf16 vision tensors")
PY
  chmod 644 "$MODEL_PATH/vision_bf16.safetensors"
fi
ls -la "$MODEL_PATH"
