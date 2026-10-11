#!/usr/bin/env bash
# One-time: build the model folder 27B-2 serves (and that can be published as one Hugging Face repo):
#   the EXL3 pack (turboderp/Qwen3.8-27B-exl3 @ SC_4.00bpw_H5_V6) copied from the HF cache, plus
#   vision_bf16.safetensors: the same model's full-precision vision tower (333 vision_tower.* bf16 tensors, copied
#   unchanged from Vontra/Qwen3.8-27B-MLX-4bit). The pack's own vision tower is EXL3-quantized (6 bits) inside the
#   shards, which TensorFold cannot load; start.sh points TENSORFOLD_VISION_WEIGHTS at vision_bf16.safetensors.
# Usage: ./27B-2/build_model_dir.sh     Env: MODEL_PATH (/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision),
#   HF_CACHE (~/.cache/huggingface), REPO, REVISION, TF_VERSION (0.6.6). Downloads what is missing (~16 + ~15 GB).
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
  vdir="$HF_CACHE/hub/models--Vontra--Qwen3.8-27B-MLX-4bit"
  [[ -d "$vdir/snapshots" ]] || hf download Vontra/Qwen3.8-27B-MLX-4bit
  docker run --rm -i -u "$(id -u):$(id -g)" --entrypoint python3 -v "$HF_CACHE:/hf:ro" -v "$MODEL_PATH:/out" \
    "tensorfold:v${TF_VERSION:-0.6.6}" - <<'PY'
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
save_file(out, "/out/vision_bf16.safetensors",
          metadata={"source": "Vontra/Qwen3.8-27B-MLX-4bit vision_tower (bf16, unchanged)"})
print(len(out), "bf16 vision tensors")
PY
  chmod 644 "$MODEL_PATH/vision_bf16.safetensors"
fi
ls -la "$MODEL_PATH"
