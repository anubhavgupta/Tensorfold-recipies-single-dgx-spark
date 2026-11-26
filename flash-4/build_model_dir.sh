#!/usr/bin/env bash
# One-time: build the model folder flash-4 serves, from files already in the Hugging Face cache (nothing is downloaded):
#   - turboderp/Qwen3.8-Flash-Next-exl3 @ 4.05bpw_h6_ng6: the nine weight shards, ngram_embedding.safetensors, tokenizer and
#     config files, hard-linked from the cache's blobs (no copy; the folder must be on the cache's filesystem)
#   - the vision tower as bf16 (vision_bf16.safetensors, `model.visual.*` names): flash-3's floating tower
#     ($HF_CACHE/vision-f16-4.05.safetensors, decoded once from the pack's 6-bit vision_k6.safetensors by TensorFold 0.6.6's
#     exl3_convert) rounded to bf16, which is what TensorFold's native tower loader reads
#   - model.safetensors.index.json with the tower's tensors added, so the loader finds them
# Usage: ./flash-4/build_model_dir.sh     Env: MODEL_PATH ($HOME/projects/models/Qwen3.8-Flash-Next-exl3-4.05bpw),
#   HF_CACHE (~/.cache/huggingface), REPO, REVISION, TOWER (the f16 tower), IMAGE (an image with torch + safetensors)
set -euo pipefail
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO="${REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
REVISION="${REVISION:-4.05bpw_h6_ng6}"
MODEL_PATH="${MODEL_PATH:-$HOME/projects/models/Qwen3.8-Flash-Next-exl3-4.05bpw}"
TOWER="${TOWER:-$HF_CACHE/vision-f16-4.05.safetensors}"
IMAGE="${IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

dir="$HF_CACHE/hub/models--${REPO//\//--}"
[[ -f "$dir/refs/$REVISION" ]] || { echo "$REPO @ $REVISION is not in $HF_CACHE (hf download $REPO --revision $REVISION)" >&2; exit 1; }
snap="$dir/snapshots/$(<"$dir/refs/$REVISION")"
[[ -z "$(find "$dir/blobs" -name '*.incomplete' -print -quit)" ]] || { echo "a download is unfinished (*.incomplete blobs)" >&2; exit 1; }
mkdir -p "$MODEL_PATH"
for f in "$snap"/*; do
  b="$(basename "$f")"
  [[ "$b" == model.safetensors.index.json || "$b" == vision_k6.safetensors || "$b" == README.md || "$b" == qbench_* ]] && continue
  [[ -e "$MODEL_PATH/$b" ]] || ln -L "$f" "$MODEL_PATH/$b" 2>/dev/null || cp -L "$f" "$MODEL_PATH/$b"
done

if [[ ! -f "$MODEL_PATH/vision_bf16.safetensors" ]]; then
  [[ -e "$TOWER" ]] || { echo "the floating vision tower $TOWER is missing: convert vision_k6.safetensors once with TensorFold 0.6.6 (flash-3/start.sh's header)" >&2; exit 1; }
  # the tower file is root-owned (a container wrote it): read and convert it in a container
  docker run --rm -v "$(dirname "$TOWER")":/in:ro -v "$MODEL_PATH":/out -v "$HERE/scripts":/s:ro --entrypoint python3 "$IMAGE" \
    /s/vision_to_bf16.py "/in/$(basename "$TOWER")" /out/vision_bf16.safetensors
  docker run --rm -v "$MODEL_PATH":/m --entrypoint chown "$IMAGE" -R "$(id -u):$(id -g)" /m
fi

python3 - "$snap/model.safetensors.index.json" "$MODEL_PATH" <<'PY'
import json, struct, sys
index, out = sys.argv[1], sys.argv[2]
idx = json.load(open(index))
with open(f"{out}/vision_bf16.safetensors", "rb") as f:
    n = struct.unpack("<Q", f.read(8))[0]
    header = json.loads(f.read(n))
tower = [k for k in header if k != "__metadata__"]
idx["weight_map"].update({k: "vision_bf16.safetensors" for k in tower})
json.dump(idx, open(f"{out}/model.safetensors.index.json", "w"), indent=1)
print(f"{len(tower)} tower tensors added to the index")
PY
ls -la "$MODEL_PATH"
