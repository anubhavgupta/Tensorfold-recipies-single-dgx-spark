#!/usr/bin/env bash
# One-time: build the model folder 27B-3 serves: the EXL3 pack turboderp/Qwen3.8-27B-exl3 @ 4.00bpw (the standard
# 4.00 bpw quant, 6-bit head; 27B-2 uses SC_4.00bpw_H5_V6) copied from the HF cache, with froggeric's v22.5 chat
# template (as in 27B-2; upstream's is kept as chat_template.upstream.jinja). This pack keeps the vision tower in bf16
# inside its shards (identical to Qwen/Qwen3.8-27B's), so TensorFold loads it directly: no vision sidecar is needed.
# Usage: ./27B-3/build_model_dir.sh     Env: MODEL_PATH (/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw),
#   HF_CACHE (~/.cache/huggingface), REPO, REVISION, TF_VERSION (0.6.6). Downloads the pack if missing (~16 GB).
set -euo pipefail
MODEL_PATH="${MODEL_PATH:-/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
REPO="${REPO:-turboderp/Qwen3.8-27B-exl3}"
REVISION="${REVISION:-4.00bpw}"
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

if ! grep -q froggeric "$MODEL_PATH/chat_template.jinja" 2>/dev/null; then
  # froggeric/Qwen-Fixed-Chat-Templates v22.5 (Apache-2.0), pinned to the commit 27B-2's model folder uses
  [[ -f "$MODEL_PATH/chat_template.upstream.jinja" ]] || cp "$MODEL_PATH/chat_template.jinja" "$MODEL_PATH/chat_template.upstream.jinja"
  curl -fsSL "https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/855bffc49448e299789730ff92c9b8d834d6cc14/chat_template.jinja" \
    -o "$MODEL_PATH/chat_template.jinja"
  echo "chat template: froggeric v22.5"
fi

ls -la "$MODEL_PATH"
