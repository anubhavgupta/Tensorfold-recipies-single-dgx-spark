#!/usr/bin/env bash
# Serve prism-ml/Ternary-Bonsai-2-27B-mlx-2bit with stock TensorFold v0.6.5 plus the patch chain in
# bonsai-27b/patches. The first four patches are the Ternary-Bonsai-2-27B capacity patches (FP8 KV, video/media limits,
# reserve 0, pinned KV pool); 0005 adds Bonsai's CUDA loader for Prism's rotated 2-bit group-128 Hadamard pack;
# 0006 makes it fast (fast lane kernels for the ternary codes, 2-bit decode reads, a fused Hadamard kernel); 0007 keeps
# one 2.125-bit PQ2 copy of the weights (~7 GiB) read by decode and prefill, and fuses the rotations into producers.
# The plain stock image has Bonsai metadata/MLX support but no CUDA serving path for this checkpoint, so PATCHES=1
# is required here.
#
# Usage: ./bonsai-27b/start.sh [extra tensorfold serve args]
#   Extra args are appended, so they override the defaults below (e.g. --parallel 4 --context 131072).
#
# Settings (environment, or bonsai-27b/.env; the environment wins):
#   TF_VERSION    TensorFold version (default: latest)      PORT         port (default: 8888)
#   HOST          bind address (default: 0.0.0.0)           SERVED_NAME  model id clients see
#   NAME          container name (default: tf-bonsai-27b)   FOREGROUND   1: run attached instead of detached
#   MODEL_ID      target model (default: prism-ml/Ternary-Bonsai-2-27B-mlx-2bit)
#   DRAFT_ID      DFlash2 drafter (default: z-lab/Qwen3.8-27B-DFlash2; empty: --no-drafts)
#   PATCHES       must be 1: the patched image with bonsai-27b/patches
#   KV_DTYPE      fp8 (default; half-size cache) or bf16
#   KV_POOL_GB    <n>: n GiB (default 92); auto: free-at-start minus 24 GiB, at most 90; 0: no pinned pool
#   MEMORY_RESERVE_GIB (0), PARALLEL (10), CONTEXT (262144), PREFILL_FP8 (0), CHECKPOINT_SLOTS
#   VISION (0 by default; text serving is the validated CUDA path), VISION_URLS, VISION_MAX_IMAGES, VISION_IMAGE_TOKENS
#   THINKING (1), MAX_TOKENS (131072); sampling follows the Bonsai/Qwen recommendations and switches with THINKING:
#     thinking TEMPERATURE=1.0 TOP_P=0.95; non-thinking TEMPERATURE=0.7 TOP_P=0.80; TOP_K=20 MIN_P=0.0.
#
# Config file: .env beside this script, KEY=value lines (# comments, quotes both optional).
# Logs: docker logs -f tf-bonsai-27b      Stop: ./bonsai-27b/end.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
if [[ -f "$ENV_FILE" ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < "$ENV_FILE"
fi

MODEL_ID="${MODEL_ID:-prism-ml/Ternary-Bonsai-2-27B-mlx-2bit}"
DRAFT_ID="${DRAFT_ID-z-lab/Qwen3.8-27B-DFlash2}"
SERVED_NAME="${SERVED_NAME:-Ternary-Bonsai-2-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-bonsai-27b}"
FOREGROUND="${FOREGROUND:-0}"

PATCHES="${PATCHES:-1}"
KV_DTYPE="${KV_DTYPE:-fp8}"
if [[ "$PATCHES" == 1 ]]; then
  KV_POOL_GB="${KV_POOL_GB-92}"; MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-0}"
else
  echo "start-bonsai-27b: PATCHES=0 cannot serve Bonsai on CUDA with stock TensorFold v0.6.5" >&2
  exit 1
fi
REQUEST_BODY_MIB="${REQUEST_BODY_MIB:-96}"
IMAGE_TOTAL_MIB="${IMAGE_TOTAL_MIB:-64}"
VIDEO_MIB="${VIDEO_MIB:-64}"
VIDEO_TOTAL_MIB="${VIDEO_TOTAL_MIB:-96}"
PARALLEL="${PARALLEL:-10}"
CONTEXT="${CONTEXT:-262144}"
PREFILL_FP8="${PREFILL_FP8:-0}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-}"
VISION="${VISION:-0}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
if [[ "$THINKING" == 1 ]]; then
  _default_temperature=1.0; _default_top_p=0.95
else
  _default_temperature=0.7; _default_top_p=0.80
fi
TEMPERATURE="${TEMPERATURE:-$_default_temperature}"
TOP_P="${TOP_P:-$_default_top_p}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

die() { echo "start-bonsai-27b: $*" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  die "container $NAME already exists; stop it first: ./bonsai-27b/end.sh"
fi

case "$KV_DTYPE" in
  fp8) export TENSORFOLD_KV_DTYPE=fp8 ;;
  bf16) unset TENSORFOLD_KV_DTYPE ;;
  *) die "KV_DTYPE: fp8 or bf16, not '$KV_DTYPE'" ;;
esac

# The pinned KV pool, "auto": what is free now minus what everything else needs (weights and per-stream buffers:
# ~24 GiB at 8 streams, ~1 GiB more a stream past 8), at most 90 GiB.
if [[ "$KV_POOL_GB" == auto ]]; then
  avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
  KV_POOL_GB=$(( avail_gb - 24 - (PARALLEL > 8 ? PARALLEL - 8 : 0) ))
  (( KV_POOL_GB <= 90 )) || KV_POOL_GB=90
  (( KV_POOL_GB >= 8 )) || die "only ${avail_gb} GiB memory available: not enough for the model and a KV pool (need ~32+). Stop other GPU workloads (docker ps), or set KV_POOL_GB=0 to let the caches grow on demand"
  echo "KV pool: auto, ${KV_POOL_GB} GiB (${avail_gb} GiB free now; KV_POOL_GB=<n> sets it, 0 turns the pin off)"
fi
[[ "$KV_POOL_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "KV_POOL_GB: auto or a number of GiB, not '$KV_POOL_GB'"

wrapper=(--tf-name "$NAME" --tf-docker-arg=--ulimit=stack=67108864)
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
if [[ "$PATCHES" == 1 ]]; then
  wrapper+=(--tf-patches "$SCRIPT_DIR/patches")
  if [[ "$KV_POOL_GB" =~ ^0*([.]0*)?$ ]]; then unset TENSORFOLD_KV_POOL_GIB; else export TENSORFOLD_KV_POOL_GIB="$KV_POOL_GB"; fi
  export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
  export TENSORFOLD_REQUEST_BODY_MIB="$REQUEST_BODY_MIB" TENSORFOLD_IMAGE_TOTAL_MIB="$IMAGE_TOTAL_MIB"
  export TENSORFOLD_VIDEO_MIB="$VIDEO_MIB" TENSORFOLD_VIDEO_TOTAL_MIB="$VIDEO_TOTAL_MIB"
fi
[[ "$FOREGROUND" == 1 ]] || wrapper+=(--tf-detach)

serve_args=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT"
            --parallel "$PARALLEL" --context "$CONTEXT"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P"
            --max-tokens "$MAX_TOKENS")
if [[ -n "$DRAFT_ID" ]]; then serve_args+=(--drafter "$DRAFT_ID"); else serve_args+=(--no-drafts); fi
if [[ "$PREFILL_FP8" == 1 ]]; then serve_args+=(--prefill-fp8); else serve_args+=(--no-prefill-fp8); fi
[[ -z "$CHECKPOINT_SLOTS" ]] || serve_args+=(--checkpoint-slots "$CHECKPOINT_SLOTS")
if [[ "$VISION" == 1 ]]; then
  serve_args+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS")
  [[ "$VISION_URLS" != 1 ]] || serve_args+=(--vision-urls)
fi
if [[ "$THINKING" == 1 ]]; then serve_args+=(--thinking); else serve_args+=(--no-thinking); fi

"$SCRIPT_DIR/../tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ID" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./bonsai-27b/end.sh"
fi
