#!/usr/bin/env bash
# Serve Qwen3.8-27B (Vontra/Qwen3.8-27B-MLX-4bit + z-lab/Qwen3.8-27B-DFlash2 drafter) with stock TensorFold through
# ./tensorfold.sh. Settings follow the Qwen3.8-27B-DGX-Spark-TensorFold recipe (TensorFold v0.6.0 + patches), but
# stock TensorFold v0.6.5 plus the patches in patches/qwen38-27b, which go into a derived image
# (tensorfold.sh --tf-patches) that only this script runs; PATCHES=0 runs the plain stock image instead. Each patch
# does nothing unless its setting is on. That recipe's image patches map as follows:
#   0001 many images / video        -> stock --vision-max-images, --vision-image-tokens, TENSORFOLD_VIDEO_TOKENS;
#                                      video for the 27B and the bigger size limits: our 0002 (stock: Flash Next
#                                      only; 32 MiB bodies, 20 MiB of images, 16/20 MiB of video)
#   0002 YaRN (1M-token windows)    -> not ported: the window stops at the native 262,144 tokens
#   0003 fp8 KV cache               -> our 0001 (KV_DTYPE=fp8, the default; stock --kv-dtype is Flash Next only).
#                                      fp8: 32 KiB a token, 8 GiB per full 262k window (bf16: 64 KiB, 16 GiB)
#   0004 memory reserve below 2 GiB -> our 0003 (MEMORY_RESERVE_GIB=0; stock floor 2)
#   0005 pinned KV pool             -> our 0004 (KV_POOL_GB=auto; stock: caches grow and shrink on demand)
#
# Usage: ./start-qwen38-27b.sh [extra tensorfold serve args]
#   Extra args are appended, so they override the defaults below (e.g. --parallel 4 --context 131072).
#   The KV pool (auto: free memory at start - 31 GiB, at most 78) holds ~2.5M tokens with fp8 KV, so 8 streams can
#   all use a full 262k window at once (bf16 KV: ~1.25M, 4-5 full windows). Past that, new requests wait and, at
#   worst, the newest running one is stopped with "ran out of memory".
#
# Settings (environment, or .env.qwen3.8-27b; the environment wins):
#   TF_VERSION    TensorFold version (default: latest)      PORT         port (default: 8888)
#   HOST          bind address (default: 0.0.0.0)           SERVED_NAME  model id clients see (default: Qwen3.8-27B)
#   NAME          container name (default: tf-qwen38-27b)   FOREGROUND   1: run attached instead of in the background
#   MODEL_ID      target model (default: Vontra/Qwen3.8-27B-MLX-4bit)
#   DRAFT_ID      drafter (default: z-lab/Qwen3.8-27B-DFlash2); DRAFT_ID= serves without drafts (--no-drafts)
#   PATCHES (1: the patched image; 0: stock image, which needs KV_DTYPE=bf16, and defaults KV_POOL_GB to 0 and
#     MEMORY_RESERVE_GIB to 2)
#   KV_DTYPE (fp8: half the cache memory; bf16)
#   KV_POOL_GB (auto: pin free-at-start minus 31 GiB, minus 1 a stream over 8, at most 78; <n>: n GiB; 0: no pool,
#     caches grow and shrink on demand)  MEMORY_RESERVE_GIB (0: budget all of MemAvailable; stock default 2)
#   REQUEST_BODY_MIB (96) IMAGE_TOTAL_MIB (64, all images of a request) VIDEO_MIB (64) VIDEO_TOTAL_MIB (96)
#   PARALLEL (8) CONTEXT (262144) PREFILL_FP8 (1: FP8 prompt activations, ~35-50% faster prefill, lower prompt
#     precision; 0: bf16) CHECKPOINT_SLOTS (empty: TensorFold's default)
#   VISION (1) VISION_URLS (0) VISION_MAX_IMAGES (50) VISION_IMAGE_TOKENS (16384)
#   THINKING (1) MAX_TOKENS (163840: TensorFold's own 4,096 can end a thinking reply before it answers);
#     sampling is Qwen's recommendation and switches with THINKING: thinking mode TEMPERATURE (1.0) TOP_P (0.95),
#     instruct/non-thinking mode TEMPERATURE (0.7) TOP_P (0.80); TOP_K (20) and MIN_P (0.0) either way. Qwen also
#     recommends a presence_penalty (1.5 in instruct mode) and a repetition_penalty, but TensorFold has neither
#     setting (it always decodes as if both were off), so they cannot be set here.
#   TENSORFOLD_VIDEO_TOKENS (default 16384) and any other TENSORFOLD_* variable are passed to the server.
#
# Config file: .env.qwen3.8-27b beside this script, KEY=value lines (# comments, quotes both optional).
#   Read before the defaults above, so it only changes what it sets; a variable already in the environment
#   (e.g. `PARALLEL=4 ./start-qwen38-27b.sh`) wins over the file either way. It is yours, never executed as a
#   script, and not required. ENV_FILE=/path/to/other.env reads another file instead.
#
# Logs: docker logs -f tf-qwen38-27b      Stop: ./stop-qwen38-27b.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.qwen3.8-27b}"
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

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-27B-MLX-4bit}"
DRAFT_ID="${DRAFT_ID-z-lab/Qwen3.8-27B-DFlash2}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-qwen38-27b}"
FOREGROUND="${FOREGROUND:-0}"

PATCHES="${PATCHES:-1}"
KV_DTYPE="${KV_DTYPE:-fp8}"
if [[ "$PATCHES" == 1 ]]; then
  KV_POOL_GB="${KV_POOL_GB-auto}"; MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-0}"
else
  KV_POOL_GB="${KV_POOL_GB:-0}"; MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-2}"
fi
REQUEST_BODY_MIB="${REQUEST_BODY_MIB:-96}"
IMAGE_TOTAL_MIB="${IMAGE_TOTAL_MIB:-64}"
VIDEO_MIB="${VIDEO_MIB:-64}"
VIDEO_TOTAL_MIB="${VIDEO_TOTAL_MIB:-96}"
PARALLEL="${PARALLEL:-8}"
CONTEXT="${CONTEXT:-262144}"
PREFILL_FP8="${PREFILL_FP8:-1}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-}"
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-163840}"
if [[ "$THINKING" == 1 ]]; then
  _default_temperature=1.0; _default_top_p=0.95
else
  _default_temperature=0.7; _default_top_p=0.80
fi
TEMPERATURE="${TEMPERATURE:-$_default_temperature}"
TOP_P="${TOP_P:-$_default_top_p}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

die() { echo "start-qwen38-27b: $*" >&2; exit 1; }

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  die "container $NAME already exists; stop it first: ./stop-qwen38-27b.sh"
fi

case "$KV_DTYPE" in
  fp8) export TENSORFOLD_KV_DTYPE=fp8 ;;
  bf16) unset TENSORFOLD_KV_DTYPE ;;
  *) die "KV_DTYPE: fp8 or bf16, not '$KV_DTYPE'" ;;
esac

# The pinned KV pool, "auto": what is free now minus what everything else needs (weights, drafter, per-stream
# buffers: ~31 GiB at 8 streams, ~1 GiB more a stream past 8), at most 78 GiB.
if [[ "$KV_POOL_GB" == auto ]]; then
  avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
  KV_POOL_GB=$(( avail_gb - 31 - (PARALLEL > 8 ? PARALLEL - 8 : 0) ))
  (( KV_POOL_GB <= 78 )) || KV_POOL_GB=78
  (( KV_POOL_GB >= 8 )) || die "only ${avail_gb} GiB memory available: not enough for the model and a KV pool (need ~40+). Stop other GPU workloads (docker ps), or set KV_POOL_GB=0 to let the caches grow on demand"
  echo "KV pool: auto, ${KV_POOL_GB} GiB (${avail_gb} GiB free now; KV_POOL_GB=<n> sets it, 0 turns the pin off)"
fi
[[ "$KV_POOL_GB" =~ ^[0-9]+([.][0-9]+)?$ ]] || die "KV_POOL_GB: auto or a number of GiB, not '$KV_POOL_GB'"

wrapper=(--tf-name "$NAME" --tf-docker-arg=--ulimit=stack=67108864)
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
if [[ "$PATCHES" == 1 ]]; then
  wrapper+=(--tf-patches "$SCRIPT_DIR/patches/qwen38-27b")
  if [[ "$KV_POOL_GB" =~ ^0*([.]0*)?$ ]]; then unset TENSORFOLD_KV_POOL_GIB; else export TENSORFOLD_KV_POOL_GIB="$KV_POOL_GB"; fi
  export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
  export TENSORFOLD_REQUEST_BODY_MIB="$REQUEST_BODY_MIB" TENSORFOLD_IMAGE_TOTAL_MIB="$IMAGE_TOTAL_MIB"
  export TENSORFOLD_VIDEO_MIB="$VIDEO_MIB" TENSORFOLD_VIDEO_TOTAL_MIB="$VIDEO_TOTAL_MIB"
else
  [[ "$KV_DTYPE" == bf16 ]] || die "PATCHES=0 (stock image) has no fp8 KV cache for the 27B: set KV_DTYPE=bf16"
  [[ "$KV_POOL_GB" =~ ^0*([.]0*)?$ ]] || die "PATCHES=0 (stock image) has no KV pool: set KV_POOL_GB=0"
  awk -v g="$MEMORY_RESERVE_GIB" 'BEGIN { exit !(g >= 2) }' \
    || die "PATCHES=0 (stock image): MEMORY_RESERVE_GIB must be 2 or more"
  export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
  unset TENSORFOLD_KV_POOL_GIB
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

"$SCRIPT_DIR/tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ID" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./stop-qwen38-27b.sh"
fi
