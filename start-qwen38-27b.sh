#!/usr/bin/env bash
# Serve Qwen3.8-27B (Vontra/Qwen3.8-27B-MLX-4bit + z-lab/Qwen3.8-27B-DFlash2 drafter) with stock TensorFold through
# ./tensorfold.sh. Settings follow the Qwen3.8-27B-DGX-Spark-TensorFold recipe (TensorFold v0.6.0 + patches), but
# every knob below is a stock TensorFold flag or TENSORFOLD_* environment variable as of v0.6.5. That recipe's
# image patches map as follows:
#   0001 many images / video        -> stock --vision-max-images, --vision-image-tokens, TENSORFOLD_VIDEO_TOKENS
#   0002 YaRN (1M-token windows)    -> no stock equivalent: the window stops at the native 262,144 tokens
#   0003 fp8 KV cache               -> no stock equivalent: --kv-dtype is Flash Next only, the 27B keeps bf16 KV
#                                      (64 KiB a token, 16 GiB per full 262k window)
#   0004 memory reserve below 2 GiB -> stock TENSORFOLD_MEMORY_RESERVE_GIB, floor 2 (2 here)
#   0005 pinned KV pool             -> no stock equivalent: caches grow on demand (TensorFold's memory gate)
#
# Usage: ./start-qwen38-27b.sh [extra tensorfold serve args]
#   Extra args are appended, so they override the defaults below (e.g. --parallel 8 --context 163840).
#   The old recipe ran 8 streams thanks to its fp8 KV and pinned pool. With stock bf16 KV, ~1.4M cache tokens fit,
#   so 4-5 streams can all use a full 262k window at once; PARALLEL=8 is safe at CONTEXT=163840 (higher C=8
#   throughput), and at 262144 it works until long requests together exceed memory (then new requests wait and,
#   at worst, the newest running one is stopped with "ran out of memory").
#
# Settings (environment, or .env.qwen3.8-27b; the environment wins):
#   TF_VERSION    TensorFold version (default: latest)      PORT         port (default: 8888)
#   HOST          bind address (default: 0.0.0.0)           SERVED_NAME  model id clients see (default: Qwen3.8-27B)
#   NAME          container name (default: tf-qwen38-27b)   FOREGROUND   1: run attached instead of in the background
#   MODEL_ID      target model (default: Vontra/Qwen3.8-27B-MLX-4bit)
#   DRAFT_ID      drafter (default: z-lab/Qwen3.8-27B-DFlash2); DRAFT_ID= serves without drafts (--no-drafts)
#   PARALLEL (4) CONTEXT (262144) PREFILL_FP8 (1: FP8 prompt activations, ~35-50% faster prefill, lower prompt
#     precision; 0: bf16) CHECKPOINT_SLOTS (empty: TensorFold's default)
#   VISION (1) VISION_URLS (0) VISION_MAX_IMAGES (50) VISION_IMAGE_TOKENS (16384)
#   THINKING (1) MAX_TOKENS (122880: TensorFold's own 4,096 can end a thinking reply before it answers);
#     sampling is Qwen's recommendation and switches with THINKING: thinking mode TEMPERATURE (1.0) TOP_P (0.95),
#     instruct/non-thinking mode TEMPERATURE (0.7) TOP_P (0.80); TOP_K (20) and MIN_P (0.0) either way. Qwen also
#     recommends a presence_penalty (1.5 in instruct mode) and a repetition_penalty, but TensorFold has neither
#     setting (it always decodes as if both were off), so they cannot be set here.
#   TENSORFOLD_VIDEO_TOKENS (default 16384), TENSORFOLD_MEMORY_RESERVE_GIB (default 2) and any other
#   TENSORFOLD_* variable are passed to the server.
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

PARALLEL="${PARALLEL:-4}"
CONTEXT="${CONTEXT:-262144}"
PREFILL_FP8="${PREFILL_FP8:-1}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-}"
VISION="${VISION:-1}"
VISION_URLS="${VISION_URLS:-0}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-122880}"
if [[ "$THINKING" == 1 ]]; then
  _default_temperature=1.0; _default_top_p=0.95
else
  _default_temperature=0.7; _default_top_p=0.80
fi
TEMPERATURE="${TEMPERATURE:-$_default_temperature}"
TOP_P="${TOP_P:-$_default_top_p}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"

export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists; stop it first: ./stop-qwen38-27b.sh" >&2
  exit 1
fi

wrapper=(--tf-name "$NAME" --tf-docker-arg=--ulimit=stack=67108864)
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
