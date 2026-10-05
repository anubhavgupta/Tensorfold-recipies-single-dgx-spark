#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP) with stock TensorFold through
# ./tensorfold.sh. Settings follow the Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold recipe, but every
# knob below is a stock TensorFold flag or TENSORFOLD_* environment variable as of v0.6.5 - no patches
# needed (that recipe's image patches up to v0.6.1 are now upstream: --vision-max-images,
# --vision-image-tokens, --kv-dtype, --mtp-drafts/--mtp-confidence, --ple-on-ssd, TENSORFOLD_PREFILL_ROWS,
# TENSORFOLD_VIDEO_TOKENS and TENSORFOLD_MEMORY_RESERVE_GIB all exist in stock TensorFold now). Left out
# because stock TensorFold has no equivalent: the DRAFT_LANGUAGE MTP-vocabulary patch and TENSORFOLD_MTP_COPY
# (prompt-lookup drafts ahead of MTP).
#
# Usage: ./start-qwen38-flash-next.sh [extra tensorfold serve args]
#   Extra args are appended, so they override the defaults below (e.g. --parallel 3 --kv-dtype bf16).
#   --parallel 5 --context 262144 --kv-dtype int8 is ~102.6 GiB (~4.5 GiB/stream); other fits: 4 streams
#   bf16 is tighter, 3 streams bf16 at 262k, 6-8 streams int8/int4 at a shorter --context.
#
# Settings (environment):
#   TF_VERSION    TensorFold version (default: latest)      PORT         port (default: 8888)
#   HOST          bind address (default: 0.0.0.0)           SERVED_NAME  model id clients see (default: Qwen3.8-Flash-Next)
#   NAME          container name (default: tf-qwen38-flash-next)   FOREGROUND   1: run attached instead of in the background
#   MODEL_ID      target model (default: Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP)
#   PARALLEL (5) CONTEXT (262144) KV_DTYPE (int8: bf16|int8|int4) PLE_ON_SSD (1)
#   VISION (1) VISION_MAX_IMAGES (50) VISION_IMAGE_TOKENS (16384)
#   MTP_DRAFTS (6) / MTP_CONFIDENCE (0.60): TensorFold's own Flash Next default is 6 / 0.70, but the
#     recipe's swept 0.60 beat it ~3-4% with identical output, so this script matches it; empty: TensorFold's own default
#   THINKING (1) MAX_TOKENS (32768); sampling is Qwen's recommendation and switches with THINKING:
#     thinking mode TEMPERATURE (1.0) TOP_P (0.95), instruct/non-thinking mode TEMPERATURE (0.7)
#     TOP_P (0.80); TOP_K (20) and MIN_P (0.0) are the same either way. Qwen also recommends a
#     presence_penalty (1.5 in instruct mode) and a repetition_penalty, but TensorFold has neither
#     setting (it always decodes as if both were off), so they cannot be set here.
#   TENSORFOLD_VIDEO_TOKENS (default 16384), TENSORFOLD_MEMORY_RESERVE_GIB (default 2),
#   TENSORFOLD_PREFILL_ROWS (default 2048 when PLE_ON_SSD=1, matching the recipe's measured-faster choice
#     with the n-gram tables on SSD; empty: TensorFold's own choice),
#   TENSORFOLD_VISION_WORKSPACE_MIB (default 0: scratch comes from the system reserve only while an image or
#     video encodes, freeing ~4 GiB for the KV cache pool instead of a standing reservation)
#   and any other TENSORFOLD_* variable are passed to the server.
#
# Config file: .env.flash-next beside this script, KEY=value lines (# comments, quotes both optional).
#   Read before the defaults above, so it only changes what it sets; a variable already in the environment
#   (e.g. `PARALLEL=3 ./start-qwen38-flash-next.sh`) wins over the file either way. It is yours, never
#   executed as a script, and not required - delete it to go back to pure script defaults.
#
# Logs: docker logs -f tf-qwen38-flash-next      Stop: ./stop-qwen38-flash-next.sh (or docker stop tf-qwen38-flash-next)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.flash-next}"
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

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-Flash-Next-MLX-4bit-MTP}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-qwen38-flash-next}"
FOREGROUND="${FOREGROUND:-0}"

PARALLEL="${PARALLEL:-5}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
PLE_ON_SSD="${PLE_ON_SSD:-1}"
VISION="${VISION:-1}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
# Qwen's recommended sampling: thinking mode (temperature 1.0, top_p 0.95) vs instruct/non-thinking
# mode (0.7, 0.80). top_k 20 and min_p 0.0 are the same in both. presence_penalty 1.5 (instruct mode)
# and repetition_penalty are also Qwen's recommendation, but TensorFold has neither setting (it always
# decodes as if both were off), so they cannot be applied here.
if [[ "$THINKING" == 1 ]]; then
  _default_temperature=1.0; _default_top_p=0.95
else
  _default_temperature=0.7; _default_top_p=0.80
fi
TEMPERATURE="${TEMPERATURE:-$_default_temperature}"
TOP_P="${TOP_P:-$_default_top_p}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"

export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"
[[ "$PLE_ON_SSD" != 1 ]] || export TENSORFOLD_PREFILL_ROWS="${TENSORFOLD_PREFILL_ROWS:-2048}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists; stop it first: ./stop-qwen38-flash-next.sh" >&2
  exit 1
fi

wrapper=(--tf-name "$NAME" --tf-docker-arg=--ulimit=stack=67108864)
[[ "$FOREGROUND" == 1 ]] || wrapper+=(--tf-detach)

serve_args=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT"
            --parallel "$PARALLEL" --context "$CONTEXT" --kv-dtype "$KV_DTYPE"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K" --min-p "$MIN_P"
            --max-tokens "$MAX_TOKENS")
[[ "$PLE_ON_SSD" == 1 ]] && serve_args+=(--ple-on-ssd)
[[ "$VISION" == 1 ]] && serve_args+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS")
if [[ "$THINKING" == 1 ]]; then serve_args+=(--thinking); else serve_args+=(--no-thinking); fi
[[ -z "${MTP_DRAFTS:-}" ]] || serve_args+=(--mtp-drafts "$MTP_DRAFTS")
[[ -z "${MTP_CONFIDENCE:-}" ]] || serve_args+=(--mtp-confidence "$MTP_CONFIDENCE")

"$SCRIPT_DIR/tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ID" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./stop-qwen38-flash-next.sh"
fi
