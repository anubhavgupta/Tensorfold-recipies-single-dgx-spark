#!/usr/bin/env bash
# Serve the EXL3 pack turboderp/Qwen3.8-Flash-Next-exl3 with TensorFold through ../tensorfold.sh.
# Download first:  hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6
# TensorFold has no HF-revision option, so the script resolves the revision's snapshot in the HF cache and serves
# that directory (the cache is mounted into the container at /root/.cache/huggingface).
#
# Usage: ./flash-2/start.sh [extra tensorfold serve args]    (extra args are appended and override the defaults)
# Env (a .env beside this script is read first; real environment variables win):
#   REVISION (4.05bpw_h6_ng6)  HF branch of the pack      MODEL_DIR  explicit local pack dir (skips the cache lookup)
#   REPO (turboderp/Qwen3.8-Flash-Next-exl3)              TF_VERSION (default 0.6.5)
#   PARALLEL (9) CONTEXT (262144) KV_DTYPE (int8) VISION (1; EXL3 needs PARALLEL >= 2) VISION_MAX_IMAGES (50)
#   VISION_IMAGE_TOKENS (16384) THINKING (1) MAX_TOKENS (32768) TEMPERATURE/TOP_P/TOP_K (1.0/0.95/20; 0.7/0.80 without thinking)
#   OVERLAY (1: mount overlay/exl3, the faster prompt kernels; 0: stock)  TENSORFOLD_MTL (5)
#   MTP_DRAFTS (6) MTP_CONFIDENCE (0.60)  PATCHES (0: use ../qwen38-flash-next/patches, untested on EXL3)
#   VISION_CACHE (1: overlay/cache, prefix caching for image requests)  VISION_LIMITS (1: overlay/limits, bigger image/video byte limits)  PORT (8888) HOST (0.0.0.0) NAME (tf-flash-2-exl3) SERVED_NAME (Qwen3.8-Flash-Next) FOREGROUND (0)
# Not available for EXL3 packs: --ple-on-ssd, --tp 2. Logs: docker logs -f tf-flash-2-exl3   Stop: ./flash-2/end.sh
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

REPO="${REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
REVISION="${REVISION:-4.05bpw_h6_ng6}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-flash-2-exl3}"
FOREGROUND="${FOREGROUND:-0}"
PATCHES="${PATCHES:-0}"
TF_VERSION="${TF_VERSION:-0.6.5}"

PARALLEL="${PARALLEL:-9}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-int8}"
VISION="${VISION:-1}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
if [[ "$THINKING" == 1 ]]; then _t=1.0; _p=0.95; else _t=0.7; _p=0.80; fi
TEMPERATURE="${TEMPERATURE:-$_t}"
TOP_P="${TOP_P:-$_p}"
TOP_K="${TOP_K:-20}"
MTP_DRAFTS="${MTP_DRAFTS:-6}"
MTP_CONFIDENCE="${MTP_CONFIDENCE:-0.60}"

export TENSORFOLD_MAX_BODY_MIB="${TENSORFOLD_MAX_BODY_MIB:-128}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The pack's vision tower is a separate vision_k6.safetensors sidecar that TensorFold loads only after a one-time
# conversion to a floating tower (the output is in the mounted HF cache). Convert once with:
#   docker run --rm --gpus all --entrypoint python3 -v ~/.cache/huggingface:/root/.cache/huggingface tensorfold:v0.6.5 \
#     -m tensorfold.vision.exl3_convert <snapshot>/vision_k6.safetensors /root/.cache/huggingface/vision-f16-4.05.safetensors
if [[ "${VISION:-1}" == 1 && -z "${TENSORFOLD_VISION_WEIGHTS:-}" && -f "$HF_CACHE/vision-f16-4.05.safetensors" ]]; then
  export TENSORFOLD_VISION_WEIGHTS=/root/.cache/huggingface/vision-f16-4.05.safetensors
fi
if [[ -z "${MODEL_DIR:-}" ]]; then
  _repo_dir="$HF_CACHE/hub/models--${REPO//\//--}"
  [[ -f "$_repo_dir/refs/$REVISION" ]] || { echo "revision $REVISION of $REPO is not in $HF_CACHE; run: hf download $REPO --revision $REVISION" >&2; exit 1; }
  _snap="$(<"$_repo_dir/refs/$REVISION")"
  _snap_dir="$_repo_dir/snapshots/$_snap"
  # The files are symlinks to blobs, so a missing blob means the download is unfinished.
  for _f in model.safetensors.index.json ngram_embedding.safetensors; do
    [[ -e "$_snap_dir/$_f" ]] || { echo "$_f missing in $_snap_dir: the download is not finished" >&2; exit 1; }
  done
  [[ -z "$(find "$_repo_dir/blobs" -name '*.incomplete' -print -quit)" ]] || { echo "download still in progress (*.incomplete blobs)" >&2; exit 1; }
  MODEL_ARG="/root/.cache/huggingface/hub/models--${REPO//\//--}/snapshots/$_snap"
else
  MODEL_ARG="$MODEL_DIR"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists; stop it first: ./flash-2/end.sh" >&2
  exit 1
fi

wrapper=(--tf-name "$NAME" --tf-version "$TF_VERSION" --tf-docker-arg=--ulimit=stack=67108864)
# Faster EXL3 prompt processing (bit-identical output): overlay/exl3 replaces the 0.6.5 expert kernels, see
# overlay/README.md. OVERLAY=0 runs stock 0.6.5. TENSORFOLD_MTL (4, 5, 6, 8) sets the row tiles that share one weight decode.
if [[ "${OVERLAY:-1}" == 1 && "$TF_VERSION" == 0.6.5 ]]; then
  _ex=/usr/local/lib/python3.12/dist-packages/tensorfold/cuda/exl3
  for _f in experts_grouped.cuh experts.cu experts.py; do
    wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/overlay/exl3/$_f:$_ex/$_f:ro")
  done
  # Larger media limits (overlay/limits): request bodies up to TENSORFOLD_MAX_BODY_MIB (128), videos 64 MiB each / 96 MiB in all, 64 MiB of images. VISION_LIMITS=0 keeps stock.
  if [[ "${VISION_LIMITS:-1}" == 1 ]]; then
    _tf=/usr/local/lib/python3.12/dist-packages/tensorfold
    wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/overlay/limits/request_body.py:$_tf/server/request_body.py:ro")
    for _f in images.py videos.py; do
      wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/overlay/limits/$_f:$_tf/vision/$_f:ro")
    done
  fi
  # Prefix caching for image requests (overlay/cache): kept prefixes are keyed by the images' hashes. VISION_CACHE=0 turns it off.
  if [[ "${VISION_CACHE:-1}" == 1 ]]; then
    _qc=/usr/local/lib/python3.12/dist-packages/tensorfold/families/qwen4_exp/cuda
    for _f in multi.py multi_fill.py prefixes.py state.py; do
      wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/overlay/cache/$_f:$_qc/$_f:ro")
    done
  fi
fi
[[ "$PATCHES" != 1 ]] || wrapper+=(--tf-patches "$SCRIPT_DIR/../qwen38-flash-next/patches")
[[ "$FOREGROUND" == 1 ]] || wrapper+=(--tf-detach)

serve_args=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT"
            --parallel "$PARALLEL" --context "$CONTEXT" --kv-dtype "$KV_DTYPE"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
            --max-tokens "$MAX_TOKENS")
if [[ "$VISION" == 1 ]]; then
  (( PARALLEL >= 2 )) || { echo "--vision on an EXL3 pack needs PARALLEL >= 2 (or VISION=0)" >&2; exit 1; }
  serve_args+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS")
fi
if [[ "$THINKING" == 1 ]]; then serve_args+=(--thinking); else serve_args+=(--no-thinking); fi
[[ -z "${MTP_DRAFTS:-}" ]] || serve_args+=(--mtp-drafts "$MTP_DRAFTS")
[[ -z "${MTP_CONFIDENCE:-}" ]] || serve_args+=(--mtp-confidence "$MTP_CONFIDENCE")

"$SCRIPT_DIR/../tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ARG" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./flash-2/end.sh"
fi
