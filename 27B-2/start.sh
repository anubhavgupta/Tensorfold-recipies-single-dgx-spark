#!/usr/bin/env bash
# Serve anubhav200/Qwen3.8-27B-exl3-4.00bpw-bf16vision (turboderp's EXL3 4.00 bpw pack of Qwen3.8-27B, SC_4.00bpw_H5_V6,
# plus the official bf16 vision tower and froggeric's v22.5 chat template) with the EXL3 DFlash2 drafter
# igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw on TensorFold only (./tensorfold.sh, a derived image built from ./patches).
# Download first:
#   hf download anubhav200/Qwen3.8-27B-exl3-4.00bpw-bf16vision --local-dir ~/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision
#   hf download igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw
# The model folder (MODEL_PATH) is mounted read-only into the container; the drafter is found in the HF cache.
#
# Patches (27B-2/patches, applied with tensorfold.sh --tf-patches; 0001-0004 come from ../qwen38-27b):
#   0001 fp8 KV cache (KV_DTYPE=fp8)  0002 video / bigger media limits  0003 memory reserve below 2 GiB
#   0004 pinned KV pool               0005 EXL3 drafter loader (decodes the drafter's EXL3 groups to bf16, packs to 4-bit)
#   0006 faster EXL3 prefill GEMM tiles   0007 image/video prefix cache   0008 elastic kept prompt states (KEEP_MAX)
#   0009 EXL3 verify kernel shares weight decodes across rows   0010 GB10 drafter GEMM tile   0011 overlapped GDN replay
#
# Usage: ./27B-2/start.sh [extra tensorfold serve args]    (appended, so they override the defaults)
# Env (a .env beside this script is read first; real environment variables win):
#   MODEL_PATH (/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision: the downloaded model folder, or one
#     rebuilt from the upstream repos with build_model_dir.sh;
#     the EXL3 pack plus vision_bf16.safetensors; mounted read-only at /models/<name>)
#   MODEL_DIR explicit pack dir inside the container (skips MODEL_PATH)
#   DRAFT_REPO (igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw)  DRAFT_DIR explicit drafter dir; DRAFT_REPO= serves without drafts
#   TF_VERSION (0.6.6)  PATCHES (1; 0: stock image, which needs KV_DTYPE=bf16, no drafter)
#   PARALLEL (16) CONTEXT (262144) KV_DTYPE (fp8|bf16) KV_POOL_GB (empty/0: caches grow on demand; auto; <n> GiB)
#   MEMORY_RESERVE_GIB (2) CHECKPOINT_SLOTS (empty: TensorFold's default)
#   VISION (1; needs MODEL_PATH/vision_bf16.safetensors) VISION_MAX_IMAGES (50) VISION_IMAGE_TOKENS (16384) THINKING (1) MAX_TOKENS (131072: a lower cap can end a thinking reply before it answers)
#   TEMPERATURE/TOP_P/TOP_K (1.0/0.95/20; 0.7/0.80 without thinking)
#   TENSORFOLD_MAX_BODY_MIB / TENSORFOLD_*  any TENSORFOLD_* variable is passed on
#   PORT (8888) HOST (0.0.0.0) NAME (tf-27b-2) SERVED_NAME (Qwen3.8-27B) FOREGROUND (0)
# Logs: docker logs -f tf-27b-2     Stop: ./27B-2/end.sh
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

DRAFT_REPO="${DRAFT_REPO-igor255/Qwen3.8-27B-DFlash2-EXL3-4.00bpw}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-27b-2}"
FOREGROUND="${FOREGROUND:-0}"
PATCHES="${PATCHES:-1}"
TF_VERSION="${TF_VERSION:-0.6.6}"

PARALLEL="${PARALLEL:-16}"
CONTEXT="${CONTEXT:-262144}"
KV_DTYPE="${KV_DTYPE:-fp8}"
KV_POOL_GB="${KV_POOL_GB:-0}"
MEMORY_RESERVE_GIB="${MEMORY_RESERVE_GIB:-2}"
CHECKPOINT_SLOTS="${CHECKPOINT_SLOTS:-}"
VISION="${VISION:-1}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-16384}"
THINKING="${THINKING:-1}"
MAX_TOKENS="${MAX_TOKENS:-131072}"
if [[ "$THINKING" == 1 ]]; then _t=1.0; _p=0.95; else _t=0.7; _p=0.80; fi
TEMPERATURE="${TEMPERATURE:-$_t}"
TOP_P="${TOP_P:-$_p}"
TOP_K="${TOP_K:-20}"

die() { echo "start-27b-2: $*" >&2; exit 1; }

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# Resolve the drafter's cached snapshot to its path inside the container; refuse unfinished downloads.
resolve() {
  local repo=$1 rev=$2 dir snap
  dir="$HF_CACHE/hub/models--${repo//\//--}"
  [[ -f "$dir/$rev" || -f "$dir/refs/$rev" ]] || die "$repo ($rev) is not in $HF_CACHE; run: hf download $repo${rev:+ --revision $rev}"
  [[ -f "$dir/refs/$rev" ]] && snap="$(<"$dir/refs/$rev")" || snap="$(<"$dir/$rev")"
  [[ -e "$dir/snapshots/$snap/model.safetensors.index.json" || -e "$dir/snapshots/$snap/model.safetensors" ]] \
    || die "$repo: the download is not finished (no weights in $dir/snapshots/$snap)"
  [[ -z "$(find "$dir/blobs" -name '*.incomplete' -print -quit)" ]] || die "$repo: download still in progress (*.incomplete blobs)"
  echo "/root/.cache/huggingface/hub/models--${repo//\//--}/snapshots/$snap"
}
MODEL_PATH="${MODEL_PATH:-/home/gb10/projects/models/Qwen3.8-27B-exl3-4.00bpw-bf16vision}"
MODEL_MOUNT=""
if [[ -n "${MODEL_DIR:-}" ]]; then
  MODEL_ARG="$MODEL_DIR"
else
  [[ -f "$MODEL_PATH/model.safetensors.index.json" ]] || die "no model in $MODEL_PATH: hf download anubhav200/Qwen3.8-27B-exl3-4.00bpw-bf16vision --local-dir $MODEL_PATH (or ./27B-2/build_model_dir.sh)"
  MODEL_MOUNT="/models/$(basename "$MODEL_PATH")"
  MODEL_ARG="$MODEL_MOUNT"
fi
DRAFT_ARG=""
if [[ -n "${DRAFT_DIR:-}" ]]; then DRAFT_ARG="$DRAFT_DIR"; elif [[ -n "$DRAFT_REPO" ]]; then DRAFT_ARG="$(resolve "$DRAFT_REPO" refs/main)"; fi

# The pack's own vision tower is EXL3-quantized (6 bits) in the shards, which TensorFold cannot load: serve the
# official bf16 tower the model folder carries as vision_bf16.safetensors instead.
if [[ "$VISION" == 1 && -z "${TENSORFOLD_VISION_WEIGHTS:-}" ]]; then
  [[ -n "$MODEL_MOUNT" ]] || die "VISION=1 with MODEL_DIR: set TENSORFOLD_VISION_WEIGHTS to a container path, or VISION=0"
  [[ -f "$MODEL_PATH/vision_bf16.safetensors" ]] || die "no $MODEL_PATH/vision_bf16.safetensors: download the model folder again (see the header) or VISION=0"
  export TENSORFOLD_VISION_WEIGHTS="$MODEL_MOUNT/vision_bf16.safetensors"
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  die "container $NAME already exists; stop it first: ./27B-2/end.sh"
fi

export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_KEEP_MAX="${TENSORFOLD_KEEP_MAX:-128}"
export TENSORFOLD_KEEP_HEADROOM_GIB="${TENSORFOLD_KEEP_HEADROOM_GIB:-8}"
export TENSORFOLD_MEMORY_RESERVE_GIB="$MEMORY_RESERVE_GIB"
export TENSORFOLD_REQUEST_BODY_MIB="${TENSORFOLD_REQUEST_BODY_MIB:-192}"
export TENSORFOLD_IMAGE_TOTAL_MIB="${TENSORFOLD_IMAGE_TOTAL_MIB:-64}"
export TENSORFOLD_VIDEO_MIB="${TENSORFOLD_VIDEO_MIB:-64}"
export TENSORFOLD_VIDEO_TOTAL_MIB="${TENSORFOLD_VIDEO_TOTAL_MIB:-192}"
case "$KV_DTYPE" in
  fp8) export TENSORFOLD_KV_DTYPE=fp8 ;;
  bf16) unset TENSORFOLD_KV_DTYPE ;;
  *) die "KV_DTYPE: fp8 or bf16, not '$KV_DTYPE'" ;;
esac
if [[ "$KV_POOL_GB" == auto ]]; then
  avail_gb=$(free -g | awk '/^Mem:/ {print $7}')
  KV_POOL_GB=$(( avail_gb - 31 - (PARALLEL > 8 ? PARALLEL - 8 : 0) ))
  (( KV_POOL_GB <= 78 )) || KV_POOL_GB=78
  (( KV_POOL_GB >= 8 )) || die "only ${avail_gb} GiB available for a KV pool"
fi

wrapper=(--tf-name "$NAME" --tf-version "$TF_VERSION" --tf-docker-arg=--ulimit=stack=67108864)
[[ -z "$MODEL_MOUNT" ]] || wrapper+=(--tf-docker-arg="--volume=$MODEL_PATH:$MODEL_MOUNT:ro")
if [[ "$PATCHES" == 1 ]]; then
  wrapper+=(--tf-patches "$SCRIPT_DIR/patches")
  if [[ "$KV_POOL_GB" =~ ^0*([.]0*)?$ ]]; then unset TENSORFOLD_KV_POOL_GIB; else export TENSORFOLD_KV_POOL_GIB="$KV_POOL_GB"; fi
else
  [[ "$KV_DTYPE" == bf16 ]] || die "PATCHES=0 (stock image) needs KV_DTYPE=bf16"
  unset TENSORFOLD_KV_POOL_GIB TENSORFOLD_REQUEST_BODY_MIB TENSORFOLD_IMAGE_TOTAL_MIB TENSORFOLD_VIDEO_MIB TENSORFOLD_VIDEO_TOTAL_MIB
fi
[[ "$FOREGROUND" == 1 ]] || wrapper+=(--tf-detach)

serve_args=(--name "$SERVED_NAME" --host "$HOST" --port "$PORT"
            --parallel "$PARALLEL" --context "$CONTEXT"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K"
            --max-tokens "$MAX_TOKENS")
if [[ -n "$DRAFT_ARG" ]]; then serve_args+=(--drafter "$DRAFT_ARG"); else serve_args+=(--no-drafts); fi
[[ -z "$CHECKPOINT_SLOTS" ]] || serve_args+=(--checkpoint-slots "$CHECKPOINT_SLOTS")
if [[ "$VISION" == 1 ]]; then
  serve_args+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS")
fi
if [[ "$THINKING" == 1 ]]; then serve_args+=(--thinking); else serve_args+=(--no-thinking); fi

"$SCRIPT_DIR/../tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ARG" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./27B-2/end.sh"
fi
