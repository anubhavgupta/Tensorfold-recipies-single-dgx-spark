#!/usr/bin/env bash
# Serve the EXL3 pack turboderp/Qwen3.8-Flash-Next-exl3 with TensorFold 0.6.6 (the last Python engine) through
# ../tensorfold.sh, drafting with the DFlash2 model incoai/Qwen3.8-Flash-Next-DFlash2 instead of the checkpoint's MTP head.
# Download first:  hf download turboderp/Qwen3.8-Flash-Next-exl3 --revision 4.05bpw_h6_ng6
#                  hf download incoai/Qwen3.8-Flash-Next-DFlash2
# TensorFold has no HF-revision option, so the script resolves the revision's snapshot in the HF cache and serves
# that directory (the cache is mounted into the container at /root/.cache/huggingface).
#
# Usage: ./flash-3/start.sh [extra tensorfold serve args]    (extra args are appended and override the defaults)
# Env (a .env beside this script is read first; real environment variables win):
#   REVISION (4.05bpw_h6_ng6)  HF branch of the pack      MODEL_DIR  explicit local pack dir (skips the cache lookup)
#   REPO (turboderp/Qwen3.8-Flash-Next-exl3)              TF_VERSION (0.6.6; the patches are for 0.6.6 only)
#   DRAFTER (incoai/Qwen3.8-Flash-Next-DFlash2; none: the MTP head, as flash-2)  DRAFTER_DIR (explicit drafter dir)
#   DRAFTS (7: DFlash2 drafts a round, at most its block of 8 less one)  DRAFT_CONFIDENCE (0.5: a chain stops where its
#     probability product falls under this)  TENSORFOLD_DFLASH_VOCAB (draft: Flash Next's draft vocabulary; full)
#   MTP_DRAFTS (6) MTP_CONFIDENCE (0.60)  with DRAFTER=none
#   PARALLEL (9; DFlash2 needs >= 2) CONTEXT (262144) KV_DTYPE (int8) VISION (1; EXL3 needs PARALLEL >= 2) VISION_MAX_IMAGES (50)
#   VISION_IMAGE_TOKENS (16384) THINKING (1) MAX_TOKENS (32768) TEMPERATURE/TOP_P/TOP_K (1.0/0.95/20; 0.7/0.80 without thinking)
#   OVERLAY (1: patches/0002, the faster EXL3 prompt kernels; 0: stock)  TENSORFOLD_MTL (5)
#   TOOL_PARSER_FIX (1: patches/0001, boundary-aware calls and union types; 0: stock parser)
#   TOOL_TRACE (0: opt-in sensitive tool request/output capture)  TOOL_TRACE_DIR (./tool-traces beside this script)
#   VISION_CACHE (1: patches/0005, prefix caching for image requests, elastic kept states)
#   VISION_LIMITS (1: patches/0003, bigger image/video byte limits)
#   TENSORFOLD_NGRAM_LOCK_GIB (0: patches/0004, n-gram tables left unpinned; stock: pin them all)
#   PORT (8888) HOST (0.0.0.0) NAME (tf-flash-3-exl3) SERVED_NAME (Qwen3.8-Flash-Next) FOREGROUND (0)
# The enabled patches are copied to runtime/patches; ../tensorfold.sh builds an image for each distinct set.
# Not available for EXL3 packs: --ple-on-ssd, --tp 2. Logs: docker logs -f tf-flash-3-exl3   Stop: ./flash-3/end.sh
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
NAME="${NAME:-tf-flash-3-exl3}"
FOREGROUND="${FOREGROUND:-0}"
TF_VERSION="${TF_VERSION:-0.6.6}"
DRAFTER="${DRAFTER:-incoai/Qwen3.8-Flash-Next-DFlash2}"

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
if [[ "$DRAFTER" == none ]]; then
  DRAFTS="${MTP_DRAFTS-6}"; DRAFT_CONFIDENCE="${MTP_CONFIDENCE-0.60}"
else
  DRAFTS="${DRAFTS-7}"; DRAFT_CONFIDENCE="${DRAFT_CONFIDENCE-0.5}"
fi

export TENSORFOLD_NGRAM_LOCK_GIB="${TENSORFOLD_NGRAM_LOCK_GIB:-0}"
export TENSORFOLD_MAX_BODY_MIB="${TENSORFOLD_MAX_BODY_MIB:-192}"
export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
export TENSORFOLD_VISION_WORKSPACE_MIB="${TENSORFOLD_VISION_WORKSPACE_MIB:-0}"

HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
# The pack's vision tower is a separate vision_k6.safetensors sidecar that TensorFold loads only after a one-time
# conversion to a floating tower (the output is in the mounted HF cache; flash-2's conversion is the same file). Convert once with:
#   docker run --rm --gpus all --entrypoint python3 -v ~/.cache/huggingface:/root/.cache/huggingface tensorfold:v0.6.6 \
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

# The drafter: a snapshot in the HF cache (DRAFTER_DIR: a directory inside the container instead)
DRAFTER_ARG=""
if [[ "$DRAFTER" != none ]]; then
  if [[ -n "${DRAFTER_DIR:-}" ]]; then
    DRAFTER_ARG="$DRAFTER_DIR"
  else
    _drepo="$HF_CACHE/hub/models--${DRAFTER//\//--}"
    [[ -f "$_drepo/refs/main" ]] || { echo "$DRAFTER is not in $HF_CACHE; run: hf download $DRAFTER (or DRAFTER=none for MTP)" >&2; exit 1; }
    _dsnap="$(<"$_drepo/refs/main")"
    [[ -e "$_drepo/snapshots/$_dsnap/model.safetensors" ]] || { echo "$DRAFTER: model.safetensors missing, the download is not finished" >&2; exit 1; }
    DRAFTER_ARG="/root/.cache/huggingface/hub/models--${DRAFTER//\//--}/snapshots/$_dsnap"
  fi
  (( PARALLEL >= 2 )) || { echo "DFlash2 drafts on Flash Next need PARALLEL >= 2 (or DRAFTER=none)" >&2; exit 1; }
fi

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists; stop it first: ./flash-3/end.sh" >&2
  exit 1
fi
[[ "$TF_VERSION" == 0.6.6 ]] || { echo "flash-3's patches are for TensorFold 0.6.6, not $TF_VERSION" >&2; exit 1; }

# The enabled patches, in order, into runtime/patches (one derived image per distinct set)
RUNTIME_PATCHES="$SCRIPT_DIR/runtime/patches"
rm -rf -- "$RUNTIME_PATCHES"; mkdir -p -- "$RUNTIME_PATCHES"
_use() { cp -- "$1" "$RUNTIME_PATCHES/"; }
if [[ "${TOOL_PARSER_FIX:-1}" == 1 ]]; then _use "$SCRIPT_DIR/patches/0001-tool-parser.patch"
elif [[ "${TOOL_TRACE:-0}" == 1 ]]; then _use "$SCRIPT_DIR/tools/patches/0001-tool-trace.patch"; fi
[[ "${OVERLAY:-1}" != 1 ]] || _use "$SCRIPT_DIR/patches/0002-exl3-prompt-kernels.patch"
[[ "${VISION_LIMITS:-1}" != 1 ]] || _use "$SCRIPT_DIR/patches/0003-media-limits.patch"
[[ "${TENSORFOLD_NGRAM_LOCK_GIB:-0}" == stock ]] || _use "$SCRIPT_DIR/patches/0004-ngram-unpinned.patch"
[[ "${VISION_CACHE:-1}" != 1 ]] || _use "$SCRIPT_DIR/patches/0005-vision-prefix-cache.patch"
if [[ "$DRAFTER" != none ]]; then                          # 0006 builds on 0005's multi.py and multi_fill.py
  [[ "${VISION_CACHE:-1}" == 1 ]] || { echo "the DFlash2 patch (0006) needs patches/0005: VISION_CACHE=0 needs DRAFTER=none" >&2; exit 1; }
  _use "$SCRIPT_DIR/patches/0006-dflash2-drafter.patch"
fi

wrapper=(--tf-name "$NAME" --tf-version "$TF_VERSION" --tf-docker-arg=--ulimit=stack=67108864 --tf-patches "$RUNTIME_PATCHES")
_cuda=/usr/local/lib/python3.12/dist-packages/tensorfold/cuda
if [[ "${TOOL_PARSER_FIX:-1}" == 1 || "${TOOL_TRACE:-0}" == 1 ]]; then
  wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/tools/tool_trace.py:$_cuda/tool_trace.py:ro")
  [[ "${TOOL_PARSER_FIX:-1}" != 1 ]] || wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$SCRIPT_DIR/tools/tool_calls.py:$_cuda/tool_calls.py:ro")
fi
if [[ "${TOOL_TRACE:-0}" == 1 ]]; then
  TOOL_TRACE_DIR="${TOOL_TRACE_DIR:-$SCRIPT_DIR/tool-traces}"
  mkdir -p -- "$TOOL_TRACE_DIR"
  chmod 700 -- "$TOOL_TRACE_DIR"
  TOOL_TRACE_DIR="$(cd "$TOOL_TRACE_DIR" && pwd)"
  export TENSORFOLD_TOOL_TRACE_DIR=/tool-traces
  wrapper+=(--tf-docker-arg=-v "--tf-docker-arg=$TOOL_TRACE_DIR:/tool-traces")
  echo "WARNING: tool tracing captures prompts, tool arguments and model output in $TOOL_TRACE_DIR; disable after reproducing." >&2
fi
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
if [[ -n "$DRAFTER_ARG" ]]; then serve_args+=(--drafter "$DRAFTER_ARG"); else serve_args+=(--drafter none); fi
# --mtp-drafts / --mtp-confidence set the DFlash2 drafts and chain floor too (patches/0006)
[[ -z "${DRAFTS:-}" ]] || serve_args+=(--mtp-drafts "$DRAFTS")
[[ -z "${DRAFT_CONFIDENCE:-}" ]] || serve_args+=(--mtp-confidence "$DRAFT_CONFIDENCE")

"$SCRIPT_DIR/../tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ARG" "${serve_args[@]}" \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: ./flash-3/end.sh"
fi
