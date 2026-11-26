# Shared settings for start.sh, end.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
#   1. the environment: `PORT=9000 ./start.sh`
#   2. ./.env (KEY=value lines, read, never run)
#   3. the defaults below
if [[ -f .env ]]; then
  while IFS= read -r _line || [[ -n "$_line" ]]; do
    [[ "$_line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    _key=${BASH_REMATCH[2]}; _value=${BASH_REMATCH[3]}
    if [[ "$_value" =~ ^\"([^\"]*)\"[[:space:]]*(#.*)?$ || "$_value" =~ ^\'([^\']*)\'[[:space:]]*(#.*)?$ ]]; then
      _value=${BASH_REMATCH[1]}
    else
      _value=${_value%%#*}; _value=${_value%"${_value##*[![:space:]]}"}
    fi
    [[ -n "${!_key+set}" ]] || export "$_key=$_value"
  done < .env
fi
unset _line _key _value

# The pack and the drafter, both in the Hugging Face cache already (flash-3's): nothing is downloaded.
REPO="${REPO:-turboderp/Qwen3.8-Flash-Next-exl3}"
REVISION="${REVISION:-4.05bpw_h6_ng6}"
DRAFTER_ID="${DRAFTER_ID:-incoai/Qwen3.8-Flash-Next-DFlash2}"
DRAFTER="${DRAFTER:-dflash}"                       # dflash | mtp (the checkpoint's own MTP head) | none
MODEL_PATH="${MODEL_PATH:-$HOME/projects/models/Qwen3.8-Flash-Next-exl3-4.05bpw}"   # build_model_dir.sh makes it

# TensorFold's latest release (v1.3.5, the native Zig engine) plus patches/*.patch, built in NVIDIA's PyTorch container.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-02b324ae9767253d9911d753d5c835a5070eed1b}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
JPEG_TURBO_REF="${JPEG_TURBO_REF:-3.1.4.1}"
WEBP_REF="${WEBP_REF:-v1.6.0}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
IMAGE="${IMAGE:-tensorfold-flash4:zig-${TF_REF:0:7}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $JPEG_TURBO_REF $WEBP_REF $BASE_IMAGE flash4"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-tf-flash-4}"

SERVED_NAME="${SERVED_NAME:-Qwen3.8-Flash-Next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-262144}"
PARALLEL="${PARALLEL:-4}"                       # full-window streams; 5 fit (7.9 GiB each); KV_PAGES shares the memory of more
KV_PAGES="${KV_PAGES:-256}"                     # >0: streams beside the first take pages of this many rows from one pool (TF_FLASHNEXT_KV_PAGES)
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-0}"       # 0: what is left after the streams
VISION="${VISION:-1}"
VISION_MAX_IMAGES="${VISION_MAX_IMAGES:-50}"
VISION_IMAGE_TOKENS="${VISION_IMAGE_TOKENS:-65536}"   # shared by a request's images, each 722 tokens at 50 up to the 65,536 maximum (fixed share keeps the prefix cache)
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
if [[ "$THINKING" == 1 ]]; then _t=1.0; _p=0.95; else _t=0.7; _p=0.80; fi
TEMPERATURE="${TEMPERATURE:-$_t}"
TOP_P="${TOP_P:-$_p}"
TOP_K="${TOP_K:-20}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"
export TENSORFOLD_MAX_BODY_MIB="${TENSORFOLD_MAX_BODY_MIB:-192}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tf-flash-4}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tf-flash-4/logs}"
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }
cache_dir() { echo "$HF_CACHE/hub/models--${1//\//--}"; }
drafter_snapshot() { echo "$(cache_dir "$DRAFTER_ID")/snapshots/$(cat "$(cache_dir "$DRAFTER_ID")/refs/main" 2>/dev/null)"; }
