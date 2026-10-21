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

# Ornith's own MLX affine 4-bit group-64 export (router gates 8-bit), the only Qwen3.5-MoE format TensorFold's CUDA
# engine reads (the NVFP4 checkpoint the SGLang recipe serves is not one). 68.7 GB. Empty revision: refs/main.
MODEL_ID="${MODEL_ID:-ornith-ai/Ornith-1.5-122B-MLX-4bit}"
MODEL_REVISION="${MODEL_REVISION:-}"
# The DFlash2 drafter (bf16; TensorFold quantizes its projections to affine 4-bit at load). DRAFTER=none: no drafting.
DRAFTER_ID="${DRAFTER_ID:-incoai/Ornith-1.5-122B-DFlash2}"
DRAFTER="${DRAFTER:-dflash}"

# TensorFold v1.1.1 (the latest release). patches/ add DFlash2 drafting to the qwen3_5_moe family.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-b10231b94f8adf3d4132acebdc0a1ac87e53a838}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-ornith122:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE ornith122"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-tf-ornith-122b}"

SERVED_NAME="${SERVED_NAME:-Ornith-1.5-122B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
CONTEXT="${CONTEXT:-262144}"
PARALLEL="${PARALLEL:-4}"
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-0}"
VISION="${VISION:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
# The card's coding profile (also generation_config.json and the drafter's evaluation setting).
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-8}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tf-ornith-122b}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tf-ornith-122b/logs}"
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
cache_dir() { echo "$HF_CACHE/hub/models--${1//\//--}"; }
snapshot_of() {   # snapshot dir of repo $1 at revision $2 (empty: refs/main)
  local rev="${2:-}"; [[ -n "$rev" ]] || rev=$(cat "$(cache_dir "$1")/refs/main" 2>/dev/null)
  echo "$(cache_dir "$1")/snapshots/$rev"
}
snapshot_dir() { snapshot_of "$MODEL_ID" "$MODEL_REVISION"; }
drafter_dir() { snapshot_of "$DRAFTER_ID" ""; }
weights_ready() {
  local snap; snap="$(snapshot_dir)"
  [[ -f "$snap/config.json" && -f "$snap/tokenizer.json" && -f "$snap/model.safetensors.index.json" ]] || return 1
  python3 -I -c 'import json, os, sys; d = sys.argv[1]; m = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]; sys.exit(0 if all(os.path.exists(os.path.join(d, f)) for f in set(m.values())) else 1)' "$snap" || return 1
  [[ "$DRAFTER" == none || -f "$(drafter_dir)/model.safetensors" ]]
}
