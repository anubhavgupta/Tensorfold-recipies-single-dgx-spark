# Shared settings for start.sh, stop.sh and scripts/*.sh. A setting's value comes from the first of these that sets it:
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

# turboderp's EXL3 pack at 2.05 bpw (branch 2.05bpw): routed experts 2-bit, attention and shared experts 4-bit, the
# dense MLPs 3-bit, the head 5-bit (mul1 codebook), 80 GiB in 12 shards. Revision 51058cd5 is that branch's commit.
MODEL_ID="${MODEL_ID:-turboderp/GLM-5.3-Flash-exl3}"
MODEL_REVISION="${MODEL_REVISION:-51058cd551c7e570d87bd32a4adee720edce2349}"

# TensorFold main at 1.0.12 (the latest release; later commits are documentation). patches/ add EXL3 to the CUDA engine
# (upstream PR #837) and load GLM-5.3-Flash's EXL3 pack on one GPU. Zig ZIG_VERSION builds the native CUDA server.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-46a08c18608153bfc566703a88acc883bc6140f4}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-glm-exl3:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE glm-exl3"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-tf-glm53-flash}"

SERVED_NAME="${SERVED_NAME:-GLM-5.3-Flash}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8890}"
# Prompt plus reply tokens a request may use (the model's window is 1,048,576). A stream's DSA caches and KDA states grow
# with it (~1.5 GiB a stream at 65,536, ~2.6 GiB at 131,072), and the 80 GiB of weights leave ~16 GiB on a 128 GB Spark.
CONTEXT="${CONTEXT:-131072}"
# Requests decoded together. The weights are read once a round for all of them, but more rows route to more experts.
PARALLEL="${PARALLEL:-4}"
# Prompt cache (TensorFold's prefix reuse): GiB of kept prompt states; 0 lets the server size it from what is left.
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-0}"
# The checkpoint's MTP head drafts (and TensorFold's copies): verify windows of the drafted rows. DRAFTS=0: one row at a time.
DRAFTS="${DRAFTS:-1}"
MAX_TOKENS="${MAX_TOKENS:-32768}"
THINKING="${THINKING:-1}"
# generation_config.json: temperature 1.0, top_p 0.95. TOP_K empty omits --top-k.
TEMPERATURE="${TEMPERATURE:-1.0}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-10}"
export TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}"

HF_CACHE="${HF_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tf-glm53-flash}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tf-glm53-flash/logs}"
_c() { [[ -t "$1" ]] && printf '\033[%sm' "$2" || true; }
log()  { printf '%s[%s]%s %s\n' "$(_c 1 '1;36')" "$(basename "$0")" "$(_c 1 0)" "$*"; }
warn() { printf '%s[%s] WARN:%s %s\n' "$(_c 2 '1;33')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; }
die()  { printf '%s[%s] ERROR:%s %s\n' "$(_c 2 '1;31')" "$(basename "$0")" "$(_c 2 0)" "$*" >&2; exit 1; }

[[ -z "${HF_TOKEN:-}" ]] || export HF_TOKEN
model_cache_dir() { echo "$HF_CACHE/hub/models--${MODEL_ID//\//--}"; }
snapshot_rev() {
  if [[ -n "$MODEL_REVISION" ]]; then echo "$MODEL_REVISION"
  else cat "$(model_cache_dir)/refs/main" 2>/dev/null; fi
}
snapshot_dir() { echo "$(model_cache_dir)/snapshots/$(snapshot_rev)"; }
weights_ready() {
  local snap; snap="$(snapshot_dir)"
  [[ -f "$snap/config.json" && -f "$snap/tokenizer.json" && -f "$snap/model.safetensors.index.json" ]] || return 1
  python3 -I -c 'import json, os, sys; d = sys.argv[1]; m = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]; sys.exit(0 if all(os.path.exists(os.path.join(d, f)) for f in set(m.values())) else 1)' "$snap"
}
