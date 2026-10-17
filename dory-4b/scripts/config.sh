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

# NVIDIA's NVFP4 export (W4A16, group 16, one 3.2 GB model.safetensors). Revision 41f02335 is the Hub commit read here.
MODEL_ID="${MODEL_ID:-nvidia/NVIDIA-Nemotron-Labs-Dory1.0-4B-NVFP4}"
MODEL_REVISION="${MODEL_REVISION:-41f02335e1a1ad3289b8b32485e2265cbe97cfdf}"

# TensorFold main at 1.0.12 (the latest release; later commits are documentation). patches/ register the Dory family;
# engine/ is libdory (the forward pass) and the lane backend. Zig ZIG_VERSION builds the native CUDA server.
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
TF_REF="${TF_REF:-46a08c18608153bfc566703a88acc883bc6140f4}"
ZIG_VERSION="${ZIG_VERSION:-0.17.0}"
ZIG_SHA256="${ZIG_SHA256:-9e8d11661d4ae3bd57702a3832781e23ad151dde5798e16a5ccd503f65234ff8}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
TF_TAG="zig-${TF_REF:0:7}"
IMAGE="${IMAGE:-tensorfold-dory:${TF_TAG}}"
image_hash() {
  ( cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
    cat patches/*.patch engine/* 2>/dev/null
    sed -n "/<<'DOCKERFILE'/,/^DOCKERFILE\$/p" scripts/prepare.sh
    echo "$TF_REPO $TF_REF zig-$ZIG_VERSION $ZIG_SHA256 $BASE_IMAGE dory"
  ) | sha256sum | cut -c1-12
}
CONTAINER_NAME="${CONTAINER_NAME:-tf-dory-4b}"

SERVED_NAME="${SERVED_NAME:-Dory1.0-4B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
# The model's native window (max_position_embeddings) for every stream. Full-attention rows are mapped as a stream grows
# (56 KiB a token with two per-loop caches, 36 KiB with KV_MODE=last_loop), so streams cost what they use.
CONTEXT="${CONTEXT:-262144}"
# Requests decoded together, 1 to 16: their rows share one forward, so the weights are read once for all of them. Each
# holds 0.33 GiB of sliding-window rings from the start.
PARALLEL="${PARALLEL:-16}"
# Recurrent loops (1-4): NVIDIA recommends 2; 1 is fastest, 4 the most accurate on agentic tasks. Each loop runs the
# 40-sublayer recurrent block again, so decode time grows with it (see README).
LOOPS="${LOOPS:-2}"
# per_loop: a KV cache per loop of every recurrent attention layer (the reference mode); last_loop: one, overwritten by
# each loop (the cache of a one-loop model whatever LOOPS is; NVIDIA reports accuracy close to per_loop).
KV_MODE="${KV_MODE:-per_loop}"
# Prompt cache (TensorFold's prefix reuse): GiB of kept prompt states (every attention cache's rows at a mark).
PROMPT_CACHE_GIB="${PROMPT_CACHE_GIB:-8}"
# Speculation by copying: the round loop proposes tokens copied from the context (suffix matches), verified in windows of
# up to 8 rows. Free when nothing matches; large gains on edits and repeated code. DRAFTS=0 decodes one row at a time.
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
STATE_DIR="${STATE_DIR:-$HOME/.local/state/tf-dory-4b}"
LOG_DIR="${LOG_DIR:-$HOME/.cache/tf-dory-4b/logs}"

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
weights_ready() {
  local snap; snap="$(model_cache_dir)/snapshots/$(snapshot_rev)"
  [[ -f "$snap/config.json" && -f "$snap/tokenizer.json" && -f "$snap/model.safetensors" ]]
}
