#!/usr/bin/env bash
# Serve ornith-ai/Ornith-1.5-122B-NVFP4 (ModelOpt NVFP4 MoE + FP8 attention/GDN, vision tower bf16) with SGLang on one
# DGX Spark (GB10, 128 GB unified memory), drafting with the DFlash2 block-diffusion drafter incoai/Ornith-1.5-122B-DFlash2.
# Download first:  hf download ornith-ai/Ornith-1.5-122B-NVFP4 && hf download incoai/Ornith-1.5-122B-DFlash2
# The host HF cache (~/.cache/huggingface) is mounted into the container; the server runs offline.
#
# Usage: ./ornith-1.5-122B/start.sh [extra sglang args]   (appended last: argparse last-wins, so they override defaults)
# Env (a .env beside this script is read first; real environment variables win):
#   SPEC (dflash; off = plain decoding)   DF_BLOCK (6: verify width; the drafter trained at 16, but every verified
#     token pulls in more MoE experts, so 6 is the measured single-stream optimum on GB10, see README)
#   DF_WINDOW (off: compact draft-attention window; saves draft attention reads at long context, not memory)
#   PARALLEL (4: concurrent streams, each able to reach CONTEXT at once)   CONTEXT (262144, native max)
#   MEM_FRACTION (0.91: share of MemTotal SGLang may take; >0.93 has frozen GB10 boxes)
#   MAMBA_SLOTS (auto: GDN state slots; auto = everything left after PARALLEL x CONTEXT of KV, the extra slots are the
#     hybrid prefix cache)   CHUNKED_PREFILL (8192; 16384 stalled a 32K prefill with MemAvailable at 2 GB)
#   PREFILL_DECODE_INTERVAL (16: decode rounds between prefill chunks while other streams decode; a stream gets
#     ~12 tok/s instead of ~0.6 during a long prefill, which is ~15% slower; 0 = stock)
#   CUDA_GRAPH_BS (auto: largest decode batch with a CUDA graph)
#   VISION (1; 0 = text only, skips the vision tower)   THINKING (1; per request: chat_template_kwargs.enable_thinking)
#   TEMPERATURE/TOP_P/TOP_K/MIN_P (0.6/0.95/20/0.0: the card's coding profile, used when a request omits them)
#   IMAGE (lmsysorg/sglang:dev-cu13 pinned by digest; stock, no patches needed)
#   PORT (8888) HOST (0.0.0.0) NAME (sglang-ornith-122b) SERVED_NAME (Ornith-1.5-122B) FOREGROUND (0) DROP_CACHES (1)
# Logs: docker logs -f sglang-ornith-122b      Stop: ./ornith-1.5-122B/end.sh
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

REPO="${REPO:-ornith-ai/Ornith-1.5-122B-NVFP4}"
DRAFTER="${DRAFTER:-incoai/Ornith-1.5-122B-DFlash2}"
SPEC="${SPEC:-dflash}"
DF_BLOCK="${DF_BLOCK:-6}"
DF_WINDOW="${DF_WINDOW:-off}"
PARALLEL="${PARALLEL:-4}"
CONTEXT="${CONTEXT:-262144}"
MEM_FRACTION="${MEM_FRACTION:-0.91}"
MAMBA_SLOTS="${MAMBA_SLOTS:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
PREFILL_DECODE_INTERVAL="${PREFILL_DECODE_INTERVAL:-16}"
CUDA_GRAPH_BS="${CUDA_GRAPH_BS:-auto}"
VISION="${VISION:-1}"
THINKING="${THINKING:-1}"
TEMPERATURE="${TEMPERATURE:-0.6}"
TOP_P="${TOP_P:-0.95}"
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0.0}"
DROP_CACHES="${DROP_CACHES:-1}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-sglang-ornith-122b}"
SERVED_NAME="${SERVED_NAME:-Ornith-1.5-122B}"
FOREGROUND="${FOREGROUND:-0}"
# dev-cu13 as of 2026-10-03 (sglang main 65f759144): has DFlash2DraftModel, DFLASH aux-hidden capture for
# Qwen3_5MoeForConditionalGeneration, extra_buffer_lazy for DFLASH verify and quantized lm_head loading.
IMAGE="${IMAGE:-lmsysorg/sglang@sha256:035f29e91cfdf3b7d031c28be43ef7bddc93bb1f025e35c02c62ac732b8ba59b}"
HF_HOME_HOST="${HF_HOME_HOST:-$HOME/.cache/huggingface}"
CACHE_DIR="$SCRIPT_DIR/.cache"

die() { echo "error: $*" >&2; exit 1; }
snapshot() {   # snapshot dir of a cached repo (refs/main), or empty
  local base="$HF_HOME_HOST/hub/models--${1//\//--}"
  [[ -f "$base/refs/main" ]] && echo "$base/snapshots/$(cat "$base/refs/main")"
}

# ---- memory model (measured on this box, see README "Memory") ----------------------------------------------------
# KV per token: 12 attention layers x K,V x 2 KV heads x 256 x 1 B (fp8) = 12 KiB target; DFlash2 adds its 6 layers x
# K,V x 8 heads x 128 x 1 B = 12 KiB in the same token slots. GDN state per slot (bf16 SSM): 36 layers x
# (64 heads x 128 x 128 x 2 B + conv 12288 x 3 x 2 B) = 74.5 MiB. With spec decoding every running request also
# holds DF_BLOCK intermediate GDN states for the verify (a separate buffer).
KV_TARGET=12288; KV_DRAFT=12288; STATE=78151680
WEIGHTS_GIB="${WEIGHTS_GIB:-68.1}"        # target as loaded ("Load weight end ... mem usage=68.12 GB")
DRAFT_GIB="${DRAFT_GIB:-1.2}"         # drafter as loaded (1.14 GB)
OVERHEAD_GIB="${OVERHEAD_GIB:-8.5}"       # CUDA context + OS share not free at load (8.0) + graphs/activations
SLOTS_PER_REQ=4                           # extra_buffer_lazy + overlap scheduler

D=0; KV=$KV_TARGET
if [[ "$SPEC" == dflash ]]; then D=$DF_BLOCK; KV=$((KV_TARGET + KV_DRAFT)); else DRAFT_GIB=0; fi
[[ "$SPEC" == dflash || "$SPEC" == off ]] || die "SPEC must be dflash or off"
MAX_TOTAL_TOKENS=$(( PARALLEL * (CONTEXT + D) ))
MEM_TOTAL_KB=$(awk '/^MemTotal:/{print $2}' /proc/meminfo)
read -r BUDGET_GIB AUTO_SLOTS MAX_PAR < <(awk -v memkb="$MEM_TOTAL_KB" -v f="$MEM_FRACTION" -v w="$WEIGHTS_GIB" \
  -v dw="$DRAFT_GIB" -v oh="$OVERHEAD_GIB" -v N="$PARALLEL" -v C="$CONTEXT" -v D="$D" -v kv="$KV" -v st="$STATE" \
  -v S="$SLOTS_PER_REQ" 'BEGIN{ G=1073741824
    rest = memkb/1048576*f - w - dw - oh
    left = rest - N*(C+D)*kv/G - (N+1)*D*st/G
    slots = int(left*G/st) - 1
    maxn = int((rest - (D+1)*st/G) / (((C+D)*kv + (D+S)*st)/G))
    printf "%.1f %d %d\n", rest, slots, maxn }')
if [[ "$MAMBA_SLOTS" == auto ]]; then
  (( AUTO_SLOTS >= PARALLEL * SLOTS_PER_REQ )) ||
    die "$PARALLEL x $CONTEXT-token streams do not fit ~${BUDGET_GIB} GiB (max ~$MAX_PAR at MEM_FRACTION=$MEM_FRACTION)"
  MAMBA_SLOTS=$AUTO_SLOTS
fi

# ---- inputs ------------------------------------------------------------------------------------------------------
MODEL_SNAP="$(snapshot "$REPO")"
[[ -n "$MODEL_SNAP" && -f "$MODEL_SNAP/config.json" ]] || die "$REPO not in $HF_HOME_HOST; run: hf download $REPO"
if [[ "$SPEC" == dflash ]]; then
  DRAFT_SNAP="$(snapshot "$DRAFTER")"
  [[ -n "$DRAFT_SNAP" && -f "$DRAFT_SNAP/model.safetensors" ]] || die "$DRAFTER not cached; run: hf download $DRAFTER"
fi
mkdir -p "$CACHE_DIR"/{triton,flashinfer,inductor}

# Server default sampling: SGLang takes defaults from generation_config.json (--sampling-defaults model); mount a
# patched copy over the snapshot's inside the container (the host cache is untouched). Request values always win.
GEN_CFG="$CACHE_DIR/generation_config.json"
python3 - "$MODEL_SNAP/generation_config.json" "$GEN_CFG" <<EOF
import json, sys
try: cfg = json.load(open(sys.argv[1]))
except FileNotFoundError: cfg = {}
cfg.update(do_sample=True, temperature=$TEMPERATURE, top_p=$TOP_P, top_k=$TOP_K, min_p=$MIN_P, repetition_penalty=1.0)
json.dump(cfg, open(sys.argv[2], "w"), indent=2)
EOF
CTR_SNAP="/root/.cache/huggingface/${MODEL_SNAP#"$HF_HOME_HOST"/}"

ARGS=(
  --model-path "$REPO" --served-model-name "$SERVED_NAME" --host "$HOST" --port "$PORT"
  --trust-remote-code --context-length "$CONTEXT"
  --mem-fraction-static "$MEM_FRACTION" --max-running-requests "$PARALLEL" --max-total-tokens "$MAX_TOTAL_TOKENS"
  --kv-cache-dtype fp8_e4m3 --attention-backend flashinfer --moe-runner-backend flashinfer_cutlass
  --mamba-ssm-dtype bfloat16 --mamba-radix-cache-strategy extra_buffer_lazy --max-mamba-cache-size "$MAMBA_SLOTS"
  --chunked-prefill-size "$CHUNKED_PREFILL" --prefill-decode-interval "$PREFILL_DECODE_INTERVAL" --disable-prefill-cuda-graph
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder --sampling-defaults model
  --enable-metrics --enable-cache-report --sleep-on-idle
)
[[ "$THINKING" == 1 ]] && ARGS+=(--default-chat-template-kwargs '{"preserve_thinking": true}') \
                       || ARGS+=(--default-chat-template-kwargs '{"enable_thinking": false}')
[[ "$VISION" == 1 ]] || ARGS+=(--language-only)
[[ "$CUDA_GRAPH_BS" == auto ]] || ARGS+=(--cuda-graph-max-bs "$CUDA_GRAPH_BS")
if [[ "$SPEC" == dflash ]]; then
  ARGS+=(--speculative-algorithm DFLASH --speculative-draft-model-path "$DRAFTER" --speculative-num-draft-tokens "$DF_BLOCK")
  [[ "$DF_WINDOW" == off ]] || ARGS+=(--speculative-draft-window-size "$DF_WINDOW")
fi
ARGS+=("$@")

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]] && die "$NAME is already running (./end.sh first)"
  docker rm "$NAME" >/dev/null
fi
others=$(docker ps --format '{{.Names}}' | grep -v -x "$NAME" || true)
[[ -z "$others" ]] || echo "warning: other containers are running ($others); this model needs ~110 GB" >&2

echo "model     $REPO  ($MODEL_SNAP)"
echo "spec      $SPEC${D:+ (block $D)}  drafter ${DRAFT_SNAP:-none}"
echo "streams   $PARALLEL x $CONTEXT tokens; KV pool $MAX_TOTAL_TOKENS tokens (~$(( MAX_TOTAL_TOKENS * KV / 1073741824 )) GiB)"
echo "GDN pool  $MAMBA_SLOTS slots (~$(( MAMBA_SLOTS * STATE / 1073741824 )) GiB; $((PARALLEL*SLOTS_PER_REQ)) running, rest prefix cache)"
echo "budget    ~${BUDGET_GIB} GiB for pools at MEM_FRACTION=$MEM_FRACTION (max ~$MAX_PAR full-context streams)"
echo "sampling  temperature=$TEMPERATURE top_p=$TOP_P top_k=$TOP_K min_p=$MIN_P (request values win); thinking=$THINKING"

ENV_ARGS=()
for e in ${DOCKER_ENV:-}; do ENV_ARGS+=(-e "$e"); done   # DOCKER_ENV="K=V K2=V2": extra container env
PRE=""
[[ "$DROP_CACHES" == 1 ]] && PRE='sync; echo 3 > /proc/sys/vm/drop_caches || true; '
docker run -d --name "$NAME" --network host --ipc host --privileged --gpus all --shm-size 32g \
  --cpuset-cpus "${CPUSET:-5-9,15-19}" \
  -e HF_HOME=/root/.cache/huggingface -e HF_HUB_OFFLINE=1 \
  -e TRITON_CACHE_DIR=/cache/triton -e FLASHINFER_WORKSPACE_BASE=/cache/flashinfer \
  -e TORCHINDUCTOR_CACHE_DIR=/cache/inductor -e SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN=1 \
  "${ENV_ARGS[@]}" \
  -v "$HF_HOME_HOST:/root/.cache/huggingface" -v "$GEN_CFG:$CTR_SNAP/generation_config.json:ro" \
  -v "$CACHE_DIR:/cache" \
  --entrypoint bash "$IMAGE" -c "$PRE"'exec python3 -m sglang.launch_server "$@"' bash "${ARGS[@]}" >/dev/null

echo "started $NAME; waiting for the server (first boot compiles kernels and captures CUDA graphs: several minutes)"
if [[ "$FOREGROUND" == 1 ]]; then exec docker logs -f "$NAME"; fi
api_host="$HOST"; [[ "$HOST" == 0.0.0.0 ]] && api_host=127.0.0.1
until docker logs "$NAME" 2>&1 | grep -q "The server is fired up"; do
  [[ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null)" == true ]] ||
    { docker logs --tail 60 "$NAME" 2>&1; die "server exited during startup"; }
  sleep 10
done
docker logs "$NAME" 2>&1 | grep -E "Load weight end|KV Cache is allocated|Mamba Cache is allocated|max_total_num_tokens|Capture cuda graph end|DFLASH draft runner" || true
echo "ready: http://$api_host:$PORT/v1  model=$SERVED_NAME   (logs: docker logs -f $NAME)"
