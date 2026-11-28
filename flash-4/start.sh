#!/usr/bin/env bash
# Serve Qwen3.8-Flash-Next (turboderp's EXL3 pack, 4.05 bpw) with TensorFold's latest release (v1.3.5, native Zig engine) on one DGX
# Spark: 262,144-token window, vision, tool calls, thinking, MTP or DFlash2 drafts. OpenAI API on port 8888, model Qwen3.8-Flash-Next.
# ./start.sh prepares the image and the model folder if needed (all from files already in the caches), starts the container,
# waits for "serving" and sends a smoke request. Ready in ~1 min.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
#   FOREGROUND=1 ./start.sh            # run the container in the foreground (what the systemd unit does)
# Settings (environment or ./.env, defaults in scripts/config.sh):
#   serving  CONTEXT (262144), PARALLEL (4; at most 5 whole windows fit), KV_PAGES (256; 0 = whole windows: streams beside the first share one pool of
#            pages of this many rows, e.g. 256, so more streams than whole windows are admitted while memory lasts), DRAFTER
#            (dflash|mtp|none; default dflash), VISION (1), VISION_MAX_IMAGES (50), VISION_IMAGE_TOKENS (65536), THINKING (1), MAX_TOKENS,
#            TEMPERATURE/TOP_P/TOP_K, PROMPT_CACHE_GIB (0: the memory left), TENSORFOLD_MEMORY_RESERVE_GIB (2), PORT, SERVED_NAME
#   files    MODEL_PATH, HF_CACHE, STATE_DIR, LOG_DIR, IMAGE, TF_REF
#   engine   every TENSORFOLD_* and TF_* variable goes into the container (TENSORFOLD_API_KEY by name only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

WAIT_TIMEOUT="${WAIT_TIMEOUT:-1500}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 262144 ]] || die "CONTEXT is a token count up to 262144, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
for v in THINKING VISION; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$DRAFTER" =~ ^(dflash|mtp|none)$ ]] || die "DRAFTER is dflash, mtp or none, not $DRAFTER"

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P" --top-k "$TOP_K")
[[ "$PROMPT_CACHE_GIB" != 0 ]] && SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision --vision-max-images "$VISION_MAX_IMAGES" --vision-image-tokens "$VISION_IMAGE_TOKENS" --vision-image-share fixed)
case "$DRAFTER" in
  dflash) SERVE_ARGS+=(--drafter "/root/.cache/huggingface/hub/models--${DRAFTER_ID//\//--}/snapshots/$(basename "$(drafter_snapshot)")") ;;
  none) SERVE_ARGS+=(--no-drafts) ;;
esac
# DFlash2 context copies: 0 is best on the content suite; 1 makes copy-heavy (echo) replies ~2.4x faster
export TF_FLASHNEXT_DFLASH_COPIES="${TF_FLASHNEXT_DFLASH_COPIES:-0}"
# Fused BF16 hyper-connection read-out on prompt passes >= 768 rows (bit-identical, +3-4% prefill)
export TF_FLASHNEXT_READOUT_FUSED="${TF_FLASHNEXT_READOUT_FUSED:-1}"
export TF_QMMF_HALF_BELOW="${TF_QMMF_HALF_BELOW:-0}"
# A call to an unoffered tool with this prefix (pi exposes mcp__* only inside codemode) is returned as a call, so the client
# answers "tool not found" and the model retries, instead of the markup ending the turn as text (patch 0015; "" = stock)
export TF_TOOL_UNOFFERED_PREFIX="${TF_TOOL_UNOFFERED_PREFIX-mcp__}"
# Mixed prefill+decode interleaving: 0.85 gives 4x concurrent decode throughput during long prefills (+4.3s on 65K)
export TENSORFOLD_PREFILL_SHARE="${TENSORFOLD_PREFILL_SHARE:-0.85}"
export TF_FLASHNEXT_CHUNK="${TF_FLASHNEXT_CHUNK:-3072}"
export TF_FLASHNEXT_CHUNK_SOLO="${TF_FLASHNEXT_CHUNK_SOLO:-3072}"
[[ "$KV_PAGES" != 0 ]] && export TF_FLASHNEXT_KV_PAGES="$KV_PAGES"

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
URL="http://$API_HOST:$PORT"
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }

mkdir -p "$STATE_DIR/tensorfold" "$LOG_DIR"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if [[ "$MODE" == start ]] && running; then
    log "$CONTAINER_NAME is already running on port $PORT: ./start.sh restart restarts it, ./end.sh stops it."
    exit 0
  fi
  [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$(image_hash)" && -f "$MODEL_PATH/model.safetensors.index.json" ]] ||
    ./scripts/prepare.sh
  [[ "$DRAFTER" != dflash || -f "$(drafter_snapshot)/model.safetensors" ]] || die "$DRAFTER_ID is not in $HF_CACHE (DRAFTER=mtp uses the checkpoint's own head)"
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  free_gib=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  (( free_gib >= 100 )) || warn "only ${free_gib} GiB MemAvailable: the model takes ~67 GiB plus ~8 GiB a full-window stream (stop other GPU work)"
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then die "port $PORT is in use: stop the other server or set PORT"; fi
fi

ENV_ARGS=()
while IFS='=' read -r k _; do
  [[ "$k" == TENSORFOLD_API_KEY ]] && { ENV_ARGS+=(-e TENSORFOLD_API_KEY); continue; }
  [[ "$k" =~ ^(TENSORFOLD_|TF_)[A-Z0-9_]*$ ]] && ENV_ARGS+=(-e "$k=${!k}")
done < <(env)
DETACH=(-d); [[ "${FOREGROUND:-0}" == 1 ]] && DETACH=(--rm)   # FOREGROUND=1 (systemd): the container runs in this process
CMD=(docker run "${DETACH[@]}" --name "$CONTAINER_NAME" --gpus all --ipc=host --ulimit memlock=-1 --network host --stop-timeout 30
     -v "$HF_CACHE":/root/.cache/huggingface:ro -v "$MODEL_PATH":/model:ro -v "$STATE_DIR/tensorfold":/root/.cache/tensorfold
     "${ENV_ARGS[@]}" --entrypoint tensorfold-native "$IMAGE"
     serve /model --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

log "Starting $SERVED_NAME: $PARALLEL streams x $CONTEXT tokens, drafter $DRAFTER, port $PORT"
[[ "${FOREGROUND:-0}" != 1 ]] || exec "${CMD[@]}"
"${CMD[@]}" >/dev/null
LOG="$LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
( docker logs -f "$CONTAINER_NAME" >"$LOG" 2>&1 & )
ln -sf "$LOG" "$LOG_DIR/latest.log"
t0=$SECONDS
until grep -q "serving $SERVED_NAME" "$LOG" 2>/dev/null; do
  running || { grep -v "loaded layer" "$LOG" | tail -30 >&2; die "the server exited during startup (log: $LOG)"; }
  (( SECONDS - t0 < WAIT_TIMEOUT )) || die "no 'serving' line after ${WAIT_TIMEOUT}s (log: $LOG)"
  sleep 2
done
grep -E "CUDA sm_|serving|KV page groups" "$LOG" | cut -c1-300 | sed 's/^/  /'
reply=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H 'content-type: application/json' ${TENSORFOLD_API_KEY:+-H "Authorization: Bearer $TENSORFOLD_API_KEY"} \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null) || reply=""
[[ -n "$reply" ]] || die "the smoke request failed (log: $LOG)"
log "Ready at $URL/v1 (model $SERVED_NAME) in $((SECONDS - t0)) s. Smoke reply: $reply"
log "Log: $LOG"
exit 0
