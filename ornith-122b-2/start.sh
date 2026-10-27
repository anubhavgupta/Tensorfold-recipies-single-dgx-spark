#!/usr/bin/env bash
# Serve Ornith-1.5-122B (ornith-ai's NVFP4 checkpoint, as the SGLang recipe ../ornith-1.5-122B serves it) with
# TensorFold 1.1.7's Zig engine on one DGX Spark, drafting with the DFlash2 model incoai/Ornith-1.5-122B-DFlash2.
# patches/0001 reads the checkpoint's NVFP4 experts, 0002 opens the DFlash2 drafter for qwen3_5_moe (see README).
# ./start.sh prepares the image if needed, launches tensorfold-native in a container, waits for "serving" and sends a
# smoke request. Ready in ~60 s (~90 s on the first start, which times the draft windows).
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # prepare if needed, launch
#   ./start.sh restart --context 65536
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
#   FOREGROUND=1 ./start.sh            # follow the log after the start
# Extra arguments go to tensorfold-native serve after the defaults, so they win.
# Settings (environment or ./.env, defaults in scripts/config.sh):
#   serving  CONTEXT (262144), PARALLEL (4), DRAFTER (dflash|none), KV_CACHE (bf16|fp8, default bf16), VISION (1),
#            PROMPT_CACHE_GIB (0 = auto), MAX_TOKENS (32768), THINKING (1), TEMPERATURE/TOP_P/TOP_K (0.6/0.95/20),
#            TENSORFOLD_PREFILL_SHARE (0.85), SERVED_NAME, HOST, PORT (8888)
#   files    MODEL_ID, MODEL_REVISION, DRAFTER_ID, HF_CACHE, STATE_DIR, LOG_DIR
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, CONTAINER_NAME
#   engine   every TENSORFOLD_* and TF_QWEN_* variable goes into the container (TENSORFOLD_API_KEY by name only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 262144 ]] || die "CONTEXT is a token count up to 262144, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
for v in THINKING VISION; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done
[[ "$DRAFTER" =~ ^(dflash|none)$ ]] || die "DRAFTER is dflash or none, not $DRAFTER"
[[ "$KV_CACHE" =~ ^(bf16|fp8)$ ]] || die "KV_CACHE is bf16 or fp8, not $KV_CACHE"

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P")
[[ "$PROMPT_CACHE_GIB" != 0 ]] && SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ -n "$TOP_K" ]] && SERVE_ARGS+=(--top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$VISION" == 1 ]] && SERVE_ARGS+=(--vision)
[[ "$KV_CACHE" == fp8 ]] && SERVE_ARGS+=(--kv-cache fp8)
snap="hub/models--${MODEL_ID//\//--}/snapshots/$(basename "$(snapshot_dir)")"
case "$DRAFTER" in
  dflash) SERVE_ARGS+=(--drafter "/root/.cache/huggingface/hub/models--${DRAFTER_ID//\//--}/snapshots/$(basename "$(drafter_dir)")") ;;
  none) SERVE_ARGS+=(--no-drafts) ;;
esac

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
URL="http://$API_HOST:$PORT"
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }

mkdir -p "$STATE_DIR/tensorfold" "$LOG_DIR"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if [[ "$MODE" == start ]] && running; then
    log "$CONTAINER_NAME is already running on port $PORT: ./start.sh restart restarts it, ./end.sh stops it."
    exit 0
  fi
  [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$(image_hash)" ]] && weights_ready ||
    ./scripts/prepare.sh
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  # on unified memory the checkpoint's cached pages compete with the GPU's allocations: drop them before the load
  python3 -I -c 'import glob, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(os.path.realpath(f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' "$(snapshot_dir)" || true
  free_gib=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  (( free_gib >= 105 )) || warn "only ${free_gib} GiB MemAvailable: the model takes ~71 GiB plus ~7 GiB a full-context stream (stop other GPU work)"
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then die "port $PORT is in use: stop the other server or set PORT"; fi
fi

ENV_ARGS=()
while IFS='=' read -r k _; do
  [[ "$k" == TENSORFOLD_API_KEY ]] && { ENV_ARGS+=(-e TENSORFOLD_API_KEY); continue; }
  [[ "$k" =~ ^(TENSORFOLD_|TF_QWEN_)[A-Z0-9_]*$ ]] && ENV_ARGS+=(-e "$k=${!k}")
done < <(env)
# the window costs the engine times on its first start are kept here (keyed by GPU, driver, checkpoint and drafter)
CMD=(docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host --ulimit memlock=-1 --network host
     -v "$HF_CACHE":/root/.cache/huggingface:ro -v "$STATE_DIR/tensorfold":/root/.cache/tensorfold
     "${ENV_ARGS[@]}" --entrypoint tensorfold-native "$IMAGE"
     serve "/root/.cache/huggingface/$snap" --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

log "Starting $SERVED_NAME: $PARALLEL streams x $CONTEXT tokens, drafter $DRAFTER, KV $KV_CACHE, port $PORT"
"${CMD[@]}" >/dev/null
LOG="$LOG_DIR/$(date +%Y%m%d-%H%M%S).log"
( docker logs -f "$CONTAINER_NAME" >"$LOG" 2>&1 & )
ln -sf "$LOG" "$LOG_DIR/latest.log"
t0=$SECONDS
until grep -q "serving $SERVED_NAME" "$LOG" 2>/dev/null; do
  running || { tail -30 "$LOG" >&2; die "the server exited during startup (log: $LOG)"; }
  (( SECONDS - t0 < WAIT_TIMEOUT )) || die "no 'serving' line after ${WAIT_TIMEOUT}s (log: $LOG)"
  sleep 2
done
grep -E "startup phases|streams at once|window costs" "$LOG" | sed 's/^/  /'
reply=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H 'content-type: application/json' ${TENSORFOLD_API_KEY:+-H "Authorization: Bearer $TENSORFOLD_API_KEY"} \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null) || reply=""
[[ -n "$reply" ]] || die "the smoke request failed (log: $LOG)"
log "Ready at $URL/v1 (model $SERVED_NAME) in $((SECONDS - t0)) s. Smoke reply: $reply"
log "Log: $LOG"
[[ "${FOREGROUND:-0}" == 1 ]] && exec tail -f "$LOG"
exit 0
