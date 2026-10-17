#!/usr/bin/env bash
# Serve GLM-5.3-Flash (turboderp's EXL3 2.05bpw pack) with TensorFold's Zig engine on one DGX Spark.
# ./start.sh prepares the image and the checkpoint if needed, launches tensorfold-native (the GLM family at one rank,
# TF_GLM_WORLD=1, on the EXL3 kernels) in a container, waits for "serving" and sends a smoke request. Loading takes ~4 min.
#
# Usage: ./start.sh [restart] [extra tensorfold-native serve args]
#   ./start.sh                         # prepare if needed, launch
#   ./start.sh restart --context 65536
#   DRY_RUN=1 ./start.sh               # print the docker command and exit
#   FOREGROUND=1 ./start.sh            # follow the log after the start
# Extra arguments go to tensorfold-native serve after the defaults, so they win.
# Settings (environment or ./.env, defaults in scripts/config.sh):
#   serving  CONTEXT, PARALLEL, DRAFTS, PROMPT_CACHE_GIB, MAX_TOKENS, THINKING, TEMPERATURE, TOP_P, TOP_K, SERVED_NAME,
#            HOST, PORT
#   files    MODEL_ID, MODEL_REVISION, HF_CACHE, STATE_DIR, LOG_DIR
#   image    IMAGE, TF_REPO, TF_REF, ZIG_VERSION, BASE_IMAGE, CONTAINER_NAME
#   engine   every TENSORFOLD_* and TF_GLM_* variable goes into the container (TENSORFOLD_API_KEY by name only)
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh

WAIT_TIMEOUT="${WAIT_TIMEOUT:-1200}"
MODE=start
case "${1:-}" in
  restart) MODE=restart; shift ;;
  help|-h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
esac

[[ "$CONTEXT" =~ ^[1-9][0-9]*$ && "$CONTEXT" -le 1048576 ]] || die "CONTEXT is a token count up to 1048576, not $CONTEXT"
[[ "$PARALLEL" =~ ^([1-9]|1[0-6])$ ]] || die "PARALLEL is 1 to 16, not $PARALLEL"
for v in THINKING DRAFTS; do [[ "${!v}" =~ ^[01]$ ]] || die "$v is 0 or 1, not ${!v}"; done

SERVE_ARGS=(--context "$CONTEXT" --parallel "$PARALLEL" --max-tokens "$MAX_TOKENS"
            --temperature "$TEMPERATURE" --top-p "$TOP_P")
[[ "$PROMPT_CACHE_GIB" != 0 ]] && SERVE_ARGS+=(--prompt-cache-gib "$PROMPT_CACHE_GIB")
[[ -n "$TOP_K" ]] && SERVE_ARGS+=(--top-k "$TOP_K")
if [[ "$THINKING" == 1 ]]; then SERVE_ARGS+=(--thinking); else SERVE_ARGS+=(--no-thinking); fi
[[ "$DRAFTS" == 1 ]] || SERVE_ARGS+=(--no-drafts)

API_HOST="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && API_HOST=127.0.0.1
URL="http://$API_HOST:$PORT"
running() { [[ "$(docker inspect -f '{{.State.Running}}' "$CONTAINER_NAME" 2>/dev/null)" == true ]]; }

mkdir -p "$STATE_DIR" "$LOG_DIR"
if [[ "${DRY_RUN:-0}" != 1 ]]; then
  if [[ "$MODE" == start ]] && running; then
    log "$CONTAINER_NAME is already running on port $PORT: ./start.sh restart restarts it, ./stop.sh stops it."
    exit 0
  fi
  [[ "$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)" == "$(image_hash)" ]] && weights_ready ||
    ./scripts/prepare.sh
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true
  # on unified memory the pack's cached pages compete with the GPU's allocations: drop them before the load reads it
  python3 -I -c 'import glob, os, sys
for f in glob.glob(os.path.join(sys.argv[1], "*.safetensors")):
    fd = os.open(os.path.realpath(f), os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)' "$(snapshot_dir)" || true
  free_gib=$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)
  (( free_gib >= 100 )) || warn "only ${free_gib} GiB MemAvailable: the pack needs ~84 GiB plus the streams (stop other GPU work)"
  if ss -ltn 2>/dev/null | awk '{print $4}' | grep -qE "[:.]$PORT\$"; then die "port $PORT is in use: stop the other server or set PORT"; fi
fi

snap="hub/models--${MODEL_ID//\//--}/snapshots/$(snapshot_rev)"
ENV_ARGS=(-e TF_GLM_RANK=0 -e TF_GLM_WORLD=1)
while IFS='=' read -r k _; do
  [[ "$k" == TENSORFOLD_API_KEY ]] && { ENV_ARGS+=(-e TENSORFOLD_API_KEY); continue; }
  [[ "$k" =~ ^(TENSORFOLD_|TF_GLM_)[A-Z0-9_]*$ && "$k" != TF_GLM_RANK && "$k" != TF_GLM_WORLD ]] && ENV_ARGS+=(-e "$k=${!k}")
done < <(env)
CMD=(docker run -d --name "$CONTAINER_NAME" --gpus all --ipc=host --ulimit memlock=-1 --network host
     -v "$HF_CACHE":/root/.cache/huggingface:ro "${ENV_ARGS[@]}" --entrypoint tensorfold-native "$IMAGE"
     serve "/root/.cache/huggingface/$snap" --name "$SERVED_NAME" --host "$HOST" --port "$PORT" "${SERVE_ARGS[@]}" "$@")
if [[ "${DRY_RUN:-0}" == 1 ]]; then printf '%q ' "${CMD[@]}"; echo; exit 0; fi

log "Starting $SERVED_NAME: $PARALLEL streams x $CONTEXT tokens, MTP drafts $([[ $DRAFTS == 1 ]] && echo on || echo off), port $PORT (loading ~80 GiB, ~4 min)"
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
grep -E "EXL3 checkpoint|CUDA sm_" "$LOG" | sed 's/^/  /'
reply=$(curl -s --max-time 120 "$URL/v1/chat/completions" -H 'content-type: application/json' ${TENSORFOLD_API_KEY:+-H "Authorization: Bearer $TENSORFOLD_API_KEY"} \
  -d "{\"model\":\"$SERVED_NAME\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hello in five words.\"}],\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}" |
  python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' 2>/dev/null) || reply=""
[[ -n "$reply" ]] || die "the smoke request failed (log: $LOG)"
log "Ready at $URL/v1 (model $SERVED_NAME) in $((SECONDS - t0)) s. Smoke reply: $reply"
log "Log: $LOG"
[[ "${FOREGROUND:-0}" == 1 ]] && exec tail -f "$LOG"
exit 0
