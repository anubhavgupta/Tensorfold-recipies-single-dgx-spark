#!/bin/bash
# Start the DSV4.1-Flash-1017 server (vLLM, port 8888, model dsv41) and wait until it is ready.
#
#   ./start.sh                         # defaults from scripts/env.sh
#   KVB=536870912 ./start.sh           # 0.5 GiB KV pool (use with the desktop session running)
#   WAIT=0 ./start.sh                  # launch and return without waiting
#
# Any scripts/env.sh variable (MAXLEN, KVB, API_PORT, ROW_COST, ...) can be set in the environment.
# Stop with ./end.sh.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
. scripts/env.sh

if $DOCKER ps -a --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "$CONTAINER already exists; run ./end.sh first" >&2
  exit 1
fi

bash scripts/serve.sh
[ "${WAIT:-1}" = 1 ] || exit 0

RUN=$(ls -1td "$RUNS_DIR"/*/ | head -1)
echo "waiting for ready (log: docker logs -f $CONTAINER)"
for _ in $(seq 1 1200); do
  stage=$(cat "$RUN/stage" 2>/dev/null || echo "?")
  if [ "$stage" = ready ]; then
    echo "ready: http://127.0.0.1:${API_PORT}/v1 (model ${SERVED_MODEL_NAME})"
    [ -f "$RUN/ready-mem" ] && echo "at ready: $(cat "$RUN/ready-mem")"
    [ -f "$RUN/slow-boot" ] && cat "$RUN/slow-boot" >&2
    bash scripts/smoke.sh
    exit 0
  fi
  if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "container $CONTAINER exited before ready (stage: $stage)" >&2
    tail -5 "$RUN/guard.log" 2>/dev/null >&2 || true
    $DOCKER logs --tail 20 "$CONTAINER" 2>&1 | cut -c1-200 >&2 || true
    exit 1
  fi
  printf '\r  stage: %-14s %ds' "$stage" "$SECONDS"
  sleep 2
done
echo; echo "not ready after 40 min; check: bash scripts/guard.sh status" >&2
exit 1
