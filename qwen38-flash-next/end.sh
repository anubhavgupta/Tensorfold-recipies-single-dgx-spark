#!/usr/bin/env bash
# Stop the Qwen3.8-Flash-Next server started by ./qwen38-flash-next/start.sh and remove its container,
# freeing its GPU memory. Gives it STOP_TIMEOUT seconds (default 30) to shut down; requests still running
# are cut off (no draining), so this warns when there are any.
# Usage: ./qwen38-flash-next/end.sh      Env: NAME, HOST, PORT (must match the start script's), STOP_TIMEOUT
set -euo pipefail

NAME="${NAME:-tf-qwen38-flash-next}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "no container named $NAME: nothing to stop"
  exit 0
fi

if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]; then
  api_host="$HOST"; [[ "$HOST" == 0.0.0.0 || "$HOST" == "::" ]] && api_host=127.0.0.1
  [[ "$api_host" == *:* ]] && api_host="[$api_host]"
  busy=$(curl -s --max-time 3 "http://$api_host:$PORT/health" 2>/dev/null |
         python3 -c 'import json,sys; print(json.load(sys.stdin).get("requests_running", 0))' 2>/dev/null || echo 0)
  (( busy == 0 )) || echo "warning: $busy request(s) still running will be cut off" >&2
  echo "stopping $NAME (up to ${STOP_TIMEOUT}s)"
  docker stop -t "$STOP_TIMEOUT" "$NAME" >/dev/null
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true   # tensorfold.sh runs with --rm, so stop already removed it
echo "stopped and removed $NAME; its GPU memory is free again"
