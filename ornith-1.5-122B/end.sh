#!/usr/bin/env bash
# Stop the Ornith-1.5-122B SGLang server started by ./ornith-1.5-122B/start.sh and remove its container (frees its
# memory). Running requests are cut off; the log is saved to ./last.log first (docker rm loses it).
# Usage: ./ornith-1.5-122B/end.sh      Env: NAME (sglang-ornith-122b), STOP_TIMEOUT (30)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NAME="${NAME:-sglang-ornith-122b}"
STOP_TIMEOUT="${STOP_TIMEOUT:-30}"

if ! docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "no container named $NAME: nothing to stop"; exit 0
fi
docker logs "$NAME" > "$SCRIPT_DIR/last.log" 2>&1 || true
if [[ "$(docker inspect -f '{{.State.Running}}' "$NAME")" == true ]]; then
  echo "stopping $NAME (up to ${STOP_TIMEOUT}s)"
  docker stop -t "$STOP_TIMEOUT" "$NAME" >/dev/null
fi
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "stopped and removed $NAME (log saved to $SCRIPT_DIR/last.log)"
