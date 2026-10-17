#!/usr/bin/env bash
# Stop the Dory server (the container named CONTAINER_NAME, default tf-dory-4b). Its log stays in LOG_DIR.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"
source ./scripts/config.sh
if docker inspect "$CONTAINER_NAME" >/dev/null 2>&1; then
  docker rm -f "$CONTAINER_NAME" >/dev/null
  log "Stopped $CONTAINER_NAME"
else
  log "$CONTAINER_NAME is not running"
fi
