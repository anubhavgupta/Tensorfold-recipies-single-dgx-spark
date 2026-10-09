#!/usr/bin/env bash
# Stop the GLM-5.3-Flash TabbyAPI started by serve.sh.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pids=$(pgrep -f "$HERE/state/config.yml" || true)
if [[ -z "$pids" ]]; then echo "not running"; else kill $pids && echo "stopped $pids"; fi
