#!/usr/bin/env bash
# Remove flash-3's boot-time service (the reverse of install-autostart.sh, option A in docs/autostart.md): disable the
# unit, delete /etc/systemd/system/tensorfold-flash-3.service and its drop-in directory, and reload systemd.
#
# It does NOT stop a running server and touches no model, image or cache: to free the GPU afterwards run
# ./flash-3/end.sh (no root, 30 s grace, warns about live requests).
#
# Usage: ./flash-3/systemd/uninstall-autostart.sh [--dry-run] [--stop] [--force]
#   --dry-run   print what would run (needs no privileges)
#   --stop      also stop the service now (cuts live requests after the container's 10 s grace)
#   --force     remove the installed unit even when it differs from the repo copy (default: show a diff and stop)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME=tensorfold-flash-3
UNIT_SRC="$SCRIPT_DIR/$UNIT_NAME.service"
# AUTOSTART_UNIT_DIR only exists so the logic can be exercised against a scratch directory in a dry run
UNIT_DST="${AUTOSTART_UNIT_DIR:-/etc/systemd/system}/$UNIT_NAME.service"
UNIT_DIR="$(dirname "$UNIT_DST")"

dry=0 stop=0 force=0
for a in "$@"; do
  case "$a" in
    --dry-run) dry=1 ;;
    --stop)    stop=1 ;;
    --force)   force=1 ;;
    -h|--help) sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "uninstall-autostart.sh: unknown option $a (--dry-run, --stop, --force)" >&2; exit 1 ;;
  esac
done

run() { if (( dry )); then echo "  would run: $*"; else "$@"; fi; }
state() { systemctl is-"$1" "$UNIT_NAME.service" 2>/dev/null || true; }

# masked units are a symlink to /dev/null; unmask first so the real file underneath is what we look at
masked=0
if [[ -L "$UNIT_DST" ]]; then
  masked=1
  echo "$(basename "$UNIT_DST") is masked; unmasking it first"
  run sudo systemctl unmask "$UNIT_NAME.service"
fi

if [[ ! -e "$UNIT_DST" ]]; then
  if (( masked )); then
    echo "it was only masked, there is no unit file to remove; the service is gone from boot"
  else
    echo "$UNIT_DST is not installed; nothing to remove"
  fi
  echo "  boot state: $(state enabled)   running: $(state active)"
  echo "  enabled by something else? check: systemctl list-unit-files --no-pager | grep -i tensorfold"
  exit 0
fi

# a unit that was edited in place is not ours to throw away silently
if [[ ! -s "$UNIT_SRC" ]]; then
  echo "uninstall-autostart.sh: $UNIT_SRC is missing; cannot compare before deleting" >&2
  echo "  restore it from git, or pass --force to delete $UNIT_DST anyway" >&2
  exit 1
elif ! cmp -s "$UNIT_DST" "$UNIT_SRC"; then
  if (( force )); then
    echo "note: $UNIT_DST differs from $UNIT_SRC; removing it anyway (--force)" >&2
  else
    echo "uninstall-autostart.sh: $UNIT_DST differs from $UNIT_SRC:" >&2
    diff -u "$UNIT_DST" "$UNIT_SRC" | sed 's/^/  /' >&2
    echo "  pass --force to remove it regardless, or keep it" >&2
    exit 1
  fi
fi

active=0
if [[ "$(state active)" == active ]]; then active=1; fi
if (( active )) && (( ! stop )); then
  echo "note: $UNIT_NAME is running; it keeps running. Stop it with: ~/projects/tensorfold/flash-3/end.sh (no root)"
fi

# disable while the unit file is still readable, then delete, then tell systemd about it
if (( stop )); then run sudo systemctl disable --now "$UNIT_NAME.service"
else              run sudo systemctl disable "$UNIT_NAME.service"; fi
run sudo rm -f "$UNIT_DST"
if [[ -d "$UNIT_DIR/$UNIT_NAME.service.d" ]]; then run sudo rm -rf "$UNIT_DIR/$UNIT_NAME.service.d"; fi
run sudo systemctl daemon-reload
run sudo systemctl reset-failed "$UNIT_NAME.service"
if (( dry )); then echo "  (dry run: nothing changed)"; exit 0; fi

echo "removed: $UNIT_DST"
echo "  boot state: $(state enabled)   running: $(state active)"
if docker ps --format '{{.Names}}' | grep -qx tf-flash-3-exl3; then
  echo "  tf-flash-3-exl3 is still serving on :8888; free the GPU with: ~/projects/tensorfold/flash-3/end.sh"
else
  echo "  no tf-flash-3-exl3 container; start one by hand with: ~/projects/tensorfold/flash-3/start.sh"
fi
echo "  models, docker images and the kernel caches were left untouched; install again with: install-autostart.sh"
