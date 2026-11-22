#!/usr/bin/env bash
# Install flash-3's boot-time service (option A in docs/autostart.md): copy systemd/tensorfold-flash-3.service to
# /etc/systemd/system, reload systemd, and enable it for the next boot. It does NOT start the server: the running
# container stays yours, so `./flash-3/end.sh` and a hand-run `./flash-3/start.sh` keep working as before.
#
# Usage: ./flash-3/systemd/install-autostart.sh [--dry-run] [--now]
#   --dry-run   print what would run (needs no privileges)
#   --now       also start it now: the unit's pre-start step runs end.sh, which cuts requests in flight
# Remove it again with ./flash-3/systemd/uninstall-autostart.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
UNIT_NAME=tensorfold-flash-3
UNIT_SRC="$SCRIPT_DIR/$UNIT_NAME.service"
UNIT_DST="/etc/systemd/system/$UNIT_NAME.service"

dry=0 now=0
for a in "$@"; do
  case "$a" in
    --dry-run) dry=1 ;;
    --now)     now=1 ;;
    -h|--help) sed -n '2,9p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "install-autostart.sh: unknown option $a (--dry-run, --now)" >&2; exit 1 ;;
  esac
done

run() { if (( dry )); then echo "  would run: $*"; else "$@"; fi; }

# The unit hardcodes this checkout's paths and user; installing it somewhere else would boot a service that fails.
want_dir="$(cd "$SCRIPT_DIR/.." && pwd)"
got_dir="$(sed -n 's/^WorkingDirectory=//p' "$UNIT_SRC" | head -1)"
got_user="$(sed -n 's/^User=//p' "$UNIT_SRC" | head -1)"
if [[ "$got_dir" != "$want_dir" ]]; then
  echo "install-autostart.sh: $UNIT_SRC has WorkingDirectory=$got_dir, this checkout is $want_dir" >&2
  echo "  fix the unit (and its Environment=HOME= and the two ExecStartPre= paths) for this checkout first" >&2
  exit 1
fi
[[ -x "$want_dir/start.sh" && -x "$want_dir/end.sh" ]] || { echo "install-autostart.sh: start.sh/end.sh not executable in $want_dir" >&2; exit 1; }
if [[ "$got_user" != "$(id -un)" ]]; then
  echo "note: the unit runs as $got_user, you are $(id -un); it needs $got_user's HF cache and docker access" >&2
fi

# never silently overwrite a unit that was edited in place
if [[ -e "$UNIT_DST" ]] && ! cmp -s "$UNIT_DST" "$UNIT_SRC"; then
  echo "install-autostart.sh: $UNIT_DST exists and differs from $UNIT_SRC:" >&2
  diff -u "$UNIT_DST" "$UNIT_SRC" | sed 's/^/  /' >&2
  echo "  copy those edits into $UNIT_SRC, or remove the installed file, then rerun" >&2
  exit 1
fi

run sudo cp "$UNIT_SRC" "$UNIT_DST"
run sudo systemctl daemon-reload
if (( now )); then run sudo systemctl enable --now "$UNIT_NAME.service"
else             run sudo systemctl enable "$UNIT_NAME.service"; fi
if (( dry )); then echo "  (dry run: nothing changed)"; exit 0; fi

echo "installed: $UNIT_DST"
systemctl is-enabled "$UNIT_NAME.service" | sed 's/^/  enabled for boot: /'
echo "  state now:      $(systemctl is-active "$UNIT_NAME.service" || true)   (start it with: sudo systemctl start $UNIT_NAME)"
if docker ps --format '{{.Names}}' | grep -qx tf-flash-3-exl3; then
  echo "  tf-flash-3-exl3 is running and untouched; it will be gone at the reboot (docker run --rm), the service owns the next one"
fi
echo "after the reboot:  systemctl status $UNIT_NAME --no-pager    journalctl -b -u $UNIT_NAME    curl -s localhost:8888/health"
