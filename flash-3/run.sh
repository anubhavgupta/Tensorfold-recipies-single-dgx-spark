#!/usr/bin/env bash
# Wrapper around the vcruz305 recipe (exllamav3 fork + TabbyAPI) that keeps everything inside flash-3/runtime
# and serves the pack from the Hugging Face cache snapshot of turboderp/Qwen3.8-Flash-Next-exl3.
#   ./run.sh setup | check | serve | stop      PROFILE=single|concurrent (default concurrent), MAX_BATCH_SIZE, CACHE_SIZE, PORT (default 8899)
#   Other recipe variables (NGRAM_RAM, VISION, DRAFT_NUM_TOKENS, ...) pass through; see exllamav3-tabby/serve.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECIPE_HOME="${RECIPE_HOME:-$HERE/runtime}"
REVISION="${REVISION:-4.05bpw_h6_ng6}"
_repo="$HOME/.cache/huggingface/hub/models--turboderp--Qwen3.8-Flash-Next-exl3"
if [[ -z "${MODEL_DIR:-}" && -f "$_repo/refs/$REVISION" ]]; then
  export MODEL_DIR="$_repo/snapshots/$(<"$_repo/refs/$REVISION")"
fi
# No python3.12-dev on this host (no root): headers extracted from the libpython3.12-dev .deb into runtime/pyinc;
# the CUDA build and triton's launcher compile need Python.h.
_pi="$RECIPE_HOME/pyinc"
if [[ -d "$_pi" ]]; then
  export CPATH="$_pi:$_pi/python3.12:$_pi/aarch64-linux-gnu/python3.12${CPATH:+:$CPATH}"
fi
cd "$HERE"
case "${1:-serve}" in
  setup) exec bash exllamav3-tabby/setup.sh ;;
  check) exec bash exllamav3-tabby/setup.sh --check ;;
  serve) [[ -z "$(find "$_repo/blobs" -name '*.incomplete' -print -quit 2>/dev/null)" ]] \
           || { echo "download of $REVISION still in progress" >&2; exit 1; }
         exec bash exllamav3-tabby/serve.sh ;;
  stop)  pids=$(pgrep -f "$RECIPE_HOME/tabbyAPI/main.py" || true)
         [[ -z "$pids" ]] && echo "not running" || kill $pids ;;
  *) echo "usage: $0 setup|check|serve|stop" >&2; exit 2 ;;
esac
