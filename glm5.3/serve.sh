#!/usr/bin/env bash
# Serve GLM-5.3-Flash EXL3 (turboderp/GLM-5.3-Flash-exl3, revision 2.05bpw) on one DGX Spark.
#
# TensorFold 0.6.x cannot: its GLM-5.3-Flash CUDA engine is two-rank only ("needs two GPUs, one per machine",
# `--tp 2`), so this recipe runs the pack on the vcruz305/exllamav3 fork + TabbyAPI runtime of ../flash-3
# (set it up once with ../flash-3/run.sh setup).
#
# Download:  hf download turboderp/GLM-5.3-Flash-exl3 --revision 2.05bpw
# Usage:     ./glm5.3/serve.sh        (foreground)       ./glm5.3/stop.sh
#   curl http://127.0.0.1:8890/v1/models
# Env:
#   REVISION (2.05bpw)  MODEL_DIR (HF cache snapshot of REVISION)   PORT (8890)  HOST (127.0.0.1)  SERVED_NAME (GLM-5.3-Flash)
#   MAX_SEQ_LEN (131072)  per-request context     CACHE_SIZE (= MAX_SEQ_LEN)  shared KV pool in tokens, multiple of 256
#   MAX_BATCH_SIZE (1)    CACHE_MODE ("8,8")      CHUNK_SIZE (4096; 8192 does not fit)  prompt tokens per forward
#   DRAFT_MODE (mtp)  DRAFT_NUM_TOKENS (2)  EXL3_DRAFT_CONFIDENCE (0.6)   REASONING_EFFORT (max: low, high or max)
#   VISION (false)   DRY_RUN=1 renders the config and prints the command
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLASH3="$HERE/../flash-3"
export RECIPE_HOME="${RECIPE_HOME:-$FLASH3/runtime}"

REVISION="${REVISION:-2.05bpw}"
_repo="$HOME/.cache/huggingface/hub/models--turboderp--GLM-5.3-Flash-exl3"
if [[ -z "${MODEL_DIR:-}" ]]; then
  [[ -f "$_repo/refs/$REVISION" ]] || { echo "revision $REVISION is not in the HF cache; run: hf download turboderp/GLM-5.3-Flash-exl3 --revision $REVISION" >&2; exit 1; }
  [[ -z "$(find "$_repo/blobs" -name '*.incomplete' -print -quit 2>/dev/null)" ]] || { echo "download of $REVISION still in progress" >&2; exit 1; }
  MODEL_DIR="$_repo/snapshots/$(<"$_repo/refs/$REVISION")"
fi
export MODEL_DIR

# env.sh: venv/TabbyAPI paths, the die/say helpers, verify_runtime, drop_pack_cache, pin_cmd
source "$FLASH3/exllamav3-tabby/env.sh"
STATE_DIR="$HERE/state"
mkdir -p "$STATE_DIR"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8890}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-131072}"
CACHE_SIZE="${CACHE_SIZE:-$MAX_SEQ_LEN}"
MAX_BATCH_SIZE="${MAX_BATCH_SIZE:-1}"
CACHE_MODE="${CACHE_MODE:-8,8}"
CHUNK_SIZE="${CHUNK_SIZE:-4096}"
DRAFT_MODE="${DRAFT_MODE:-mtp}"
DRAFT_NUM_TOKENS="${DRAFT_NUM_TOKENS:-2}"
VISION="${VISION:-false}"
REASONING_EFFORT="${REASONING_EFFORT:-max}"
(( CACHE_SIZE % 256 == 0 )) || die "CACHE_SIZE must be a multiple of 256"
(( CACHE_SIZE >= MAX_SEQ_LEN )) || die "CACHE_SIZE ($CACHE_SIZE) < MAX_SEQ_LEN ($MAX_SEQ_LEN)"

case "$HOST" in
  127.0.0.1|localhost|::1) DISABLE_AUTH="${DISABLE_AUTH:-true}" ;;
  *) DISABLE_AUTH="${DISABLE_AUTH:-false}"
     [[ "$DISABLE_AUTH" == "true" ]] && echo "warning: auth disabled on $HOST" >&2 ;;
esac

MODEL_DIR="$(cd "$MODEL_DIR" 2>/dev/null && pwd || echo "$MODEL_DIR")"
MODEL_NAME="${SERVED_NAME:-GLM-5.3-Flash}"
MODEL_PARENT="$STATE_DIR/models"
mkdir -p "$MODEL_PARENT"
find "$MODEL_PARENT" -mindepth 1 -maxdepth 1 -type l -delete
ln -sfn "$MODEL_DIR" "$MODEL_PARENT/$MODEL_NAME"
export HOST PORT DISABLE_AUTH MODEL_PARENT MODEL_NAME MAX_SEQ_LEN CACHE_SIZE CACHE_MODE CHUNK_SIZE MAX_BATCH_SIZE \
       VISION DRAFT_MODE DRAFT_NUM_TOKENS REASONING_EFFORT

CONFIG="$STATE_DIR/config.yml"
"$VENV/bin/python" - "$HERE/tabby-config.yml" "$CONFIG" <<'PY'
import os, string, sys
open(sys.argv[2], "w").write(string.Template(open(sys.argv[1]).read()).substitute(os.environ))
PY
say "config: $CONFIG  (${MAX_SEQ_LEN} tokens, ${MAX_BATCH_SIZE} concurrent, ${DRAFT_MODE} x${DRAFT_NUM_TOKENS})"

CMD=( $(pin_cmd) "$VENV/bin/python" "$TABBY_DIR/main.py" --config "$CONFIG" )
if [[ -n "${DRY_RUN:-}" ]]; then echo "cd $TABBY_DIR && ${CMD[*]}"; exit 0; fi

verify_runtime
verify_pack
drop_pack_cache
free -g | sed -n 2p >&2
export PATH="$CUDA_HOME/bin:$VENV/bin:$PATH"
cd "$TABBY_DIR"
exec "${CMD[@]}"
