#!/usr/bin/env bash
# Serve Qwen3.8-27B (Vontra/Qwen3.8-27B-MLX-4bit + z-lab/Qwen3.8-27B-DFlash2 drafter) with stock TensorFold
# through ./tensorfold.sh. Settings follow the Qwen3.8-27B-DGX-Spark-TensorFold recipe. Its image limits
# (TENSORFOLD_MAX_IMAGES / TENSORFOLD_IMAGE_TOKENS patches) map to stock --vision-max-images / --vision-image-tokens
# (since v0.6.3). Not available in stock TensorFold, so left out: fp8 KV cache (--kv-dtype is bf16-only for the 27B),
# pinned KV pool, YaRN, and a memory reserve below 2 GiB.
#
# Usage: ./start-qwen38-27b.sh [extra tensorfold serve args]
#   Extra args are appended, so they override the defaults below (e.g. --parallel 8 --context 163840).
#   --parallel 4 at 262144 keeps 4 full-length requests in memory at once (~1.4M cache tokens free on this Spark).
#
# Settings (environment):
#   TF_VERSION   TensorFold version (default: latest)     PORT        port (default: 8888)
#   HOST         bind address (default: 0.0.0.0)          SERVED_NAME model id clients see (default: Qwen3.8-27B)
#   NAME         container name (default: tf-qwen38-27b)   FOREGROUND  1: run attached instead of in the background
#   MODEL_ID     target model (default: Vontra/Qwen3.8-27B-MLX-4bit)
#   DRAFT_ID     drafter (default: z-lab/Qwen3.8-27B-DFlash2); DRAFT_ID= serves without drafts (--no-drafts)
#   TENSORFOLD_VIDEO_TOKENS (default 16384), TENSORFOLD_MEMORY_RESERVE_GIB (default 2) and any other
#   TENSORFOLD_* variable are passed to the server.
#
# Logs: docker logs -f tf-qwen38-27b      Stop: docker stop tf-qwen38-27b
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_ID="${MODEL_ID:-Vontra/Qwen3.8-27B-MLX-4bit}"
DRAFT_ID="${DRAFT_ID-z-lab/Qwen3.8-27B-DFlash2}"
SERVED_NAME="${SERVED_NAME:-Qwen3.8-27B}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8888}"
NAME="${NAME:-tf-qwen38-27b}"
FOREGROUND="${FOREGROUND:-0}"

export TENSORFOLD_VIDEO_TOKENS="${TENSORFOLD_VIDEO_TOKENS:-16384}"
export TENSORFOLD_MEMORY_RESERVE_GIB="${TENSORFOLD_MEMORY_RESERVE_GIB:-2}"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  echo "container $NAME already exists; stop it first: docker stop $NAME" >&2
  exit 1
fi

wrapper=(--tf-name "$NAME" --tf-docker-arg=--ulimit=stack=67108864)
[[ "$FOREGROUND" == 1 ]] || wrapper+=(--tf-detach)

if [[ -n "$DRAFT_ID" ]]; then draft=(--drafter "$DRAFT_ID"); else draft=(--no-drafts); fi

"$SCRIPT_DIR/tensorfold.sh" "${wrapper[@]}" \
  serve "$MODEL_ID" "${draft[@]}" \
  --name "$SERVED_NAME" --host "$HOST" --port "$PORT" \
  --parallel 4 --context 262144 \
  --temperature 1.0 --top-p 0.95 --top-k 20 \
  --prefill-fp8 --vision --thinking \
  --vision-max-images 50 --vision-image-tokens 16384 \
  "$@"

if [[ "$FOREGROUND" != 1 ]]; then
  echo "started $NAME on http://$HOST:$PORT/v1 (model: $SERVED_NAME)"
  echo "logs: docker logs -f $NAME    stop: docker stop $NAME"
fi
