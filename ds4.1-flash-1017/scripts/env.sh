#!/bin/bash
# Shared pins and paths. Source this; do not execute it.
#
# Every value here is the measured configuration. Changing one without
# re-measuring means the numbers in README.md no longer describe what you serve.

# ---- model ----------------------------------------------------------------
MODEL_REPO="vcruz305/DSV4.1-Flash-1017-SAGE-EXL3-1.62bpw"
# Hub revision to download. The repo's history has been replaced more than once
# (each time a single "Initial commit" with the same files), so a commit id can
# stop resolving. scripts/SHA256SUMS is the authority: download_and_verify.sh
# falls back to main when this revision is gone, and the sha256 check decides.
# See README "Pack revision".
MODEL_REV="${MODEL_REV:-2a6059359c685cd0ea66e1e6c34e6803322ff095}"
MODEL_NAME="dsv41-1.62bpw-v3"          # directory name under $MODELS_DIR

# ---- images ---------------------------------------------------------------
# Base engine: Christopher Owen's spark-ds41f r6w (upstreams pinned at 2dc95329),
# built per justincasey/DeepSeek-V4.1-Flash-1017-4x-DGX-Spark docker/Dockerfile.
# That Dockerfile's own base is vllm/vllm-openai@sha256:3cf14c52765a0f0c2f09035e05fa79e1e4b2f9788b64afa90fa41c1ff12d29cf.
BASE_IMAGE="${BASE_IMAGE:-dsv41-1017-ds41f:r6w}"
IMAGE="${IMAGE:-dsv41-1017-exl3:r6w}"

# Runtime forks baked into the layer above the base, pinned to the commits whose
# file hashes match the measured server (branches named below).
FORK_REPO="https://github.com/vcruz305/exllamav3.git"
FORK_BRANCH="feat/moe-warp-private-stages"
FORK_REF="${FORK_REF:-909ad8564fcb9e070caca8a65d380c741681e79c}"          # exllamav3: mixed-K unified MoE + aarch64 guards + warp-private MoE pipeline stages
PLUGIN_REPO="https://github.com/vcruz305/vllm-exl3.git"
PLUGIN_BRANCH="feat/dsv41-1017-tp1"
PLUGIN_REF="${PLUGIN_REF:-30e8b941376ce2d43bc8438439937c4809106635}"    # vllm-exl3: Exl3MoEMethod / ATS loader

# MoE tile-K the served kernel was compiled at. It is a compile-time constant in
# the fork, so the build rewrites the header before compiling.
MOE_TILESIZE_K="${MOE_TILESIZE_K:-16}"

# ---- paths ----------------------------------------------------------------
RECIPE_HOME="${RECIPE_HOME:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
MODELS_DIR="${MODELS_DIR:-$HOME/models}"
MODEL_DIR="${MODEL_DIR:-$MODELS_DIR/$MODEL_NAME}"
CACHE_DIR="${CACHE_DIR:-$HOME/.cache/dsv41-ds41f}"
ALIAS_DIR="${ALIAS_DIR:-$CACHE_DIR/alias-cold}"
RUNS_DIR="${RUNS_DIR:-$RECIPE_HOME/runs}"

# ---- serve ----------------------------------------------------------------
CONTAINER="${CONTAINER:-dsv41-tp1}"
API_PORT="${API_PORT:-8000}"
MAXLEN="${MAXLEN:-229376}"      # local default: one ~230K stream (upstream ships 524288, the largest measured clean limit)
KVB="${KVB:-536870912}"         # local default 0.5 GiB: 380,035 tokens at MAXLEN=229376 (1.66 requests); upstream 1 GiB left
                                # this box (desktop session, ~2.3 GiB more host memory) under the guard floor in a 225K prefill
MNBT="${MNBT:-2112}"
LPT="${LPT:-2047}"
MTP_K="${MTP_K:-5}"
CAPTURE_SIZES="${CAPTURE_SIZES:-1,2,3,4,5,6,7}"
CG_MODE="${CG_MODE:-FULL_DECODE_ONLY}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
ROW_COST="${ROW_COST-}"                 # DSpark adaptive verification's linear step cost; upstream passes 0.01, which needs vLLM
                                        # patch 0184 (spark-ds41f r6x); the r6w base built here rejects it, so empty = its timed budget
PREFIX_CACHING="${PREFIX_CACHING:-1}"   # 1 = reuse KV of repeated prefixes (default); 0 = every request is a full prefill
DSML_STOP="${DSML_STOP:-1}"             # 1 = end tool-call replies at the first closing tag after a DSML invoke
ATS_LAYERS="${ATS_LAYERS:-38,39}"
ATS_MTP="${ATS_MTP:-1}"
ATS_EMBED="${ATS_EMBED:-1}"

export MODEL_REPO MODEL_REV MODEL_NAME BASE_IMAGE IMAGE
export FORK_REPO FORK_BRANCH FORK_REF PLUGIN_REPO PLUGIN_BRANCH PLUGIN_REF MOE_TILESIZE_K
export RECIPE_HOME MODELS_DIR MODEL_DIR CACHE_DIR ALIAS_DIR RUNS_DIR
export ROW_COST CONTAINER API_PORT MAXLEN KVB MNBT LPT MTP_K CAPTURE_SIZES CG_MODE GPU_MEM_UTIL PREFIX_CACHING DSML_STOP
export ATS_LAYERS ATS_MTP ATS_EMBED

# ---- privileges (local change: this host has docker-group access but no passwordless sudo) ----
# DOCKER: plain `docker` when the user can reach the daemon, else `sudo -n docker` (upstream's assumption).
# as_root CMD...: run a host command as root: `sudo -n` when available, else a one-shot privileged container
# that enters the host's namespaces (docker-group membership is root-equivalent anyway).
if docker info >/dev/null 2>&1; then DOCKER="docker"; else DOCKER="sudo -n docker"; fi
as_root() {
  if sudo -n true 2>/dev/null; then sudo -n "$@"
  else docker run --rm --privileged --pid=host --net=host alpine:latest nsenter -t 1 -m -u -i -n -p -- "$@"; fi
}
export DOCKER
export -f as_root
