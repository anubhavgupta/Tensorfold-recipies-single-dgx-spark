#!/usr/bin/env bash
# Prepare this Spark for Nemotron-Labs Dory1.0-4B NVFP4 on TensorFold's Zig engine:
#   1. the image: tensorfold-native and libdory built from TensorFold (TF_REF) plus patches/*.patch and engine/ with Zig,
#      on NVIDIA's PyTorch container (rebuilt when the patches, engine or this Dockerfile change)
#   2. the checkpoint in the Hugging Face cache (~3.2 GB), at its pinned revision
# ./start.sh runs this by itself when needed. Safe to re-run. --rebuild rebuilds the image from scratch.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
source ./scripts/config.sh

REBUILD=0
for arg in "$@"; do
  case "$arg" in
    --rebuild) REBUILD=1 ;;
    -h|--help) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 0 ;;
    *) die "unknown argument: $arg" ;;
  esac
done

mkdir -p "$STATE_DIR" "$HF_CACHE/hub"
exec 9>"$STATE_DIR/prepare.lock"
flock -n 9 || die "another prepare.sh is already running"
command -v docker >/dev/null || die "docker is not installed"
docker info >/dev/null 2>&1 || die "cannot talk to the docker daemon"

PATCHES_HASH=$(image_hash)
built_hash=$(docker image inspect -f '{{index .Config.Labels "tf.patches"}}' "$IMAGE" 2>/dev/null || true)
if [[ $REBUILD -eq 1 || "$built_hash" != "$PATCHES_HASH" ]]; then
  docker image inspect "$BASE_IMAGE" >/dev/null 2>&1 || { log "Pulling base image $BASE_IMAGE"; docker pull "$BASE_IMAGE"; }
  log "Building $IMAGE (TensorFold ${TF_REF:0:12}, Zig $ZIG_VERSION, patches $PATCHES_HASH): a few minutes"
  ctx=$(mktemp -d "$STATE_DIR/build.XXXXXX")
  trap 'rm -rf -- "$ctx"' EXIT
  mkdir -p "$ctx/patches" "$ctx/engine"
  cp -- patches/*.patch "$ctx/patches/"
  cp -- engine/* "$ctx/engine/"
  nocache=(); [[ $REBUILD -eq 1 ]] && nocache=(--no-cache)
  docker build "${nocache[@]}" -t "$IMAGE" --build-arg BASE_IMAGE="$BASE_IMAGE" \
    --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$TF_REF" \
    --build-arg ZIG_VERSION="$ZIG_VERSION" --build-arg ZIG_SHA256="$ZIG_SHA256" \
    --build-arg PATCHES_HASH="$PATCHES_HASH" \
    -f - "$ctx" <<'DOCKERFILE'
ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:26.07-py3
FROM ${BASE_IMAGE} AS build
ARG ZIG_VERSION
ARG ZIG_SHA256
RUN curl -fsSL -o /tmp/zig.tar.xz "https://ziglang.org/download/${ZIG_VERSION}/zig-aarch64-linux-${ZIG_VERSION}.tar.xz" && \
    echo "${ZIG_SHA256}  /tmp/zig.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/zig && tar -xJf /tmp/zig.tar.xz -C /opt/zig --strip-components=1 && rm /tmp/zig.tar.xz && \
    /opt/zig/zig version
ARG TF_REPO
ARG TF_REF
RUN git init -q /tensorfold && cd /tensorfold && git remote add origin "${TF_REPO}" && \
    git fetch -q --depth 1 origin "${TF_REF}" && git checkout -q --detach FETCH_HEAD && \
    test "$(git rev-parse HEAD)" = "${TF_REF}"
# structured output (response_format, forced tool_choice) loads libtfgrammar, built from the xgrammar release the tree pins
RUN tag="$(sed -n 's/^pub const pinned = "\(.*\)";/\1/p' /tensorfold/zig/build/grammar.zig)" && test -n "$tag" && \
    git clone -q --branch "$tag" --recurse-submodules https://github.com/mlc-ai/xgrammar /opt/xgrammar && \
    test "$(git -C /opt/xgrammar describe --tags --exact-match)" = "$tag"
COPY patches /opt/tf-patches
COPY engine /opt/dory-engine
RUN cd /tensorfold && \
    for p in /opt/tf-patches/*.patch; do echo "applying $p"; git apply --check "$p" && git apply "$p" || exit 1; done && \
    cp /opt/dory-engine/dory_native.zig /opt/dory-engine/dory_api.zig zig/src/families/nemotron/
RUN cd /tensorfold && /opt/zig/zig build -Dnvcc=/usr/local/cuda/bin/nvcc -Ddory-src=/opt/dory-engine \
      -Dxgrammar=/opt/xgrammar -Doptimize=ReleaseFast --prefix /opt/tensorfold -j"$(nproc)" install native dory grammar && \
    cp -a /opt/tensorfold/lib/libtfgrammar.so /opt/tensorfold/native/lib/ && \
    test -x /opt/tensorfold/native/bin/tensorfold-native && test -f /opt/tensorfold/native/lib/libdory.so && \
    test -f /opt/tensorfold/native/lib/libtfgrammar.so
RUN mkdir -p /opt/tensorfold/share/doc/tensorfold/LICENSES && cd /tensorfold && \
    for f in LICENSE LICENSE.md LICENSES NOTICE THIRD_PARTY_NOTICES.md; do \
      if [ -e "$f" ]; then cp -a "$f" /opt/tensorfold/share/doc/tensorfold/ || exit 1; fi; done && \
    cp /opt/xgrammar/LICENSE /opt/tensorfold/share/doc/tensorfold/LICENSES/xgrammar-Apache-2.0.txt

FROM ${BASE_IMAGE}
COPY --from=build /opt/tensorfold /opt/tensorfold
RUN ln -s /opt/tensorfold/native/bin/tensorfold-native /usr/local/bin/tensorfold-native
ARG PATCHES_HASH
LABEL tf.patches=${PATCHES_HASH} tf.gate=dory
ENV HF_HOME=/root/.cache/huggingface CUDA_CACHE_PATH=/cache/nv LD_LIBRARY_PATH=/opt/tensorfold/native/lib
WORKDIR /workspace
DOCKERFILE
  rm -rf -- "$ctx"; trap - EXIT
else
  log "Image $IMAGE already built with patches $PATCHES_HASH"
fi

if ! weights_ready; then
  log "Downloading $MODEL_ID @ ${MODEL_REVISION:0:8} into $HF_CACHE/hub"
  if command -v hf >/dev/null; then
    hf download "$MODEL_ID" ${MODEL_REVISION:+--revision "$MODEL_REVISION"} --cache-dir "$HF_CACHE/hub" >/dev/null
  else
    docker run --rm --user "$(id -u):$(id -g)" --network host --entrypoint python ${HF_TOKEN:+-e HF_TOKEN} \
      -v "$HF_CACHE":/hf -e HF_HOME=/hf -e HOME=/tmp "$IMAGE" -c \
      'import sys; from huggingface_hub import snapshot_download; snapshot_download(sys.argv[1], revision=sys.argv[2] or None)' "$MODEL_ID" "$MODEL_REVISION"
  fi
fi
weights_ready || die "$MODEL_ID @ ${MODEL_REVISION:0:8}: no complete snapshot in $HF_CACHE after the download"
log "Checkpoint: $(model_cache_dir)/snapshots/$(snapshot_rev)"
