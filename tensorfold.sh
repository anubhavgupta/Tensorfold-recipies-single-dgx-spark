#!/usr/bin/env bash
# Run TensorFold (https://github.com/ashhart/TensorFold) in NVIDIA's PyTorch container on this DGX Spark.
#
# Usage: ./tensorfold.sh [wrapper options] <tensorfold args...>
#
# Leading --tf-* options are read by this script; everything from the first other argument
# onward is passed to `tensorfold` unchanged, so any current or future tensorfold flag works.
#
# Wrapper options (must come first):
#   --tf-version REF     TensorFold tag, branch or commit (default: latest release tag; env TF_VERSION)
#                        "0.6.4" is accepted for "v0.6.4"; "latest" resolves the newest tag.
#   --tf-rebuild         Rebuild the image for REF even if it exists.
#   --tf-offline         Skip GitHub; use a locally built image and cached models only (env TF_OFFLINE=1).
#                        Also used automatically when GitHub is unreachable.
#   --tf-name NAME       Container name (default: auto-generated).
#   --tf-detach          Run in the background (follow with: docker logs -f NAME).
#   --tf-docker-arg ARG  Extra `docker run` argument; repeatable (e.g. --tf-docker-arg=-eFOO=1).
#   --tf-shell           Open a bash shell in the container instead of running tensorfold.
#   --tf-patches DIR     Run a derived image with DIR/*.patch (unified diffs against site-packages, `patch -p0`)
#                        applied on top of REF's image, tagged <image>-p<hash of the patches>. Built once, rebuilt
#                        when the patches change (or with --tf-rebuild); the plain image stays unpatched.
#   --tf-help            Show this help.
#
# Host environment variables named TENSORFOLD_*, NCCL_*, HF_*, CUDA_VISIBLE_DEVICES are forwarded.
#
# Examples:
#   ./tensorfold.sh --version
#   ./tensorfold.sh --tf-version v0.6.4 serve Vontra/Qwen3.8-27B-MLX-4bit --drafter z-lab/Qwen3.8-27B-DFlash2 \
#       --name local-model --host 0.0.0.0 --port 8080
#   TENSORFOLD_CUDA_MEMORY_LIMIT_GB=80 ./tensorfold.sh serve --help
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TF_REPO="${TF_REPO:-https://github.com/ashhart/TensorFold.git}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io/nvidia/pytorch:26.07-py3}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
TF_CACHE="${TF_CACHE:-$HOME/.cache/tensorfold-docker}"

version="${TF_VERSION:-latest}"
rebuild=0 detach=0 shell=0 name="" patches=""
offline="${TF_OFFLINE:-0}"
docker_extra=()

die() { echo "tensorfold.sh: $*" >&2; exit 1; }
need() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tf-version) need "$@"; version="$2"; shift 2 ;;
    --tf-version=*) version="${1#*=}"; shift ;;
    --tf-rebuild) rebuild=1; shift ;;
    --tf-offline) offline=1; shift ;;
    --tf-name) need "$@"; name="$2"; shift 2 ;;
    --tf-name=*) name="${1#*=}"; shift ;;
    --tf-detach) detach=1; shift ;;
    --tf-docker-arg) need "$@"; docker_extra+=("$2"); shift 2 ;;
    --tf-docker-arg=*) docker_extra+=("${1#*=}"); shift ;;
    --tf-shell) shell=1; shift ;;
    --tf-patches) need "$@"; patches="$2"; shift 2 ;;
    --tf-patches=*) patches="${1#*=}"; shift ;;
    --tf-help) sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

[[ "$version" =~ ^[0-9]+(\.[0-9]+)+$ ]] && version="v$version"

remote_refs=""
if [[ $offline -eq 0 ]]; then
  remote_refs="$(GIT_TERMINAL_PROMPT=0 timeout 15 git ls-remote --tags --heads "$TF_REPO" 2>/dev/null)" || {
    echo ">> cannot reach $TF_REPO; using locally built images (offline mode)" >&2
    offline=1
  }
fi

if [[ $offline -eq 1 ]]; then
  # Offline: pick an already-built image; the commit is read from the image label.
  if [[ "$version" == "latest" ]]; then
    version="$(docker images tensorfold --format '{{.Tag}}' | grep -E '^v[0-9.]+-[0-9a-f]{12}$' \
      | sed 's/-[0-9a-f]\{12\}$//' | sort -V | tail -1)"
    [[ -n "$version" ]] || die "offline and no TensorFold image has been built yet"
  fi
  tag_safe="$(sed 's/[^A-Za-z0-9_.-]/_/g' <<<"$version")"
  docker image inspect "tensorfold:${tag_safe}" >/dev/null 2>&1 \
    || die "offline and no local image for $version (built: $(docker images tensorfold --format '{{.Tag}}' | grep -v -- '-[0-9a-f]\{12\}$' | sort -V | tr '\n' ' '))"
  [[ $rebuild -eq 1 ]] && die "--tf-rebuild needs network access"
  image="tensorfold:${tag_safe}"
  commit="$(docker image inspect "$image" --format '{{index .Config.Labels "tensorfold.ref"}}')"
  [[ -n "$commit" ]] || commit="$tag_safe"
  echo ">> offline: running $image (${commit:0:12})" >&2
else
  # Resolve the requested ref to a commit so moving refs (branches, "latest") rebuild when they change.
  if [[ "$version" == "latest" ]]; then
    version="$(awk '{print $2}' <<<"$remote_refs" | grep -E '^refs/tags/v[0-9.]+$' | sed 's#refs/tags/##' | sort -V | tail -1)"
    [[ -n "$version" ]] || die "no release tags found"
  fi
  commit="$(awk -v t="refs/tags/$version^{}" -v l="refs/tags/$version" -v h="refs/heads/$version" \
    '$2==t{p=$1} $2==l&&!p{p=$1} $2==h&&!b{b=$1} END{print (p?p:b)}' <<<"$remote_refs")"
  if [[ -z "$commit" ]]; then
    [[ "$version" =~ ^[0-9a-f]{7,40}$ ]] || die "unknown TensorFold version/ref: $version"
    commit="$version"
  fi
  tag_safe="$(sed 's/[^A-Za-z0-9_.-]/_/g' <<<"$version")"
  image="tensorfold:${tag_safe}-${commit:0:12}"
fi

if [[ $offline -eq 0 ]] && { [[ $rebuild -eq 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; }; then
  echo ">> building $image (TensorFold $version @ ${commit:0:12})" >&2
  build_opts=()
  [[ $rebuild -eq 1 ]] && build_opts+=(--no-cache)
  docker build "${build_opts[@]}" \
    --build-arg BASE_IMAGE="$BASE_IMAGE" --build-arg TF_REPO="$TF_REPO" --build-arg TF_REF="$commit" \
    -t "$image" -t "tensorfold:${tag_safe}" "$SCRIPT_DIR" >&2
fi

# Compiled kernels are tied to the TensorFold build, so they are cached per commit (and per patch set).
kernel_cache="$TF_CACHE/kernels/${commit:0:12}"

if [[ -n "$patches" ]]; then
  [[ -d "$patches" ]] || die "--tf-patches: no directory $patches"
  patch_files=("$patches"/*.patch)
  [[ -e "${patch_files[0]}" ]] || die "--tf-patches: no *.patch files in $patches"
  patch_hash="$(cat "${patch_files[@]}" | sha256sum | cut -c1-12)"
  base_image="$image"
  image="${image}-p${patch_hash}"
  kernel_cache="${kernel_cache}-p${patch_hash}"
  if [[ $rebuild -eq 1 ]] || ! docker image inspect "$image" >/dev/null 2>&1; then
    echo ">> building $image ($base_image + $(cd "$patches" && ls *.patch | paste -sd' '))" >&2
    docker build --build-arg BASE="$base_image" -t "$image" -f - "$patches" >&2 <<'DOCKERFILE'
ARG BASE
FROM ${BASE}
COPY *.patch /opt/tf-patches/
RUN cd "$(python -c 'import os, tensorfold; print(os.path.dirname(os.path.dirname(tensorfold.__file__)))')" \
 && for p in /opt/tf-patches/*.patch; do echo "applying $p"; patch -p0 --forward --no-backup-if-mismatch < "$p" || exit 1; done \
 && python -m compileall -q tensorfold
DOCKERFILE
    docker image inspect "$image" >/dev/null 2>&1 || die "could not build $image"
  fi
fi
mkdir -p "$HF_CACHE" "$TF_CACHE/state" "$kernel_cache"

run=(docker run --rm --init --gpus all --ipc=host --network host
     --ulimit memlock=-1 --cap-add IPC_LOCK
     -v "$HF_CACHE:/root/.cache/huggingface"
     -v "$TF_CACHE/state:/root/.cache/tensorfold"
     -v "$kernel_cache:/cache"
     -e TENSORFOLD_NO_UPDATE_CHECK="${TENSORFOLD_NO_UPDATE_CHECK:-1}")
[[ $offline -eq 1 ]] && run+=(-e HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}")
[[ -e /dev/infiniband ]] && run+=(--device /dev/infiniband)
[[ -n "$name" ]] && run+=(--name "$name")
if [[ $detach -eq 1 ]]; then run+=(-d)
elif [[ -t 0 && -t 1 ]]; then run+=(-it)
else run+=(-i); fi

while IFS='=' read -r var _; do
  [[ "$var" =~ ^(TENSORFOLD_|NCCL_|HF_)[A-Za-z0-9_]*$ || "$var" == CUDA_VISIBLE_DEVICES ]] || continue
  [[ "$var" == TENSORFOLD_NO_UPDATE_CHECK || "$var" == HF_CACHE || "$var" == HF_HOME || "$var" == HF_HUB_CACHE ]] && continue
  run+=(-e "$var")
done < <(env)

run+=("${docker_extra[@]}")
[[ $shell -eq 1 ]] && run+=(--entrypoint /bin/bash "$image") || run+=("$image")

exec "${run[@]}" "$@"
