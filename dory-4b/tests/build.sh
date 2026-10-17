#!/usr/bin/env bash
# Build libdory.so for the tests (inside the pytorch container): tests/build.sh [OUT_DIR], default /cache/build.
set -euo pipefail
cd "$(dirname "$0")/.."
out="${1:-/cache/build}"
mkdir -p "$out"
nvcc -shared -Xcompiler -fPIC -O3 -gencode arch=compute_121,code=sm_121 -Iengine -o "$out/libdory.so" \
  engine/dory_kernels.cu engine/dory_mma.cu engine/dory_attn.cu engine/dory_host.cu -lcuda
