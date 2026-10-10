#!/usr/bin/env bash
# Run a Python test script inside the TensorFold image with a working source tree mounted over the installed package.
# usage: tf-run.sh IMAGE TREE SCRIPT.py [docker -e args...]   e.g. tf-run.sh tensorfold:v0.6.6 /tmp/tfwork /tmp/x.py -e N=16
set -euo pipefail
img=$1; tree=$2; script=$3; shift 3
exec docker run --rm --gpus all --entrypoint python3 -e TENSORFOLD_KV_DTYPE="${TENSORFOLD_KV_DTYPE:-fp8}" "$@" \
  -v "$(realpath "$script")":/x.py \
  -v "$tree/tensorfold":/usr/local/lib/python3.12/dist-packages/tensorfold:ro \
  -v "$HOME/.cache/huggingface":/root/.cache/huggingface:ro "$img" /x.py
