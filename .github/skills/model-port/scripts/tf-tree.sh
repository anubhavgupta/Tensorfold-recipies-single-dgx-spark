#!/usr/bin/env bash
# Extract TensorFold's installed package from an image into DIR (pristine), optionally applying patches in order.
# usage: tf-tree.sh IMAGE DIR [PATCH_DIR]     e.g. tf-tree.sh tensorfold:v0.6.6 /tmp/tfwork ~/projects/tensorfold/27B-2/patches
set -euo pipefail
img=$1; dir=$2; patches=${3:-}
rm -rf "$dir"; mkdir -p "$dir"
cid=$(docker create "$img")
docker cp "$cid:/usr/local/lib/python3.12/dist-packages/tensorfold" "$dir/tensorfold"
docker rm "$cid" >/dev/null
if [[ -n "$patches" ]]; then
  for p in "$patches"/*.patch; do
    (cd "$dir" && patch -p0 -s --forward --no-backup-if-mismatch < "$p") || { echo "FAILED: $p"; exit 1; }
  done
fi
find "$dir" -name "*.rej" -o -name "*.orig" | head
echo "tree ready: $dir/tensorfold"
