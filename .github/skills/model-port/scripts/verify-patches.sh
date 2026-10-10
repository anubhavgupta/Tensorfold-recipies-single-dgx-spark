#!/usr/bin/env bash
# Apply every PATCH_DIR/*.patch in order to a pristine copy and diff against the working tree.
# usage: verify-patches.sh PRISTINE_DIR PATCH_DIR WORK_DIR
set -euo pipefail
tmp=$(mktemp -d); cp -r "$1/tensorfold" "$tmp/"
for p in "$2"/*.patch; do (cd "$tmp" && patch -p0 -s --forward --no-backup-if-mismatch < "$p") || { echo "FAIL $p"; exit 1; }; done
if diff -rq "$tmp/tensorfold" "$3/tensorfold" --exclude=__pycache__ --exclude="*.orig"; then echo "SAME: patches reproduce $3"; fi
rm -rf "$tmp"
