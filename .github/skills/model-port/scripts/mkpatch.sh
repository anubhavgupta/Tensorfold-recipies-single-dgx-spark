#!/usr/bin/env bash
# Patch file (patch -p0 format, paths tensorfold/...) from BASE to NEW trees, with a header paragraph.
# usage: mkpatch.sh BASEDIR NEWDIR OUT.patch "header text"   (BASEDIR/NEWDIR each contain tensorfold/)
set -euo pipefail
base=$(cd "$1" && pwd); new=$(cd "$2" && pwd); out=$3; hdr=$4
cd /
{ diff -u -r "$base/tensorfold" "$new/tensorfold" --exclude=__pycache__ --exclude="*.orig" || true; } | python3 -c "
import re, sys
s = sys.stdin.read()
s = re.sub(r'^diff .*\n', '', s, flags=re.M)
s = re.sub(r'^Only in.*\n', '', s, flags=re.M)
s = re.sub(r'^(---|\+\+\+) \S*?/tensorfold/', r'\1 tensorfold/', s, flags=re.M)
open(sys.argv[2], 'w').write(sys.argv[1] + '\n\n' + s)
" "$hdr" "$out"
echo "wrote $out ($(grep -c '^@@' "$out") hunks)"
