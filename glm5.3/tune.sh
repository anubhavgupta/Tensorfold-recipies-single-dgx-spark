#!/usr/bin/env bash
# One tuning trial: restart the server with the given environment, then measure decode and prefill.
#   ./glm5.3/tune.sh <tag> [VAR=value ...]      e.g.  ./glm5.3/tune.sh mtp2 DRAFT_NUM_TOKENS=2
# Results go to ../flash-3/results/*.<tag>.json (suite.py) and are summarised on stdout.
#   PREFILL (8192,32768)  GEN (256)  CLS (code,prose)  N (1)
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
tag="$1"; shift
"$HERE/stop.sh" >/dev/null; for _ in $(seq 1 30); do pgrep -f "$HERE/state/config.yml" >/dev/null || break; sleep 1; done
sync; "$HERE/../flash-3/exllamav3-tabby/drop-model-cache.sh" "$HOME/.cache/huggingface/hub/models--turboderp--GLM-5.3-Flash-exl3/snapshots/"*/ >/dev/null 2>&1
env "$@" "$HERE/serve.sh" > "$HERE/state/serve.$tag.log" 2>&1 &
for _ in $(seq 1 90); do
  grep -q "Serving OAI API" "$HERE/state/serve.$tag.log" && break
  grep -q "Traceback" "$HERE/state/serve.$tag.log" && { tail -5 "$HERE/state/serve.$tag.log"; exit 1; }
  sleep 4
done
export BASE_URL="http://127.0.0.1:${PORT:-8890}/v1"
cd "$HERE/../flash-3/bench"
python3 suite.py prefill "$tag" "${PREFILL:-8192,32768}" 2>&1 | grep -o "prompt_tokens.*prefill_tok_s': [0-9.]*"
CLS="${CLS:-code,prose}" python3 suite.py content "$tag" "${N:-1}" "${GEN:-256}" 2>&1 | grep -o "^\[.*per_stream_mean': [0-9.]*"
