#!/bin/bash
# Stop the DSV4.1-Flash-1017 server: removes the container, stops the memory guard and
# restores the host sysctls that scripts/serve.sh changed.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
bash scripts/guard.sh stop
