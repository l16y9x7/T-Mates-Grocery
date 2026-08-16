#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_FILE:-$PROJECT_ROOT/config/runtime.production.yaml}"

# The web endpoint detaches this process before returning. Delay long enough for
# the HTTP response to reach the browser before restarting the serving process.
sleep "${RUNTIME_RESTART_DELAY_SECONDS:-1}"
"$PROJECT_ROOT/scripts/pick-place.sh" restart
"$PROJECT_ROOT/scripts/tasks.sh" restart
