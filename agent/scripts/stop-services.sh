#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_ROOT/web/stop.sh"
"$PROJECT_ROOT/scripts/stop-pick-place.sh"

echo "pick-place 和网页服务均已停止"
