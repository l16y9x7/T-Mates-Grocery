#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
"$PROJECT_ROOT/scripts/start-pick-place-bg.sh" "$@"
"$PROJECT_ROOT/web/start-bg.sh"

echo "pick-place 和网页服务均已启动"
