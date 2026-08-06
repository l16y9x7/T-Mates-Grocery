#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export PYTHONPATH="$PROJECT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if ! command -v uv >/dev/null 2>&1; then
  echo "错误：未找到 uv，请先安装 uv 并运行 scripts/setup.sh。" >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec uv run --project "$PROJECT_ROOT" --frozen python -m agent.main "$@"
