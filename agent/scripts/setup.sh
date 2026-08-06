#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"

if ! command -v uv >/dev/null 2>&1; then
  echo "错误：未找到 uv，请先安装 uv：https://docs.astral.sh/uv/" >&2
  exit 1
fi

echo "正在使用锁文件创建开发环境..."
uv sync --project "$PROJECT_ROOT" --frozen
echo "环境就绪：$PROJECT_ROOT/.venv"
