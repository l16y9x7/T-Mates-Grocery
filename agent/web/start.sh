#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${LOCATE_WEB_PORT:-8090}"
HOST="${LOCATE_WEB_HOST:-0.0.0.0}"
export PICK_PLACE_LOG_DIR="${PICK_PLACE_LOG_DIR:-$PROJECT_ROOT/log}"
cd "$PROJECT_ROOT"
exec uvicorn web.app:app --host "$HOST" --port "$PORT"
