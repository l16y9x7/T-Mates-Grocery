#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
config_values="$(python -m web.settings)"
IFS=$'\t' read -r HOST PORT <<< "$config_values"
exec uvicorn web.app:app --host "$HOST" --port "$PORT"
