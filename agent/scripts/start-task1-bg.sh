#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/run"
LOG_DIR="$PROJECT_ROOT/log/process"
PID_FILE="$RUN_DIR/task1.pid"
mkdir -p "$RUN_DIR" "$LOG_DIR"

if [[ -f "$PID_FILE" ]]; then
  pid="$(<"$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "task1 服务已经在运行，PID=$pid"
    exit 0
  fi
  rm -f "$PID_FILE"
fi

timestamp="$(date +%Y%m%d-%H%M%S)"
log_file="$LOG_DIR/task1-$timestamp.log"

cd "$PROJECT_ROOT"
nohup "$PROJECT_ROOT/scripts/start-task1.sh" "$@" >"$log_file" 2>&1 < /dev/null &
pid=$!
printf '%s\n' "$pid" > "$PID_FILE"

sleep 1
if ! kill -0 "$pid" 2>/dev/null; then
  echo "task1 服务启动失败，日志：$log_file" >&2
  rm -f "$PID_FILE"
  tail -n 40 "$log_file" >&2 || true
  exit 1
fi

echo "task1 服务已后台启动"
echo "PID: $pid"
echo "地址: http://127.0.0.1:8108"
echo "日志: $log_file"
echo "停止: $PROJECT_ROOT/scripts/stop-task1.sh"
