#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export PYTHONPATH="$PROJECT_ROOT/src"
export RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_FILE:-$PROJECT_ROOT/config/runtime.production.yaml}"
RUN_DIR="$PROJECT_ROOT/run"
LOG_DIR="$PROJECT_ROOT/log/process"
PID_FILE="$RUN_DIR/tasks.pid"

usage() { echo "用法: $0 {start|stop|restart}"; }

start() {
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "统一任务服务已经在运行，PID=$pid"
      return 0
    fi
    rm -f "$PID_FILE"
  fi
  command -v uv >/dev/null 2>&1 || {
    echo "错误：未找到 uv，请先运行 scripts/setup.sh。" >&2
    return 1
  }
  local timestamp log_file pid
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/tasks-$timestamp.log"
  cd "$PROJECT_ROOT"
  nohup uv run --project "$PROJECT_ROOT" --frozen python -m task_service \
    --config "$RUNTIME_CONFIG_FILE" >"$log_file" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "统一任务服务启动失败，日志：$log_file" >&2
    rm -f "$PID_FILE"
    tail -n 40 "$log_file" >&2 || true
    return 1
  fi
  echo "统一任务服务已后台启动，PID=$pid"
  echo "地址: http://127.0.0.1:8108"
  echo "日志: $log_file"
}

stop() {
  if [[ ! -f "$PID_FILE" ]]; then
    echo "统一任务服务未运行"
    return 0
  fi
  local pid
  pid="$(<"$PID_FILE")"
  if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "统一任务服务未运行，已清理 PID 文件"
    return 0
  fi
  echo "正在停止统一任务服务，PID=$pid"
  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      rm -f "$PID_FILE"
      echo "统一任务服务已停止"
      return 0
    fi
    sleep 0.5
  done
  kill -KILL "$pid" 2>/dev/null || true
  rm -f "$PID_FILE"
  echo "统一任务服务已强制停止"
}

case "${1:-}" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
