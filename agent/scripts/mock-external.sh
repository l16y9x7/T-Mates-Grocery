#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export PYTHONPATH="$PROJECT_ROOT/src"
CONFIG_FILE="${MOCK_EXTERNAL_CONFIG_FILE:-$PROJECT_ROOT/config/runtime.mock.yaml}"
RUN_DIR="${MOCK_EXTERNAL_RUN_DIR:-$PROJECT_ROOT/run}"
LOG_DIR="${MOCK_EXTERNAL_LOG_DIR:-$PROJECT_ROOT/log/process}"
PID_FILE="${MOCK_EXTERNAL_PID_FILE:-$RUN_DIR/mock-external.pid}"
MOCK_EXTERNAL_PORT="${MOCK_EXTERNAL_PORT:-8109}"

usage() {
  cat <<EOF
用法: $0 {start|stop|restart}

环境变量:
  MOCK_EXTERNAL_CONFIG_FILE  配置文件，默认 config/runtime.mock.yaml
  MOCK_EXTERNAL_PORT         端口提示，默认 8109
  MOCK_EXTERNAL_RUN_DIR      PID 文件目录
  MOCK_EXTERNAL_LOG_DIR      日志目录
  MOCK_EXTERNAL_PID_FILE     PID 文件路径
EOF
}

port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -nP -iTCP:"$MOCK_EXTERNAL_PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$MOCK_EXTERNAL_PORT" 2>/dev/null | tr ' ' '\n' || true
  fi
}

stop_port_listeners() {
  local pids pid
  pids="$(port_pids)"
  [[ -z "$pids" ]] && return 0
  echo "正在清理端口 $MOCK_EXTERNAL_PORT 的监听进程：$pids"
  for pid in $pids; do
    [[ "$pid" == "$$" ]] && continue
    kill "$pid" 2>/dev/null || true
  done
  for _ in {1..20}; do
    [[ -z "$(port_pids)" ]] && return 0
    sleep 0.5
  done
  for pid in $(port_pids); do
    [[ "$pid" == "$$" ]] && continue
    kill -KILL "$pid" 2>/dev/null || true
  done
}

start() {
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  if [[ -f "$PID_FILE" ]]; then
    local pid
    pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
      echo "外部接口 Mock 已经在运行，PID=$pid"
      return 0
    fi
    rm -f "$PID_FILE"
  fi
  if [[ -n "$(port_pids)" ]]; then
    stop_port_listeners
  fi
  command -v uv >/dev/null 2>&1 || {
    echo "错误：未找到 uv，请先运行 scripts/setup.sh。" >&2
    return 1
  }
  local timestamp log_file pid
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/mock-external-$timestamp.log"
  cd "$PROJECT_ROOT"
  nohup uv run --project "$PROJECT_ROOT" --frozen python -m mock_external_service \
    --config "$CONFIG_FILE" --port "$MOCK_EXTERNAL_PORT" >"$log_file" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "外部接口 Mock 启动失败，日志：$log_file" >&2
    rm -f "$PID_FILE"
    tail -n 40 "$log_file" >&2 || true
    return 1
  fi
  echo "外部接口 Mock 已后台启动，PID=$pid"
  echo "地址: http://127.0.0.1:$MOCK_EXTERNAL_PORT"
  echo "日志: $log_file"
}

stop() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
  fi
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "正在停止外部接口 Mock，PID=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.5
    done
    kill -0 "$pid" 2>/dev/null && kill -KILL "$pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
  stop_port_listeners
}

ACTION="${1:-}"
case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  -h|--help) usage ;;
  *) usage >&2; exit 2 ;;
esac
