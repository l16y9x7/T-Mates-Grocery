#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="$PROJECT_ROOT/agent/run"
LOG_DIR="$PROJECT_ROOT/agent/log/process"
PID_FILE="$RUN_DIR/external-demo.pid"
DEMO_HOST="${DEMO_HOST:-0.0.0.0}"
DEMO_PORT="${DEMO_PORT:-8765}"
ROBOT_TASK_URL="${ROBOT_TASK_URL:-http://127.0.0.1:8108}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

detect_public_host() {
  if [[ -n "${DEMO_PUBLIC_HOST:-}" ]]; then
    printf '%s\n' "$DEMO_PUBLIC_HOST"
    return
  fi
  local detected=""
  if command -v hostname >/dev/null 2>&1; then
    detected="$(hostname -I 2>/dev/null | awk '{print $1}')"
  fi
  printf '%s\n' "${detected:-127.0.0.1}"
}

DEMO_PUBLIC_HOST="$(detect_public_host)"
DEMO_CALLBACK_URL="${DEMO_CALLBACK_URL:-http://$DEMO_PUBLIC_HOST:$DEMO_PORT/api/callback}"

usage() {
  cat <<EOF
用法: $0 {start|stop|restart|status|logs}

环境变量：
  DEMO_HOST          监听地址，默认 0.0.0.0
  DEMO_PORT          监听端口，默认 8765
  DEMO_PUBLIC_HOST   局域网访问地址，默认自动检测本机 IP
  DEMO_CALLBACK_URL  Agent 状态回调地址，默认使用 DEMO_PUBLIC_HOST
  ROBOT_TASK_URL     Agent 地址，默认 http://127.0.0.1:8108
  PYTHON_BIN         Python 命令，默认 python3
EOF
}

port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -nP -iTCP:"$DEMO_PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$DEMO_PORT" 2>/dev/null | tr ' ' '\n' || true
  fi
}

running_pid() {
  [[ -f "$PID_FILE" ]] || return 1
  local pid
  pid="$(<"$PID_FILE")"
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    printf '%s\n' "$pid"
    return 0
  fi
  rm -f "$PID_FILE"
  return 1
}

stop_pid() {
  local pid="$1"
  kill "$pid" 2>/dev/null || true
  for _ in {1..20}; do
    if ! kill -0 "$pid" 2>/dev/null; then
      return
    fi
    sleep 0.25
  done
  kill -KILL "$pid" 2>/dev/null || true
}

start() {
  local pid
  if pid="$(running_pid)"; then
    echo "模拟前端已经在运行，PID=$pid"
    echo "局域网地址: http://$DEMO_PUBLIC_HOST:$DEMO_PORT"
    return
  fi
  if [[ -n "$(port_pids)" ]]; then
    echo "错误：端口 $DEMO_PORT 已被占用，请先执行 $0 stop 或 $0 restart。" >&2
    return 1
  fi
  command -v "$PYTHON_BIN" >/dev/null 2>&1 || {
    echo "错误：未找到 Python 命令：$PYTHON_BIN" >&2
    return 1
  }
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  local timestamp log_file
  timestamp="$(date +%Y%m%d-%H%M%S)"
  log_file="$LOG_DIR/external-demo-$timestamp.log"
  nohup env \
    DEMO_HOST="$DEMO_HOST" \
    DEMO_PORT="$DEMO_PORT" \
    DEMO_CALLBACK_URL="$DEMO_CALLBACK_URL" \
    ROBOT_TASK_URL="$ROBOT_TASK_URL" \
    "$PYTHON_BIN" -u "$PROJECT_ROOT/external-demo/server.py" \
    >"$log_file" 2>&1 < /dev/null &
  pid=$!
  printf '%s\n' "$pid" > "$PID_FILE"
  sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then
    echo "模拟前端启动失败，日志：$log_file" >&2
    rm -f "$PID_FILE"
    tail -n 40 "$log_file" >&2 || true
    return 1
  fi
  echo "模拟前端已后台启动，PID=$pid"
  echo "局域网地址: http://$DEMO_PUBLIC_HOST:$DEMO_PORT"
  echo "Agent 地址: $ROBOT_TASK_URL"
  echo "回调地址: $DEMO_CALLBACK_URL"
  echo "日志: $log_file"
}

stop() {
  local pid=""
  if pid="$(running_pid)"; then
    echo "正在停止模拟前端，PID=$pid"
    stop_pid "$pid"
  fi
  rm -f "$PID_FILE"

  local listeners
  listeners="$(port_pids)"
  if [[ -n "$listeners" ]]; then
    echo "正在清理端口 $DEMO_PORT 的监听进程：$listeners"
    local listener
    for listener in $listeners; do
      stop_pid "$listener"
    done
  fi
  echo "模拟前端已停止"
}

status() {
  local pid
  if pid="$(running_pid)"; then
    echo "模拟前端正在运行，PID=$pid"
    echo "局域网地址: http://$DEMO_PUBLIC_HOST:$DEMO_PORT"
    echo "Agent 地址: $ROBOT_TASK_URL"
    echo "回调地址: $DEMO_CALLBACK_URL"
    return
  fi
  local listeners
  listeners="$(port_pids)"
  if [[ -n "$listeners" ]]; then
    echo "端口 $DEMO_PORT 存在未由脚本管理的监听进程：$listeners"
    return 1
  fi
  echo "模拟前端未运行"
  return 1
}

logs() {
  local log_file
  log_file="$(ls -t "$LOG_DIR"/external-demo-*.log 2>/dev/null | head -1 || true)"
  if [[ -z "$log_file" ]]; then
    echo "尚无模拟前端日志。" >&2
    return 1
  fi
  echo "查看日志: $log_file"
  tail -f "$log_file"
}

ACTION="${1:-}"
case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
  status) status ;;
  logs) logs ;;
  -h|--help|help) usage ;;
  *) usage >&2; exit 2 ;;
esac
