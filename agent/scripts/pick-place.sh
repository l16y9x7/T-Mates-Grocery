#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
export PYTHONPATH="$PROJECT_ROOT/src"
export RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_FILE:-$PROJECT_ROOT/config/runtime.production.yaml}"
RUN_DIR="$PROJECT_ROOT/run"
LOG_DIR="$PROJECT_ROOT/log/process"
PID_FILE="$RUN_DIR/pick-place.pid"
PICK_PLACE_PORT="${PICK_PLACE_PORT:-8086}"
ROBOT_IP_OVERRIDE="${ROBOT_IP:-}"
usage() {
  cat <<EOF
用法: $0 {start|stop|restart} [--robot-ip ADDRESS]

--robot-ip ADDRESS  仅对本次启动/重启覆盖配置中的机器人 IPv4 地址
EOF
}

port_pids() {
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -nP -iTCP:"$PICK_PLACE_PORT" -sTCP:LISTEN 2>/dev/null || true
  elif command -v fuser >/dev/null 2>&1; then
    fuser -n tcp "$PICK_PLACE_PORT" 2>/dev/null | tr ' ' '\n' || true
  fi
}

stop_port_listeners() {
  local pids pid
  pids="$(port_pids)"
  [[ -z "$pids" ]] && return 0
  echo "正在清理端口 $PICK_PLACE_PORT 的监听进程：$pids"
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
  for _ in {1..10}; do
    [[ -z "$(port_pids)" ]] && return 0
    sleep 0.2
  done
  echo "无法释放端口 $PICK_PLACE_PORT" >&2
  return 1
}

start() {
  if [[ -n "$ROBOT_IP_OVERRIDE" ]] && ! python3 -c \
    'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1])' "$ROBOT_IP_OVERRIDE" 2>/dev/null; then
    echo "错误：机器人地址必须是有效的 IPv4 地址：$ROBOT_IP_OVERRIDE" >&2
    return 1
  fi
  mkdir -p "$RUN_DIR" "$LOG_DIR"
  if [[ -f "$PID_FILE" ]]; then
    local pid; pid="$(<"$PID_FILE")"
    if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then echo "pick-place 服务已经在运行，PID=$pid"; return 0; fi
    rm -f "$PID_FILE"
  fi
  if [[ -n "$(port_pids)" ]]; then
    stop_port_listeners
  fi
  command -v uv >/dev/null 2>&1 || { echo "错误：未找到 uv，请先运行 scripts/setup.sh。" >&2; return 1; }
  local timestamp log_file pid
  timestamp="$(date +%Y%m%d-%H%M%S)"; log_file="$LOG_DIR/pick-place-$timestamp.log"
  cd "$PROJECT_ROOT"
  if [[ -n "$ROBOT_IP_OVERRIDE" ]]; then export ROBOT_IP="$ROBOT_IP_OVERRIDE"; else unset ROBOT_IP; fi
  nohup uv run --project "$PROJECT_ROOT" --frozen python -m pick_place_service --config "$RUNTIME_CONFIG_FILE" >"$log_file" 2>&1 < /dev/null &
  pid=$!; printf '%s\n' "$pid" > "$PID_FILE"; sleep 1
  if ! kill -0 "$pid" 2>/dev/null; then echo "pick-place 服务启动失败，日志：$log_file" >&2; rm -f "$PID_FILE"; tail -n 40 "$log_file" >&2 || true; return 1; fi
  echo "pick-place 服务已后台启动，PID=$pid"; echo "地址: http://127.0.0.1:8086"; echo "日志: $log_file"
}

stop() {
  local pid=""
  if [[ -f "$PID_FILE" ]]; then
    pid="$(<"$PID_FILE")"
  fi
  if [[ "$pid" =~ ^[0-9]+$ ]] && kill -0 "$pid" 2>/dev/null; then
    echo "正在停止 pick-place 服务，PID=$pid"
    kill "$pid" 2>/dev/null || true
    for _ in {1..20}; do
      if ! kill -0 "$pid" 2>/dev/null; then
        break
      fi
      sleep 0.5
    done
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL "$pid" 2>/dev/null || true
      echo "pick-place 服务已强制停止"
    else
      echo "pick-place 服务已停止"
    fi
  else
    [[ -f "$PID_FILE" ]] && echo "pick-place 服务未运行，已清理 PID 文件"
  fi
  rm -f "$PID_FILE"
  stop_port_listeners
}

ACTION=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    start|stop|restart)
      [[ -z "$ACTION" ]] || { usage >&2; exit 2; }
      ACTION="$1"
      ;;
    --robot-ip)
      [[ $# -ge 2 ]] || { echo "错误：--robot-ip 需要地址。" >&2; usage >&2; exit 2; }
      ROBOT_IP_OVERRIDE="$2"
      shift
      ;;
    --robot-ip=*) ROBOT_IP_OVERRIDE="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *) usage >&2; exit 2 ;;
  esac
  shift
done

[[ -n "$ACTION" ]] || { usage >&2; exit 2; }
case "$ACTION" in
  start) start ;;
  stop) stop ;;
  restart) stop; start ;;
esac
