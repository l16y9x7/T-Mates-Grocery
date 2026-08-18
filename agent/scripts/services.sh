#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_IP_OVERRIDE="${ROBOT_IP:-}"
ACTION=""

usage() {
  cat <<EOF
用法: $0 {start|stop|restart} [ROBOT_IP]
      $0 {start|stop|restart} [--robot-ip ADDRESS]

start     启动 pick-place 和统一任务服务
stop      停止两个服务
restart   停止后按指定地址重新启动两个服务
EOF
}

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
    *)
      if [[ -z "$ROBOT_IP_OVERRIDE" ]]; then
        ROBOT_IP_OVERRIDE="$1"
      else
        usage >&2
        exit 2
      fi
      ;;
  esac
  shift
done

[[ -n "$ACTION" ]] || { usage >&2; exit 2; }

if [[ "$ACTION" != "stop" && -n "$ROBOT_IP_OVERRIDE" ]] && ! python3 -c \
  'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1])' "$ROBOT_IP_OVERRIDE" 2>/dev/null; then
  echo "错误：机器人地址必须是有效的 IPv4 地址：$ROBOT_IP_OVERRIDE" >&2
  exit 2
fi

case "$ACTION" in
  start|restart)
    if [[ -n "$ROBOT_IP_OVERRIDE" ]]; then
      "$PROJECT_ROOT/scripts/pick-place.sh" "$ACTION" --robot-ip "$ROBOT_IP_OVERRIDE"
      "$PROJECT_ROOT/scripts/tasks.sh" "$ACTION" --robot-ip "$ROBOT_IP_OVERRIDE"
    else
      "$PROJECT_ROOT/scripts/pick-place.sh" "$ACTION"
      "$PROJECT_ROOT/scripts/tasks.sh" "$ACTION"
    fi
    ;;
  stop)
    "$PROJECT_ROOT/scripts/tasks.sh" stop
    "$PROJECT_ROOT/scripts/pick-place.sh" stop
    ;;
esac
