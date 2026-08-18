#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT_ROOT/.cache/uv}"
ROBOT_IP_OVERRIDE="${ROBOT_IP:-}"
TASKS_HOST="${TASKS_HOST:-127.0.0.1}"
TASKS_PORT="${TASKS_PORT:-8108}"
PICK_PLACE_HOST="${PICK_PLACE_HOST:-127.0.0.1}"
PICK_PLACE_PORT="${PICK_PLACE_PORT:-8086}"
PERCEPTION_URL="${PERCEPTION_URL:-http://127.0.0.1:8083}"
SKU_URL="${SKU_URL:-http://127.0.0.1:25540}"

usage() {
  cat <<EOF
用法: $0 [ROBOT_IP]
      $0 [--robot-ip ADDRESS]

检查统一任务、pick-place、感知、SKU，以及机器人导航/位姿/相机健康接口。
未指定地址时使用配置中的 robot.ip。
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip)
      [[ $# -ge 2 ]] || { echo "错误：--robot-ip 需要地址。" >&2; usage >&2; exit 2; }
      ROBOT_IP_OVERRIDE="$2"
      shift
      ;;
    --robot-ip=*) ROBOT_IP_OVERRIDE="${1#*=}" ;;
    -h|--help) usage; exit 0 ;;
    *)
      [[ -z "$ROBOT_IP_OVERRIDE" ]] || { usage >&2; exit 2; }
      ROBOT_IP_OVERRIDE="$1"
      ;;
  esac
  shift
done

if [[ -z "$ROBOT_IP_OVERRIDE" ]]; then
  ROBOT_IP_OVERRIDE="$(ROBOT_IP= PYTHONPATH="$PROJECT_ROOT/src" uv run \
    --project "$PROJECT_ROOT" --frozen python -c \
    'import sys; from runtime_config import load_runtime_document; print(load_runtime_document(sys.argv[1]).robot.ip)' \
    "$PROJECT_ROOT/config/runtime.production.yaml" 2>/dev/null)" || {
    echo "错误：无法从运行配置读取 robot.ip。" >&2
    exit 1
  }
fi

if ! python3 -c \
  'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1])' "$ROBOT_IP_OVERRIDE" 2>/dev/null; then
  echo "错误：机器人地址必须是有效的 IPv4 地址：$ROBOT_IP_OVERRIDE" >&2
  exit 2
fi

check_url() {
  local name="$1" url="$2" code body
  body="$(mktemp)"
  code="$(curl -sS --connect-timeout 3 --max-time 8 -o "$body" -w '%{http_code}' "$url" || true)"
  if [[ "$code" == "200" ]]; then
    echo "[通过] $name: $url"
    rm -f "$body"
    return 0
  fi
  echo "[失败] $name: $url (HTTP ${code:-000})"
  [[ -s "$body" ]] && sed -n '1,2p' "$body" | sed 's/^/       /'
  rm -f "$body"
  return 1
}

checks=0
failed=0
run_check() { checks=$((checks + 1)); check_url "$1" "$2" || failed=$((failed + 1)); }

run_check "统一任务" "http://${TASKS_HOST}:${TASKS_PORT}/health"
run_check "pick-place" "http://${PICK_PLACE_HOST}:${PICK_PLACE_PORT}/health"
run_check "感知" "${PERCEPTION_URL%/}/perception/health"
run_check "SKU" "${SKU_URL%/}/sku/health"
run_check "机器人导航" "http://${ROBOT_IP_OVERRIDE}:8081/navigation/health"
run_check "机器人位姿" "http://${ROBOT_IP_OVERRIDE}:8084/pose/health"
run_check "机器人相机" "http://${ROBOT_IP_OVERRIDE}:8085/camera/health"

echo "健康检查完成：$((checks - failed))/$checks 通过，机器人地址 $ROBOT_IP_OVERRIDE"
[[ "$failed" -eq 0 ]]
