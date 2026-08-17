#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASKS_HOST="${TASKS_HOST:-127.0.0.1}"
TASKS_PORT="${TASKS_PORT:-8108}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-3600}"
BASE_URL="http://${TASKS_HOST}:${TASKS_PORT}"

TASK_NAMES=(
  [0]="Task0 基准采集"
  [1]="Task1 小票分拣"
  [2]="Task2 缺货补货"
  [3]="Task3 乱放交换"
)

usage() {
  cat <<EOF
用法: $0 [--ensure-services] {0|1|2|3|health}

  0       启动 Task0 基准采集
  1       启动 Task1 小票分拣
  2       启动 Task2 缺货补货（需先成功完成 Task0）
  3       启动 Task3 乱放交换（需先成功完成 Task0）
  health  查询聚合健康状态

选项：
  --ensure-services  若 8108 不可达，先启动 pick-place 和统一任务服务

环境变量：
  TASKS_HOST          默认 127.0.0.1
  TASKS_PORT          默认 8108
  IDEMPOTENCY_KEY     可选；未设置时自动生成
  MAX_TIME_SECONDS    curl 最长等待秒数，默认 3600
EOF
}

print_json() {
  local body="$1"
  if command -v python3 >/dev/null 2>&1; then
    python3 -m json.tool <<<"$body" 2>/dev/null && return 0
  fi
  printf '%s\n' "$body"
}

service_reachable() {
  curl -sS --connect-timeout 3 --max-time 8 -o /dev/null "$BASE_URL/health" >/dev/null 2>&1
}

ensure_services() {
  if service_reachable; then
    return 0
  fi
  echo "统一任务服务不可达，正在启动 pick-place 和统一任务服务…"
  "$PROJECT_ROOT/scripts/pick-place.sh" start
  "$PROJECT_ROOT/scripts/tasks.sh" start
  local _
  for _ in {1..30}; do
    if service_reachable; then
      echo "统一任务服务已就绪：$BASE_URL"
      return 0
    fi
    sleep 0.5
  done
  echo "错误：服务启动后仍无法连接 $BASE_URL/health" >&2
  return 1
}

require_service() {
  if service_reachable; then
    return 0
  fi
  cat >&2 <<EOF
错误：无法连接统一任务服务 $BASE_URL

请先启动服务：
  scripts/pick-place.sh start
  scripts/tasks.sh start

或使用：
  $0 --ensure-services <任务号>
EOF
  return 1
}

run_health() {
  local tmp http_code
  tmp="$(mktemp)"
  http_code="$(
    curl -sS --connect-timeout 5 --max-time 15 \
      -o "$tmp" -w '%{http_code}' \
      "$BASE_URL/health" || true
  )"
  echo "GET $BASE_URL/health"
  echo "HTTP $http_code"
  print_json "$(<"$tmp")"
  rm -f "$tmp"
  [[ "$http_code" == 200 ]]
}

run_task() {
  local task_id="$1"
  local key="${IDEMPOTENCY_KEY:-task${task_id}-$(date +%Y%m%d-%H%M%S)-$$}"
  local tmp http_code
  tmp="$(mktemp)"

  echo "正在启动 ${TASK_NAMES[$task_id]}"
  echo "POST $BASE_URL/tasks/${task_id}/run"
  echo "Idempotency-Key=$key"
  echo "等待任务结束（最长 ${MAX_TIME_SECONDS}s）…"
  echo

  http_code="$(
    curl -sS --connect-timeout 5 --max-time "$MAX_TIME_SECONDS" \
      -o "$tmp" -w '%{http_code}' \
      -X POST "$BASE_URL/tasks/${task_id}/run" \
      -H 'Content-Type: application/json' \
      -H "Idempotency-Key: $key" \
      -d '{}' || true
  )"

  if [[ -z "$http_code" || "$http_code" == "000" ]]; then
    echo "错误：请求失败，服务无响应或已超时。" >&2
    [[ -s "$tmp" ]] && print_json "$(<"$tmp")" >&2
    rm -f "$tmp"
    return 1
  fi

  echo "HTTP $http_code"
  print_json "$(<"$tmp")"
  rm -f "$tmp"

  case "$http_code" in
    2*) return 0 ;;
    409)
      echo "已有任务正在执行，请等待结束后再启动。" >&2
      return 1
      ;;
    *) return 1 ;;
  esac
}

ENSURE_SERVICES=0
TARGET=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ensure-services) ENSURE_SERVICES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    0|1|2|3|health|task0|task1|task2|task3|Task0|Task1|Task2|Task3)
      if [[ -n "$TARGET" ]]; then
        usage >&2
        exit 2
      fi
      TARGET="$1"
      shift
      ;;
    *) usage >&2; exit 2 ;;
  esac
done

[[ -n "$TARGET" ]] || { usage >&2; exit 2; }

command -v curl >/dev/null 2>&1 || {
  echo "错误：未找到 curl。" >&2
  exit 1
}

case "$TARGET" in
  task0|Task0) TARGET=0 ;;
  task1|Task1) TARGET=1 ;;
  task2|Task2) TARGET=2 ;;
  task3|Task3) TARGET=3 ;;
esac

if [[ "$ENSURE_SERVICES" -eq 1 ]]; then
  ensure_services
else
  require_service
fi

if [[ "$TARGET" == "health" ]]; then
  run_health
else
  run_task "$TARGET"
fi
