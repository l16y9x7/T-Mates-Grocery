#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TASK_TYPE="${1:-SORTING}"
SCENARIO="${2:-success}"
MOCK_LOG=""
MOCK_PID=""
MOCK_READY=false

usage() {
  cat <<'EOF'
用法：scripts/run-demo.sh [TASK_TYPE] [SCENARIO]

自动启动独立 Mock、等待六个端口就绪、运行一个 Agent 任务，最后停止 Mock。

TASK_TYPE：SORTING（默认）、SHORTAGE、MISPLACED
SCENARIO：success（默认）、slow、random-delay、health-error、navigation-failure、
          late-findings、timeout-recovery、timeout-unknown

所有 Mock 场景使用 config/agent.mock.yaml。
EOF
}

if [[ "$TASK_TYPE" == "-h" || "$TASK_TYPE" == "--help" ]]; then
  usage
  exit 0
fi

case "$TASK_TYPE" in
  SORTING|SHORTAGE|MISPLACED) ;;
  *)
    echo "错误：未知任务类型 $TASK_TYPE" >&2
    usage >&2
    exit 2
    ;;
esac

case "$SCENARIO" in
  success|slow|random-delay|health-error|navigation-failure|late-findings|timeout-recovery|timeout-unknown) ;;
  *)
    echo "错误：未知 Mock 场景 $SCENARIO" >&2
    usage >&2
    exit 2
    ;;
esac

if ! command -v curl >/dev/null 2>&1; then
  echo "错误：run-demo.sh 需要 curl 来检查 Mock 是否就绪。" >&2
  exit 1
fi

cleanup() {
  local exit_code=$?
  trap - EXIT INT TERM
  if [[ -n "$MOCK_PID" ]] && kill -0 "$MOCK_PID" 2>/dev/null; then
    kill "$MOCK_PID" 2>/dev/null || true
    wait "$MOCK_PID" 2>/dev/null || true
  fi
  if [[ -n "$MOCK_LOG" && -f "$MOCK_LOG" ]]; then
    rm -f "$MOCK_LOG"
  fi
  exit "$exit_code"
}
trap cleanup EXIT INT TERM

MOCK_LOG="$(mktemp "${TMPDIR:-/tmp}/robot-games-agent-mock.XXXXXX.log")"
"$PROJECT_ROOT/scripts/start-mock.sh" --scenario "$SCENARIO" >"$MOCK_LOG" 2>&1 &
MOCK_PID=$!

echo "正在启动 Mock 场景：$SCENARIO"
for _ in {1..50}; do
  if ! kill -0 "$MOCK_PID" 2>/dev/null; then
    echo "错误：Mock 启动失败，日志如下：" >&2
    sed -n '1,160p' "$MOCK_LOG" >&2
    exit 1
  fi
  if curl --fail --silent --max-time 1 http://127.0.0.1:8101/navigation/health >/dev/null \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8102/perception/health >/dev/null \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8103/pose/health >/dev/null \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8104/manipulation/health >/dev/null \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8106/health >/dev/null \
    && curl --fail --silent --max-time 1 http://127.0.0.1:8107/sku/health >/dev/null; then
    MOCK_READY=true
    break
  fi
  sleep 0.1
done

if [[ "$MOCK_READY" != true ]]; then
  echo "错误：等待 Mock 就绪超时，日志如下：" >&2
  sed -n '1,160p' "$MOCK_LOG" >&2
  exit 1
fi

CONFIG_PATH="config/agent.mock.yaml"

echo "开始运行任务：$TASK_TYPE（配置：$CONFIG_PATH）"
set +e
"$PROJECT_ROOT/scripts/run-task.sh" "$TASK_TYPE" --config "$CONFIG_PATH"
TASK_EXIT=$?
set -e

echo "Mock 请求统计："
curl --fail --silent http://127.0.0.1:8101/mock/state || true
echo

if [[ $TASK_EXIT -ne 0 ]]; then
  echo "Agent 进程退出码：$TASK_EXIT" >&2
fi
exit "$TASK_EXIT"
