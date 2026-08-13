#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PID_FILE="$PROJECT_ROOT/run/pick-place.pid"

if [[ ! -f "$PID_FILE" ]]; then
  echo "pick-place 服务未发现 PID 文件，可能未运行"
  exit 0
fi

pid="$(<"$PID_FILE")"
if [[ ! "$pid" =~ ^[0-9]+$ ]] || ! kill -0 "$pid" 2>/dev/null; then
  echo "pick-place 服务进程不存在，清理过期 PID 文件"
  rm -f "$PID_FILE"
  exit 0
fi

echo "正在停止 pick-place 服务，PID=$pid"
kill "$pid" 2>/dev/null || true
for _ in {1..20}; do
  if ! kill -0 "$pid" 2>/dev/null; then
    rm -f "$PID_FILE"
    echo "pick-place 服务已停止"
    exit 0
  fi
  sleep 0.5
done

echo "服务未在 10 秒内退出，发送强制终止信号" >&2
kill -KILL "$pid" 2>/dev/null || true
rm -f "$PID_FILE"
echo "pick-place 服务已停止"
