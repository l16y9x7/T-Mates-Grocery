#!/usr/bin/env bash
set -euo pipefail

# 8086 取放服务地址；需要测试远程 8086 时可在命令前覆盖此变量。
PICK_PLACE_URL="${PICK_PLACE_URL:-http://127.0.0.1:8086}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-900}"
TASK_TYPE="${TASK_TYPE:-SORTING}"
PRODUCT_NAME="${PRODUCT_NAME:-妙芙绵醇奶油味}"
HAND="${HAND:-left}"

# 每次运行使用新 key，避免 8086 的进程内幂等缓存直接返回上一次结果。
IDEMPOTENCY_KEY="${IDEMPOTENCY_KEY:-coca-cola-can-$(date +%Y%m%d-%H%M%S)-$$}"

printf '调用取放服务: %s/pick\n' "${PICK_PLACE_URL%/}"
printf 'task_type=%s product_name=%s hand=%s\n' "$TASK_TYPE" "$PRODUCT_NAME" "$HAND"
printf 'Idempotency-Key=%s\n\n' "$IDEMPOTENCY_KEY"

curl -i --max-time "$MAX_TIME_SECONDS" \
  -X POST "${PICK_PLACE_URL%/}/pick" \
  -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -d "$(printf '{\n  \"task_type\": \"%s\",\n  \"product_name\": \"%s\",\n  \"hand\": \"%s\"\n}\n' "$TASK_TYPE" "$PRODUCT_NAME" "$HAND")"
