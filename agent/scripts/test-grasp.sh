#!/usr/bin/env bash
set -euo pipefail

# 直接测试真实抓取执行接口，不经过 8086 的定位、取图和位姿估计步骤。
GRASP_URL="${GRASP_URL:-http://192.168.130.50:8084/manipulation/grasp}"
MAX_TIME_SECONDS="${MAX_TIME_SECONDS:-600}"
TASK_TYPE="${TASK_TYPE:-SORTING}"
HAND="${HAND:-left}"
POSE_JSON="${POSE_JSON:-[-67.60958582162857,44.49313133955002,637.5830173492432,-2.42480556254728,-0.8991547469296523,-0.5789370005365923]}"
PRODUCT_TYPE="${PRODUCT_TYPE:-}"
IDEMPOTENCY_KEY="${IDEMPOTENCY_KEY:-direct-grasp-$(date +%Y%m%d-%H%M%S)-$$}"

if [[ -n "$PRODUCT_TYPE" ]]; then
  PAYLOAD="$(printf '{\n  \"task_type\": \"%s\",\n  \"pose\": %s,\n  \"hand\": \"%s\",\n  \"product_type\": \"%s\"\n}\n' \
    "$TASK_TYPE" "$POSE_JSON" "$HAND" "$PRODUCT_TYPE")"
else
  PAYLOAD="$(printf '{\n  \"task_type\": \"%s\",\n  \"pose\": %s,\n  \"hand\": \"%s\"\n}\n' \
    "$TASK_TYPE" "$POSE_JSON" "$HAND")"
fi

printf '调用抓取执行接口: %s\n' "$GRASP_URL"
printf 'task_type=%s hand=%s pose=%s\n' "$TASK_TYPE" "$HAND" "$POSE_JSON"
printf 'Idempotency-Key=%s\n\n' "$IDEMPOTENCY_KEY"

# 不使用 --fail，保留 400/404/500 的响应体，便于判断接口路径和参数问题。
curl -i --max-time "$MAX_TIME_SECONDS" \
  -X POST "$GRASP_URL" \
  -H 'Content-Type: application/json' \
  -H "Idempotency-Key: $IDEMPOTENCY_KEY" \
  -d "$PAYLOAD"
