# 外部系统接口 Postman 联调说明

更新日期：2026-09-04

本文供其他系统后端快速调用机器人外部接口。完整字段定义和状态说明见
`外部系统接口设计.md`。

## 1. 服务地址

联调时可以使用 Mock 服务：

```text
http://192.168.200.65:8109
```

真实任务服务默认地址：

```text
http://192.168.200.65:8108
```

Mock 和真实服务使用相同的外部接口路径、请求格式和响应格式。

## 2. Postman 环境变量

建议在 Postman Environment 中配置：

| 变量 | 示例 | 说明 |
|---|---|---|
| `base_url` | `http://192.168.200.66:8109` | Mock 地址；真实联调时将端口改为 `8108` |
| `access_token` | `your-access-token` | 调用接口使用的 Bearer 密钥 |
| `callback_url` | `http://192.168.200.65:8765/api/callback` | 外部系统接收状态回调的接口 |
| `task_run_id` | 留空 | 触发任务成功后自动保存 |

所有请求建议添加：

```http
Authorization: Bearer {{access_token}}
X-Request-Id: postman-{{$guid}}
```

任务触发接口还必须添加：

```http
Content-Type: application/json
Idempotency-Key: postman-{{$guid}}
```

## 3. 健康检查

### 请求

```http
GET {{base_url}}/api/external/v1/health
Authorization: Bearer {{access_token}}
X-Request-Id: postman-health-{{$guid}}
```

系统可以接收任务时返回 HTTP `200`：

```json
{
  "schema_version": "1.0",
  "status": "READY",
  "ready_for_task0": true,
  "ready_for_task1": true,
  "ready_for_task2": false,
  "accepting_tasks": true,
  "active_task": null
}
```

`ready_for_task2=false` 表示尚未成功完成 Task0，应先触发一次 Task0。

## 4. 触发 Task0 理货

### 请求

```http
POST {{base_url}}/api/external/v1/tasks/0/runs
Authorization: Bearer {{access_token}}
Content-Type: application/json
Idempotency-Key: task0-{{$guid}}
X-Request-Id: postman-task0-{{$guid}}
```

Body 选择 `raw` 和 `JSON`：

```json
{
  "external_task_id": "PREP-20260904-0001",
  "status_callback_url": "{{callback_url}}"
}
```

成功时立即返回 HTTP `202 Accepted`：

```json
{
  "schema_version": "1.0",
  "request_id": "postman-task0-example",
  "external_task_id": "PREP-20260904-0001",
  "task_run_id": "task0-20260904-103000-a1b2c3d4",
  "task_type": "TASK0_INVENTORY",
  "task_name": "理货",
  "status": "ACCEPTED",
  "accepted_at": "2026-09-04T10:30:00+08:00",
  "status_callback_enabled": true
}
```

## 5. 触发 Task1 取货

### 请求

```http
POST {{base_url}}/api/external/v1/task1/orders
Authorization: Bearer {{access_token}}
Content-Type: application/json
Idempotency-Key: task1-{{$guid}}
X-Request-Id: postman-task1-{{$guid}}
```

Body：

```json
{
  "external_task_id": "ORD-20260904-0001",
  "external_order_id": "ORD-20260904-0001",
  "items": [
    {
      "sku_id": "SKU_001"
    },
    {
      "sku_id": "SKU_002"
    }
  ],
  "status_callback_url": "{{callback_url}}"
}
```

当前 V1 要求：

- `items` 必须恰好包含两个商品。
- 两个 `sku_id` 必须不同。
- 每件商品数量固定为 1，不需要传 `quantity`。

成功时立即返回 HTTP `202 Accepted`：

```json
{
  "schema_version": "1.0",
  "request_id": "postman-task1-example",
  "external_task_id": "ORD-20260904-0001",
  "external_order_id": "ORD-20260904-0001",
  "task_run_id": "task1-20260904-103000-a1b2c3d4",
  "task_type": "TASK1_PICKUP",
  "task_name": "取货",
  "status": "ACCEPTED",
  "accepted_at": "2026-09-04T10:30:00+08:00",
  "status_callback_enabled": true
}
```

## 6. 触发 Task2 补货

Task2 必须在 Task0 成功完成后触发。

### 请求

```http
POST {{base_url}}/api/external/v1/tasks/2/runs
Authorization: Bearer {{access_token}}
Content-Type: application/json
Idempotency-Key: task2-{{$guid}}
X-Request-Id: postman-task2-{{$guid}}
```

Body：

```json
{
  "external_task_id": "REPLENISH-20260904-0001",
  "status_callback_url": "{{callback_url}}"
}
```

成功时立即返回 HTTP `202 Accepted`，并返回 `task_run_id`。如果 Task0 尚未完成，返回
HTTP `503` 和 `BASELINE_NOT_READY`。

## 7. 自动保存 task_run_id

在 Task0、Task1、Task2 请求的 Postman `Tests` 中加入：

```javascript
pm.test("任务已受理", function () {
    pm.expect(pm.response.code).to.be.oneOf([200, 202]);
});

const body = pm.response.json();
if (body.task_run_id) {
    pm.environment.set("task_run_id", body.task_run_id);
}
```

## 8. 查询任务状态

### 请求

```http
GET {{base_url}}/api/external/v1/tasks/{{task_run_id}}/status
Authorization: Bearer {{access_token}}
X-Request-Id: postman-status-{{$guid}}
```

返回当前最新状态。常见顶层状态：

- `ACCEPTED`：任务已接收。
- `RUNNING`：正在执行。
- `SUCCEEDED`：任务成功完成。
- `PARTIAL_SUCCESS`：部分完成。
- `FAILED`：执行失败。

Mock 默认每个阶段间隔约 1 秒，可以重复点击 `Send` 查看状态变化。

## 9. 状态回调

`status_callback_url` 不是本系统提供的接口，而是外部系统需要提供的 HTTP `POST` 接口。
机器人或 Mock 会主动向该地址发送完整任务状态。

### 9.1 回调请求结构

请求体统一使用以下结构。下面以 Task1 取货过程中的一条状态为例：

```json
{
  "schema_version": "1.0",
  "event_id": "evt-task1-20260904-0008",
  "sequence": 8,
  "event_type": "TASK_PROGRESS",
  "occurred_at": "2026-09-04T10:30:08+08:00",
  "external_task_id": "ORD-20260904-0001",
  "external_order_id": "ORD-20260904-0001",
  "task_run_id": "task1-20260904-103000-a1b2c3d4",
  "task_type": "TASK1_PICKUP",
  "task_name": "取货",
  "status": "RUNNING",
  "display_title": "正在为您取货",
  "display_message": "已取到 1 件商品，正在处理第 2 件商品",
  "current_step": {
    "code": "PICKING",
    "label": "正在取第 2 件商品",
    "progress_percent": 40,
    "message": "机器人正在货架区域取货，请稍候"
  },
  "location": {
    "code": "SHELF",
    "label": "货架区域"
  },
  "next_step": {
    "code": "NAVIGATING_TO_DELIVERY",
    "label": "前往交付台",
    "message": "两件商品取货完成后，将送到交付台"
  },
  "estimated_remaining_seconds": 90,
  "summary": {
    "total_items": 2,
    "items_completed": 1,
    "items_in_progress": 1,
    "items_failed": 0,
    "items_held": 1
  },
  "items": [
    {
      "sku_id": "SKU_001",
      "product_name": "可口可乐罐装",
      "status": "PICKED",
      "status_label": "已取到",
      "picked": true,
      "placed": false,
      "message": "商品已取到，等待送到交付台"
    },
    {
      "sku_id": "SKU_002",
      "product_name": "百事可乐瓶装",
      "status": "PENDING",
      "status_label": "等待处理",
      "picked": false,
      "placed": false,
      "message": "等待处理"
    }
  ],
  "user_notice": {
    "level": "INFO",
    "code": "PICKING_IN_PROGRESS",
    "message": "取货正在进行中，请稍候"
  },
  "last_updated_at": "2026-09-04T10:30:08+08:00",
  "error": null
}
```

### 9.2 公共字段说明

| 字段 | 类型 | 说明 |
|---|---|---|
| `schema_version` | string | 当前固定为 `1.0` |
| `event_id` | string | 当前状态事件唯一 ID，用于去重 |
| `sequence` | integer | 同一 `task_run_id` 下严格递增的序号 |
| `event_type` | string | `TASK_ACCEPTED`、`TASK_PROGRESS`、`TASK_HEARTBEAT` 或 `TASK_COMPLETED` |
| `occurred_at` | string | 状态发生时间，ISO 8601 格式并带时区 |
| `external_task_id` | string | 外部任务号 |
| `external_order_id` | string/null | Task1 订单号；Task0、Task2 为 `null` 或不传 |
| `task_run_id` | string | 本次任务运行 ID |
| `task_type` | string | `TASK0_INVENTORY`、`TASK1_PICKUP` 或 `TASK2_REPLENISHMENT` |
| `task_name` | string | `理货`、`取货` 或 `补货` |
| `status` | string | `ACCEPTED`、`RUNNING`、`SUCCEEDED`、`PARTIAL_SUCCESS` 或 `FAILED` |
| `display_title` | string | 面向用户的状态标题 |
| `display_message` | string | 面向用户的当前状态说明 |
| `current_step` | object | 当前执行阶段和展示进度 |
| `location` | object | 当前业务位置 |
| `next_step` | object/null | 下一执行阶段；任务结束时为 `null` |
| `estimated_remaining_seconds` | integer | 预计剩余秒数，仅用于展示 |
| `summary` | object | 任务统计信息，随任务类型变化 |
| `user_notice` | object | 用户提示级别、提示码和文案 |
| `last_updated_at` | string | 最近一次状态更新时间 |
| `error` | object/null | 失败信息；正常状态为 `null` |

### 9.3 Task0 和 Task1/Task2 的明细字段

Task0 使用 `captures`：

```json
{
  "captures": [
    {
      "target_id": "H2_INSPECT",
      "target_label": "2 号货架",
      "view": "UPPER",
      "status": "COMPLETED",
      "status_label": "已记录",
      "message": "2 号货架上层信息已记录"
    }
  ]
}
```

Task1 使用 `items` 表示订单商品，商品状态包括：
`PENDING`、`LOCATING`、`PICKING`、`PICKED`、`PLACING`、`PLACED`。

Task2 使用 `items` 表示缺货和补货商品，商品状态包括：
`SHORTAGE_FOUND`、`LOCATING`、`PICKING`、`NAVIGATING_TO_SHELF`、`PLACING`、`REPLENISHED`。

Task2 在完成货架巡检、确认缺货商品之前，`items` 可以是空数组。

外部回调接口收到请求并成功处理后，应返回任意 `2xx`，建议响应：

```json
{
  "received": true,
  "event_id": "收到的 event_id"
}
```

回调请求会包含以下请求头：

```http
Authorization: Bearer <callback-access-token>
X-Event-Id: <event-id>
X-Task-Run-Id: <task-run-id>
X-Signature-Timestamp: <ISO-8601-time>
X-Signature: sha256=<hmac-signature>
```

外部系统应：

- 使用 `event_id` 去重。
- 同一个 `task_run_id` 只接受更大的 `sequence`。
- 对成功处理的回调返回 HTTP `2xx`。
- 不要仅依赖回调时间判断状态新旧。

## 10. 常见错误

| HTTP 状态码 | `error_code` | 处理建议 |
|---:|---|---|
| `400` | `INVALID_REQUEST` | 检查请求头和请求体 |
| `401` | `UNAUTHORIZED` | 检查 Bearer 密钥 |
| `409` | `TASK_BUSY` | 等当前任务结束后重试 |
| `409` | `TASK_CONFLICT` | 更换任务号或幂等键 |
| `422` | `INVALID_CALLBACK_URL` | 检查回调地址是否为合法 HTTP(S) URL |
| `503` | `BASELINE_NOT_READY` | 先完成 Task0，再触发 Task2 |

注意：重试同一业务请求时应继续使用原来的 `Idempotency-Key`，不要为每次网络重试生成新值。
