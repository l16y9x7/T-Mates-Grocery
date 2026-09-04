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
