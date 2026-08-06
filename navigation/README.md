# 导航模块（Agent 调度接口）

本目录向 **Agent 调度模块** 暴露导航能力的 HTTP 约定与 Python 客户端。  
真机服务由 TianJi `retail_nav_http_gateway` 提供（默认 `http://127.0.0.1:8081`）。

## 接口一览

| 方法 | 路径 | 用途 |
|------|------|------|
| `GET` | `/navigation/health` | 探活；仅 `READY` 时允许发导航 |
| `POST` | `/navigation/navigate` | 按 `target_id` 导航，到达后返回 |

### 1. `GET /navigation/health`

无请求体。

成功：

```json
{"status": "READY"}
```

`status` 可能为：`READY` / `STARTING` / `ERROR`。只有 `READY` 时 Agent 才应调用导航。

### 2. `POST /navigation/navigate`

**必须**在请求头携带 `Idempotency-Key`（不要放进 JSON 体）。相同 Key 重试会返回首次结果，不会重复执行物理动作。

请求体：

```json
{"target_id": "H1_F_L1_C01"}
```

成功：

```json
{"status": "SUCCEEDED"}
```

`target_id` 支持：

- 商品货位：`H<1|2>_<F|B>_L<1..5>_C<两位列号>`（如 `H1_F_L1_C01`）
- 业务点：`delivery_place`、`replenishment_pickup`、`receipt_viewpoint`、`task_boundary`、`start` 等

### 错误码（响应体 `error_code`）

| HTTP | error_code | 含义 |
|------|------------|------|
| 400 | `MISSING_IDEMPOTENCY_KEY` | 缺少幂等头 |
| 400 | `INVALID_REQUEST` | JSON / 字段非法 |
| 400 | `INVALID_TARGET` | `target_id` 无法解析 |
| 404 | `NOT_FOUND` | 路径不存在 |
| 409 | `IDEMPOTENCY_KEY_CONFLICT` | 同 Key 不同目标 |
| 503 | `MODULE_NOT_READY` | 导航未就绪 |
| 500 / 504 | `EXECUTION_FAILED` | 执行失败或超时 |

## 调用示例

```bash
curl -s http://127.0.0.1:8081/navigation/health

curl -X POST http://127.0.0.1:8081/navigation/navigate \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: agent:nav-001' \
  -d '{"target_id":"H1_F_L1_C01"}'
```

Python（Agent 调度侧）：

```python
from navigation.client import NavigationClient

nav = NavigationClient("http://127.0.0.1:8081")
assert nav.health() == "READY"
result = nav.navigate("H1_F_L1_C01", idempotency_key="agent:nav-001")
assert result["status"] == "SUCCEEDED"
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `client.py` | Agent 用的薄 HTTP 客户端 |
| `test_client.py` | 客户端单测（mock HTTP，无需真机） |
| `inspect/` | 巡检相关占位 |

## 依赖与真机启动

客户端仅依赖 Python 标准库。真机需先在 TianJi 侧拉起导航桥与 HTTP 网关，例如：

```bash
cd /path/to/TianJi
./scripts/start_nav_camera.sh
```

详见 TianJi：`docs/导航与相机最小使用文档.md`。
