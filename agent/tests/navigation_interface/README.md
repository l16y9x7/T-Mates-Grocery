# 导航模块接口实现与联调说明

本目录交给导航模块负责人 Nora，用于实现和测试 Agent 所需的两个 HTTP 接口。整个目录可以单独拷贝，不依赖 `robot_games` 项目或项目虚拟环境。测试脚本只使用 Python 标准库；接口示例使用 FastAPI 语法展示写法，不要求导航模块采用 FastAPI。

|文件|用途|
|---|---|
|`navigation_server_example.py`|接口结构示例，只说明路由、参数、响应和幂等要求|
|`test_navigation_api.py`|模拟 Agent 的调用方式，测试健康检查、导航和幂等行为|

## 1. 接口一：健康检查

```http
GET /navigation/health
```

请求体：无。

模块完成初始化、能够接收导航任务时返回：

```http
HTTP/1.1 200 OK
Content-Type: application/json
```

```json
{"status": "READY"}
```

`status` 只能是：

|值|含义|
|---|---|
|`STARTING`|模块仍在初始化|
|`READY`|可以执行导航|
|`ERROR`|模块故障|

Agent 只有在四个模块全部为 `READY` 时才启动任务。四个健康检查接口现统一为：

```text
GET /navigation/health
GET /manipulation/health
GET /perception/health
GET /pose/health
```

## 2. 接口二：执行导航

```http
POST /navigation/navigate
Content-Type: application/json
Idempotency-Key: <task_run_id>:<action_id>
```

请求体：

```json
{"target_id": "由导航模块测试人员填写的目标点"}
```

|字段|类型|必填|要求|
|---|---|---|---|
|`target_id`|string|是|非空，必须是导航模块能够识别的固定点位|

这是阻塞式长任务接口。收到请求后开始导航，机器人真正到达目标点后再返回：

```http
HTTP/1.1 200 OK
```

```json
{"status": "SUCCEEDED"}
```

不能在“已经接收任务”或“开始移动”时提前返回 `SUCCEEDED`。当前 Agent 给单次导航请求配置 600 秒读取超时。

执行失败必须返回非 `2xx`，响应至少包含：

```json
{"error_code": "EXECUTION_FAILED"}
```

常用错误建议：

|HTTP 状态|响应示例|场景|
|---:|---|---|
|400|`{"error_code":"INVALID_REQUEST"}`|JSON 或 `target_id` 不合法|
|400|`{"error_code":"MISSING_IDEMPOTENCY_KEY"}`|缺少幂等键|
|409|`{"error_code":"IDEMPOTENCY_KEY_CONFLICT"}`|相同键对应了不同目标|
|503|`{"error_code":"MODULE_NOT_READY"}`|导航模块未就绪|
|500|`{"error_code":"EXECUTION_FAILED"}`|规划、控制或到达判定失败|

## 3. 幂等要求

Agent 构建的键为：

```text
Idempotency-Key = task_run_id + ":" + action_id
```

网络超时后，Agent 最多重试一次，并复用原请求体和原幂等键。导航服务必须做到：

1. 第一次收到新键时创建记录并执行一次导航。
2. 导航执行中再次收到相同键时，等待第一次执行完成，不创建第二个导航任务。
3. 导航完成后再次收到相同键时，直接返回第一次保存的结果。
4. 相同键携带不同 `target_id` 时返回 HTTP 409。

幂等记录使用内存、数据库还是其他方式由导航模块决定。如果要求导航服务进程重启后仍能识别旧键，就需要使用持久化存储。

## 4. 接口示例代码

`navigation_server_example.py` 采用 FastAPI 语法展示接口应该怎么写，但它不是导航模块的完整实现，也不应该直接用于控制机器人。

导航模块只需要按自身框架实现同样的接口，并自行完成示例中的占位函数：

```python
async def navigate_once(target_id: str, idempotency_key: str) -> None:
    ...
```

这个函数必须等待底盘确认到达，且相同 `Idempotency-Key` 只能执行一次真实导航。缓存、并发等待、持久化和底盘调用逻辑均由导航模块自行实现。

## 5. 使用 Agent 模拟脚本测试

先确认机器人周围安全，并选择一个允许实际移动的目标点。`target_id` 由导航模块测试人员通过命令行填写：

```bash
python3 test_navigation_api.py \
  --base-url http://127.0.0.1:8101 \
  --target-id 请填写实际目标点
```

脚本按 Agent 的方式执行：

1. 调用 `GET /navigation/health`，严格检查 `{"status":"READY"}`。
2. 生成 `<UUID>:navigation.interface_test.navigate` 幂等键。
3. 调用 `POST /navigation/navigate`。
4. 仅在网络错误或超时时重试一次，并复用原幂等键。
5. 严格检查成功响应为 `{"status":"SUCCEEDED"}`。

需要额外验证重复请求不会重复移动时，显式增加：

```bash
python3 test_navigation_api.py \
  --base-url http://127.0.0.1:8101 \
  --target-id 请填写实际目标点 \
  --verify-idempotency
```

`--verify-idempotency` 会在第一次成功后，用完全相同的键和请求体再调用一次。启用前必须有人在现场观察，并同时检查导航服务日志中只有一次真实导航执行。

## 6. 通过标准

- 健康检查路径和响应结构正确。
- 导航请求必须包含 `target_id` 和 `Idempotency-Key`。
- HTTP 连接在长程导航期间保持等待，最终到达后才返回成功。
- 非 `2xx` 能明确表达失败，不能用 HTTP 200 搭配失败字符串。
- 相同幂等键的重试不会产生第二次真实移动。
- 服务端日志能够输出目标点、幂等键、首次执行或重复请求、最终结果。
