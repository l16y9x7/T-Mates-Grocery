# 独立 Mock 服务试跑说明

独立 Mock 服务用于按照实机方式测试 Agent：Agent 读取 `agent.yaml`，通过真实 TCP/HTTP 请求调用四个端口，不注入测试 Transport。

## 1. 启动 Mock 服务

在项目根目录打开终端一：

```bash
PYTHONPATH=src .venv/bin/python -m agent.mock_server --scenario success
```

一个进程会同时监听：

|模块|地址|
|---|---|
|导航|`http://127.0.0.1:8101`|
|场景理解|`http://127.0.0.1:8102`|
|位姿控制|`http://127.0.0.1:8103`|
|抓放|`http://127.0.0.1:8104`|

检查服务：

```bash
curl http://127.0.0.1:8101/health
curl http://127.0.0.1:8102/health
curl http://127.0.0.1:8103/health
curl http://127.0.0.1:8104/health
```

## 2. 运行 Agent

打开终端二：

```bash
PYTHONPATH=src .venv/bin/python -m agent.main SORTING
PYTHONPATH=src .venv/bin/python -m agent.main SHORTAGE
PYTHONPATH=src .venv/bin/python -m agent.main MISPLACED
```

每次命令创建一个独立任务。任务完成后输出最终 State。

查看 Mock 收到的请求数、实际动作数和幂等键数量：

```bash
curl http://127.0.0.1:8101/mock/state
```

结束 Mock 服务时在终端一按 `Ctrl+C`。

## 3. 场景选择

每次切换场景前先停止原 Mock 服务，再重新启动。

|场景|启动参数|预期|
|---|---|---|
|正常流程|`success`|三个任务均成功|
|慢动作|`slow`|动作延迟但不超过正式长超时，任务成功|
|模块未就绪|`health-error`|健康检查失败，不执行动作|
|导航失败|`navigation-failure`|首次导航返回 500，任务立即失败且不重试|
|延迟发现|`late-findings`|`SHORTAGE` 前两轮无结果，第三轮发现目标并继续完成任务|
|首次超时后恢复|`timeout-recovery`|首次导航超时，原幂等键重试后成功|
|连续超时|`timeout-unknown`|两次导航超时，任务以 `ACTION_RESULT_UNKNOWN` 停止|

示例：

```bash
PYTHONPATH=src .venv/bin/python -m agent.mock_server --scenario navigation-failure
```

超时场景需要使用短超时配置运行 Agent：

```bash
PYTHONPATH=src .venv/bin/python -m agent.main \
  SORTING --config config/agent.mock-fast.yaml
```

## 4. 幂等行为

物理动作第一次收到 `Idempotency-Key` 后开始模拟动作。即使 Agent 因读取超时断开，动作仍继续执行。相同键的重试只等待并返回原动作结果，不创建第二次实际动作。

Mock 默认关闭 Python HTTP Server 的英文访问日志，统一输出中文请求、动作和响应日志：

```text
2026-08-04 16:30:00.100 | Mock 导航模块 | 收到请求 | 方法=POST | 路径=/navigation/navigate | 请求体={"target_id":"receipt_viewpoint"} | 幂等键=...
2026-08-04 16:30:00.101 | Mock 导航模块 | 开始模拟动作 | 动作=导航 | 处理方式=首次执行 | 延迟秒=0.05 | 幂等键=...
2026-08-04 16:30:00.151 | Mock 导航模块 | 发送响应 | 状态码=200 | 响应体={"status":"SUCCEEDED"}
```

重复幂等键会显示 `处理方式=重复请求，不重复执行`。每次调用都能看到被触发的模块、接收的请求方法/路径/JSON/幂等键，以及发送的状态码和 JSON。

`GET /mock/state` 中：

- `request_counts` 是 HTTP 请求次数，包含重试。
- `actual_action_counts` 是实际模拟动作次数。
- `idempotency_keys` 是已记录的逻辑动作数量。

在超时恢复场景中，导航 HTTP 请求数会大于导航实际动作数，这是正确的幂等行为。
