# Robot Games Agent

面向机器人零售比赛任务的 LangGraph 编排服务。Agent 本身不实现导航、视觉、位姿或抓放算法，而是通过 HTTP 调用四个能力模块，并将它们编排成商品分拣、缺货补货和乱放归位工作流。

## 运行要求

- Linux 或 macOS（启动脚本使用 Bash）
- Python 3.11 或更高版本
- [uv](https://docs.astral.sh/uv/)
- `curl`（仅一键 Mock 联调脚本需要）

首次使用时，在项目根目录执行：

```bash
scripts/setup.sh
```

该命令严格按照 `uv.lock` 创建 `.venv`，并安装运行及测试依赖。所有脚本都能从任意当前目录调用。

## 启动选项

项目提供以下脚本：

|脚本|用途|
|---|---|
|`scripts/setup.sh`|创建或同步开发环境|
|`scripts/start-mock.sh`|启动四个独立能力 Mock 服务|
|`scripts/run-task.sh`|连接配置中的能力服务，运行一个任务|
|`scripts/run-demo.sh`|自动启动 Mock、运行任务、输出统计并回收 Mock|
|`scripts/run-tests.sh`|运行自动化测试，可透传 pytest 参数|

Agent 支持三类任务：

|任务类型|含义|
|---|---|
|`SORTING`|识别小票、拣取两件商品并送到交付台|
|`SHORTAGE`|巡检货架，找到两处缺货后完成补货|
|`MISPLACED`|巡检货架，找到两件乱放商品后交换归位|

命令行还支持 `--config <文件>` 切换服务地址和超时配置，以及 `--log-level DEBUG|INFO|WARNING|ERROR` 调整日志级别。

## 一键本地联调

最快的试跑方式是：

```bash
scripts/run-demo.sh SORTING success
scripts/run-demo.sh SHORTAGE success
scripts/run-demo.sh MISPLACED success
```

参数均可省略，默认执行 `SORTING success`。脚本会等待 `8101-8104` 四个 Mock 端口就绪，任务结束后输出 Mock 请求统计，并停止它启动的 Mock 进程。

可选 Mock 场景如下：

|场景|预期行为|
|---|---|
|`success`|正常完成任务|
|`slow`|各能力动作有延迟，但不超过正式配置超时|
|`health-error`|位姿模块状态为 `ERROR`，任务在动作前失败|
|`navigation-failure`|首次导航返回 HTTP 500，任务停止且不重试|
|`late-findings`|补货巡检到第三轮才发现目标，适合 `SHORTAGE`|
|`timeout-recovery`|首次导航超时，使用相同幂等键重试后成功|
|`timeout-unknown`|两次导航均超时，以 `ACTION_RESULT_UNKNOWN` 停止|

一键脚本会为两个超时场景自动选择短超时配置 `config/agent.mock-fast.yaml`，其余场景使用 `config/agent.yaml`。

## 分开启动 Mock 与 Agent

需要持续观察 Mock 日志或连续运行多个任务时，打开两个终端。

终端一启动 Mock：

```bash
scripts/start-mock.sh --scenario success
```

还可以用 `--host` 修改监听地址：

```bash
scripts/start-mock.sh --host 0.0.0.0 --scenario slow
```

终端二运行任务：

```bash
scripts/run-task.sh SORTING
scripts/run-task.sh SHORTAGE --log-level DEBUG
scripts/run-task.sh MISPLACED --config config/agent.yaml
```

查看 Mock 内部统计：

```bash
curl http://127.0.0.1:8101/mock/state
```

Mock 默认监听以下地址：

|模块|地址|
|---|---|
|导航|`http://127.0.0.1:8101`|
|场景理解|`http://127.0.0.1:8102`|
|位姿控制|`http://127.0.0.1:8103`|
|抓放|`http://127.0.0.1:8104`|

## 连接真实能力服务

复制 `config/agent.yaml` 并修改以下内容：

- `services`：四个能力模块的 HTTP 基础地址。
- `inspection_points`：现场标定的巡检导航点及访问顺序。
- `timeouts`：各动作超时；长动作建议按最长实测耗时预留 30%-50%。
- `product_slots`：货位、商品名称和访问顺序映射。

然后显式传入现场配置：

```bash
scripts/run-task.sh SORTING --config config/agent.robot.yaml --log-level INFO
```

运行前需确保四个服务的健康检查均返回 `{"status":"READY"}`。导航、位姿、抓取和放置属于物理动作，服务端必须按 `Idempotency-Key` 去重，并在动作真正结束后才返回 `{"status":"SUCCEEDED"}`。接口与端口约定见 [机器人能力服务端口规划表](doc/服务器端口规划表.md)。

## 测试

运行完整测试：

```bash
scripts/run-tests.sh
```

只运行一个用例或显示更详细输出：

```bash
scripts/run-tests.sh tests/test_workflow.py::test_sorting_success -v
```

测试使用进程内 HTTP Mock，不需要提前启动独立 Mock 服务，也不会控制真实机器人。

## 项目结构

```text
config/                    Agent 服务地址、超时和场地配置
scripts/                   环境、启动、联调和测试脚本
src/agent/main.py          CLI 与 Python 调用入口
src/agent/workflow.py      三类 LangGraph 工作流
src/agent/client.py        能力模块 HTTP 客户端
src/agent/mock_server.py   四端口独立 Mock 服务
tests/                     工作流与接口测试
doc/                       协议、比赛规则和设计文档
```

直接使用 Python 时，等价入口为 `PYTHONPATH=src uv run python -m agent.main ...` 和 `PYTHONPATH=src uv run python -m agent.mock_server ...`；日常使用建议通过脚本启动，以避免当前目录和模块搜索路径差异。
