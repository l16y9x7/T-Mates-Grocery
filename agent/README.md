# Robot Games Agent

本项目是机器人零售比赛的 LangGraph 编排程序。主 Agent 通过 HTTP 调用导航、感知、姿态、商品库和 8086 取放服务，运行 `SORTING`、`SHORTAGE`、`MISPLACED` 三类任务。

## 环境安装

要求 Python 3.11+、Bash 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```bash
scripts/setup.sh
```

## 启动 8086 取放服务

8086 是独立服务，负责一次完整的 `/pick` 或 `/place` 子流程。它需要能访问 8083 视觉理解、8084 抓放和 8085 相机服务。

启动脚本默认在后台运行，并保存服务日志：

```bash
scripts/pick-place.sh start
scripts/pick-place.sh stop
scripts/pick-place.sh restart
```

网页也默认在后台启动：

```bash
scripts/web.sh start
scripts/web.sh stop
scripts/web.sh restart
```

后台进程 PID 保存在 `run/`，服务输出日志保存在 `log/process/`，每次取放任务的详细接口记录仍保存在 `log/<时间>-<幂等键>/`。

默认监听 `0.0.0.0:8086`。生产环境先按现场修改 `config/pick-place.yaml`，再启动服务；主 Agent 的配置必须将 `services.pick_place` 指向该服务地址。

## 启动商品定位工作台

`web/` 提供一个局域网可访问的定位结果页面。它由服务器代理正式感知接口，并在原图上叠加 `bbox` 和 `mask`，所以连接同一服务器的电脑不需要直接访问内网感知接口或服务器文件路径。

```bash
scripts/web.sh start
```

然后在其他电脑打开 `http://服务器IP:8090`。端口可通过 `LOCATE_WEB_PORT` 修改，正式定位接口可通过 `LOCATE_FORMAL_API_URL` 修改。详细说明见 [`web/README.md`](web/README.md)。

## 运行任务

```bash
scripts/run-task.sh SORTING
scripts/run-task.sh SHORTAGE --log-level DEBUG
scripts/run-task.sh MISPLACED
```

任务类型含义：

|类型|用途|
|---|---|
|`SORTING`|识别小票，抓取两件商品并放到交付台|
|`SHORTAGE`|巡检货架，补齐两处缺货|
|`MISPLACED`|巡检货架，将一对乱放商品交换归位|

## 启动任务一独立服务

`task1_service` 执行任务一完整闭环：移动到小票点并调整拍摄位姿，识别两个商品，按
货位手能力映射抓取，移动到交付台调整放置位姿并调用 `pick-place` 放置，最后移动到
`task_boundary` 判定区。任务入口固定处理小票上的两件商品，请求体使用 `{}`。

使用生产配置启动：

```bash
scripts/task1.sh start
```

健康检查：

```bash
curl http://127.0.0.1:8108/health
```

执行完整任务一：

```bash
curl -X POST http://127.0.0.1:8108/task1/run \
  -H 'Content-Type: application/json' \
  -d '{}'
```

任务一实际环境配置见 `config/task1.production.yaml`。

## 使用生产配置

生产地址已经写在 `config/agent.production.yaml`：商品库 `192.168.130.59:25540`，导航 `8081`，姿态 `8082`，感知 `8083`，抓放 `8084`，取放编排 `8086`。运行前确认各服务健康检查返回 `{"status":"READY"}`：

```bash
scripts/run-task.sh SORTING --config config/agent.production.yaml
```

现场需要调整的内容主要是 `services`、`inspection_points`、`timeouts` 和 `product_slots`。8086 的下游地址、相机、标定文件和临时目录位于 `config/pick-place.yaml`。

## 测试

```bash
scripts/run-tests.sh
```

等价的直接命令为：

```bash
PYTHONPATH=src .venv/bin/pytest -q
```

测试使用进程内 HTTP Mock，不会驱动真实机器人。

## 配置与代码入口

|入口|职责|
|---|---|
|`src/agent/main.py`|任务 CLI 和 `run_task` Python 入口|
|`src/agent/workflow.py`|三类 LangGraph 工作流|
|`src/agent/client.py`|主 Agent 的 HTTP 客户端、超时、重试和幂等键|
|`src/pick_place_service/`|8086 独立取放服务|
|`src/task1_service/`|任务一小票识别到抓取独立服务|
|`config/agent.production.yaml`|主 Agent 实际环境服务地址和场地配置|
|`config/task1.production.yaml`|任务一独立服务地址、点位、手能力映射和超时配置|
|`config/pick-place.yaml`|8086 下游服务和相机配置|
