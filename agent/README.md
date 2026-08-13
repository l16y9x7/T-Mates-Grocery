# Robot Games Agent

本项目是机器人零售比赛的 LangGraph 编排程序。主 Agent 通过 HTTP 调用导航、感知、姿态、商品库和 8086 取放服务，运行 `SORTING`、`SHORTAGE`、`MISPLACED` 三类任务。

## 环境安装

要求 Python 3.11+、Bash 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```bash
scripts/setup.sh
```

## 本地 Mock 运行

`start-mock.sh` 会同时启动导航、感知、姿态、抓放、8086 取放编排和商品库 Mock，端口为 `8101`、`8102`、`8103`、`8104`、`8106`、`8107`。

终端一：

```bash
scripts/start-mock.sh --scenario success
```

终端二：

```bash
scripts/run-task.sh SORTING
scripts/run-task.sh SHORTAGE
scripts/run-task.sh MISPLACED
```

可用场景如下：

|场景|模拟行为|预期结果/验证目标|
|---|---|---|
|`success`|所有健康检查就绪，导航、姿态和取放动作正常返回|三类任务均可完整成功，是默认场景|
|`slow`|小票识别、巡检、导航、姿态和取放动作增加延迟，但仍在 `config/agent.mock.yaml` 超时内|验证长耗时请求不会被误判为超时|
|`random-delay`|小票识别、巡检、导航、姿态、抓取和放置每次调用分别随机等待 5-10 秒后返回|验证所有动作按阻塞接口工作，并模拟每次耗时不同的现场情况|
|`health-error`|姿态服务健康检查返回 `ERROR`|任务在业务动作前失败，验证启动前健康检查和失败路由|
|`navigation-failure`|第一次导航请求返回 HTTP 500|任务立即失败，验证明确的 HTTP/业务错误不会自动重试|
|`late-findings`|`SHORTAGE` 前若干次巡检返回空结果，后续巡检才返回两个缺货位|验证巡检会按正反向持续循环，直到累计两个有效结果|
|`timeout-recovery`|第一次导航超时，复用相同幂等键重试后成功|验证网络超时的一次重试、幂等键复用和恢复后继续流程|
|`timeout-unknown`|同一次导航连续两次超时|任务以 `ACTION_RESULT_UNKNOWN` 失败并停止，避免在无法确认机器人状态时继续动作|

例如运行超时恢复场景：

```bash
scripts/run-demo.sh SORTING timeout-recovery
```

也可以使用一键联调：

```bash
scripts/run-demo.sh SORTING success
```

## 启动 8086 取放服务

8086 是独立服务，负责一次完整的 `/pick` 或 `/place` 子流程。它需要能访问 8083 视觉理解、8084 抓放和 8085 相机服务。

```bash
scripts/start-pick-place.sh --config config/pick-place.yaml
```

后台启动并保存服务日志：

```bash
scripts/start-pick-place-bg.sh --config config/pick-place.yaml
scripts/stop-pick-place.sh
```

网页也支持后台启动和停止：

```bash
web/start-bg.sh
web/stop.sh
```

也可以一键管理两个服务：

```bash
scripts/start-services-bg.sh --config config/pick-place.yaml
scripts/stop-services.sh
```

后台进程 PID 保存在 `run/`，服务输出日志保存在 `log/process/`，每次取放任务的详细接口记录仍保存在 `log/<时间>-<幂等键>/`。

默认监听 `0.0.0.0:8086`。生产环境先按现场修改 `config/pick-place.yaml`，再启动服务；主 Agent 的配置必须将 `services.pick_place` 指向该服务地址。

## 启动商品定位工作台

`web/` 提供一个局域网可访问的定位结果页面。它由服务器代理正式感知接口，并在原图上叠加 `bbox` 和 `mask`，所以连接同一服务器的电脑不需要直接访问内网感知接口或服务器文件路径。

```bash
./web/start.sh
```

然后在其他电脑打开 `http://服务器IP:8090`。端口可通过 `LOCATE_WEB_PORT` 修改，正式定位接口可通过 `LOCATE_FORMAL_API_URL` 修改。详细说明见 [`web/README.md`](web/README.md)。

## 运行任务

```bash
scripts/run-task.sh SORTING
scripts/run-task.sh SHORTAGE --log-level DEBUG
scripts/run-task.sh MISPLACED --config config/agent.mock.yaml
```

任务类型含义：

|类型|用途|
|---|---|
|`SORTING`|识别小票，抓取两件商品并放到交付台|
|`SHORTAGE`|巡检货架，补齐两处缺货|
|`MISPLACED`|巡检货架，将一对乱放商品交换归位|

## 启动任务一独立服务

`task1_service` 用文档规定的接口验证任务一从小票识别到抓取的流程。它会先调用
`POST /perception/parse` 得到商品名，再通过商品库 `25540` 的
`GET /sku/search_by_name` 得到标准货位，最后按“导航 -> 位姿 -> 抓取”的顺序执行。
它不会执行放置动作。

使用生产配置启动：

```bash
PYTHONPATH=src python -m task1_service --config config/task1.production.yaml
```

健康检查：

```bash
curl http://127.0.0.1:8108/health
```

只抓取第一个商品：

```bash
curl -X POST http://127.0.0.1:8108/task1/run \
  -H 'Content-Type: application/json' \
  -d '{"pick_count":1}'
```

抓取两个商品：

```bash
curl -X POST http://127.0.0.1:8108/task1/run \
  -H 'Content-Type: application/json' \
  -d '{"pick_count":2}'
```

本地配置见 `config/task1.mock.yaml` 和 `config/task1.production.yaml`。

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
|`config/agent.mock.yaml`、`config/agent.production.yaml`|主 Agent 服务地址和场地配置|
|`config/task1.mock.yaml`、`config/task1.production.yaml`|任务一独立服务地址和超时配置|
|`config/pick-place.yaml`|8086 下游服务和相机配置|
