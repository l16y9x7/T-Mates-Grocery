# Robot Games Task Services

本项目提供机器人零售比赛的统一任务编排服务、8086 取放编排服务和局域网调试页面。
Task0、Task1、Task2、Task3 由同一个 FastAPI 进程提供，旧的主 Agent 和 LangGraph
工作流已退役。

## 环境安装

要求 Python 3.11+、Bash 和 [uv](https://docs.astral.sh/uv/)。在项目根目录执行：

```bash
scripts/setup.sh
```

## 服务入口

| 服务 | 默认地址 | 入口 | 用途 |
|---|---|---|---|
| 统一任务服务 | `127.0.0.1:8108` | `POST /tasks/{0|1|2|3}/run` | 运行 Task0-3 |
| Web | `127.0.0.1:8108` | `GET /` | 统一任务、取放和机器人接口控制台 |
| pick-place | `127.0.0.1:8086` | `POST /pick`、`POST /place` | 完成单次定位、取图、位姿估计和抓放 |

四个任务请求体均为 `{}`，支持可选请求头 `Idempotency-Key`。`GET /health` 返回
四个任务的聚合健康状态；任一 Task0-3 正在执行时，启动其他任务会返回 HTTP
`409 TASK_IN_PROGRESS`。

完整流程、真实端口及请求响应见
[`doc/任务流程与pick-place真实端口接口清单.md`](doc/任务流程与pick-place真实端口接口清单.md)。

## 启动

```bash
scripts/pick-place.sh start
scripts/tasks.sh start
```

浏览器打开 `http://127.0.0.1:8108/`。停止或重启时把 `start` 替换为 `stop` 或
`restart`。统一任务服务的 PID 保存在 `run/tasks.pid`，进程输出保存在
`log/process/tasks-*.log`，每次任务的接口和事件记录保存在
`log/<时间>-<任务键>/`。

健康检查和任务调用示例：

```bash
curl http://127.0.0.1:8108/health

curl -X POST http://127.0.0.1:8108/tasks/0/run \
  -H 'Content-Type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8108/tasks/1/run \
  -H 'Content-Type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8108/tasks/2/run \
  -H 'Content-Type: application/json' -d '{}'
curl -X POST http://127.0.0.1:8108/tasks/3/run \
  -H 'Content-Type: application/json' -d '{}'
```

### test1 双腕相机采集

`test1` 是独立的一次性采集任务，不接入前端，也不占用服务端口。在项目根目录执行：

```bash
UV_CACHE_DIR="$PWD/.cache/uv" PYTHONPATH="$PWD/src" \
uv run --frozen python -m test1_service \
  --config "$PWD/config/runtime.production.yaml"
```

命令启动后会立即驱动机器人完成整套采集流程，成功或失败后自动退出。机器人从
`start` 出发，按 Task0 的八个巡检点顺序导航，并在每次导航前执行
`START_POSITION` 复位。每个巡检点固定按 L1-L5 调整位姿，每层等待 2 秒后依次采集
左、右腕部相机的彩色图和深度图，最后复位并返回 `start`。

每次运行的数据保存在独立的 `output/test1/<时间-运行ID>/` 批次目录中。单次成功
运行生成 80 个采集目录，命名格式为 `<导航点>-<层数>-<LEFT|RIGHT>`，例如
`H1_F_L_INSPECT-L1-LEFT`；每个目录包含 `rgb.jpg`（相机返回 PNG 时为
`rgb.png`）和 `depth.png`。运行事件及接口调用记录保存在 `log/`。

## 任务说明

- Task0 先到 `start`，再以蛇形姿态顺序巡检八个点；每次拍摄前等待 2 秒，完成后
  返回 `start`，并将 RGB-D 数据保存到 `output/task0/`。
- Task1 识别小票并完成 SKU 货位转换、两件商品抓取、交付台放置和任务收尾。
- Task2 往返巡检货架，由感知服务自行取图识别缺货商品；累计两件后从补货台抓取，
  并恢复发现位置完成放置。
- Task3 由感知服务自行取图识别一对乱放商品，校验安全手能力后交换货位。
- test1 按八个巡检点和 L1-L5 固定顺序采集左右腕部相机的 80 组 RGB-D 数据，仅通过
  上述一次性命令启动。

Task2 和 Task3 运行前必须先成功完成一次 Task0 基准采集。

## 配置

| 文件 | 用途 |
|---|---|
| `config/runtime.production.yaml` | 机器人 IP、服务端口、Task0-3、test1、pick-place 与 Web 的全部运行配置；任务同名通用项位于 `tasks.shared` |
| `config/product-hand-options.yaml` | 商品货位、巡检导航点及安全手能力 |
| `config/camera/*.json` | 三台相机的独立标定数据 |

机器人 IP 只配置在 `robot.ip`。修改后需要依次重启 pick-place 和统一任务服务；也可
在 Web 控制台使用“应用并重启”。统一服务固定使用一个 Uvicorn worker，以保证全局
任务锁对 Task0-3 全部生效。

## 测试

```bash
scripts/run-tests.sh
```

测试使用进程内 HTTP Mock，不会驱动真实机器人。
