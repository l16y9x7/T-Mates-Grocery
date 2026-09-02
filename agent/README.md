# Robot Games Task Services

本项目提供机器人零售比赛的统一任务编排服务、8086 取放编排服务和局域网调试页面。
Task0、Task1、Task2、Task3 由同一个 FastAPI 进程提供，旧的主 Agent 和 LangGraph
工作流已退役。

## 可用脚本

全部在项目根目录执行。

| 脚本 | 用法 | 说明 |
|---|---|---|
| `scripts/setup.sh` | `scripts/setup.sh` | 按锁文件安装 Python 依赖 |
| `scripts/pick-place.sh` | `scripts/pick-place.sh {start\|stop\|restart}` | 启动/停止 8086 取放编排服务 |
| `scripts/tasks.sh` | `scripts/tasks.sh {start\|stop\|restart}` | 启动/停止 8108 统一任务服务（含 Web） |
| `scripts/services.sh` | `scripts/services.sh {start\|stop\|restart} [机器人IP]` | 使用指定机器人地址统一控制两个服务 |
| `scripts/health-check.sh` | `scripts/health-check.sh [机器人IP]` | 检查本地服务、机器人及依赖健康状态 |
| `scripts/run-task.sh` | `scripts/run-task.sh [--ensure-services] {0\|1\|2\|3\|health}` | 终端启动 Task0-3，或查询健康状态；加 `--ensure-services` 时若服务未就绪会先拉起 pick-place 和统一任务服务 |
| `scripts/restart-runtime.sh` | `scripts/restart-runtime.sh` | 依次重启 pick-place 和统一任务服务（Web「应用并重启」也调用它） |
| `scripts/run-tests.sh` | `scripts/run-tests.sh [pytest 参数…]` | 跑进程内 HTTP Mock 测试，不驱动真实机器人 |
| `scripts/test-grasp.sh` | `scripts/test-grasp.sh` | 直连抓取执行接口，跳过 8086 的定位、取图和位姿估计 |

常用顺序：

```bash
scripts/setup.sh
scripts/pick-place.sh start
scripts/tasks.sh start
scripts/run-task.sh 0    # 再换成 1、2、3
```

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

Task0、Task2、Task3 的请求体为 `{}`。Task1 也可用 `{}` 让服务端从当前 SKU 商品池
随机生成订单；Web 会先调用 `POST /api/task1/mock-order` 展示两件不同商品，允许重新随机，
再把同一 `order_id` 和 `product_names` 交给 Task1 执行。所有任务均支持可选请求头
`Idempotency-Key`。`GET /health` 返回四个任务的聚合健康状态；任一 Task0-3 正在执行时，
启动其他任务会返回 HTTP `409 TASK_IN_PROGRESS`。

完整流程、真实端口及请求响应见
[`doc/任务流程与pick-place真实端口接口清单.md`](doc/任务流程与pick-place真实端口接口清单.md)。

## 启动

```bash
scripts/pick-place.sh start
scripts/tasks.sh start
```

也可以用一个地址统一启动、停止或重启两个服务；地址只对本次启动的进程生效，不会改写生产配置：

```bash
scripts/services.sh start 192.168.200.66
scripts/health-check.sh 192.168.200.66
scripts/services.sh restart 192.168.200.66
scripts/services.sh stop
```

`services.sh` 省略地址时使用配置文件中的 `robot.ip`。健康检查脚本会检查
8108、8086、感知、SKU，以及机器人 8081/8084/8085 的健康接口；任一检查失败时以
非零状态退出。

浏览器打开 `http://127.0.0.1:8108/`。停止或重启时把 `start` 替换为 `stop` 或
`restart`。统一任务服务的 PID 保存在 `run/tasks.pid`，进程输出保存在
`log/process/tasks-*.log`，每次任务的接口和事件记录保存在
`log/<时间>-<任务键>/`。

健康检查和任务调用可直接用脚本（会阻塞直到该次任务结束）：

```bash
scripts/run-task.sh health
scripts/run-task.sh 0
scripts/run-task.sh 1
scripts/run-task.sh 2
scripts/run-task.sh 3
```

若服务尚未启动，可加 `--ensure-services` 自动拉起 pick-place 和统一任务服务。也可用 curl：

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
`start` 出发，按 Task0 的五个巡检点顺序导航，并在每次导航前执行
`START_POSITION` 复位。每个巡检点固定按 L1-L5 调整位姿，每层等待 2 秒后依次采集
左、右腕部相机的彩色图和深度图，最后复位并返回 `start`。

每次运行的数据保存在独立的 `output/test1/<时间-运行ID>/` 批次目录中。单次成功
运行生成 50 个采集目录，命名格式为 `<导航点>-<层数>-<LEFT|RIGHT>`，例如
`H1_INSPECT-L1-LEFT`；每个目录包含 `rgb.jpg`（相机返回 PNG 时为
`rgb.png`）和 `depth.png`。运行事件及接口调用记录保存在 `log/`。

## 任务说明

- Task0 先到 `start`，再依次巡检五个点；每次拍摄前等待 2 秒，完成后
  返回 `start`，并将 RGB-D 数据保存到 `output/task0/`。
- Task1 从 SKU 服务 `GET /sku/get_all_names` 返回的当前商品池（现为 43 个 SKU）中模拟
  点单两个不同商品；Web 可预览并重新随机，执行时会重新读取目录并复核同一订单。通过
  navigation、pose、pick-place、SKU 四项健康检查后，继续完成 SKU 货位转换、左右手联合
  规划、抓取、交付台放置和任务收尾。单件失败时保留已得分步骤并继续处理另一件。
- Task1 对每个真实下游 HTTP 尝试分别统计调用次数、成功/失败次数、本次耗时、累计耗时和
  平均耗时；重试和失败尝试也计数，并在 Web 的“Task 1 接口统计”中实时展示。
- Task2 先巡检 `H1/H12/H2/H23/H3` 五个点并按精确 `slot_id` 去重；若两件缺货商品
  可分配给左右手，则一次到补货台各抓一件后依次放回，不能分配不同手时退化为两次
  往返。单个巡检点、候选或机械手失败时继续后续候选，成功放置两件后结束。
- Task3 由感知服务自行取图识别一对乱放商品，校验安全手能力后交换货位。
- test1 按五个巡检点和 L1-L5 固定顺序采集左右腕部相机的 50 组 RGB-D 数据，仅通过
  上述一次性命令启动。

Task2 和 Task3 运行前必须先成功完成一次 Task0 基准采集。
Task1/Task2 的启动健康检查仍是严格前置条件；检查通过后的部分完成也保持原有
`status="SUCCEEDED"`，实际结果以响应中每个 `target_items` 的 `picked`、`placed` 为准。

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
