# 机器人任务控制台

Web 控制台与 Task0-3 API 由同一个进程在 `0.0.0.0:8108` 提供。页面使用一个任务
面板切换 Task0、Task1、Task2 和 Task3，共用启动、状态、时间线、错误和结果模块，
并保留 8086 pick/place、导航、位姿和小票识别调试工具。

## 启动

```bash
scripts/pick-place.sh start
scripts/tasks.sh start
```

本机打开 `http://127.0.0.1:8108/`，局域网电脑打开
`http://服务器IP:8108/`。停止服务：

```bash
scripts/tasks.sh stop
scripts/pick-place.sh stop
```

统一服务 PID 位于 `run/tasks.pid`，进程日志写入 `log/process/tasks-*.log`。监听地址、
机器人 IP、Task0-3、pick-place、Web 下游服务和日志目录集中在
`config/runtime.production.yaml`。修改 `robot.ip` 会统一更新所有机器人服务地址。
页面会显示当前生效 IP，并可保存新 IPv4 后依次重启 8086 和 8108；运行中操作需要
再次确认才能强制重启。运行控制按钮不依赖 PID 文件是否存在：服务缺失时会启动，
PID 文件失效但端口仍被占用时会按 8086/8108 监听端口清理旧进程后重启。

## 任务接口

页面不会通过 HTTP 反向调用本机任务服务，而是直接使用统一任务协调器启动后台任务。
四个任务共享全局执行锁。

```http
POST /api/tasks/<0|1|2|3>/start
Content-Type: application/json

{}
```

启动响应包含 `run_id`、`operation_key`、`events_url` 和 `visual_url`。实时接口：

```http
GET /api/task-runs/<run_id>/events
Accept: text/event-stream

GET /api/task-runs/<run_id>/visual

POST /api/task-runs/<run_id>/terminate
```

终止接口取消指定 Task0-3 后台任务、释放全局执行锁，并通过 SSE 返回
`TASK_TERMINATED` 结果；它不调用机器人急停接口。

任务面板按任务展示专属内容：

- Task0：RGB-D 基准采集进度和输出目录。
- Task1：抓取 RGB、mask、bbox 和 6D 位姿。
- Task2：缺货发现、巡检位置、手臂分配和补货结果。
- Task3：乱放商品、正确商品、来源/目标货位和交换结果。

## pick/place 调试

抓取和放置仍通过同源代理调用 `127.0.0.1:8086`：

```http
POST /api/pick/start
GET /api/pick/<run_id>/events
GET /api/pick/<run_id>/visual

POST /api/place/start
GET /api/place/<run_id>/events
GET /api/place/<run_id>/visual
```

请求包含 `task_type`、`product_name` 和 `hand`。旧的 `POST /api/locate` 定位代理继续
保留。

## 其他调试接口

```http
POST /api/perception/parse
POST /api/robot/prepare
POST /api/robot/navigate
GET /api/robot/health
POST /api/robot/gripper/open
```

夹爪松手请求体为 `{"hand":"LEFT"}` 或 `{"hand":"RIGHT"}`，Web 代理会转发到机器人
`/manipulation/gripper/open`，并自动携带 `Idempotency-Key`。

每次请求自动携带 `Idempotency-Key`，页面显示下游请求、HTTP 状态和响应正文。任务
事件来自 `log/<时间>-<任务键>/events.jsonl`，通过 SSE 实时发送。
