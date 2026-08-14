# 取放流程控制台

这是一个局域网网页控制台。网页支持两类流程：输入商品名称调用 8086 `/pick`，以及调用任务一独立服务完成小票识别到抓取。两类流程都通过同源代理异步执行，并实时显示对应服务写入的 `events.jsonl` 流程事件。

## 启动

确保 8086 已经启动：

```bash
scripts/pick-place.sh start
```

任务一测试还需要启动 8108 独立服务。启动脚本默认在后台运行：

```bash
scripts/task1.sh start
scripts/task1.sh restart
```

停止后台服务：

```bash
scripts/task1.sh stop
```

8108 服务的 PID 保存在 `run/task1.pid`，进程日志写入 `log/process/task1-*.log`。

启动网页（默认在后台运行）：

```bash
scripts/web.sh start
```

完整的推荐启动顺序：

```bash
scripts/task1.sh start
scripts/pick-place.sh start
scripts/web.sh start
```

停止顺序：

```bash
scripts/web.sh stop
scripts/pick-place.sh stop
scripts/task1.sh stop
```

后台进程 PID 保存在 `run/web.pid`，网页进程日志按启动时间写入 `log/process/web-*.log`。

取放服务也可以后台运行：

```bash
scripts/pick-place.sh start
scripts/pick-place.sh stop
```

取放服务 PID 保存在 `run/pick-place.pid`，服务日志写入 `log/process/pick-place-*.log`；每次取放任务的详细接口记录仍保存在 `log/<时间>-<幂等键>/`。

默认监听：

```text
0.0.0.0:8090
```

在同一局域网的电脑浏览器打开：

```text
http://服务器IP:8090
```

例如本机访问地址是 `127.0.0.1`：

```text
http://127.0.0.1:8090
```

如果 8090 已被占用，请修改 `web/config.yaml` 中的 `server.port` 后重启服务。

网页服务默认调用：

```text
http://127.0.0.1:8086/pick
```

任务一服务默认调用：

```text
http://127.0.0.1:8108/task1/run
```

所有下游服务地址、监听端口、请求超时和日志目录都集中配置在 `web/config.yaml`。修改配置后重启网页服务即可生效。网页中的“任务一 · 小票到抓取”区域执行实际环境配置中的完整两件商品流程，服务会把每次运行的事件写入与 8086 相同的 `log/<时间>-<幂等键>/events.jsonl`，网页通过 SSE 实时读取。

页面中的“小票识别”按钮通过同源代理调用视觉理解服务：

```http
POST /api/perception/parse
Content-Type: application/json

{}
```

代理默认请求 `http://127.0.0.1:8083/perception/parse`，并将返回的
`product_names` 列表显示在页面中。视觉理解服务地址配置在 `web/config.yaml` 的 `services.perception_url`。

机器人移动控制默认调用以下真实服务：

```text
位姿准备：      http://192.168.3.226:8084/pose/prepare
位姿健康检查：  http://192.168.3.226:8084/pose/health
导航移动：      http://192.168.3.226:8081/navigation/navigate
导航健康检查：  http://192.168.3.226:8081/navigation/health
```

如需使用其他机器人地址，请直接修改 `web/config.yaml` 中的 `services.navigation_url` 和 `services.pose_url`。

## 页面操作

填写商品名称，选择任务类型和左右手，点击“开始拣取”。页面会显示：

- 任务幂等键
- 定位开始/成功
- 相机 RGB 和 depth 获取
- 深度标准化
- 位姿估计开始/成功
- grasp 请求和响应
- 视觉校验开始/成功或失败
- 8086 最终 HTTP 响应

页面下方的“机器人移动控制”区域可以直接触发：

- `START_POSITION`：回到机器人初始位姿
- `SHELF_PICK_READY`：前往货架预抓取位姿，可选择 `L1/L2/L3`
- `SHELF_VIEW_UPPER`、`SHELF_VIEW_LOWER`：货架上下层扫描位姿
- 在“导航移动”中选择固定点位：任务判定点 `task_boundary`、起点 `start`（同一点），小票识别点 `receipt_viewpoint`、交付台 `delivery_place`（同一点），补货台 `replenishment_pickup`，或 8 个货架巡检点 `H1_F_L_INSPECT` 至 `H2_B_R_INSPECT`
- 切换到“自定义 target_id”后可输入导航地图中的其他站点

每次控制请求都自动生成并发送 `Idempotency-Key`，页面会显示下游请求 JSON、HTTP 状态和响应正文。导航请求字段是 `target_id`，不是带空格的 `target id`。

每次任务的完整输入输出仍保存在 8086 的：

```text
log/<时间>-<幂等键>/
```

任务一服务使用相同目录结构，日志中的步骤包括小票点导航、小票位姿、小票识别、SKU 货位转换、商品导航、抓取位姿和抓取结果。

网页实时日志接口为：

```http
GET /api/pick/<task_id>/events
Accept: text/event-stream
```

放置流程使用与抓取相同的实时日志和视觉展示：

```http
POST /api/place/start
GET /api/place/<task_id>/events
GET /api/place/<task_id>/visual
```

`POST /api/place/start` 接收 `task_type`、`product_name` 和 `hand`，代理调用
8086 的 `/place`。页面会依次显示放置定位、相机取图、放置位姿、释放执行和视觉校验的
流程事件，以及每个已落盘下游接口的完整请求和响应。

旧的 `POST /api/locate` 定位代理接口仍保留，以兼容已有调用方。
