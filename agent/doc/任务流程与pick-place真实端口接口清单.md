# 任务流程与 pick-place 真实接口清单

> 依据当前工作树的生产配置、HTTP 客户端和流程编排代码整理。本文覆盖 Task0 基准采集、Task1（`SORTING`）、Task2（`SHORTAGE`）、Task3（`MISPLACED`）和 pick-place 支持的全部任务类型；接口地址表只记录主机 `127.0.0.1` 与 `<robot_ip>`。

四个任务的领域流程仍位于 `src/task0_service` 至 `src/task3_service`，由 `src/task_service` 统一提供 FastAPI、生命周期和全局执行锁；旧的统一主 Agent 和 LangGraph 工作流已退役。

## 1. pick-place 完整调用接口

pick-place 编排服务默认绑定 `0.0.0.0:8086`，本机可通过 `127.0.0.1:8086` 访问。下表覆盖 `SORTING`、`SHORTAGE` 和 `MISPLACED` 分支可能实际调用的接口：

| 主机:端口 | 接口 | 调用方/用途 |
|---|---|---|
| `127.0.0.1:8086` | `GET /health` | 检查感知、位姿估计、机器人操作和相机下游 |
| `127.0.0.1:8086` | `POST /pick` | 对外接收任务抓取请求 |
| `127.0.0.1:8086` | `POST /place` | 对外接收任务放置请求 |
| `127.0.0.1:8083` | `GET /perception/health` | pick-place 健康检查 |
| `127.0.0.1:8083` | `POST /perception/pick/locate` | 抓取目标定位 |
| `127.0.0.1:8083` | `POST /perception/place/locate` | 非 `SORTING` 放置目标定位 |
| `127.0.0.1:8084` | `GET /manipulation/health` | 物体位姿估计健康检查 |
| `127.0.0.1:8084` | `POST /manipulation/pick_pose` | 根据 RGB、depth、mask 和标定文件估算抓取位姿 |
| `127.0.0.1:8084` | `POST /manipulation/place_pose` | 根据 RGB、depth、mask 和标定文件估算放置位姿 |
| `<robot_ip>:8084` | `GET /manipulation/health` | 机器人抓放服务健康检查 |
| `<robot_ip>:8084` | `POST /manipulation/grasp` | 执行抓取 |
| `<robot_ip>:8084` | `POST /manipulation/release` | 执行释放；SORTING 放置直接调用 |
| `<robot_ip>:8085` | `GET /camera/health` | 相机网关健康检查 |
| `<robot_ip>:8085` | `GET /camera/list` | 相机列表可用性检查 |
| `<robot_ip>:8085` | `GET /camera/snapshot` | 获取相机彩色图 |
| `<robot_ip>:8085` | `GET /camera/stream` | 获取相机 depth 流的第一帧 |

## 2. Task0 实际调用的接口

Task0 由统一任务服务 `0.0.0.0:8108` 编排，领域配置为 `config/runtime.production.yaml`。它在正式任务前采集八个巡检点上下视角的头部 RGB-D 基准数据。

| 主机:端口 | 接口 | Task0 用途 |
|---|---|---|
| `127.0.0.1:8108` | `GET /health` | 聚合检查 Task0-3；Task0 检查导航、姿态、头部相机和彩色/depth 流 |
| `127.0.0.1:8108` | `POST /tasks/0/run` | 启动一次完整基准采集，请求体为空对象 |
| `<robot_ip>:8081` | `GET /navigation/health` | Task0 健康检查 |
| `<robot_ip>:8081` | `POST /navigation/navigate` | 前往开始点、八个巡检点并返回开始点 |
| `<robot_ip>:8084` | `GET /pose/health` | Task0 健康检查 |
| `<robot_ip>:8084` | `POST /pose/prepare` | 准备上下货架观察姿态 |
| `<robot_ip>:8085` | `GET /camera/health` | Task0 健康检查 |
| `<robot_ip>:8085` | `GET /camera/list` | 确认 `head` 的 color/depth 流在线 |
| `<robot_ip>:8085` | `GET /camera/rgbd?camera=head` | 获取包含 RGB、depth 和元数据的 ZIP |

## 3. Task1 实际调用的接口

Task1 由统一任务服务 `0.0.0.0:8108` 编排，领域配置为 `config/runtime.production.yaml`。聚合 `GET /health` 会检查 Task1 的五个依赖；当前 `POST /tasks/1/run` 不执行运行前健康检查，任务会直接进入小票点导航。

| 主机:端口 | 接口 | Task1 用途 |
|---|---|---|
| `127.0.0.1:8108` | `GET /health` | 对外检查 Task1 依赖是否全部就绪 |
| `127.0.0.1:8108` | `POST /tasks/1/run` | 启动一次 SORTING 任务，请求体为空对象 |
| `<robot_ip>:8081` | `GET /navigation/health` | Task1 `GET /health` 依赖检查 |
| `<robot_ip>:8081` | `POST /navigation/navigate` | 前往小票点、商品货位、交付台、任务判定区 |
| `<robot_ip>:8081` | `POST /navigation/nudge` | 抓取或放置失败后后退 3 cm，重试完成后回到微调前位置 |
| `127.0.0.1:8083` | `GET /perception/health` | Task1 `GET /health` 依赖检查 |
| `127.0.0.1:8083` | `POST /perception/parse` | 从小票识别两个商品名 |
| `<robot_ip>:8084` | `GET /pose/health` | Task1 `GET /health` 依赖检查 |
| `<robot_ip>:8084` | `POST /pose/prepare` | 准备起始、看小票、货架抓取、交付台放置姿态 |
| `127.0.0.1:8086` | `GET /health` | Task1 `GET /health` 依赖检查 |
| `127.0.0.1:8086` | `POST /pick` | 委托 pick-place 抓取商品 |
| `127.0.0.1:8086` | `POST /place` | 委托 pick-place 放置商品 |
| `127.0.0.1:25540` | `GET /sku/health` | Task1 `GET /health` 依赖检查 |
| `127.0.0.1:25540` | `GET /sku/search_by_name` | 将商品名解析为唯一货位 |

## 4. Task2 实际调用的接口

Task2 由统一任务服务 `0.0.0.0:8108` 编排，领域配置为 `config/runtime.production.yaml`。任务启动时仍检查 Task0 预先生成的 16 张基准 `rgb.jpg`；巡检识别由感知服务自行获取基准图和当前图。`POST /tasks/2/run` 开始时先并发检查五个直接服务依赖。

| 主机:端口 | 接口 | Task2 用途 |
|---|---|---|
| `127.0.0.1:8108` | `GET /health` | 聚合检查 Task0-3；Task2 检查五个下游、头部彩色流和 Task0 基准图 |
| `127.0.0.1:8108` | `POST /tasks/2/run` | 启动一次 `SHORTAGE` 补货任务，请求体为空对象 |
| `<robot_ip>:8081` | `GET /navigation/health` | 任务启动健康检查 |
| `<robot_ip>:8081` | `POST /navigation/navigate` | 前往巡检点、补货台和任务判定区 |
| `<robot_ip>:8081` | `POST /navigation/nudge` | 抓取或放置失败后后退 3 cm，重试完成后回到微调前位置 |
| `127.0.0.1:8083` | `GET /perception/health` | 任务启动健康检查 |
| `127.0.0.1:8083` | `POST /perception/inspect` | 在当前巡检点的上/下观察姿态识别缺货商品；感知服务自行取图 |
| `<robot_ip>:8084` | `GET /pose/health` | 任务启动健康检查 |
| `<robot_ip>:8084` | `POST /pose/prepare` | 准备起始、货架观察和补货台抓取姿态 |
| `<robot_ip>:8085` | `GET /camera/health`、`GET /camera/list` | 检查头部彩色流 |
| `127.0.0.1:8086` | `GET /health` | 任务启动健康检查 |
| `127.0.0.1:8086` | `POST /pick` | 使用 `SHORTAGE` 类型从补货台抓取商品 |
| `127.0.0.1:8086` | `POST /place` | 使用 `SHORTAGE` 类型向货架放置商品 |

Task2 Agent 不直接调用 SKU 服务，也不再上传巡检图片；商品手臂能力以及巡检导航点到代表货位 `location_id` 的映射来自 `config/product-hand-options.yaml`。感知服务会在 `/perception/inspect` 内部自行获取基准图和当前图，并调用 SKU 服务获取候选商品。

## 5. Task3 实际调用的接口

Task3 由统一任务服务 `0.0.0.0:8108` 编排，领域配置为 `config/runtime.production.yaml`。任务启动时仍检查 Task0 预先生成的 16 张基准 `rgb.jpg`；乱放识别由感知服务自行获取基准图和当前图。

| 主机:端口 | 接口 | Task3 用途 |
|---|---|---|
| `127.0.0.1:8108` | `GET /health` | 聚合检查 Task0-3；Task3 检查六个下游、头部彩色流和 Task0 基准图 |
| `127.0.0.1:8108` | `POST /tasks/3/run` | 启动一次 `MISPLACED` 交换任务，请求体为空对象 |
| `<robot_ip>:8081` | `GET /navigation/health` | 任务启动健康检查 |
| `<robot_ip>:8081` | `POST /navigation/navigate` | 前往巡检点、两个商品货位对应导航点和任务判定区 |
| `<robot_ip>:8081` | `POST /navigation/nudge` | 抓取或放置失败后后退 3 cm，重试完成后回到微调前位置 |
| `<robot_ip>:8084` | `GET /pose/health` | 任务启动健康检查 |
| `<robot_ip>:8084` | `POST /pose/prepare` | 准备复位、观察、货架抓取和货架放置姿态 |
| `<robot_ip>:8085` | `GET /camera/health`、`GET /camera/list` | 检查头部彩色流 |
| `127.0.0.1:8083` | `GET /perception/health` | 任务启动健康检查 |
| `127.0.0.1:8083` | `POST /perception/inspect` | 在当前巡检点的上/下观察姿态识别乱放商品名对；感知服务自行取图 |
| `127.0.0.1:25540` | `GET /sku/health` | 任务启动健康检查 |
| `127.0.0.1:25540` | `GET /sku/search_by_name` | 将两个商品名转换为 P1/P2 标准货位 |
| `127.0.0.1:8086` | `GET /health` | 任务启动健康检查 |
| `127.0.0.1:8086` | `POST /pick`、`POST /place` | 以 `MISPLACED` 类型完成两抓两放 |

## 6. pick-place 调用顺序

### 6.1 全部任务类型 `/pick`

Task1 使用 `task_type=SORTING`：

```json
{"task_type":"SORTING","product_name":"商品名","hand":"LEFT","level":"L2"}
```

Task2 在补货台完成 `REPLENISHMENT_TABLE_PICK_READY` 姿态准备后，使用 `task_type=SHORTAGE` 调用同一个 `8086 /pick` 接口：

```json
{"task_type":"SHORTAGE","product_name":"商品名","hand":"LEFT"}
```

Task3 使用来源货位的层号调用同一个接口，例如：

```json
{"task_type":"MISPLACED","product_name":"商品名","hand":"RIGHT","level":"L4"}
```

三种任务进入 8086 后执行相同的定位、取图、位姿估计和抓取步骤；`task_type` 保持原值并传给定位和机器人抓取接口。Task1 和 Task3 的 `level` 也会继续传给抓取定位接口；Task2 在补货台抓取，没有货架层号，因此省略该字段。相机选择不同：`SHORTAGE` 无论左手还是右手都使用 `head`；`SORTING` 和 `MISPLACED` 按手臂使用左/右腕相机：

```text
Task1 (SORTING) / Task2 (SHORTAGE) / Task3 (MISPLACED) -> POST 127.0.0.1:8086/pick
  -> POST 127.0.0.1:8083/perception/pick/locate
  -> GET <robot_ip>:8085/camera/snapshot?camera=<按任务类型选择>&type=color
  -> GET <robot_ip>:8085/camera/stream?camera=<同一相机>&type=depth
  -> POST 127.0.0.1:8084/manipulation/pick_pose (multipart/form-data)
  -> POST <robot_ip>:8084/manipulation/grasp
  <- {"status":"SUCCEEDED"}
```

上述 `/pick` 分支在 `grasp` 成功后直接返回；结果视觉校验代码已注释，三种任务都不调用 `/perception/pick/check`。

### 6.2 SORTING `/place`

Task1 已先调用 `<robot_ip>:8084/pose/prepare` 准备 `DELIVERY_TABLE_PLACE_READY`，因此 `8086` 不再定位、取图或估算放置位姿：

```text
Task1 -> POST 127.0.0.1:8086/place
  -> POST <robot_ip>:8084/manipulation/release
  <- {"status":"SUCCEEDED"}
```

### 6.3 SHORTAGE 与 MISPLACED `/place`

Task2 使用 `task_type=SHORTAGE` 调用 `/place`。该分支不使用 Task2 恢复的观察姿态作为最终放置位姿，而是继续完成放置定位、头部 RGB-D 取图、放置位姿估计和释放。`MISPLACED /place` 执行相同链路：

```text
Task2 (SHORTAGE) / Task3 (MISPLACED) -> POST 127.0.0.1:8086/place
  -> POST 127.0.0.1:8083/perception/place/locate
  -> GET <robot_ip>:8085/camera/snapshot?camera=head&type=color
  -> GET <robot_ip>:8085/camera/stream?camera=head&type=depth
  -> POST 127.0.0.1:8084/manipulation/place_pose (multipart/form-data)
  -> POST <robot_ip>:8084/manipulation/release
  <- {"status":"SUCCEEDED"}
```

`release` 成功后直接返回；结果视觉校验代码已注释，不调用 `/perception/place/check`。

### 6.4 `8086 /health`

```text
GET 127.0.0.1:8086/health
  -> 并发 GET 127.0.0.1:8083/perception/health
  -> 并发 GET 127.0.0.1:8084/manipulation/health
  -> 并发 GET <robot_ip>:8084/manipulation/health
  -> 并发 GET <robot_ip>:8085/camera/health
  -> GET <robot_ip>:8085/camera/list
```

## 7. Task0 调用顺序

```text
POST 127.0.0.1:8108/tasks/0/run {}
  -> 检查 navigation、pose、camera 健康状态和 head RGB-D 流
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"start"}
  -> 对八个巡检点按蛇形姿态顺序：
       POST <robot_ip>:8081/navigation/navigate {"target_id":"<巡检点>"}
       奇数点：SHELF_VIEW_UPPER -> 等待 2 秒 -> 拍摄 -> SHELF_VIEW_LOWER -> 等待 2 秒 -> 拍摄
       偶数点：复用 SHELF_VIEW_LOWER -> 等待 2 秒 -> 拍摄 -> SHELF_VIEW_UPPER -> 等待 2 秒 -> 拍摄
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"start"}
  -> 原子替换 output/task0/<巡检点>_<UPPER|LOWER>/ 下的三项数据
  <- Task0 SUCCEEDED
```

相邻点的首个姿态与上一点末姿态相同时不会重复调用 `/pose/prepare`。每个 ZIP 必须包含非空 `rgb.jpg`、NumPy 格式 `depth_mm.npy` 和 JSON 对象 `meta.json`；无效的新数据不会覆盖已有基准。

## 8. Task1 调用顺序

`POST /tasks/1/run` 当前不执行运行前健康检查。商品手臂分配由配置决定；以下为两件商品使用不同手臂时的主顺序。

```text
POST 127.0.0.1:8108/tasks/1/run {}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"receipt_viewpoint"}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"RECEIPT_VIEW"}
  -> POST 127.0.0.1:8083/perception/parse
  -> 对两个商品分别：
       GET 127.0.0.1:25540/sku/search_by_name?name=<商品名>
  -> 对每件商品分别：
       POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
       POST <robot_ip>:8081/navigation/navigate {"target_id":"<货位对应导航点>"}
       POST <robot_ip>:8084/pose/prepare {"pose_type":"SHELF_PICK_READY","shelf_level":"Lx"}
       POST 127.0.0.1:8086/pick {"task_type":"SORTING","level":"Lx", ...}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"delivery_place"}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"DELIVERY_TABLE_PLACE_READY"}
  -> 对每件商品分别：POST 127.0.0.1:8086/place {"task_type":"SORTING", ...}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"task_boundary"}
  <- Task1 SUCCEEDED
```

如果两件商品只能使用同一只手，Task1 会对每件商品执行“抓取 -> 前往交付台 -> 准备放置姿态 -> 放置”，再处理下一件；重复导航到当前已在的目标点时会跳过重复的复位和导航请求。

Task1 的 `/pick` 或 `/place` 明确报错或结果未知时，会先调用 `/navigation/nudge` 向 `back` 后退 3 cm，再使用新的动作键重试一次相同的 8086 请求，最后无论重试成功或失败都调用 `return`。抓取最终失败的商品跳过放置；放置最终失败时保留持物状态。其他手的独立动作继续执行，单手仍被占用时跳过下一件商品。全部抓放成功时仍前往 `task_boundary`；存在最终失败时跳过判定区、返回 `start` 后统一报错。

## 9. Task2 调用顺序

Task2 的巡检点按以下顺序配置：

```text
H1_F_L_INSPECT -> H1_F_R_INSPECT -> H1_B_L_INSPECT -> H1_B_R_INSPECT
-> H2_F_L_INSPECT -> H2_F_R_INSPECT -> H2_B_L_INSPECT -> H2_B_R_INSPECT
```

`POST /tasks/2/run` 先并发检查五个直接服务依赖、头部彩色流和 Task0 基准图，然后严格串行巡检。每个巡检点先恢复 `START_POSITION` 并导航，再分别以 `SHELF_VIEW_UPPER` 和 `SHELF_VIEW_LOWER` 调用缺货识别；基准图和当前图由感知服务自行获取。找到两件后立即停止；一轮不足两件时沿原路反向巡检，并持续往返直到累计两个唯一发现。

两件商品可使用不同手时，主顺序如下：

```text
POST 127.0.0.1:8108/tasks/2/run {}
  -> 并发健康检查：
       GET <robot_ip>:8081/navigation/health
       GET 127.0.0.1:8083/perception/health
       GET <robot_ip>:8084/pose/health
       GET <robot_ip>:8085/camera/health
       GET <robot_ip>:8085/camera/list
       GET 127.0.0.1:8086/health
       检查 output/task0 下 16 张基准 rgb.jpg
  -> 对每个巡检点，直到找到两件缺货商品：
       POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
       POST <robot_ip>:8081/navigation/navigate {"target_id":"<巡检点>"}
       POST <robot_ip>:8084/pose/prepare {"pose_type":"SHELF_VIEW_UPPER"}
       POST 127.0.0.1:8083/perception/inspect {"task_type":"SHORTAGE", "location_id":"<代表货位>", "pose_type":"SHELF_VIEW_UPPER"}
       POST <robot_ip>:8084/pose/prepare {"pose_type":"SHELF_VIEW_LOWER"}
       POST 127.0.0.1:8083/perception/inspect {"task_type":"SHORTAGE", "location_id":"<代表货位>", "pose_type":"SHELF_VIEW_LOWER"}
  -> 根据商品名、发现时巡检点和上/下观察姿态分配抓取手
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"replenishment_pickup"}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"REPLENISHMENT_TABLE_PICK_READY"}
  -> 对两件商品分别：POST 127.0.0.1:8086/pick {"task_type":"SHORTAGE", ...}
  -> 对每件商品分别：
       POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
       POST <robot_ip>:8081/navigation/navigate {"target_id":"<发现时巡检点>"}
       POST <robot_ip>:8084/pose/prepare {"pose_type":"<SHELF_VIEW_UPPER|SHELF_VIEW_LOWER>"}
       POST 127.0.0.1:8086/place {"task_type":"SHORTAGE", ...}
  -> POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
  -> POST <robot_ip>:8081/navigation/navigate {"target_id":"task_boundary"}
  <- Task2 SUCCEEDED
```

Task2 会把商品与发现时的“巡检点 + 上/下观察姿态”绑定，放置前恢复该上下文。Task2 不调用 `SHELF_PLACE_READY`，也不介入 8086 内部的放置定位和位姿估计。

如果两件商品只能使用同一只手，Task2 按“前往补货台 -> 抓取一件 -> 前往对应巡检点 -> 放置一件”处理完当前商品后，再处理下一件。已在目标点时同样会跳过重复复位和导航。

Task2 对 `/pick` 和 `/place` 使用与 Task1 相同的后退、重试和回原点策略。抓取最终失败时不再前往对应货架放置；放置最终失败时该手继续记录为持物状态，另一只手的商品仍继续处理。单手分支中如果上一件放置失败，下一轮会在前往补货台前因手被占用而跳过。存在最终抓放失败时直接返回 `start` 并统一报错，不前往 `task_boundary`。

## 10. Task3 调用顺序

Task3 与 Task2 使用相同的八点往返巡检路线和 8083 缺货/乱放识别契约；区别是 Task3 请求 `MISPLACED` 并在发现一组乱放结果后立即停止巡检。感知服务自行获取基准图和当前图，任务编排不再上传图片。

```text
POST 127.0.0.1:8108/tasks/3/run {}
  -> 检查 navigation、perception、pose、pick-place、sku、camera 和 Task0 基准图
  -> 对每个巡检点的 UPPER/LOWER 视角，直到发现一组乱放商品：
       POST <robot_ip>:8084/pose/prepare {"pose_type":"START_POSITION"}
       POST <robot_ip>:8081/navigation/navigate {"target_id":"<巡检点>"}
       POST <robot_ip>:8084/pose/prepare {"pose_type":"<SHELF_VIEW_UPPER|SHELF_VIEW_LOWER>"}
       POST 127.0.0.1:8083/perception/inspect
         {task_type, location_id, pose_type}
  -> 对 misplaced_product_name 和 gt_product_name 分别调用 SKU search_by_name
  -> 推导 P1=gt_product_name 当前应在的货位，P2=misplaced_product_name 标准货位
  -> 按 product-hand-options 为两项交换作业分配不同且对来源/目标均安全的手
  -> 到 P1，准备 SHELF_PICK_READY + P1 层号，以 level=P1 层号抓取 misplaced_product_name
  -> 到 P2，准备 SHELF_PICK_READY + P2 层号，以 level=P2 层号抓取 gt_product_name
  -> 保持在 P2，准备 SHELF_PLACE_READY + P2 层号，放置第一件商品
  -> 返回 P1，准备 SHELF_PLACE_READY + P1 层号，放置第二件商品
  -> 返回 task_boundary
  <- Task3 SUCCEEDED
```

Task3 要求感知只返回一组不同的非空商品名。P1 会使用发现时的巡检点和上下观察层级消歧；P2 必须是唯一配置货位。无法确定货位或无法为两件商品分配不同的安全手时，在任何抓放动作前以 HTTP `422` 失败。

Task3 的两件交换商品使用不同手，因此一件商品在后退微调后仍抓取或放置失败时，只跳过依赖该结果的动作，另一只手继续完成仍可执行的交换步骤。交换不完整时不进入 `task_boundary`，而是导航回 `start` 后统一报错。

## 11. 接口规范

### 11.1 Task0 编排接口：`127.0.0.1:8108`

#### `POST /tasks/0/run`

请求体为 `{}`，支持可选 `Idempotency-Key`。流程先导航到 `start`，按 UPPER/LOWER 蛇形顺序采集并在每次拍摄前等待 2 秒，完成后返回 `start`。成功响应包含全部 16 项采集结果及其 `rgb_path`、`depth_path`、`meta_path`；同一进程已有任务时返回 HTTP `409 TASK_IN_PROGRESS`。

#### `GET /health`

该接口聚合 Task0-3 健康状态，响应包含 `tasks` 对象。全部任务就绪时返回 `{"status":"READY","tasks":{"0":"READY","1":"READY","2":"READY","3":"READY"}}`；任一任务异常时返回 HTTP `503`、顶层 `status=ERROR`，并将对应任务标记为 `ERROR`。

### 11.2 Task1 编排接口：`127.0.0.1:8108`

#### `POST /tasks/1/run`

请求头可选：`Idempotency-Key: <任务键>`。请求体必须是空对象，额外字段返回 HTTP `422`：

```json
{}
```

成功响应：

```json
{
  "task_run_id": "<任务键或自动生成的 ID>",
  "task_type": "SORTING",
  "status": "SUCCEEDED",
  "product_names": ["东方树叶茉莉花茶", "绿豆冰沙"],
  "target_items": [
    {
      "product_name": "东方树叶茉莉花茶",
      "product_slot_id": "H2_F_L2_C04",
      "target_id": "H2_F_R_INSPECT",
      "shelf_level": "L2",
      "hand": "LEFT",
      "picked": true,
      "placed": true
    },
    {
      "product_name": "绿豆冰沙",
      "product_slot_id": "H2_F_L2_C05",
      "target_id": "H2_F_R_INSPECT",
      "shelf_level": "L2",
      "hand": "RIGHT",
      "picked": true,
      "placed": true
    }
  ],
  "held_items": {}
}
```

同一进程已有任务运行时返回 HTTP `409`：

```json
{"error_code":"TASK_IN_PROGRESS","message":"task 0 is already running"}
```

#### `GET /health`

使用统一的聚合 `GET /health`，响应格式见 11.1。

### 11.3 Task2 编排接口：`127.0.0.1:8108`

#### `POST /tasks/2/run`

请求头可选：`Idempotency-Key: <任务键>`。请求体必须是空对象，额外字段返回 HTTP `422`：

```json
{}
```

成功响应：

```json
{
  "task_run_id":"<任务键或自动生成的 ID>",
  "task_type":"SHORTAGE",
  "status":"SUCCEEDED",
  "inspection_pass":1,
  "product_names":["商品1","商品2"],
  "target_items":[
    {
      "product_name":"商品1",
      "inspection_target_id":"H1_F_L_INSPECT",
      "inspection_pose_type":"SHELF_VIEW_UPPER",
      "hand":"LEFT",
      "picked":true,
      "placed":true
    },
    {
      "product_name":"商品2",
      "inspection_target_id":"H2_B_R_INSPECT",
      "inspection_pose_type":"SHELF_VIEW_LOWER",
      "hand":"RIGHT",
      "picked":true,
      "placed":true
    }
  ],
  "held_items":{}
}
```

同一进程已有任务运行时返回 HTTP `409` `TASK_IN_PROGRESS`。

#### `GET /health`

使用统一的聚合 `GET /health`，响应格式见 11.1；Task2 的状态还要求五个下游、头部彩色流和 16 张 Task0 基准图全部就绪。

### 11.4 Task3 编排接口：`127.0.0.1:8108`

#### `POST /tasks/3/run`

请求头可选 `Idempotency-Key`，请求体必须是空对象。成功响应示例：

```json
{
  "task_run_id":"<任务键或自动生成的 ID>",
  "task_type":"MISPLACED",
  "status":"SUCCEEDED",
  "inspection_pass":1,
  "finding":{
    "misplaced_product_name":"错误商品",
    "gt_product_name":"应放商品",
    "inspection_target_id":"H1_F_L_INSPECT",
    "inspection_location_id":"H1_F_L1_C01",
    "inspection_pose_type":"SHELF_VIEW_UPPER"
  },
  "product_names":["错误商品","应放商品"],
  "target_items":[
    {
      "product_name":"错误商品",
      "source_slot_id":"H1_F_L1_C02",
      "destination_slot_id":"H1_F_L1_C04",
      "source_target_id":"H1_F_L_INSPECT",
      "destination_target_id":"H1_F_R_INSPECT",
      "hand":"LEFT",
      "picked":true,
      "placed":true
    },
    {
      "product_name":"应放商品",
      "source_slot_id":"H1_F_L1_C04",
      "destination_slot_id":"H1_F_L1_C02",
      "source_target_id":"H1_F_R_INSPECT",
      "destination_target_id":"H1_F_L_INSPECT",
      "hand":"RIGHT",
      "picked":true,
      "placed":true
    }
  ],
  "held_items":{}
}
```

额外请求字段返回 HTTP `422 INVALID_REQUEST`；并发运行返回 HTTP `409 TASK_IN_PROGRESS`。

#### `GET /health`

使用统一的聚合 `GET /health`，响应格式见 11.1；Task3 的状态还要求六个下游、头部彩色流和 16 张 Task0 基准图全部就绪。

### 11.5 导航接口：`<robot_ip>:8081`

#### `GET /navigation/health`

无请求体；就绪响应必须为 `{"status":"READY"}`。

#### `POST /navigation/navigate`

请求头：`Content-Type: application/json`、`Idempotency-Key: <唯一动作键>`。

```json
{"target_id":"receipt_viewpoint"}
```

Task0、Task2、Task3 使用八个巡检点；Task1 和 Task3 会使用 SKU 货位在 `product-hand-options.yaml` 中对应的导航点。各任务还使用各自的固定业务点和 `task_boundary`。成功响应至少包含：

```json
{"status":"SUCCEEDED"}
```

#### `POST /navigation/nudge`

机器人通过 `/navigation/navigate` 到站后，可使用该阻塞式接口在车体坐标系内短距离离站，并在操作完成后返回离站前的初始位置。请求头必须包含：

```http
Content-Type: application/json
Idempotency-Key: <唯一动作键>
```

离站请求：

```json
{"action":"approach","direction":"left"}
```

`direction` 必须是以下值之一：

| 值 | 车体方向 |
|---|---|
| `left` | 左 |
| `right` | 右 |
| `forward` | 前 |
| `back` | 后 |

每次 `approach` 固定移动 3 cm。从同一初始位置开始，最多连续调用两次 `approach`，两次的方向可以分别指定；不得在未 `return` 的情况下调用第三次。回到首次离站前初始位置的请求为：

```json
{"action":"return"}
```

`return` 会一次性回到第一次 `approach` 前记录的初始位置，而不是只撤销最后一次 3 cm 移动。成功响应示例：

```json
{"status":"SUCCEEDED","station_id":"shelf_group_b_front_1","nudge_count":1}
```

接口与 `/navigation/navigate` 一样，会阻塞到实际移动完成后再返回。`station_id` 是当前离站序列关联的导航站点，`nudge_count` 是尚未回站的连续离站次数：第一次和第二次 `approach` 成功后分别为 `1` 和 `2`，`return` 成功后重置为 `0`。

每个逻辑动作必须使用不同的 `Idempotency-Key`，因此两次 `approach` 和一次 `return` 共使用三个键。仅当同一个逻辑动作因网络错误或超时需要重试时，才复用该动作原有的键和请求体，避免重复产生物理移动。完整顺序为：

```text
POST /navigation/nudge {"action":"approach","direction":"left"}  Idempotency-Key: <动作键-1>
  <- {"status":"SUCCEEDED","station_id":"<站点>","nudge_count":1}
POST /navigation/nudge {"action":"approach","direction":"forward"}  Idempotency-Key: <动作键-2>
  <- {"status":"SUCCEEDED","station_id":"<站点>","nudge_count":2}
POST /navigation/nudge {"action":"return"}  Idempotency-Key: <动作键-3>
  <- {"status":"SUCCEEDED","station_id":"<站点>","nudge_count":0}
```

Task1、Task2、Task3 目前在每次 8086 `/pick` 或 `/place` 返回服务错误后自动使用一次微调额度，固定顺序为：

```text
POST 8086/<pick|place>  Idempotency-Key: <动作键>
  <- 失败
POST 8081/navigation/nudge {"action":"approach","direction":"back"}
  Idempotency-Key: <动作键>:recovery.approach
POST 8086/<pick|place>  Idempotency-Key: <动作键>:recovery.retry
  <- 成功或失败
POST 8081/navigation/nudge {"action":"return"}
  Idempotency-Key: <动作键>:recovery.return.1
```

第一次 `return` 失败时使用 `<动作键>:recovery.return.2` 再调用一次；两次都失败则停止后续物理动作。单个 HTTP 请求自己的网络重试仍复用该请求原有的键。任务没有最终失败时按原流程前往 `task_boundary`；Task1-3 任意整体失败都会先准备 `START_POSITION` 并导航到 `start`。如果失败回 `start` 也失败，对外返回 `FAILURE_RECOVERY_FAILED`，原始失败和回程失败同时记录在任务日志中。

### 11.6 姿态接口：`<robot_ip>:8084`

#### `GET /pose/health`

无请求体；就绪响应为 `{"status":"READY"}`。

#### `POST /pose/prepare`

请求头：`Idempotency-Key: <唯一动作键>`。Task1 实际发送以下请求体：

```jsonl
{"pose_type":"START_POSITION"}
{"pose_type":"RECEIPT_VIEW"}
{"pose_type":"SHELF_PICK_READY","shelf_level":"L2"}
{"pose_type":"DELIVERY_TABLE_PLACE_READY"}
```

Task2 另外发送：

```jsonl
{"pose_type":"SHELF_VIEW_UPPER"}
{"pose_type":"SHELF_VIEW_LOWER"}
{"pose_type":"REPLENISHMENT_TABLE_PICK_READY"}
```

成功响应至少包含 `{"status":"SUCCEEDED"}`；扩展执行字段由任务服务忽略。

### 11.7 感知接口：`127.0.0.1:8083`

#### `GET /perception/health`

无请求体；就绪响应为 `{"status":"READY"}`。

#### `POST /perception/parse`

无请求体。成功响应必须有两个不同的非空商品名：

```json
{"product_names":["商品1","商品2"]}
```

#### `POST /perception/inspect`

Task2 和 Task3 在每个巡检点的上/下观察姿态各调用一次。感知服务自行获取 Task0 基准图和当前头部彩色图，任务编排只发送识别上下文。Task2 的 `location_id` 是从 `product-hand-options.yaml` 中按当前导航点反查并排序后得到的第一个合法货位：

```json
{
  "task_type":"SHORTAGE",
  "location_id":"H1_F_L1_C01",
  "pose_type":"SHELF_VIEW_UPPER"
}
```

成功响应：

```json
{"findings":[{"shortage_product_name":"缺货商品名"}]}
```

感知接口还接受可选的正数 `reference_item_area`，Task2 和 Task3 当前都不发送。Task2 Agent 单次最多接受两个非空商品名，并以“商品名 + 当前巡检点 + 当前观察姿态”去重，累计两个唯一发现后停止巡检。

Task3 使用相同字段，将任务类型改为：

```json
{
  "task_type":"MISPLACED",
  "location_id":"H1_F_L1_C01",
  "pose_type":"SHELF_VIEW_UPPER"
}
```

无发现返回 `{"findings":[]}`；发现乱放商品返回：

```json
{
  "findings":[
    {
      "misplaced_product_name":"当前放错的商品名",
      "gt_product_name":"当前位置应有的商品名"
    }
  ]
}
```

Task3 只接受一组不同的非空商品名。

#### `POST /perception/pick/locate`

请求体：

```json
{
  "task_type":"SORTING",
  "product_name":"商品名",
  "level":"L2",
  "hand":"left"
}
```

`task_type` 可为 `SORTING`、`SHORTAGE` 或 `MISPLACED`。`level` 接受 `L1` 至 `L5`；Task1 和 Task3 的抓取请求传入来源货位层号，Task2 的 `SHORTAGE` 抓取省略该字段。

成功响应至少包含：

```json
{
  "product_name":"商品名",
  "bbox":[853,404,983,797],
  "mask":"<可选，base64 PNG>",
  "image_path":"<可选>"
}
```

`product_name` 必须与请求一致（比较前会去掉空格和符号，因此 `Lay's乐事薯片...` 与 `Lays乐事薯片...` 视为相同），`bbox` 必须恰好四个数。

#### `POST /perception/place/locate`

`SHORTAGE` 和 `MISPLACED` 的 `/place` 使用此接口。请求与定位响应结构和 `/perception/pick/locate` 相同，但路径中的操作类型为 `place`：

```json
{"task_type":"SHORTAGE","product_name":"商品名","hand":"left"}
```

#### `POST /perception/{pick|place}/check`

pick-place 中两类结果视觉校验的调用代码均已注释；当前所有任务在 `grasp` 或 `release` 成功后直接返回，不调用这两个接口。

### 11.8 SKU 接口：`127.0.0.1:25540`

#### `GET /sku/health`

无请求体；就绪响应为 `{"status":"READY"}`。

#### `GET /sku/search_by_name?name=<商品名>`

输入是 URL query，不是 GET JSON body。成功响应：

```json
{
  "sku_id":"SKU_076",
  "name":"商品名",
  "images":["images/SKU_076.jpg"],
  "locations":["H2_F_L2_C04"]
}
```

Task1 要求 `name` 与查询值一致、`locations` 恰好一个。Task3 同样校验名称，并结合发现时巡检上下文解析 P1、要求 P2 唯一；货位格式为 `H[12]_[FB]_L[1-5]_Cdd`。

### 11.9 pick-place 对外接口：`127.0.0.1:8086`

#### `GET /health`

无请求体。下游健康检查和相机列表均通过后返回：

```json
{"status":"READY"}
```

否则返回 HTTP `503` 和 `{"status":"ERROR"}`。

#### `POST /pick` 与 `POST /place`

请求头必须包含非空 `Idempotency-Key`，并使用 `Content-Type: application/json`。请求模型禁止额外字段：

```json
{
  "task_type":"SORTING",
  "product_name":"商品名",
  "hand":"LEFT",
  "level":"L2",
  "product_type":"<可选字符串或整数>"
}
```

`hand` 接受 `left`、`right`、`LEFT`、`RIGHT`，服务内部规范化为小写。

`task_type` 接受 `SORTING`、`SHORTAGE` 和 `MISPLACED`。Task1 只发送 `SORTING`，Task2 只发送 `SHORTAGE`。`level` 可选且只在 `/pick` 定位时使用，接受 `L1` 至 `L5`；Task1 和 Task3 传入该字段，Task2 省略。兼容字段 `product_type` 不会传给机器人；调用 `grasp/release` 时统一传递 `product_name`。

成功响应：

```json
{"status":"SUCCEEDED"}
```

缺少幂等键返回 HTTP `400` `MISSING_IDEMPOTENCY_KEY`；相同幂等键对应不同请求体返回 HTTP `409` `IDEMPOTENCY_KEY_CONFLICT`。幂等缓存在 8086 进程内由 `/pick` 和 `/place` 共享，因此每个物理动作都必须使用不同的键，不得在抓取和放置之间复用。

### 11.10 位姿估计接口：`127.0.0.1:8084`

#### `GET /manipulation/health`

无请求体；就绪响应为 `{"status":"READY"}`。

#### `POST /manipulation/pick_pose` 与 `POST /manipulation/place_pose`

`/pick` 调用 `pick_pose`；非 `SORTING /place` 调用 `place_pose`。两者都使用 `multipart/form-data`，字段如下：

| 字段 | 类型 | 内容 |
|---|---|---|
| `product_name` | 表单字段 | 当前商品名 |
| `rgb` | 文件 | 当前选中相机的 RGB 图 |
| `depth` | 文件 | depth 首帧；尺寸匹配的裸 little-endian `uint16` 数据会转换为 16 位 PNG，其他编码原样上传 |
| `camera` | 文件 | `SHORTAGE /pick` 和非 `SORTING /place` 使用 `head.json`；其他 `/pick` 使用左/右腕标定 |
| `mask` | 文件 | 定位 mask；无 mask 时由 bbox 生成 |

成功响应必须包含六维 `pose`：

```json
{
  "pose":[237.9558,33.8016,547.2908,-0.6210,1.4421,2.5262],
  "frame":"camera",
  "pose_unit":"mm_rad",
  "rotation_order":"zyx"
}
```

可选的 `corners_mm` 也会被 8086 接受。缺少 `frame`、`pose_unit`、`rotation_order` 时，8086 默认分别使用 `camera`、`mm_rad`、`zyx`。

### 11.11 机器人抓放接口：`<robot_ip>:8084`

#### `GET /manipulation/health`

无请求体；就绪响应为 `{"status":"READY"}`。

#### `POST /manipulation/grasp`

请求头：`Idempotency-Key: <8086 幂等键>:execute`。请求体使用位姿估计结果：

```json
{
  "task_type":"SORTING",
  "product_name":"商品名",
  "pose":[237.9558,33.8016,547.2908,-0.6210,1.4421,2.5262],
  "hand":"left",
  "frame":"camera",
  "pose_unit":"mm_rad",
  "rotation_order":"zyx"
}
```

`task_type`、`product_name` 和 `hand` 来自 8086 对外请求，`hand` 在此通用分支为小写；`product_type` 不会传给机器人接口。

成功响应至少包含：

```json
{"status":"SUCCEEDED"}
```

#### `POST /manipulation/release`

请求头：`Idempotency-Key: <8086 幂等键>:execute`。SORTING 放置使用全零 pose 和大写 `hand`，因为机器人已由 Task1 的 `DELIVERY_TABLE_PLACE_READY` 姿态接口准备好：

```json
{
  "task_type":"SORTING",
  "product_name":"商品名",
  "hand":"LEFT",
  "pose":[0,0,0,0,0,0],
  "frame":"camera",
  "pose_unit":"mm_rad",
  "rotation_order":"zyx"
}
```

成功响应至少包含 `{"status":"SUCCEEDED"}`。

`SHORTAGE` 和 `MISPLACED` 放置使用 `place_pose` 返回的六维位姿，`hand` 为小写；释放成功后直接返回，不执行结果视觉校验。

### 11.12 相机接口：`<robot_ip>:8085`

Task0 和 pick-place 直接调用相机网关。Task2 和 Task3 只在启动时检查头部彩色流是否在线，巡检取图改由感知服务完成。pick-place 在需要视觉位姿的 `/pick` 和非 `SORTING /place` 中先取彩色快照，再从 depth 流读取第一帧。`SHORTAGE /pick` 无论使用哪只手都使用头部相机；`SORTING /pick` 和 `MISPLACED /pick` 使用与抓取手对应的腕部相机：

| 场景 | 相机 | 实际请求 |
|---|---|---|
| Task0 基准采集 | `head` | `GET /camera/rgbd?camera=head` |
| Task2 / Task3 启动健康检查 | `head` | `GET /camera/health`，然后 `GET /camera/list` |
| `SORTING /pick`，`hand=left` | `left_wrist` | `GET /camera/snapshot?camera=left_wrist&type=color`，然后 `GET /camera/stream?camera=left_wrist&type=depth` |
| `SORTING /pick`，`hand=right` | `right_wrist` | `GET /camera/snapshot?camera=right_wrist&type=color`，然后 `GET /camera/stream?camera=right_wrist&type=depth` |
| `SHORTAGE /pick`，`hand=left` 或 `hand=right` | `head` | `GET /camera/snapshot?camera=head&type=color`，然后 `GET /camera/stream?camera=head&type=depth` |
| `MISPLACED /pick` | `left_wrist` 或 `right_wrist` | 根据 `hand` 选择对应腕部相机的 color 和 depth 接口 |
| `SORTING /place` | 不使用相机 | Task1 已准备固定放置姿态，8086 直接调用 `release` |
| `SHORTAGE /place` 或 `MISPLACED /place` | `head` | `GET /camera/snapshot?camera=head&type=color`，然后 `GET /camera/stream?camera=head&type=depth` |
| `GET 8086 /health` | 不取图 | 只调用 `GET /camera/health` 和 `GET /camera/list` |

#### `GET /camera/health`、`GET /camera/list`

无请求体。`/camera/health` 必须返回 `{"status":"READY"}`。Task0 要求列表中的 `head`、color 和 depth 均在线；Task2 和 Task3 要求 `head` 和 color 在线；8086 只要求 `/camera/list` 返回 HTTP `2xx`。

#### `GET /camera/rgbd`

```http
GET /camera/rgbd?camera=head
```

Task0 使用此接口。响应为 ZIP，必须包含非空 `rgb.jpg`、`depth_mm.npy` 和 `meta.json`。

#### `GET /camera/snapshot`

```http
GET /camera/snapshot?camera=left_wrist&type=color
```

左手使用 `left_wrist`，右手使用 `right_wrist`。响应为非空 RGB 图像二进制，通常为 `image/jpeg`。

#### `GET /camera/stream`

```http
GET /camera/stream?camera=left_wrist&type=depth
Accept: multipart/x-mixed-replace
```

响应为 depth 长连接流；8086 读取第一帧。如果返回内容是与 RGB 尺寸匹配的裸 little-endian `uint16` 数据，8086 会将其转换为 16 位单通道 PNG；已编码的 PNG、JPEG 或其他内容当前会原样上传位姿估计接口。
