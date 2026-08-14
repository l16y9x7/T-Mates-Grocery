# Task1 与 pick-place 真实端口接口清单

> 整理日期：2026-08-14。本文以当前工作树中的 `config/task1.production.yaml`、`config/pick-place.yaml` 和 `src/task1_service`、`src/pick_place_service` 实现为准；真实响应示例来自 `log/20260814-164239-876711-web-task1-20260814-164239-009c18610a` 及其 pick/place 子日志。

## 1. 结论

当前 Task1 从入口到实际硬件共涉及以下地址。注意两个 `8084` 是不同主机、不同职责，不能互换。

| 地址 | 服务角色 | Task1 业务主链实际使用 |
|---|---|---|
| `0.0.0.0:8108` | Task1 编排服务 | 对外接收 `/task1/run` |
| `192.168.3.226:8081` | 导航 | 导航到小票点、商品货位、交付台和任务判定区 |
| `127.0.0.1:8083` | 感知 | 识别小票；pick 时定位商品 |
| `127.0.0.1:25540` | SKU 商品库 | 按商品名查询货位 |
| `127.0.0.1:8086` | pick-place 编排服务 | Task1 调用 `/pick`、`/place` |
| `127.0.0.1:8084` | 物体位姿估计 | pick 时根据 RGB-D、mask 和标定计算抓取位姿 |
| `192.168.3.226:8084` | 机器人姿态与操作 | Task1 准备全身位姿；pick-place 执行抓取和释放 |
| `192.168.3.226:8085` | 相机网关 | pick 时读取腕部 RGB 和 depth |

Task1 当前主链不是统一的五阶段 pick/place：

- `/pick`：定位 -> 腕部 RGB/depth -> 抓取位姿估计 -> `grasp`。抓取后的视觉校验代码目前被注释，不调用 `/perception/pick/check`。
- `SORTING /place`：Task1 先通过 `/pose/prepare` 到交付台放置准备位，然后 `8086` 只调用一次 `release`。不调用放置定位、相机、放置位姿估计或视觉校验。
- `/task1/run` 中的全量健康检查当前被注释；只有显式请求 Task1 的 `GET /health` 时才会检查所有下游健康状态。

## 2. 当前 Task1 调用顺序

两件商品可分配给不同手时，当前流程如下。每次导航前都会先调用 `START_POSITION` 收回机器人姿态。

```text
POST :8108/task1/run {}
  -> :8084/pose/prepare {"pose_type":"START_POSITION"}
  -> :8081/navigation/navigate {"target_id":"receipt_viewpoint"}
  -> :8084/pose/prepare {"pose_type":"RECEIPT_VIEW"}
  -> :8083/perception/parse
  -> :25540/sku/search_by_name?name=<商品1>
  -> :25540/sku/search_by_name?name=<商品2>
  -> 对每件商品：
       :8084/pose/prepare {"pose_type":"START_POSITION"}
       :8081/navigation/navigate {"target_id":"<商品货位>"}
       :8084/pose/prepare {"pose_type":"SHELF_PICK_READY","shelf_level":"Lx"}
       :8086/pick -> 见第 4 节
  -> :8084/pose/prepare {"pose_type":"START_POSITION"}
  -> :8081/navigation/navigate {"target_id":"delivery_place"}
  -> :8084/pose/prepare {"pose_type":"DELIVERY_TABLE_PLACE_READY"}
  -> 对每件商品：:8086/place -> 仅调用 :8084/manipulation/release
  -> :8084/pose/prepare {"pose_type":"START_POSITION"}
  -> :8081/navigation/navigate {"target_id":"task_boundary"}
  <- Task1 SUCCEEDED
```

如果两件商品只能使用同一只手，则按“取一件 -> 去交付台放一件 -> 再取下一件”的顺序串行执行。

## 3. Task1 层接口输入输出

### 3.1 `8108` Task1 编排服务

#### `POST /task1/run`

请求头可选：

```http
Idempotency-Key: <本次任务 ID>
Content-Type: application/json
```

请求体固定为空对象，额外字段会返回 `422 INVALID_REQUEST`：

```json
{}
```

成功输出：

```json
{
  "task_run_id": "web-task1-...",
  "task_type": "SORTING",
  "status": "SUCCEEDED",
  "product_names": ["东方树叶茉莉花茶", "绿豆冰沙"],
  "target_items": [
    {
      "product_name": "东方树叶茉莉花茶",
      "product_slot_id": "H2_F_L2_C04",
      "shelf_level": "L2",
      "hand": "LEFT",
      "picked": true,
      "placed": true
    },
    {
      "product_name": "绿豆冰沙",
      "product_slot_id": "H2_F_L2_C05",
      "shelf_level": "L2",
      "hand": "RIGHT",
      "picked": true,
      "placed": true
    }
  ],
  "held_items": {}
}
```

同一 Task1 进程同时只允许一个任务；已有任务运行时返回 HTTP 409：

```json
{"error_code":"TASK_IN_PROGRESS","message":"another task is already running"}
```

#### `GET /health`

无输入。该接口会检查第 6 节列出的下游健康接口。全部正常时输出：

```json
{"status":"READY"}
```

任一下游异常时返回 HTTP 503：

```json
{"status":"ERROR"}
```

### 3.2 `192.168.3.226:8081` 导航

#### `POST /navigation/navigate`

请求头：`Idempotency-Key: <唯一动作键>`。

请求体：

```json
{"target_id":"H2_F_L2_C04"}
```

`target_id` 在 Task1 中可能为：`receipt_viewpoint`、SKU 返回的货位 ID、`delivery_place`、`task_boundary`。

成功输出：

```json
{"status":"SUCCEEDED"}
```

### 3.3 `192.168.3.226:8084` 姿态准备

#### `POST /pose/prepare`

请求头：`Idempotency-Key: <唯一动作键>`。

请求体有以下实际形式：

```json
{"pose_type":"START_POSITION"}
```

```json
{"pose_type":"RECEIPT_VIEW"}
```

```json
{"pose_type":"SHELF_PICK_READY","shelf_level":"L2"}
```

```json
{"pose_type":"DELIVERY_TABLE_PLACE_READY"}
```

Task1 只要求输出包含：

```json
{"status":"SUCCEEDED"}
```

真实服务还会返回 `executed`、`pose_type`、`current_pose`、`memory_point`、`arms` 等执行详情；Task1 会忽略这些扩展字段。

### 3.4 `127.0.0.1:8083` 小票感知

#### `POST /perception/parse`

无请求体、无必需请求头。

成功输出必须恰好包含两个不同的非空商品名：

```json
{"product_names":["东方树叶茉莉花茶","绿豆冰沙"]}
```

### 3.5 `127.0.0.1:25540` SKU 商品库

#### `GET /sku/search_by_name?name=<商品名>`

输入是 URL query，不是 GET JSON body：

```http
GET /sku/search_by_name?name=东方树叶茉莉花茶
```

成功输出：

```json
{
  "sku_id": "SKU_076",
  "name": "东方树叶茉莉花茶",
  "images": ["images/SKU_076.jpg"],
  "locations": ["H2_F_L2_C04"]
}
```

Task1 要求返回的 `name` 与查询值一致，且主流程要求 `locations` 恰好一个，格式为 `H[12]_[FB]_L[1-5]_Cdd`。

代码客户端还实现了 `GET /sku/search_by_location?location=<货位>`，但当前 Task1 主流程没有调用它，因此不计入业务主链。

### 3.6 `127.0.0.1:8086` pick-place 对外接口

#### `POST /pick`

请求头：

```http
Idempotency-Key: <唯一动作键>
Content-Type: application/json
```

Task1 实际请求：

```json
{
  "task_type": "SORTING",
  "product_name": "东方树叶茉莉花茶",
  "hand": "LEFT"
}
```

`hand` 接受 `left/right/LEFT/RIGHT`；`product_type` 是可选的字符串或整数。成功输出：

```json
{"status":"SUCCEEDED"}
```

#### `POST /place`

输入输出合同与 `/pick` 相同。Task1 的 `SORTING` 请求在 `8086` 内走固定释放分支，见第 5 节。

`Idempotency-Key` 缺失时返回 HTTP 400 `MISSING_IDEMPOTENCY_KEY`；同一 key 配不同请求体时返回 HTTP 409 `IDEMPOTENCY_KEY_CONFLICT`。

## 4. `8086 /pick` 内部真实接口

### 4.1 `127.0.0.1:8083/perception/pick/locate`

请求：

```json
{
  "task_type": "SORTING",
  "product_name": "东方树叶茉莉花茶",
  "hand": "left"
}
```

成功输出：

```json
{
  "product_name": "东方树叶茉莉花茶",
  "bbox": [853, 404, 983, 797],
  "mask": "<base64 编码的 PNG，可选>",
  "image_path": "/data/.../image.jpg"
}
```

`8086` 校验商品名完全一致、`bbox` 恰好四个值。`mask` 存在时作为目标 PNG 使用；缺失时按 `bbox` 生成矩形 mask。

### 4.2 `192.168.3.226:8085/camera/snapshot`

请求：

```http
GET /camera/snapshot?camera=left_wrist&type=color
```

`camera` 由抓取手决定：左手为 `left_wrist`，右手为 `right_wrist`。输出是非空 RGB 图像二进制；真实响应为 `Content-Type: image/jpeg`，示例大小 45491 bytes。

### 4.3 `192.168.3.226:8085/camera/stream`

请求：

```http
GET /camera/stream?camera=left_wrist&type=depth
Accept: multipart/x-mixed-replace
```

输出是 depth 长连接流。真实响应为：

```http
Content-Type: multipart/x-mixed-replace; boundary=tianjiframe
```

`8086` 读取第一帧；裸 little-endian `uint16` 深度会被转换为与 RGB 同尺寸的 16 位单通道 PNG，再传给位姿服务。

### 4.4 `127.0.0.1:8084/manipulation/pick_pose`

请求类型：`multipart/form-data`。

| 字段 | 类型 | 输入 |
|---|---|---|
| `product_name` | form 字段 | 当前商品名 |
| `rgb` | 文件 | 腕部 RGB JPEG/PNG |
| `depth` | 文件 | 与 RGB 对齐的 16 位 depth PNG |
| `camera` | 文件 | 当前相机标定 JSON，如 `left_wrist.json` |
| `mask` | 文件 | 定位 mask PNG，或 bbox 生成的 PGM |

成功输出：

```json
{
  "pose": [237.9558, 33.8016, 547.2908, -0.6210, 1.4421, 2.5262],
  "corners_mm": [[282.86,112.33,577.74]],
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

`pose` 必须恰好六个数；`corners_mm`、`frame`、`pose_unit`、`rotation_order` 可选。后三项缺失时 `8086` 分别按 `camera`、`mm_rad`、`zyx` 处理。

### 4.5 `192.168.3.226:8084/manipulation/grasp`

请求头：`Idempotency-Key: <8086 动作键>:execute`。

请求：

```json
{
  "task_type": "SORTING",
  "pose": [237.9558, 33.8016, 547.2908, -0.6210, 1.4421, 2.5262],
  "hand": "left",
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

原始 `/pick` 带 `product_type` 时，这里也会带同名字段。成功输出必须包含：

```json
{"status":"SUCCEEDED"}
```

服务可以附加执行详情，`8086` 不解析扩展字段。

## 5. `SORTING /place` 内部真实接口

Task1 已通过 `POST /pose/prepare {"pose_type":"DELIVERY_TABLE_PLACE_READY"}` 准备放置位姿。因此 `8086 /place` 当前只调用以下接口。

### `192.168.3.226:8084/manipulation/release`

请求头：`Idempotency-Key: <8086 动作键>:execute`。

请求：

```json
{
  "task_type": "SORTING",
  "hand": "LEFT",
  "pose": [0, 0, 0, 0, 0, 0],
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

注意此分支的 `hand` 是大写。成功输出至少包含：

```json
{"status":"SUCCEEDED"}
```

真实服务还返回过 `executed`、`operation`、`hand`、`place_joints_deg`、`gripper`、`current_pose` 等执行详情，`8086` 只检查 `status`。

对于非 `SORTING` 的通用 `/place`，代码仍保留 `place/locate -> head RGB-D -> place_pose -> release -> place/check` 流程；它不是当前 Task1 实际路径。

## 6. 仅健康检查使用的接口

这些接口不在当前 `/task1/run` 业务主链中。请求 Task1 的 `GET /health` 时会依次检查五个服务，其中 `8086 GET /health` 又会检查其四类下游和相机列表。

| 地址 | 方法与路径 | 输入 | `READY` 判定 |
|---|---|---|---|
| `192.168.3.226:8081` | `GET /navigation/health` | 无 | JSON `status == "READY"` |
| `127.0.0.1:8083` | `GET /perception/health` | 无 | JSON `status == "READY"` |
| `192.168.3.226:8084` | `GET /pose/health` | 无 | JSON `status == "READY"` |
| `127.0.0.1:8086` | `GET /health` | 无 | JSON `status == "READY"` |
| `127.0.0.1:25540` | `GET /sku/health` | 无 | JSON `status == "READY"` |
| `127.0.0.1:8084` | `GET /manipulation/health` | 无 | JSON `status == "READY"` |
| `192.168.3.226:8084` | `GET /manipulation/health` | 无 | JSON `status == "READY"` |
| `192.168.3.226:8085` | `GET /camera/health` | 无 | JSON `status == "READY"` |
| `192.168.3.226:8085` | `GET /camera/list` | 无 | 任意 HTTP 2xx；当前代码不校验响应体 |

前五项由 Task1 健康检查发起；后四项由 `8086 /health` 发起。`127.0.0.1:8083/perception/health` 会被 Task1 和 `8086` 各检查一次。

## 7. 当前明确不调用的历史接口

为避免联调时误判，以下接口虽然代码中有实现或旧文档曾列出，但当前 Task1 主链不调用：

| 接口 | 当前状态 |
|---|---|
| `POST /perception/pick/check` | pick 后视觉校验已注释跳过 |
| `POST /perception/place/locate` | `SORTING /place` 固定释放分支跳过 |
| `GET /camera/*?camera=head...` | `SORTING /place` 不取图 |
| `POST /manipulation/place_pose` | `SORTING /place` 不估算放置位姿 |
| `POST /perception/place/check` | `SORTING /place` 固定释放分支跳过 |
| `GET /sku/search_by_location` | 客户端已实现，但 Task1 主流程只按商品名查询 |

## 8. 验证范围

- 当前接口清单由生产 YAML 与代码调用点交叉核对。
- 2026-08-14 的真实 Task1 日志确认了 `8081` 导航、远端 `8084` 姿态/抓取/释放、`8085` 相机、本机 `8083` 感知、本机 `8084` 位姿估计、本机 `8086` 编排和 `25540` SKU 的上述请求及成功响应。
- 自动化测试结果：`tests/test_pick_place_service.py` 为 20/20 通过；`tests/test_task1_service.py` 为 11/13 通过。两个失败测试都要求导航健康为 `ERROR` 时 `/task1/run` 返回 503，实际因运行前的 `check_all_health()` 被注释而继续执行并返回成功。这与第 1 节记录的当前行为一致，但属于需要明确决定是否恢复的安全检查。
- 本次从受限执行环境对所有健康 URL 做只读探测时均得到连接失败（HTTP 000），包括 `127.0.0.1` 和 `192.168.3.226`；该结果只能说明当前执行环境不可达，不能据此判断现场服务已经停止。端口合同和成功响应仍以上述生产配置、代码及同日真实运行日志为依据。
- 本文整理过程只运行自动化测试，不重新触发 `/task1/run`、`grasp` 或 `release` 等机器人物理动作。
