# `src/pick_place_service` 实际调用流程与接口说明

本文记录当前代码真实执行的取放流程，以及 `code/` 目录中各模块同事提供的独立参考脚本。地址、字段和响应以当前代码与最近一次真实联调日志为准。

## 一、当前服务地址

配置文件：`config/pick-place.yaml`

| 能力 | 地址 | 当前用途 |
|---|---|---|
| 正式定位 | `http://192.168.130.59:8083` | `/perception/pick/locate`、`/perception/place/locate` |
| 感知校验 | `http://192.168.130.59:8083` | `/perception/pick/check`、`/perception/place/check` |
| 物体位姿估计 | `http://192.168.130.59:8084` | `/manipulation/pick_pose`、`/manipulation/place_pose` |
| 抓取/释放执行 | `http://192.168.130.50:8084` | `/manipulation/grasp`、`/manipulation/release` |
| 相机网关 | `http://192.168.130.50:8085` | RGB 快照、depth 流、相机健康检查 |
| 8086 编排服务 | `http://127.0.0.1:8086` | 对外 `/pick`、`/place` |

当前相机配置：

```yaml
pick_cameras:
  left: left_wrist
  right: right_wrist
place_camera: head
calibration_files:
  head: config/camera/head.json
  left_wrist: config/camera/left_wrist.json
  right_wrist: config/camera/right_wrist.json
log_dir: log
```

8086 会根据实际传入的相机 ID 选择标定文件：`head` 使用 `head.json`，`left_wrist` 使用 `left_wrist.json`，`right_wrist` 使用 `right_wrist.json`。旧配置中的单一 `calibration_file` 仍作为兼容 fallback，但正式配置应使用 `calibration_files`。

每次 `/pick` 或 `/place` 运行都会在 `log/` 下创建独立目录，目录名包含时间和幂等键。典型结构：

```text
log/<时间>-<幂等键>/
├── operation.json
├── request.json
├── interfaces/
│   ├── perception_pick_locate/request.json
│   ├── perception_pick_locate/response.json
│   ├── camera_snapshot_color/request.json
│   ├── camera_snapshot_color/response.json
│   ├── camera_stream_depth/request.json
│   ├── camera_stream_depth/response.json
│   ├── camera_stream_depth/frame.bin
│   ├── manipulation_pick_pose/request.json
│   ├── manipulation_pick_pose/response.json
│   ├── manipulation_grasp/request.json
│   └── manipulation_grasp/response.json
└── camera/
    ├── rgb.jpg
    ├── depth.png
    ├── mask.png
    └── <camera>.json
```

JSON 接口日志包含请求 URL、请求体、响应状态码、响应头和响应体；位姿接口还会保存 multipart 文件元数据及相机输入文件；临时 `/tmp/pick-place` 目录仍会在流程结束时清理，但 `log/` 下的记录会保留。

## 二、8086 对外输入输出

### 2.1 拣取

```http
POST http://127.0.0.1:8086/pick
Content-Type: application/json
Idempotency-Key: <唯一任务键>
```

请求：

```json
{
  "task_type": "SORTING",
  "product_name": "妙芙绵醇奶油味",
  "hand": "left"
}
```

允许的 `task_type`：`SORTING`、`SHORTAGE`、`MISPLACED`。`hand` 为 `left` 或 `right`，代码会规范化为小写。

成功响应：

```json
{"status":"SUCCEEDED"}
```

常见编排层错误：

| HTTP | `error_code` | 含义 |
|---:|---|---|
| 400 | `MISSING_IDEMPOTENCY_KEY` | 缺少幂等键 |
| 409 | `IDEMPOTENCY_KEY_CONFLICT` | 同一个幂等键对应不同请求 |
| 502 | `VISION_INPUT_UNAVAILABLE` | 相机输入不可用 |
| 502 | `POSE_INPUT_UNAVAILABLE` | 位姿文件或标定文件不可读 |
| 502 | 下游错误码 | 8083、8084、8085 返回非 2xx |
| 504 | `NETWORK_ERROR` / `ACTION_RESULT_UNKNOWN` | 超时或动作结果未知 |

### 2.2 放置

```http
POST http://127.0.0.1:8086/place
Content-Type: application/json
Idempotency-Key: <唯一任务键>
```

外部请求结构与 `/pick` 相同。内部接口依次替换为 `place/locate`、`place_pose`、`release`、`place/check`。

## 三、`src/pick_place_service` 实际流程

代码入口：`src/pick_place_service/app.py`；流程编排：`src/pick_place_service/service.py`。

```text
POST /pick 或 /place
  -> 1. 正式目标定位
  -> 2. 相机获取 RGB-D 和 mask
  -> 3. 物体位姿估计
  -> 4. 机械臂抓取或释放
  -> 5. 视觉校验
  -> 返回 {"status":"SUCCEEDED"}
```

任何一步失败都会停止后续步骤，并记录：

```text
取放流程失败 step=<当前步骤> kind=<pick/place> product=<商品> key=<幂等键>
```

### 3.1 正式目标定位

拣取调用：

```http
POST http://192.168.130.59:8083/perception/pick/locate
Content-Type: application/json
```

请求体：

```json
{
  "task_type": "SORTING",
  "product_name": "妙芙绵醇奶油味",
  "hand": "left"
}
```

8086 当前使用的响应字段：

```json
{
  "product_name": "妙芙绵醇奶油味",
  "bbox": [659, 297, 873, 833],
  "mask": "<base64 PNG>",
  "image_path": "/some/path/image.jpg"
}
```

代码会校验商品名、四元素 `bbox` 和 mask。mask 存在时优先保存为原尺寸 `mask.png`，没有 mask 才根据 bbox 生成兼容的矩形 `mask.pgm`。

### 3.2 相机获取与深度标准化

拣取根据请求 `hand` 选择 `pick_cameras.left` 或 `pick_cameras.right`。两项均为必填配置，缺少任一项时服务配置加载失败；放置使用 `place_camera`。

RGB：

```http
GET /camera/snapshot?camera=left_wrist&type=color
```

Depth：

```http
GET /camera/stream?camera=left_wrist&type=depth
Accept: multipart/x-mixed-replace
```

当前 8085 depth 流的实际内容是 multipart 中的裸 `uint16` 帧。例如左腕相机：

```text
640 x 480 x 2 = 614400 bytes
Content-Type: application/octet-stream
```

8086 会：

1. 解析 multipart boundary 和 `Content-Length`，提取第一帧；
2. 按 little-endian `uint16` 读取深度值；
3. 转成 16 位单通道 PNG；
4. 与 RGB 尺寸、mask 尺寸保持一致后上传 8084。

因此日志中出现以下内容表示深度转换成功：

```text
深度输入标准化完成 ... format=PNG size=640x480 bit_depth=16 color=grayscale
```

### 3.3 物体位姿估计

拣取调用：

```http
POST http://192.168.130.59:8084/manipulation/pick_pose
Content-Type: multipart/form-data
```

multipart 字段：

| 字段 | 类型 | 内容 |
|---|---|---|
| `rgb` | 文件 | RGB JPEG |
| `depth` | 文件 | 与 RGB 对齐的 16 位单通道 PNG |
| `camera` | 文件 | `left_wrist.json`、`head.json` 等标定 JSON |
| `mask` | 文件 | 定位服务返回的目标 mask |
| `product_name` | 表单字段 | 商品名 |

成功响应中的核心字段：

```json
{
  "pose": [139.006868, 13.573344, 497.991979, -2.442349, -1.529998, -0.794928],
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

代码要求 `pose` 必须有 6 个数字。

### 3.4 抓取或释放执行

拣取调用：

```http
POST http://192.168.130.50:8084/manipulation/grasp
Content-Type: application/json
Idempotency-Key: <任务键>:execute
```

请求：

```json
{
  "task_type": "SORTING",
  "pose": [139.006868, 13.573344, 497.991979, -2.442349, -1.529998, -0.794928],
  "hand": "left",
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

如果原始 `/pick` 请求包含 `product_type`，也会传给执行服务。

放置调用 `/manipulation/release`，请求结构相同。

接口文档示例响应为：

```json
{"status":"SUCCEEDED"}
```

当前真实 `.50:8084` 服务实际返回过以下扩展结构：

```json
{
  "executed": true,
  "simulated": true,
  "operation": "GRASP",
  "request": {"task_type":"SORTING", "pose":[1,2,3,4,5,6], "hand":"left"},
  "message": "模拟模式：已跳过位姿转换及机器人控制"
}
```

8086 只以响应中的 `status=SUCCEEDED` 判断执行成功，并允许执行服务返回其他扩展字段。上述历史响应缺少 `status`，因此仍会在 8086 本地解析失败；`simulated=true` 也表示当时没有实际驱动机器人。

### 3.5 视觉校验

抓取成功后调用：

```http
POST http://192.168.130.59:8083/perception/pick/check
Content-Type: application/json
```

请求：

```json
{
  "task_type": "SORTING",
  "product_name": "妙芙绵醇奶油味",
  "hand": "left"
}
```

放置对应 `/perception/place/check`。

当前实际检查结果：`192.168.130.59:8083` 曾返回连接拒绝，因此该服务需要先启动或确认端口配置。8086 只有在执行响应校验成功后才会进入此步骤。

## 四、`code/` 目录参考脚本

### 4.1 `code/request_formal_api.py`：正式定位接口最小调用

用途：只测试 8083 正式定位接口。

调用：

```bash
python code/request_formal_api.py SORTING "可口可乐罐装" left
```

默认地址：

```text
http://192.168.130.59:8083/perception/pick/locate
```

请求 JSON：

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐罐装",
  "hand": "left"
}
```

输出：直接打印定位服务返回 JSON。

### 4.2 `code/test_formal_api.py`：SKU 查询 + 正式定位批量测试

用途：先按商品名查 SKU，再从 `image_name_mapping.json` 找对应本地图片，逐张调用正式定位接口。

脚本默认把测试图片目录解析为项目上一级的 `test_data/2026-08-04`。当前工作区的真实图片位于 `perception/test_data/2026-08-04`，因此直接执行前需要确认脚本默认路径与本机目录布局一致，或调整脚本中的 `DEFAULT_IMAGE_DIRECTORY`。

调用：

```bash
python code/test_formal_api.py SORTING "可口可乐罐装" left
```

调用的 SKU 接口：

```http
GET http://192.168.130.59:25540/sku/search_by_name?name=<商品名>
```

调用的定位接口：

```http
POST http://192.168.130.59:8083/perception/pick/locate
```

与 `request_formal_api.py` 不同，它还上传：

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐罐装",
  "hand": "left",
  "image_name": "record_xxx_rgb.jpg",
  "image_base64": "<base64 RGB>"
}
```

它严格要求定位响应包含：

```json
{
  "product_name": "...",
  "bbox": [1, 2, 3, 4],
  "mask": "<base64>",
  "image_path": "/absolute/path/image.jpg"
}
```

这个脚本是离线图片驱动的定位测试，不是 8086 当前运行时请求的完整格式。

### 4.3 `code/test_inference.py`：视觉推理/可视化接口

用途：测试另一套视觉推理接口，并把实例 mask、bbox 画到结果图片上。

该脚本也默认读取项目上一级的 `test_data/2026-08-04`，当前图片目录同样需要确认是否应改为 `perception/test_data/2026-08-04`。

调用：

```bash
python code/test_inference.py "可口可乐罐装"
```

它先调用 SKU：

```http
GET http://192.168.130.59:25540/sku/search_by_name?name=<商品名>
```

然后调用：

```http
POST http://192.168.130.59:8081/visual/pick/locate
```

请求：

```json
{
  "name": "<SKU 商品名>",
  "image_name": "record_xxx_rgb.jpg",
  "image_base64": "<base64 RGB>"
}
```

响应主要使用：

```json
{
  "instances": [
    {
      "bbox": [x1, y1, x2, y2],
      "mask": "<base64 PNG>",
      "score": 0.95
    }
  ]
}
```

该接口当前没有直接接入 `src/pick_place_service`；8086 使用的是正式 `/perception/{pick|place}/locate` 合同。

### 4.4 `code/pick_pose_request.py`：位姿估计独立测试

用途：绕过 8086，直接测试 8084 的物体位姿估计。

调用示例：

```bash
python code/pick_pose_request.py \
  --url http://192.168.130.59:8084/manipulation/pick_pose \
  --rgb test/2026-08-04/record_20260804_144405_673341_rgb.jpg \
  --depth test/2026-08-04/record_20260804_144405_673341_depth_mm.png \
  --camera test/camera.json \
  --mask test/record_20260804_144405_673341_rgb.png \
  --product-name "可口可乐罐装"
```

实际发送 multipart 字段：

```text
rgb、depth、camera、mask、product_name
```

其中 MIME 类型由文件扩展名自动推断。该脚本的请求格式与 8086 位姿上传基本一致，但默认 URL 是 `.59:8084`，不能改成 `.50:8084` 作为位姿估计地址。

### 4.5 `code/image.py`：8085 相机诊断

调用：

```bash
python code/image.py --camera head --output-dir /tmp/camera-test
python code/image.py --camera left_wrist --stream-type depth
```

检查接口：

```http
GET /camera/health
GET /camera/list
GET /camera/snapshot?camera=<camera>&type=color
GET /camera/stream?camera=<camera>&type=<color|depth>
```

输出并保存：

```text
health JSON
camera list JSON
RGB JPEG 快照
流的第一帧样本
```

这是相机独立诊断脚本，不直接调用定位、位姿或抓取接口。

### 4.6 `code/8084接口.md`：机械臂姿态与动作接口示例

健康检查：

```bash
curl http://192.168.130.50:8084/pose/health
curl http://192.168.130.50:8084/manipulation/health
```

预备姿态：

```http
POST /pose/prepare
```

请求示例：

```json
{"pose_type":"SHELF_PICK_READY","shelf_level":"L3"}
```

该接口当前没有接入 `src/pick_place_service` 的 `/pick` 主流程。

抓取：

```http
POST http://192.168.130.50:8084/manipulation/grasp
```

请求字段为 `task_type`、`pose`、`hand`，以及可选的 `product_type`、`frame`、`pose_unit`、`rotation_order`。

释放：

```http
POST http://192.168.130.50:8084/manipulation/release
```

请求结构与抓取相同。

## 五、辅助脚本

### `scripts/test-pick-coca-cola.sh`

调用本地完整编排：

```bash
./scripts/test-pick-coca-cola.sh
```

默认请求 `127.0.0.1:8086/pick`，自动生成新的 `Idempotency-Key`。

### `scripts/test-grasp.sh`

绕过 8086，直接测试真实抓取执行：

```bash
./scripts/test-grasp.sh
```

默认地址：

```text
http://192.168.130.50:8084/manipulation/grasp
```

可通过环境变量覆盖：

```bash
HAND=right ./scripts/test-grasp.sh
POSE_JSON='[1,2,3,4,5,6]' ./scripts/test-grasp.sh
```

## 六、当前联调结论

截至当前代码和日志：

```text
8083 正式定位：已通（请求字段为 `task_type`）
8085 RGB：已通
8085 depth 裸 uint16 multipart：已通，8086 已转 16 位 PNG
`.59:8084` 位姿估计：已通
`.50:8084` grasp/release：已收到请求，曾返回模拟执行响应
`.59:8083` 视觉校验：已接入代码，但当前实测连接被拒绝
```

另外，`.50:8084/manipulation/grasp` 当前曾返回：

```json
{
  "executed": true,
  "simulated": true,
  "operation": "GRASP"
}
```

这表示当时服务处于模拟模式，并不代表机械臂真的完成了抓取。当前 8086 要求执行响应包含 `"status":"SUCCEEDED"`，同时兼容真实服务返回的其他扩展字段。
