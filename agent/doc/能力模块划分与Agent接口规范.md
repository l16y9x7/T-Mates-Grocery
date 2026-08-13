# 能力模块划分与Agent接口规范

# 能力模块与 Agent 接口规范

## 1\. 模块与调用边界

|模块|负责人|对 Agent 提供的能力|
|---|---|---|
|底盘能力模块|Nora|导航至固定点位|
|场景理解模块<br>|Alexander<br>|小票识别、抓取目标识别、货架缺货和乱放识别<br><br>访问摆放表|
|位姿控制模块<br>|Mui<br>|小票拍摄、货架观察、抓放预备位姿|
|抓放能力模块|Mui / Stephen / Samuel|双手抓取和放置|
|Agent 编排模块<br>|Skylar<br>|State、任务规则、作业生成、流程路由和接口调用<br><br>访问摆放表|
|头部视频流采集？|Nora|头部相机视频流获取方式：<br>cd /home/nvidia/smt/robot\_device<br>\./start\.sh start\-camera|

Agent 通过 HTTP/JSON API 调用能力模块，不直接调用模块内部 Python 函数，也不使用 ROS。各能力模块自行完成初始化并持续运行；Agent 通过各模块约定的健康检查接口判断模块是否可用。主 Agent 的抓放动作统一调用 8086 的取放编排服务，8086 再调用 8083、8084、8085 完成一次完整子流程。

`WorkflowState` 只在 Agent 内部流转，不作为能力接口的请求体。商品货位表、底盘固定点位表和巡检点配置由 Agent 在启动时加载并校验。

## 2\. 公共约定

### 2\.1 传输与响应

- 请求体和响应体使用 UTF\-8 JSON。

- 未在本规范中定义的字段不属于当前最简接口。

- 请求无效、模块未就绪或执行失败时返回非 `2xx`，响应至少包含：

```JSON
{"error_code": "EXECUTION_FAILED"}
```

### 2\.2 物理动作接口

以下接口属于物理动作接口：

- `POST /navigation/navigate`

- `POST /pose/prepare`

- `POST /pick`（8086）

- `POST /place`（8086）

- 8084 的 `POST /manipulation/grasp` 和 `POST /manipulation/release` 仅由 8086 内部调用，Agent 不直接调用。

物理动作接口必须携带 `Idempotency-Key`，并在动作最终成功后返回：

```JSON
{"status": "SUCCEEDED"}
```

同一逻辑动作的 Agent 重试必须复用同一个 `Idempotency-Key`。模块收到相同键时返回原执行结果，不重复执行动作。

### 2\.3 商品货位编号

`product_slot_id` 表示商品的标准货位，格式为：

```Plaintext
H1_F_L1_C01
│  │  │  └─ C01：面对货架面时从左到右的商品位
│  │  └──── L1：从上到下的货架层
│  └─────── F：正面；B：反面
└────────── H1：货架编号
```

格式约束：`^H[12]_[FB]_L[1-5]_C\d{2}$`。

Agent 按下列规则派生导航点和层号：

|`product_slot_id`|导航 `target_id`|`shelf_level`|
|---|---|---|
|`H1_F_L1_C01`|`H1_F_L1_C01`|`L1`|
|`H2_B_L3_C06`|`H2_B_L3_C06`|`L3`|

Agent 必须校验 `product_slot_id` 存在于本地商品货位表，且派生的 `target_id` 存在于[第 6 章的底盘固定点位表](https://my.feishu.cn/wiki/EWa1w3TpAiq2jmkYySOcGQuNnvf?fromScene=spaceOverview#share-DQtVd1CkTosH9gxdrlkceEZDnPY)。

## 3\. 能力接口

### 3\.1 健康检查

```HTTP
GET /navigation/health     ---- Nora
GET /manipulation/health   ---- Mui&Stephen
GET /perception/health     ---- Alexander
GET /pose/health           ---- Mui
```

请求体：无。

响应：

```JSON
{"status": "READY"}
```

`status` 取值为 `STARTING`、`READY` 或 `ERROR`。主 Agent 直接依赖的导航、姿态、视觉、商品库以及 8086 取放编排服务均为 `READY` 时，Agent 才能开始任务。8086 自身还会检查 8083、8084、8085 的健康状态和相机列表。

#### 涉及模块：底盘能力、场景理解、控制、抓放

### 3\.2 底盘导航

```HTTP
POST /navigation/navigate
```

|请求字段|类型|必填|约束|
|---|---|---|---|
|`target_id`<br>|string<br>|是<br>|商品抓取和放置会传入完整的商品货位编号。其他可用点位必须存在于[第 6 章的底盘固定点位表](https://my.feishu.cn/wiki/EWa1w3TpAiq2jmkYySOcGQuNnvf?fromScene=spaceOverview#share-DQtVd1CkTosH9gxdrlkceEZDnPY)|

请求示例：

```JSON
{"target_id": "H1_F_L1_C01"}
```

物理动作接口必须携带 `Idempotency-Key`，

```JSON
成功时返回 200： {"status": "SUCCEEDED"}                                                          
失败时返回非 2xx：{"error_code": "EXECUTION_FAILED"} 
```

#### 涉及模块：底盘能力

### 3\.3 场景理解

#### 3\.3\.1 小票识别

```HTTP
POST /perception/parse
```

请求体：无

成功响应为对象，`product_names` 必须恰好包含两个不同的非空商品名。Agent 随后对每个商品名调用商品库 `GET /sku/search_by_name`，并校验返回恰好一个有效的标准货位。

响应示例：

```JSON
{"product_names": ["可口可乐罐装", "百事可乐瓶装"]}
```

##### 涉及模块：控制、头部视频流采集、场景理解

### 3\.4 货架巡检

```HTTP
POST /perception/inspect
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string|是|`SHORTAGE` 或 `MISPLACED`|

`SHORTAGE` 用于任务二，`MISPLACED` 用于任务三。

响应字段：

|`task_type`|`findings` 元素类型|说明|
|---|---|---|
|`SHORTAGE`|List\[String\]|缺货货位编号|
|`MISPLACED`|Tuple\[String\]|乱放商品的名字，以及当前槽位商品名|

`findings` 必须为数组。未发现目标时返回空数组。

缺货响应示例：

```JSON
{
  "findings": ["H1_F_L1_C01","H1_F_L2_C04"]
}
```

乱放响应示例：

```JSON
{
  "findings":["H1_F_L1_C01","H1_F_L2_C04"]
}
```

`MISPLACED` 未发现乱放商品时返回空数组；发现一件乱放商品时固定返回两个不同的有效货位编号：`findings[0]` 是该商品当前所在货位，`findings[1]` 是识别出的商品标准货位。两件乱放商品确定互换，因此 Agent 收到这两个编号后即可停止巡检。

`SHORTAGE` 允许单次返回 0、1 或 2 个结果。Agent 跨货架面去重并累计；得到两个不同缺货货位后停止巡检，全部巡检完成仍不足两个或累计结果超过两个时任务失败。

以上两项需要\>=x\(5\)帧检出才确定；建议用贪心算法，确定即肯定

#### 涉及模块：底盘能力、控制、头部视频流采集、场景理解

### 3\.5 位姿准备

```HTTP
POST /pose/prepare
```

|请求字段|类型|必填|约束|
|---|---|---|---|
|`pose_type`|string|是|见下表|
|`shelf_level`|string|条件必填|货架抓取或放置时必传，取值为 `L1` 至 `L5`|

|`pose_type`|含义|`shelf_level`|
|---|---|---|
|`RECEIPT_VIEW`|小票拍摄位姿|不传|
|SHELF\_VIEW\_UPPER|整面货架观察高位姿|不传|
|SHELF\_VIEW\_LOWER|整面货架观察低位姿|不传|
|`SHELF_INSPECT`|定点高度观察位姿|必传|
|`SHELF_PICK_READY`|货架抓取预备位姿|必传|
|`REPLENISHMENT_TABLE_PICK_READY`|补货台抓取预备位姿|不传<br>|
|`SHELF_PLACE_READY`|货架放置预备位姿|必传|
|`DELIVERY_TABLE_PLACE_READY`|交付台放置预备位姿|不传|
|`START_POSITION`|复位|不传|

请求示例：

```JSON
{"pose_type": "SHELF_PICK_READY", "shelf_level": "L1"}
```

物理动作接口必须携带 `Idempotency-Key`，

```JSON
成功时返回 200： {"status": "SUCCEEDED"}                                                          
失败时返回非 2xx：{"error_code": "EXECUTION_FAILED"} 
```

#### 涉及模块：控制、头部视频流采集、场景理解

### 3\.6 抓取

```HTTP
POST http://192.168.130.59:8086/pick
```

|请求字段|类型|必填|约束|
|---|---|---|---|
|`task_type`|string|是<br>|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`<br>拣选、短缺、放错|
|`product_name`|string|是|完整商品名|
|`hand`|string|是|`LEFT` 或 `RIGHT`|

调用前提：Agent 已完成导航，再完成抓取预备位姿调整；导航与位姿不能并行。

`SORTING`：任务一，拣选，从货架上抓取需要的商品。

`SHORTAGE`:任务二，短缺，从补货台抓取需要的商品。

`MISPLACED`：任务三，放错，从货架上抓取放错的商品。

货架抓取示例：

```JSON
{
  "task_type": "SORTING",
  "product_name": "可口可乐",
  "hand": "LEFT"
}
```

补货台抓取示例：

```JSON
{
  "pick_type": "SHORTAGE",
  "product_name": "可口可乐",
  "hand": "LEFT"
}
```

物理动作接口必须携带 `Idempotency-Key`，

```JSON
成功时返回 200： {"status": "SUCCEEDED"}                                                          
失败时返回非 2xx：{"error_code": "EXECUTION_FAILED"} 
```

#### 涉及模块：控制、头部视频流采集、场景理解、抓放

### 3\.7 放置

```HTTP
POST http://192.168.130.59:8086/place
```

|请求字段|类型|必填|约束|
|---|---|---|---|
|`task_type`|string|是|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`<br>拣选、短缺、放错|
|`product_name`|string|是|完整商品名|
|`hand`|string|是|`LEFT` 或 `RIGHT`|

前提：Agent 已完成导航，再完成放置预备位姿调整；导航与位姿不能并行。抓放模块将指定手中的商品放到机器人当前工作位置，因此无需再传放置位置。

`SORTING`：任务一，拣选，向交付台放置商品。

`SHORTAGE`:任务二，短缺，向货架放置商品。

`MISPLACED`：任务三，放错，向货架放置商品。

货架放置示例：

```JSON
{
  "task_type": "SHORTAGE",
  "product_name": "可口可乐",
  "hand": "LEFT"
}
```

交付台放置示例：

```JSON
{
  "task_type": "SORTING",
  "product_name": "可口可乐",
  "hand": "LEFT"
}
```

物理动作接口必须携带 `Idempotency-Key`，

```JSON
成功时返回 200： {"status": "SUCCEEDED"}                                                          
失败时返回非 2xx：{"error_code": "EXECUTION_FAILED"} 
```

#### 涉及模块：控制、头部视频流采集、场景理解、抓放

## 4\. Agent State

```Python
class WorkflowState(TypedDict):
    task_type: str
    status: str
    current_product_slot_id: str | None
    navigation_target_id: str | None
    shelf_level: str | None
    navigation_status: str
    pose_request: PoseRequest | None
    pose_status: str
    inspection_points: list[InspectionPoint]
    inspection_index: int
    target_items: list[str]
    findings: list[str]
    held_items: dict[Hand, HeldItem]
    jobs: list[Job]
```

- 每次任务创建新的 State，并将列表和 `held_items` 初始化为空。

- `held_items` 记录左右手当前持有的商品。

- `navigation_target_id` 和 `shelf_level` 从当前作业的 `product_slot_id` 派生。

## 5\. 三个任务主流程

### 5\.1 通用规则

- 同一物理动作的重试复用原 `Idempotency-Key`。

    - 导航目标和位姿已知时，Agent 必须先调用 `POST /navigation/navigate`，收到 `SUCCEEDED` 后再调用 `POST /pose/prepare`。两者不得并行。

- 接口返回非 `2xx`、响应字段无效或本地配置校验失败时，停止当前任务并将 `status` 置为 `FAILED`。

- 所有作业完成且 `held_items` 为空后，机器人导航至 `task_boundary`，到达后将 `status` 置为 `SUCCEEDED`。

### 5\.2 任务一：商品拣选

**小票固定包含两个商品，每个商品抓取一件。**

|节点|处理|调用|成功后|
|---|---|---|---|
|T1\-N01|创建 State，检查导航、姿态、视觉、商品库和 8086 均为 `READY`|`GET /navigation/health`、`/pose/health`、`/perception/health`、`/sku/health`、8086 `GET /health`|进入 N02|
|T1\-N02|先到达小票识别位置，再进入拍摄位姿|先导航 `POST /navigation/navigate`（`receipt_viewpoint`），成功后 `POST /pose/prepare`（`RECEIPT_VIEW`）|进入 N03|
|T1\-N03|识别小票；校验恰好两个不同的非空商品名|`POST /perception/parse`|写入 `target_items`（商品名），进入 N04|
|T1\-N04|通过商品名查询并校验唯一标准货位，派生导航点和层号，生成作业|商品库 `GET /sku/search_by_name`（每个商品一次）+ 本地配置|写入 `jobs`，进入 N05|
|T1\-N05|按 `route_order` 排序作业，依次分配 `LEFT`、`RIGHT`|本地处理|进入 N06|
|T1\-N06|先到达当前商品货位，再进入对应层抓取预备位姿|先 `POST /navigation/navigate`，成功后 `POST /pose/prepare`（`SHELF_PICK_READY` + `shelf_level`）|进入 N07|
|T1\-N07|在当前工作位置抓取指定商品|8086 `POST /pick`（`task_type=SORTING`）|写入 `held_items`；有待取商品则回 N06，否则进入 N08|
|T1\-N08|两件商品均持有后先到达交付台，再进入放置预备位姿|先导航 `delivery_place`，成功后准备 `DELIVERY_TABLE_PLACE_READY`|进入 N09|
|T1\-N09|按 `held_items` 逐件放置商品|8086 `POST /place`（`task_type=SORTING`）|清除对应持物；全部放置后进入 N10|
|T1\-N10|到达任务判定区|导航 `task_boundary`|任务成功|

### 5\.3 任务二：货架补货

**任务目标为识别并完成两个不同缺货货位的补货。**

|节点|处理|调用|成功后|
|---|---|---|---|
|T2\-N01|创建 State，检查模块，加载四个巡检点|`GET /health`、本地配置|进入 N02|
|T2\-N02|先到达当前货架面的巡检点，再进入观察位姿|先导航 `POST /navigation/navigate`，成功后依次 `POST /pose/prepare`（`SHELF_VIEW_UPPER`、`SHELF_VIEW_LOWER`）|进入 N03|
|T2\-N03|识别当前货架面的全部缺货项|`POST ``/perception/inspect`，类型 `SHORTAGE`|进入 N04|
|T2\-N04|校验、去重并累计结果|本地处理|得到两个缺货货位则进入 N05；不足两个则巡检下一面；巡检结束仍不足两个再来一遍|
|T2\-N05|为两个缺货货位生成补货作业，分配 `LEFT`、`RIGHT`|本地配置|进入 N06|
|T2\-N06|先到达补货台，再进入抓取预备位姿|先导航 `replenishment_pickup`，成功后 `POST /pose/prepare`（`REPLENISHMENT_TABLE_PICK_READY`）|进入 N07|
|T2\-N07|按作业逐件抓取补货商品|8086 `POST /pick`（`task_type=SHORTAGE`）|写入 `held_items`；两件均持有后进入 N08|
|T2\-N08|先到达当前目标货位，再进入放置预备位姿并放置|先导航目标货位，成功后 `POST /pose/prepare`（`SHELF_PLACE_READY` + 层号），随后 8086 `POST /place`（`task_type=SHORTAGE`）|完成当前作业；有剩余作业则重复 N08，否则进入 N09|
|T2\-N09|到达任务判定区|导航 `task_boundary`|任务成功|

### 5\.4 任务三：乱放归位

当前已确定任务三的两件乱放商品互换位置。场景模块识别到一件乱放商品时，同时返回其当前位置和标准货位。Agent 将 `findings[0]` 记为 P1，将 `findings[1]` 记为 P2，即可推断 P1 中的商品应归位至 P2，P2 中的商品应归位至 P1。

|节点|处理|调用|成功后|
|---|---|---|---|
|T3\-N01|创建 State，检查模块，加载四个巡检点|`GET /health`、本地配置|进入 N02|
|T3\-N02|先到达当前货架面的巡检点，再进入观察位姿|先导航 `POST /navigation/navigate`，成功后依次 `POST /pose/prepare`（`SHELF_VIEW_UPPER`、`SHELF_VIEW_LOWER`）|进入 N03|
|T3\-N03|识别当前货架面的乱放商品|`POST /``perception``/inspect`，类型 `MISPLACED`|有结果则进入 N04；无结果则巡检下一面；全部巡检完成仍无结果则再来一遍|
|T3\-N04|校验 `findings` 恰好包含两个不同的有效编号，将 `findings[0]` 记为 P1、`findings[1]` 记为 P2|本地处理|派生 P1、P2 的导航点和层号，进入 N05|
|T3\-N05|到达 P1 后进入抓取预备位姿，左手抓取应归位至 P2 的商品|先导航 P1，成功后 `POST /pose/prepare`（`SHELF_PICK_READY` + P1 层），随后 8086 `POST /pick`（`task_type=MISPLACED`）|写入 `held_items.LEFT`，进入 N06|
|T3\-N06|到达 P2 后进入抓取预备位姿，右手抓取应归位至 P1 的商品|先导航 P2，成功后 `POST /pose/prepare`（`SHELF_PICK_READY` + P2 层），随后 8086 `POST /pick`（`task_type=MISPLACED`）|写入 `held_items.RIGHT`，进入 N07|
|T3\-N07|在 P2 进入放置预备位姿，用左手放置第一件商品|已在 P2，调用 `POST /pose/prepare`（`SHELF_PLACE_READY` + P2 层），随后 8086 `POST /place`（`task_type=MISPLACED`）|清除 `held_items.LEFT`，进入 N08|
|T3\-N08|返回 P1，进入放置预备位姿，用右手放置第二件商品|先导航 P1，成功后 `POST /pose/prepare`（`SHELF_PLACE_READY` + P1 层），随后 8086 `POST /place`（`task_type=MISPLACED`）|清除 `held_items.RIGHT`，进入 N09|
|T3\-N09|到达任务判定区|导航 `task_boundary`|任务成功|

## 6\. 底盘固定点位表

|`target_id`|点位类型|语义位置|
|---|---|---|
|`task_boundary`|任务判定点<br>|三个任务共用的起点和任务结束判定区|
|`receipt_viewpoint`|固定业务点|小票识别位置|
|`replenishment_pickup`|固定业务点|补货台取货位置|
|`delivery_place`|固定业务点|交付台放货位置|
|`H1_F_``L_``INSPECT`|巡检拍摄点|货架 1 正面左侧巡检位置|
|`H1_F_R_INSPECT`|巡检拍摄点|货架 1 正面右侧巡检位置|
|`H1_B_``L_``INSPECT`|巡检拍摄点|货架 1 反面左侧巡检位置|
|`H1_B_R_INSPECT`|巡检拍摄点|货架 1 反面右侧巡检位置|
|`H2_F_``L_``INSPECT`|巡检拍摄点|货架 2 正面左侧巡检位置|
|`H2_F_R_INSPECT`|巡检拍摄点|货架 2 正面右侧巡检位置|
|`H2_B_``L_``INSPECT`|巡检拍摄点|货架 2 反面左侧巡检位置|
|`H2_B_R_INSPECT`|巡检拍摄点|货架 2 反面右侧巡检位置|

# 内部流程

## 8086 `/pick` —— 取放子流程

调用前提：Agent 已到达抓取预备位姿，已确定左右手。

流程：抓取物品定位\(Alexander\) ——\> 位姿估计\(Stephen\) ——\> 执行\(Mui\) ——\> 视觉校验\(Alexander\)

### 抓取物品定位\(Alexander\)

```HTTP
POST /perception/pick/locate
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string<br>|是|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|
|`product_name`|string|是|完整商品名|
|`hand`|string|是|left/right，影响`SORTING`策略|

成功响应为\{"`product_name`": "", "bbox": \[x1, y1, x2, y2\], "mask": "base64 code", "image\_path": "image path"\}。

\(x1, y1\), \(x2, y2\)分别为左上角和右下角坐标, 标准化到\[1, 1000\]

响应示例：

```JSON
{“product_name”: "可口可乐", "bbox": [100, 200, 200, 400], "mask": "base64 code", "image_path": "image path"}
```

### 位姿估计模块\(Stephen\)

```HTTP
POST /manipulation/pick_pose
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`rgb`<br>|string<br>|是|RGB 图像文件|
|`depth`|string|是|与 RGB 对齐的深度图文件|
|`camera`|string|是|相机参数 JSON 文件|
|`mask`|string|是<br>|放置目标区域的掩码文件，尺寸应与 RGB 一致|
|`product_name`|string|可选||

成功响应为目标物体在深度相机坐标系下的 6D 位姿，以及模型估计的三维有向包围盒八个角点。

pose 格式为：\[x, y, z, rx, ry, rz\]

其中：x、y、z 的单位为 mm；rx、ry、rz 的单位为 rad；旋转采用 ZYX 欧拉角顺序；corners\_mm 为三维有向包围盒的 8 个角点，单位为 mm；本接口不直接判断机械臂是否可达。应考虑能否到达抓取目标位置

响应示例（请Stephen补充）：

```JSON
{
  "pose": [
    95.964,
    -84.310,
    382.659,
    -0.616429,
    0.134398,
    3.057678
  ],
  "corners_mm": [
    [120.98, 10.67, 384.45],
    [87.29, 13.50, 379.88],
    [88.68, -115.41, 289.76],
    [122.37, -118.25, 294.33],
    [103.25, -53.21, 475.56],
    [69.56, -50.37, 470.99],
    [70.94, -179.29, 380.87],
    [104.63, -182.12, 385.44]
  ],
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

### 执行模块\(Mui\)

```HTTP
POST /manipulation/grasp
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string|是？<br>|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`<br>区分IK策略|
|`pose`<br>|<br>|是||
|`hand`|string|是|left/right|
|`product_type`|string/int|可选|根据预先定好的策略|

响应为Success/Fail/Unreachable

### 视觉校验模块\(Alexander\)

```HTTP
POST /perception/pick/check
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string|是|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|
|`product_name`|string|是|完整商品名|
|`hand`|string|是|left/right|

响应为\{"pick\_status":"Success"\}或者\{"pick\_status":"Fail"\}



## 8086 `/place` —— 取放子流程

调用前提：Agent 已到达放置预备位姿，已确定左右手。

流程：目标空闲区域定位\(Alexander\) ——\> 位姿估计模块\(Stephen\) ——\> 执行模块\(Mui\) ——\> 视觉校验\(Alexander\)

### 目标空闲区域定位\(Alexander\)

```HTTP
POST /perception/place/locate
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string|是<br>|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|
|`product_name`|string|是|完整商品名|
|`hand`|string|是|left/right，影响`SORTING`策略|

成功响应为\{“`product_name`”: "", "bbox": \[x1, y1, x2, y2\]\}。

\(x1, y1\), \(x2, y2\)分别为左上角和右下角坐标, 标准化到\[1, 1000\]

响应示例：

```JSON
{“product_name”: "可口可乐", "bbox": [100, 200, 200, 400]}
```

### 位姿估计模块\(Stephen\)

```HTTP
POST /manipulation/place_pose
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`rgb`<br>|string<br>|是|RGB 图像文件|
|`depth`|string<br>|是|与 RGB 对齐的深度图文件|
|`camera`|string|是|相机参数 JSON 文件|
|`mask`|string|是<br>|放置目标区域的掩码文件，尺寸应与 RGB 一致|
|`product_name`|string|可选||

成功响应为目标物体在深度相机坐标系下的 6D 位姿，以及模型估计的三维有向包围盒八个角点。

pose 格式为：\[x, y, z, rx, ry, rz\]

其中：x、y、z 的单位为 mm；rx、ry、rz 的单位为 rad；旋转采用 ZYX 欧拉角顺序；corners\_mm 为三维有向包围盒的 8 个角点，单位为 mm；本接口不直接判断机械臂是否可达。应考虑能否到达抓取目标位置

响应示例（请Stephen补充）：

```JSON
{
  "pose": [
    95.964,
    -84.310,
    382.659,
    -0.616429,
    0.134398,
    3.057678
  ],
  "corners_mm": [
    [120.98, 10.67, 384.45],
    [87.29, 13.50, 379.88],
    [88.68, -115.41, 289.76],
    [122.37, -118.25, 294.33],
    [103.25, -53.21, 475.56],
    [69.56, -50.37, 470.99],
    [70.94, -179.29, 380.87],
    [104.63, -182.12, 385.44]
  ],
  "frame": "camera",
  "pose_unit": "mm_rad",
  "rotation_order": "zyx"
}
```

### 执行模块\(Mui\)

```HTTP
POST /manipulation/release
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`|string<br>|是？<br>|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|
|`pose`<br>|<br>|是||
|`hand`|string|是|left/right|
|`product_type`|string/int|可选|根据预先定好的策略|

响应为Success/Fail/Unreachable

### 视觉校验模块\(Alexander\)

```HTTP
POST /perception/place/check
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string|是|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|
|`product_name`|string|是|完整商品名|
|`bbox`|list|是|\[x1, y1, x2, y2\]|

响应为Success/Fail







## `/``perception``/inspect` —— sub\-agent

流程：先由底盘导航到巡检点；导航成功后由位姿模块依次准备观察姿态，再调用视觉巡查。导航、位姿和视觉请求不得并行。

建议预先定好几个导航的检查点位，需要\>=x帧检出才确定，贪心算法，确定即肯定

### 视觉巡查模块\(Alexander\)

```HTTP
POST /perception/inspect
```

|请求字段|类型|必填|取值|
|---|---|---|---|
|`task_type`<br>|string<br>|是<br>|`SORTING`或<br>`SHORTAGE` 或 `MISPLACED`|

响应字段 finding：

|`task_type`|`finding` 元素类型|说明|
|---|---|---|
|`SHORTAGE`|列表|缺货商品的名字|
|`MISPLACED`|元组|乱放商品的名字，以及当前槽位商品名|

应支持至少5并发

## 商品位置表

# 设计文档

[视觉感知](https://my.feishu.cn/wiki/SoNLwPrHmi95ipkHqxQcwVYvnVc)

# 服务器端口规划

服务器：192\.168\.130\.59

> 以现场联调接口为准：商品库查询使用 `/sku/search_by_name`、`/sku/search_by_location`；主 Agent 的抓放使用 8086 的 `/pick`、`/place`。8083 的 `/perception/inspect`、`/perception/place/locate` 和 8084 的 `/manipulation/release` 当前不可用，因此任务二、任务三只能在巡检接口补齐后执行，放置动作须先完成 8086 `/place` 联调。

|端口|接口|请求方法|请求体样例|返回|负责人|是否ready|
|---|---|---|---|---|---|---|
|25540<br>\(商品库\)<br>|/sku/health|GET|\{\}|\{"status": "READY"\}|Alex|ready|
||/sku/search\_by\_SKU<br>|GET|\{"sku": "SKU\_088"\}<br>|\{<br>"sku\_id": "SKU\_088",<br>"name": "外星人电解质水青柠口味0糖",<br>"images": \[<br>"images/SKU\_088\.jpg"<br>\],<br>"locations": \[<br>"H2\_F\_L4\_C05"<br>\]<br>\}|Alex<br>|ready|
||/sku/search\_by\_name|GET<br>|\{"name": "外星人电解质水青柠口味0糖"\}|\{<br>"sku\_id": "SKU\_088",<br>"name": "外星人电解质水青柠口味0糖",<br>"images": \[<br>"images/SKU\_088\.jpg"<br>\],<br>"locations": \[<br>"H2\_F\_L4\_C05"<br>\]<br>\}|Alex|ready|
||/sku/search\_by\_location<br>|GET<br>|\{"location": "H2\_F\_L4\_C05"\}|\{<br>"sku\_id": "SKU\_088",<br>"name": "外星人电解质水青柠口味0糖",<br>"images": \[<br>"images/SKU\_088\.jpg"<br>\],<br>"locations": \[<br>"H2\_F\_L4\_C05"<br>\]<br>\}|Alex|ready|
||/sku/get\_image|GET|\{"name": "外星人电解质水青柠口味0糖"\}|\["images/SKU\_088\.jpg"\]<br>|Alex|ready|
||/sku/get\_all\_names|GET|\{\}|\[\]|Alex|ready|
|8081<br>\(导航\)|/navigation/health|GET|无|\{"status": "READY"\}|Nora|ready|
||/navigation/navigate<br>||`{"target_id": "H1_F_L1_C01"}`|\{"status": "SUCCEEDED"\}<br>|Nora|需要Skylar确认<br>补货缺失、交付台缺失|
|8082<br>\(躯干控制\)|/pose/health|GET|无|\{<br>"status": "READY"<br>\}|Mui|ready|
||/pose/prepare||\{<br>"pose\_type": "SHELF\_VIEW\_UPPER"<br>\}|\{<br>"status": "SUCCEEDED",<br>"executed": true,<br>"pose\_type": "SHELF\_VIEW\_UPPER",<br>"shelf\_level": null,<br>"memory\_point": "扫上"<br>\}|Mui|通，需要测试小票点位，功能和skylar对齐<br><br>补货、交付台缺失|
|8083<br>\(视觉理解\)<br>|/perception/health|GET|||Alex|ready|
||/perception/pick/locate<br>|GET|\{“product\_name”: "可口可乐罐装", "task\_type": "SORTING", "hand": "left"\}|\{“product\_name”: "可口可乐", "bbox": \[100, 200, 200, 400\], "mask": "base64 code", "image\_path": "image path"\}|Alex<br>|SORTING \+ 放错 货架上识别<br>补货缺失<br>|
||/perception/pick/check||\{“product\_name”: "可口可乐罐装", "task\_type": "SORTING", "hand": "left"\}|\{"pick\_status":"Success"\}<br>|Alex|有，待测试<br>|
||/perception/place/locate||||Alex|缺失|
||/perception/place/check||\{“product\_name”: "可口可乐罐装", "task\_type": "SORTING", "hand": "left"\}|\{"place\_status":"Success"\}<br>|Alex|有，待测试<br>|
||/perception/parse|POST|\{\}|\{"product\_names": \["可口可乐罐装", "百事可乐瓶装"\]\}|Alex|有，待测试|
||/perception/inspect||||Alex|缺失|
|8084<br>\(抓放\)<br>|/manipulation/health|GET||\{"status":"READY"\}|Stephen/Mui|有|
||/manipulation/pick\_pose<br>|POST<br>|\{"rgb:string","depth":"string","camera":"string","mask":"string"\}<br>|\{"pose": \[x, y, z, rx, ry, rz\],<br>"corners\_mm": \[\[x, y, z\], "\.\.\.共8个角点"\],<br>"frame": "camera",<br>"pose\_unit": "mm\_rad",<br>"rotation\_order": "zyx" \}|Stephen|有<br>|
||/manipulation/grasp||\{<br>"task\_type": "SORTING",<br>"pose": \[<br>139\.006868,<br>13\.573344,<br>497\.991979,<br>\-2\.442349,<br>\-1\.529998,<br>\-0\.794928<br>\],<br>"hand": "LEFT",<br>"product\_type": "bottle",<br>"frame": "camera",<br>"pose\_unit": "mm\_rad",<br>"rotation\_order": "zyx"<br>\}|成功：\{<br>"status": "SUCCEEDED",<br>"executed": true,<br>"operation": "GRASP\_POSITION",<br>"reachable": true,<br>"tool\_length\_mm": 200\.0,<br>"tool\_axis": "\+Z",<br>"input\_orientation\_ignored": true,<br>"message": "夹具末端已移动到目标物体中心；未执行夹爪闭合"<br>\}<br>失败：\{<br>"error\_code": "UNREACHABLE",<br>"message": "目标法兰Pose逆向运动学失败",<br>"status": "UNREACHABLE",<br>"reachable": false,<br>"executed\_pose\_arm\_base\_mm\_deg": \[<br>420\.0,<br>180\.0,<br>120\.0,<br>\-90\.0,<br>0\.0,<br>\-90\.0<br>\]<br>\}|Mui|有，能通，需要抓取策略|
||/manipulation/place\_pose<br>|POST<br>|\{"rgb:string","depth":"string","camera":"string","mask":"string"\}<br>|\{"pose": \[x, y, z, rx, ry, rz\],<br>"corners\_mm": \[\[x, y, z\], "\.\.\.共8个角点"\],<br>"frame": "camera",<br>"pose\_unit": "mm\_rad",<br>"rotation\_order": "zyx" \}|Stephen<br>|有，但是需要测试<br>|
||/manipulation/release||\{<br>"task\_type": "SHORTAGE",<br>"pose": \[<br>139\.006868,<br>13\.573344,<br>497\.991979,<br>\-2\.442349,<br>\-1\.529998,<br>\-0\.794928<br>\],<br>"hand": "LEFT",<br>"product\_type": "bottle",<br>"frame": "camera",<br>"pose\_unit": "mm\_rad",<br>"rotation\_order": "zyx"<br>\}|\{<br>"error\_code": "NOT\_IMPLEMENTED",<br>"message": "放置动作执行尚待补充",<br>"status": "TODO",<br>"operation": "RELEASE",<br>"todo": \[<br>"RELEASE\_EXECUTION"<br>\]<br>\}|Mui|缺失|
|8085<br>\(视频流获取\)|/camera/health<br><br>|GET|无|\{"status":"READY"\}<br>|Kai<br>|有<br>|
||/camera/list|GET<br>|无<br>|各相机 color/depth 在线状态与分辨率||有<br>|
||/camera/snapshot<br>|GET|camera=head\&type=color|image/jpeg<br>||有<br>|
||/camera/stream|GET|camera=right\_wrist\&type=depth|视频流||有|
|8086<br>（Agent流程）|/pick|POST<br>|POST /pick<br>Content\-Type: application/json<br>Idempotency\-Key: task\-20260807\-0001:pick<br><br>\{<br>"task\_type": "SORTING",<br>"product\_name": "可口可乐",<br>"hand": "LEFT"<br>\}|成功响应（HTTP `200`）：<br><br>\{"status":"SUCCEEDED"\}<br>|skylar<br>|已调通|
||/place|POST<br>|POST /place<br>Content\-Type: application/json<br>Idempotency\-Key: task\-20260807\-0001:place<br>\{<br>  "task\_type": "SHORTAGE",<br>  "product\_name": "可口可乐",<br>  "hand": "left"<br>\}<br>|成功响应（HTTP `200`）：<br><br>\{"status":"SUCCEEDED"\}<br>|skylar<br>|未调通|
||||||||

# 容错机制

1、导航失败

2、
