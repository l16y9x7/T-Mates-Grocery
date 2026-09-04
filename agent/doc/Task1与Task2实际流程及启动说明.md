# Task1 与 Task2 实际流程及启动说明

本文描述当前工作区（2026-09-01）的实际代码行为，重点包括：

- Task1 模拟点单分拣的真实执行顺序；
- Task2 缺货巡检与补货的真实执行顺序；
- 商品货位、五个导航点和左右手配置如何参与规划；
- 正常路径、部分完成、重试、不确定状态和失败收尾；
- 生产机器与开发环境的服务启动顺序、命令和检查方式。

本文写的是当前程序实际执行的流程，不是比赛规则中的抽象流程。

## 1. 当前运行拓扑

### 1.1 统一任务入口

Task0～Task3 共用统一任务服务：

```text
POST http://127.0.0.1:8108/tasks/{task_id}/run
```

Task2 的请求体为空对象。Task1 也可以传空对象，由服务端在执行时随机生成两件商品的模拟订单：

```json
{}
```

Web 控制台会先调用 `POST /api/task1/mock-order` 生成可预览订单，点击“重新随机”可更换。开始任务时，Web 会把当前预览的订单原样提交：

```json
{
  "order_source": "mock_random",
  "order_id": "预览时返回的订单号",
  "product_names": ["商品A", "商品B"]
}
```

`order_id` 和 `product_names` 必须同时提供；商品必须恰好两个、互不相同。Task1 执行时会重新读取 SKU 目录，确认预览时的这两件商品仍然可用，不会在点击开始后悄悄换成另一对商品。

推荐每次请求都带唯一的 `Idempotency-Key`。仓库内的 `scripts/run-task.sh` 会自动生成。

顶层 8108 编排本身不缓存整个 Task1/Task2 的最终结果；重复提交相同顶层幂等键仍会重新进入任务编排。幂等键的主要保护发生在下游物理动作和 8086 单次抓放操作，所以人工重试顶层任务时仍应换一个新键，避免与上一次残留操作混淆。

统一任务服务使用一个全局机器人执行锁。同一时间只能运行一个任务；如果已有任务运行，新请求返回 HTTP 409 和 `TASK_IN_PROGRESS`，不会让两个任务同时控制机器人。

### 1.2 当前生产地址和端口

| 能力 | 地址 | 作用 |
| --- | --- | --- |
| 统一任务服务 | `127.0.0.1:8108` | Task0～Task3 编排、Web 控制台 |
| pick-place | `127.0.0.1:8086` | 抓取、放置复合流程 |
| Perception | `127.0.0.1:8083` | 巡检、抓取定位、放置定位 |
| SKU | `127.0.0.1:25540` | 商品名与一个或多个货位的映射 |
| SAM3 | `127.0.0.1:25541` | 感知分割能力 |
| Qwen3-VL | `127.0.0.1:25542` | 感知侧视觉语言能力 |
| GenPose2 | `127.0.0.1:8084` | 本机物体/参照物位姿估计 |
| 机器人导航 | `192.168.200.66:8081` | 绝对导航、左右微调、微调返回 |
| 机器人位姿/操作 | `192.168.200.66:8084` | 预备位姿、抓取、释放、夹爪控制 |
| 机器人相机 | `192.168.200.66:8085` | head、left_wrist、right_wrist 图像和 RGB-D |

本机 GenPose2 与机器人位姿服务都使用 8084，但 IP 不同，不冲突。

### 1.3 当前五个货架导航点

三个货架在当前配置中作为一个连续货架面处理：

| 顺序 | 点位 | 含义 |
| --- | --- | --- |
| 1 | `H1_INSPECT` | 正对 1 号货架 |
| 2 | `H12_INSPECT` | 1、2 号货架连接处 |
| 3 | `H2_INSPECT` | 正对 2 号货架 |
| 4 | `H23_INSPECT` | 2、3 号货架连接处 |
| 5 | `H3_INSPECT` | 正对 3 号货架 |

此外，任务还会使用这些导航点：

| 点位 | 用途 |
| --- | --- |
| `start` | 开始点，以及全局失败后的回退点 |
| `delivery_place` | Task1 商品交付台 |
| `replenishment_pickup` | Task2 补货商品领取台 |
| `task_boundary` | Task1、Task2 正常流程结束后的任务判定区 |

这些名称必须存在于机器人导航服务实际加载的导航 YAML 中。Agent 只发送 `target_id`，不会自己把名称转换成坐标。

### 1.4 货位、导航点和手的来源

运行配置通过以下文件加载每个精确货位的可抓方案：

```text
agent/config/product-hand-options.yaml
```

当前为 `schema_version: "2.0"`。一个货位可有一个或多个方案，例如：

```yaml
H1_L01_C04:
  product_name: "舒肤佳香皂柠檬清新香型"
  grasp_options:
    - hands: [RIGHT]
      target_id: H1_INSPECT
    - hands: [LEFT]
      target_id: H12_INSPECT
```

这表示同一个商品货位既可在正对货架点用右手，也可在连接处点用左手。Task1 和 Task2 都从这里枚举方案；程序没有“交换手”动作。

## 2. 两个任务共用的执行规则

### 2.1 移动前先收回机器人姿态

Task1、Task2 每次从当前点移动到另一个点前，实际顺序都是：

```text
POST /pose/prepare        pose_type=START_POSITION
POST /navigation/navigate target_id=<目标点>
```

如果编排器确认机器人已经在同一目标点，则复用当前位置，不重复 `START_POSITION` 和导航。

导航或预备位姿遇到明确执行失败时，各自最多再尝试一次；如果返回结果未知、网络结果未知或响应格式无效，则不会盲目重复可能已经执行过的物理动作。

### 2.2 最多处理两件，不交换手

- 一次任务最多以两件商品成功放置为目标；
- 能用左、右手分别拿一件时，优先一次带走两件；
- 不能形成左右手组合时，拆成单件流程；
- 不存在把商品从一只手转到另一只手的动作；
- 一只手不会同时持有两件商品。

### 2.3 明确失败和结果未知的处理不同

程序区分两类失败：

1. **明确失败**：下游明确表示动作没有成功。抓取/放置编排可以安全地再尝试一次；部分抓取错误还会触发左右微调。
2. **结果未知**：超时、网络中断或响应无法确认，动作可能已经发生。程序把对应手记为 `uncertain`，不再用这只手重复抓放，避免二次抓取或误放。

对于 8086 的抓放请求，如果原请求结果未知，客户端会先用同一个幂等键查询：

```text
GET /operations/result?idempotency_key=...
```

最多对账约 15 秒。仍无法确认时才把该手标记为不确定。

### 2.4 正常收尾与失败收尾

正常主流程完成后，Task1、Task2 都按以下顺序离场：

1. 导航到 `task_boundary`；
2. 如果失败，再导航一次 `task_boundary`；
3. 如果仍失败，改为导航到 `start`；
4. 三次都失败，整个任务返回失败。

如果主流程出现未被“单件 best-effort”吸收的全局异常，程序会尝试回 `start`。如果原任务失败且回 `start` 也失败，返回 `FAILURE_RECOVERY_FAILED`。

### 2.5 `SUCCEEDED` 可能是部分完成

Task1、Task2 采用“单件失败不阻断另一件”的 best-effort 策略。只要主流程能够走到并完成最终离场，响应中的顶层状态仍可能是：

```json
{"status": "SUCCEEDED"}
```

即使实际只放好一件，甚至没有放好商品，也可能出现这个状态。因此现场判断必须同时查看：

- `target_items[].picked`；
- `target_items[].placed`；
- `held_items` 是否为空；
- 任务日志中的 `placed_count` 和 `partial`。

HTTP 200 代表编排正常收尾，不等同于“两件全部完成”。

## 3. Task1：模拟订单分拣实际流程

### 3.1 总流程

```text
前端预览：SKU GET /sku/get_all_names（当前 43 项）
  → 随机选两个不同商品
  → 显示 order_id 和商品，可重新随机

实际执行：并行做四项健康检查
  → 再次 GET /sku/get_all_names，复核同一 order_id 下的两件商品
  → SKU 查询所有候选货位
  → YAML 联合规划货位、导航点、左右手
  → 抓取一件或两件
  → delivery_place 放置
  → task_boundary（失败时回 start）
```

### 3.2 第一步：健康检查

Task1 开始执行时并行检查以下四项是否返回 `READY`：

- navigation：机器人导航；
- pose：机器人位姿/操作；
- pick-place：8086 抓放编排；
- SKU：商品目录和货位映射。

任一项未就绪，Task1 在操作商品前失败，并尝试回 `start`。

当前 Task1 有效超时为：SKU 10 秒、导航 90 秒、位姿 30 秒、抓取 90 秒、放置 90 秒；连接 3 秒、健康检查 5 秒。单个下游动作可能在这段总预算内发生内部重试或结果对账。

### 3.3 第二步：模拟点单和订单复核

模拟点单的商品池不在前端写死。服务每次都通过：

```text
GET /sku/get_all_names
```

取得 SKU 服务当前的商品名列表。当前生产目录返回 43 个不同商品；`43` 不是前端的固定名单，页面同时显示该次返回的 `catalog_size`。

两种入口的行为是：

- **Web 预览**：进入 Task1 页面时生成新 `order_id`，从目录中不放回地随机取两个不同商品。点击“重新随机”会再请求一对。
- **Web 开始执行**：上送已预览的 `order_id` 和两个 `product_names`。四项健康检查通过后，Task1 重新获取 SKU 目录，检查两个名称非空、互不相同且仍在当前目录中，然后执行同一订单。
- **命令行或直接 HTTP `{}`**：没有预览订单时，四项健康检查通过后，Task1 当场获取目录、随机选两件并生成 `order_id`。

目录清理后少于两个可用商品，或前端提交的某个商品已不在当前目录中时，模拟点单阶段直接失败，不会随机替换用户已看到的商品。`skip_product_names` 中的商品不进入可选商品池；当前该列表为空。

### 3.4 第三步：SKU 转换为候选货位

对模拟订单中的每个商品调用：

```text
GET /sku/search_by_name?name=<商品名>
```

SKU 可以返回一个或多个 `locations`。每个货位必须符合当前格式，例如：

```text
H1_L01_C04
H2_L03_C06
H3_L05_C02
```

同一商品有多个货位时，候选货位先按以下层级优先级排序，再按货位编号排序：

```text
L3 → L2 → L4 → L1 → L5
```

这个排序只是候选顺序。后面的双手规划会查看该商品的所有候选货位，必要时可以选择另一个货位来形成更好的左右手组合。

当前 `skip_product_names` 和 `defer_product_names` 都为空：

- `skip` 可配置为完全跳过某商品；
- `defer` 可配置为把商品放到处理顺序最后；
- 当前订单中的所有商品都按原顺序进入规划。

SKU 无货位、货位格式错误或某个 SKU 请求失败时，只跳过该商品，另一商品仍可继续。

### 3.5 第四步：联合规划货位、导航点和左右手

对每件商品，程序把 SKU 的所有候选货位与 `product-hand-options.yaml` 中的所有 `grasp_options` 展开，再联合选择两件商品的方案。

两件商品的优先级严格为：

1. 两个不同物理货位、左右手不同、抓取导航点相同；
2. 两个不同物理货位、左右手不同、抓取导航点不同；
3. 如果不能形成左右手组合，选择两个不同物理货位，按单件串行流程处理。

关键限制：

- 两件商品不能指向同一个物理货位；
- 当前三段式货位 `H1/H2/H3_Lxx_Cxx` 必须显式存在于 YAML 中；
- 未映射的新货位不会默认获得左右手能力；
- 任一进入联合规划的商品完全没有抓取方案时，规划报 `NO_FEASIBLE_HAND_ASSIGNMENT`，该次已收集的规划目标会被清空；
- 单件商品使用排序后第一个可用货位、该货位 YAML 中第一个可用抓取方案。

### 3.6 第五步 A：形成左右手组合时

若两件商品分别使用 `LEFT` 和 `RIGHT`，执行顺序为：

1. 去第一件商品选中的抓取点；
2. 准备 `SHELF_PICK_READY`，并传入该货位层级；
3. 调用 8086 `/pick` 抓第一件；
4. 去第二件商品选中的抓取点；如果与上一件相同，则不重复导航；
5. 准备第二件对应层级的 `SHELF_PICK_READY`；
6. 调用 8086 `/pick` 抓第二件；
7. 两件都先抓完，再去 `delivery_place`；
8. 准备 `DELIVERY_TABLE_PLACE_READY`；
9. 两手都成功持物时，优先调用机器人位姿服务的 `/manipulation/release/both` 一次释放两件；
10. 双手释放明确失败、且没有手处于不确定状态时，回退为逐手调用 8086 `/place`；
11. 如果只抓到一件，则只放成功抓到的那一件。

两件商品可以来自不同抓取点；只要使用不同的手，仍然会先全部抓完，最后只去一次交付台。

### 3.7 第五步 B：只能使用同一只手时

如果不能形成左右手组合，Task1 严格执行“一件一趟”：

```text
商品 1 抓取点 → 抓商品 1 → delivery_place → 放商品 1
商品 2 抓取点 → 抓商品 2 → delivery_place → 放商品 2
```

不会先用同一只手抓两件，也不会交换手。因此这种情况要去两次交付台。

### 3.8 8086 内部怎样完成 Task1 抓取和放置

Task1 抓取请求包含：

```json
{
  "task_type": "SORTING",
  "product_name": "商品名",
  "hand": "LEFT",
  "level": "L3"
}
```

8086 内部实际执行：

1. 调用 Perception 抓取定位；
2. 按手选择腕部相机：左手用 `left_wrist`，右手用 `right_wrist`；
3. 获取 RGB、深度和对应相机标定；
4. 把商品 mask、RGB、深度、标定送到本机 GenPose2；
5. 得到六维位姿；
6. 调机器人 `/manipulation/grasp`。

Task1 在交付台的单手 `/place` 是固定释放流程：外层已准备好 `DELIVERY_TABLE_PLACE_READY`，8086 不再做商品定位、取图和位姿估计，直接调用机器人 `/manipulation/release`。

当前 8086 在抓取/释放执行成功后直接返回，抓放后的视觉复核代码处于禁用状态。因此“动作接口成功”不代表又额外拍照确认了实物结果。

### 3.9 Task1 抓取失败恢复

每次抓取或单手放置最多进行两次动作尝试。

对抓取而言，如果机器人明确返回 `manipulation_grasp` 执行失败，并附带可用六维位姿：

1. 根据位姿 x 方向选择向左或向右微调；
2. 执行 `/navigation/nudge`；
3. 重新准备当前层级的 `SHELF_PICK_READY`；
4. 再抓一次；
5. 无论重试成功与否，最多两次执行 `/navigation/nudge` return 回原微调点。

没有可用位姿的明确失败，则不做横向微调，但仍会重新准备抓取位姿并重试一次。

如果两次微调返回都失败，程序清空当前位置缓存，后续依靠绝对导航继续，不把“微调回原点失败”直接升级为全局任务失败。

网络/超时/无效响应造成结果未知时，不重抓，直接锁定对应手为不确定状态。

### 3.10 Task1 返回结果如何看

核心字段：

```json
{
  "task_type": "SORTING",
  "status": "SUCCEEDED",
  "product_names": ["模拟订单商品A", "模拟订单商品B"],
  "order": {
    "order_id": "7e5f...",
    "source": "mock_random",
    "catalog_size": 43,
    "product_names": ["模拟订单商品A", "模拟订单商品B"]
  },
  "target_items": [
    {
      "product_name": "商品名",
      "product_slot_id": "H2_L03_C05",
      "target_id": "H23_INSPECT",
      "shelf_level": "L3",
      "hand": "LEFT",
      "picked": true,
      "placed": true
    }
  ],
  "held_items": {},
  "interface_metrics": [
    {
      "interface": "sku/sku/get_all_names",
      "service": "sku",
      "method": "GET",
      "url": "http://127.0.0.1:25540/sku/get_all_names",
      "call_count": 1,
      "success_count": 1,
      "failure_count": 0,
      "total_duration_ms": 12.4,
      "average_duration_ms": 12.4
    }
  ]
}
```

`product_names` 是执行时生成或复核通过的两件模拟订单商品；`order` 保留订单号、来源、当次目录大小和同一对商品。`target_items` 是成功进入抓取规划的目标；如果某件商品查不到可用货位，它仍在 `product_names` 和 `order` 中，但可能不在 `target_items` 中。

### 3.11 Task1 接口次数和耗时统计

Task1 从四项健康检查开始，对每一个真实发出的下游 HTTP 尝试计时和计数。统计口径是“HTTP 尝试”，而不是只数业务步骤：

- 同一接口因明确失败而重试两次，`call_count` 增加两次；
- HTTP 非 2xx、连接错误和超时都计入 `failure_count`，失败尝试的耗时也计入累计值；
- 原动作结果未知后调用的 `/operations/result` 对账请求，每次实际请求同样单独计数；
- `duration_ms` 是最新一次 HTTP 尝试的耗时；`total_duration_ms` 是该接口所有尝试的耗时和；`average_duration_ms` 是累计耗时除以调用次数。重试之间的等待时间不算在单次 HTTP 耗时内。

Web 控制台的“Task 1 接口统计”表会随任务事件实时刷新，显示接口、次数、HTTP 成功/失败、累计、平均和最近耗时。成功结果的 `interface_metrics` 保留最终累计值；Task1 失败时，错误结果和持久化日志也会带已采集的统计，便于定位失败接口。

## 4. Task2：缺货巡检与补货实际流程

### 4.1 总流程

```text
健康检查 + Task0 基准完整性检查
  → 五个点各做 UPPER、LOWER 巡检，共 10 个观察
  → 汇总并按精确 slot_id 去重
  → YAML 校验商品、层级、导航点、左右手
  → 规划左右手补货批次
  → replenishment_pickup 一次抓一件或两件
  → 精确货位逐件放置
  → 成功放置两件后停止，或候选耗尽
  → task_boundary（失败时回 start）
```

### 4.2 第一步：健康和 Task0 基准检查

Task2 开始时检查：

- 机器人导航；
- Perception；
- 机器人位姿/操作；
- pick-place；
- 机器人相机；
- `camera/list` 中 head 相机在线且 color 流在线。

随后根据 `agent/output/task0/current.json` 选择当前完整扫描，并检查五个点的上下观察基准。当前一共需要 10 个目录：

```text
agent/output/task0/runs/<scan_id>/H1_INSPECT_UPPER/
agent/output/task0/runs/<scan_id>/H1_INSPECT_LOWER/
agent/output/task0/runs/<scan_id>/H12_INSPECT_UPPER/
agent/output/task0/runs/<scan_id>/H12_INSPECT_LOWER/
agent/output/task0/runs/<scan_id>/H2_INSPECT_UPPER/
agent/output/task0/runs/<scan_id>/H2_INSPECT_LOWER/
agent/output/task0/runs/<scan_id>/H23_INSPECT_UPPER/
agent/output/task0/runs/<scan_id>/H23_INSPECT_LOWER/
agent/output/task0/runs/<scan_id>/H3_INSPECT_UPPER/
agent/output/task0/runs/<scan_id>/H3_INSPECT_LOWER/
```

每个目录必须有三个非空文件：

```text
rgb.jpg
depth_mm.npy
meta.json
```

任何一个缺失或为空，Task2 在巡检前返回 HTTP 503、`BASELINE_NOT_READY`。

“存在且非空”只是 Task2 readiness 的第一道门槛。真正巡检加载基准时，Perception 还会校验 `meta.json`、RGB 解码、二维数值深度、尺寸与对齐关系等内容；文件存在不代表内容一定有效。

Task0 基准代表“货架完整时”的参考状态。货架布局、点位、相机标定、相机安装姿态或观察位姿发生变化后，应重新运行 Task0，而不是继续复用旧基准。

当前 Task2 有效超时为：单次巡检 30 秒、导航 90 秒、位姿 30 秒、抓取 90 秒、放置 90 秒；连接 3 秒、健康检查 5 秒。

### 4.3 第二步：完成五点十次巡检

当前五个点符合新命名规则，因此代码把它们视为一个连续三货架面。程序先完成全部 10 次观察，再开始规划补货，不会在第一个点发现缺货后立刻补。

实际顺序固定为：

| 次序 | 导航点 | 观察位姿 |
| --- | --- | --- |
| 1 | `H1_INSPECT` | `SHELF_VIEW_UPPER` |
| 2 | `H1_INSPECT` | `SHELF_VIEW_LOWER` |
| 3 | `H12_INSPECT` | `SHELF_VIEW_UPPER` |
| 4 | `H12_INSPECT` | `SHELF_VIEW_LOWER` |
| 5 | `H2_INSPECT` | `SHELF_VIEW_UPPER` |
| 6 | `H2_INSPECT` | `SHELF_VIEW_LOWER` |
| 7 | `H23_INSPECT` | `SHELF_VIEW_UPPER` |
| 8 | `H23_INSPECT` | `SHELF_VIEW_LOWER` |
| 9 | `H3_INSPECT` | `SHELF_VIEW_UPPER` |
| 10 | `H3_INSPECT` | `SHELF_VIEW_LOWER` |

每个观察调用：

```json
{
  "task_type": "SHORTAGE",
  "location_id": "H2_INSPECT",
  "pose_type": "SHELF_VIEW_UPPER"
}
```

导航到某点失败时，跳过该点的两个观察，继续下一个点。某个观察位姿失败或该次感知失败时，只跳过这一次观察，仍继续后续观察。

### 4.4 视觉怎样判断缺货列并产生 `slot_id`

Task2 的缺货列不是由 Agent 根据数组下标临时生成，也不是 SKU 猜出来的。当前正式 SHORTAGE 感知路径实际为：

1. 根据 `location_id + pose_type` 加载 Task0 的基准 RGB-D；
2. 从机器人 head 相机采集当前 RGB-D；
3. 对基准图和当前图分别检测货架层并裁出行区域；
4. 对两边使用相同的货架 mask 处理；
5. 读取 `perception/inspect/shortage_mapping_config.json` 中该点、该层的商品组、期望前排数量、商品顺序和精确 `slot_ids`；
6. SAM3 分别检测基准和当前画面中的商品组实例；
7. 使用实例横向位置、前排匹配和深度信息比较基准与当前槽位；
8. 对确认缺失的位置取配置中同一列的 `slot_id` 和商品名；
9. 返回：

```json
{
  "findings": [
    {
      "shortage_product_name": "商品名",
      "slot_id": "H2_L03_C05"
    }
  ]
}
```

因此，视觉参与的是“当前画面中哪个已配置槽位缺货”的判断；槽位编号文本本身来自固定映射配置。两者必须结合，才会得到最终 `slot_id`。

正式 SHORTAGE 路径目前使用 SAM3 的基准/当前前排槽位比较和深度判断。旧的 comparison/Qwen 路径仍用于其他场景或离线诊断，不是当前 Task2 SHORTAGE 的主判断路径。

### 4.5 第三步：严格校验和去重

当前手配置是 schema 2.0，因此每条巡检结果必须同时有非空商品名和合法 `slot_id`。缺少 `slot_id`、货位格式错误或响应结构错误时，该次感知视为 `INVALID_RESPONSE`，不会退回到“只按商品名猜货位”。

五点存在视野重叠，同一货位可能被正对点和连接点都看到。程序完成全脸巡检后用精确 `slot_id` 去重，同一货位只保留一次。

每条候选还必须通过：

- `slot_id` 在 `product-hand-options.yaml` 中存在；
- 感知商品名与该货位配置商品名规范化后一致；
- `L01/L02` 只能来自 `SHELF_VIEW_UPPER`；
- `L03/L04/L05` 只能来自 `SHELF_VIEW_LOWER`；
- 该货位至少有一个抓取点/手方案。

不通过的发现只被跳过，不会使用不安全的默认手。

注意：最终用于补货的 `inspection_target_id` 来自该精确货位 YAML 中的 `grasp_options.target_id`，不一定是最初发现缺货的观察点。这使连接处商品可以选择更适合指定手的点位。

### 4.6 第四步：规划补货批次

程序对所有有效缺货货位进行贪心分批，优先级为：

1. 两个不同物理货位、不同手、相同补货导航点；
2. 两个不同物理货位、不同手、不同补货导航点；
3. 没有左右手组合时，把候选拆成单件批次。

同一批最多两件，必须一件 `LEFT`、一件 `RIGHT`。程序不会把同一个槽位配两次，也不会交换手。

Task2 会遍历批次，直到成功放置两件或所有候选耗尽。`target_items` 只记录实际进入抓取尝试的目标，不是所有视觉发现。

### 4.7 第五步：在补货台抓商品

每个批次先执行：

1. 导航到 `replenishment_pickup`；
2. 准备 `REPLENISHMENT_TABLE_PICK_READY`；
3. 如果上一批有放置失败而仍持有商品，先在补货台调用 `/manipulation/gripper/open` 弃置，单手最多尝试两次；
4. 按批次顺序抓商品，最多抓到剩余成功容量。例如已经成功放置一件，本批最多再抓一件；
5. 双手批次会先抓完两件，再去货架放置。

Task2 的 8086 `/pick` 使用 head 相机，不使用腕部相机。内部仍执行：商品定位 → RGB/深度/标定 → GenPose2 → `/manipulation/grasp`。

如果一只手已持物或状态不确定，该手对应目标被跳过；另一只安全手的目标仍可以尝试。

### 4.8 第六步：逐件放回精确货位

对每一件已抓成功的商品，实际顺序为：

1. 导航到该 YAML 方案选定的 `inspection_target_id`；
2. 恢复该货位对应的 `SHELF_VIEW_UPPER` 或 `SHELF_VIEW_LOWER`；
3. 调用 8086 `/place`，同时传商品名、原抓取手、导航点、观察位姿和精确 `slot_id`；
4. 8086 要求 SHORTAGE 放置必须有 `slot_id`；
5. Perception 在当前画面中重新确认同一个商品名、同一个 `slot_id` 的缺口，并返回周围参照物；
6. 如果感知返回的 `slot_id` 与请求不一致，8086 拒绝继续；
7. 8086 获取参照物对应的当前 RGB-D 和 mask；
8. GenPose2 分别估计一个或两个参照物位姿；
9. 根据缺口在参照物左、右、两者之间或上方的位置合成目标放置位姿；
10. 调机器人准备 `SHELF_PLACE_READY`，并传入精确层级；
11. 调机器人 `/manipulation/release`，仍使用抓取时的同一只手。

两件商品即使在同一导航点，也仍按件完成放置定位和释放；第二件只会省掉重复导航。

8086 当前同样跳过放置后的视觉结果复核。因此 `placed=true` 表示放置动作链返回成功，不表示随后又用独立视觉检查确认商品最终状态。

### 4.9 Task2 抓放失败恢复

补货台抓取明确失败且返回了可用抓取位姿时：

1. 先重新准备 `REPLENISHMENT_TABLE_PICK_READY`；
2. 根据失败位姿做左/右微调；
3. 再抓一次；
4. 最后尝试回微调原点。

没有可用于定向微调的信息时，明确失败仍会原地重试一次。

货架放置当前显式关闭横向微调恢复：明确失败会在同一状态下再调用一次放置，但不会自动左右挪机器人。

抓取或放置如果结果未知，则对应手进入不确定状态，不重复物理动作。放置失败后商品仍记在 `held_items`；如果还有下一批，回补货台时先尝试弃置。如果没有后续批次，最终响应会把仍持有的商品暴露在 `held_items` 中。

最后一个批次放置失败且没有下一批时，当前实现不会额外安排一次回补货台弃置，而是继续执行结束导航；现场必须根据 `held_items` 和事件日志确认机器人是否仍带着商品。

### 4.10 Task2 返回结果如何看

核心字段：

```json
{
  "task_type": "SHORTAGE",
  "status": "SUCCEEDED",
  "inspection_pass": 1,
  "product_names": ["实际进入抓取尝试的商品"],
  "target_items": [
    {
      "product_name": "商品名",
      "product_slot_id": "H2_L03_C05",
      "inspection_target_id": "H23_INSPECT",
      "inspection_pose_type": "SHELF_VIEW_LOWER",
      "hand": "LEFT",
      "picked": true,
      "placed": true
    }
  ],
  "held_items": {}
}
```

当前 `inspection_pass` 固定为 1；五个点属于同一次全脸巡检。

## 5. 推荐启动流程

### 5.1 启动前必须满足的机器人侧条件

先在机器人本体确认以下三个服务可访问：

```bash
curl -fsS http://192.168.200.66:8081/navigation/health
curl -fsS http://192.168.200.66:8084/pose/health
curl -fsS http://192.168.200.66:8084/manipulation/health
curl -fsS http://192.168.200.66:8085/camera/health
curl -fsS http://192.168.200.66:8085/camera/list
```

相机列表至少应满足：

- Task0、Task2：`head` 的 color 和 depth 可用；
- Task1 分拣抓取：`left_wrist`、`right_wrist` 对应 RGB-D 可用。

并确认机器人导航配置已经加载：

```text
start
delivery_place
replenishment_pickup
task_boundary
H1_INSPECT
H12_INSPECT
H2_INSPECT
H23_INSPECT
H3_INSPECT
```

仅修改 Agent 的 YAML 不会在机器人导航服务中自动创建这些点。

### 5.2 生产机器一键启动

部署脚本假定仓库位于：

```text
/home/nora/tianji/T-Mates-Grocery
```

在生产机器执行：

```bash
cd /home/nora/tianji/T-Mates-Grocery
./deploy/start_all_services.sh --robot-ip 192.168.200.66 --restart-perception
```

脚本按以下顺序启动并等待：

1. SAM3；
2. Qwen3-VL；
3. GenPose2；
4. Perception；
5. SKU；
6. pick-place 8086；
7. 统一任务服务 8108；
8. 七项总健康检查。

脚本会保留已经健康的 SAM3、Qwen3-VL、GenPose2、Perception 和 SKU，不主动重启。`--restart-perception` 会先释放 8083，使新的机器人相机 IP 环境变量生效。

如果修改了机器人 IP，而 pick-place 或统一任务进程已经运行，仅执行 `start` 不会替换旧进程；应再执行：

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/services.sh restart 192.168.200.66
```

当前仓库生产配置本身已经是 `192.168.200.66`。

一键脚本按“健康则保留”工作。完成 `git pull`、修改运行配置或修改 `product-hand-options.yaml` 后，旧的健康进程不会自动重载新代码/新配置；至少应重启 8086、8108。商品手与点位 YAML 由统一任务服务启动时加载，修改后必须重启 8108。

### 5.3 手动启动 Agent 两个服务

只需要手动启动 Agent 时：

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/setup.sh
scripts/services.sh start 192.168.200.66
```

它等价于分别启动：

```bash
scripts/pick-place.sh start --robot-ip 192.168.200.66
scripts/tasks.sh start --robot-ip 192.168.200.66
```

注意：这两个脚本只启动 8086 和 8108，不会启动 Perception、SKU、SAM3、Qwen3-VL 或 GenPose2。

`scripts/run-task.sh --ensure-services ...` 也只保证 8086、8108 进程可达，不能替代完整能力服务启动。

### 5.4 总健康检查

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/health-check.sh 192.168.200.66
```

脚本要求以下七项都返回 HTTP 200：

- 统一任务 8108；
- pick-place 8086；
- Perception 8083；
- SKU 25540；
- 机器人导航 8081；
- 机器人位姿 8084；
- 机器人相机 8085。

统一任务 `/health` 会聚合 Task0～Task3 的 readiness。只要 Task2/Task3 所需 Task0 基准缺失，它可能返回 HTTP 503，即使 8108 进程本身已经正常启动。此时应查看响应中的 `tasks` 字段，并先运行 Task0。

生产一键脚本最后也执行同一个总健康检查。如果新部署还没有 Task0 基准，脚本可能在最后以非零状态结束，但此前已经启动的服务仍然在运行，可以继续执行 Task0。

`health-check.sh` 不直接检查 SAM3、Qwen3-VL 和本机 GenPose2；Perception 的 `/perception/health` 也主要表示进程可用，不等于整条视觉链已经验证。赛前还应单独执行：

```bash
curl -fsS http://127.0.0.1:25541/health
curl -fsS http://127.0.0.1:25542/v1/models
curl -fsS http://127.0.0.1:8084/manipulation/health
```

8086 `/health` 会进一步检查 Perception、本机 GenPose2、机器人 manipulation、机器人相机及相机列表。若一键启动后立即总检失败，先查看进程日志并重新执行健康检查，避免把短暂启动等待误判成服务未启动。

### 5.5 第一次部署或基准失效时运行 Task0

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/run-task.sh 0
```

Task0 会访问五个巡检点，各保存 UPPER、LOWER 一份对齐 RGB-D，共 10 份。成功后检查：

```bash
find output/task0/runs -maxdepth 3 -type f | sort
```

当前 `scan_id` 下应看到 10 个目录，每个目录有 `rgb.jpg`、`depth_mm.npy`、`meta.json`，并且扫描根目录包含 `manifest.json`。然后重新执行：

```bash
scripts/health-check.sh 192.168.200.66
```

### 5.6 启动 Task1

推荐在浏览器打开 `http://<生产机IP>:8108/`，选择 Task1。页面会从 SKU 服务的当前 43 个商品中随机显示两个不同商品；如果不接受这一对，点击“重新随机”。点击开始后，页面提交当前的 `order_id` 和两个商品，Task1 执行前复核这一订单。

如果使用命令行，空请求体会让服务端自动随机一对：

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/run-task.sh 1
```

等价 HTTP 请求：

```bash
curl -X POST http://127.0.0.1:8108/tasks/1/run \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: task1-manual-001' \
  -d '{}'
```

Task1 本身不读取 Task0 基准。它的启动前置会直接检查 navigation、pose、pick-place、SKU 四项；pick-place 在内部抓放时仍会使用 Perception、GenPose2、腕部相机和机器人操作服务。

### 5.7 启动 Task2

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/run-task.sh 2
```

等价 HTTP 请求：

```bash
curl -X POST http://127.0.0.1:8108/tasks/2/run \
  -H 'Content-Type: application/json' \
  -H 'Idempotency-Key: task2-manual-001' \
  -d '{}'
```

Task2 必须在完整且有效的 Task0 基准存在后运行。

Web 控制台默认由统一任务服务提供。生产脚本当前打印：

```text
http://192.168.200.65:8108/
```

### 5.8 停止或重启 Agent

```bash
cd /home/nora/tianji/T-Mates-Grocery/agent
scripts/services.sh stop
scripts/services.sh restart 192.168.200.66
```

停止顺序是先统一任务服务，再 pick-place；启动顺序是先 pick-place，再统一任务服务。

## 6. 日志和现场排查

### 6.1 日志位置

| 日志 | 默认位置 |
| --- | --- |
| Task1/Task2 编排事件与接口调用 | `agent/log/<时间戳>-<operation_key>/` |
| pick-place 每次复合操作 | `agent/log/<时间戳>-<operation_key>/` |
| 8086、8108 进程 stdout/stderr | `agent/log/process/` |
| 一键脚本启动的推理、感知、SKU 日志 | `/home/nora/tianji/logs/` |
| Task0 基准 | `agent/output/task0/` |

Task1、Task2 的编排目录中：

- `operation.json`：任务类型、幂等键、请求和创建时间；
- `events.jsonl`：每一步状态、商品、手、点位、重试和完整下游接口记录。

Task1 的每条下游接口事件还包含单次 `duration_ms`、累计 `call_count`、`success_count`、`failure_count`、`total_duration_ms` 和 `average_duration_ms`；最终聚合值保存在任务结果或错误的 `interface_metrics` 中。

### 6.2 常见现象对应检查

| 现象 | 优先检查 |
| --- | --- |
| 8108 `/health` 为 503 | 响应中哪个 task 为 `ERROR`；Task0 基准是否完整 |
| Task1 没有生成或复核订单 | `/sku/get_all_names` 是否返回至少两个不同商品；前端上送的两个名称是否仍在当前目录 |
| Task1 订单已有但没有规划商品 | SKU `search_by_name` 是否返回货位；货位是否存在于手配置 YAML |
| Task1 接口次数高于业务步骤数 | 重试、失败尝试和结果对账均按真实 HTTP 请求单独计数，属于预期行为 |
| Task1 分两趟 | 两件商品没有可行的左右手组合，属于预期串行回退 |
| Task2 返回 `BASELINE_NOT_READY` | 五点 × 上下共 10 个 Task0 目录及三个文件 |
| Task2 看出缺货但没补 | slot、商品名、UPPER/LOWER 和 YAML 是否一致；是否没有安全手方案 |
| 连接处重复发现 | 正常情况下会按精确 `slot_id` 去重；检查感知是否返回同一编号 |
| 某只手后续都被跳过 | 前一动作结果未知，手被标为 `uncertain` |
| 顶层 `SUCCEEDED` 但不足两件 | 查看每个 `target_items[].placed`、`held_items` 和日志 `partial` |
| 抓放动作接口成功但现场不一致 | 当前抓放后视觉复核被禁用；查看 8086 执行日志和机器人现场 |

## 7. 上场前最小检查清单

1. `runtime.production.yaml` 中机器人 IP 为 `192.168.200.66`；
2. 机器人导航 YAML 中存在全部任务点和五个货架点；
3. `product-hand-options.yaml` 为 schema 2.0，货位商品名与 SKU、感知映射一致；
4. 七项健康检查通过，或明确知道 8108 仅因 Task0 基准缺失而为 503；
5. Task2 前已采集当前场地的 10 份 Task0 RGB-D；
6. Task1 先确认前端能显示当前 43 项 SKU 中的两件模拟订单、可重新随机，执行后订单号和商品不变；
7. Task2 先用一个明确缺货槽位验证返回的精确 `slot_id`；
8. 结果判断同时看 `placed` 和 `held_items`，不要只看顶层 `SUCCEEDED`；
9. 操作期间不要并发启动其他任务；
10. 现场改过机器人 IP 后，重启 Perception、pick-place 和统一任务服务。

## 8. 对应实现文件

- `agent/config/runtime.production.yaml`：生产地址、端口、点位和超时；
- `agent/config/product-hand-options.yaml`：精确货位对应商品、抓取点和手；
- `agent/src/task1_service/service.py`：Task1 主编排和抓取恢复；
- `agent/src/task1_service/mock_order.py`：从当前 SKU 目录生成或复核两件商品的模拟订单；
- `agent/src/task1_service/client.py`：Task1 下游接口、动作对账和每次 HTTP 尝试的耗时/次数统计；
- `agent/web/app.py` 与 `agent/web/static/`：模拟订单预览、重新随机和 Task1 接口统计展示；
- `agent/src/task2_service/service.py`：Task2 巡检、去重、批次和补货编排；
- `agent/src/task2_service/client.py`：Task2 下游接口和严格巡检响应校验；
- `agent/src/pick_place_service/service.py`：8086 抓放内部定位、取图、位姿和执行；
- `perception/inspect/main.py`：正式 SHORTAGE 巡检入口；
- `perception/inspect/sam_shortage_pipeline.py`：SAM3 槽位比较和精确缺货列判断；
- `perception/inspect/shortage_mapping_config.json`：观察点、层、商品组和精确槽位顺序；
- `deploy/start_all_services.sh`：生产一键启动；
- `agent/scripts/services.sh`：8086、8108 统一启停；
- `agent/scripts/health-check.sh`：七项总健康检查；
- `agent/scripts/run-task.sh`：命令行运行 Task0～Task3。
