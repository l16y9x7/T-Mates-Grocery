# Task2 流程与接口说明

## 1. 服务边界

`task2_service` 负责编排导航、观察位姿、缺货巡检、补货台抓取和货架放置调用。
感知服务自行完成相机取图和缺货识别；task2 不传图片、不实现视觉算法，也不调用 SKU
服务。8086 的 `/place` 内部定位、位姿估计和机械臂动作后续单独实现。

Task2 默认监听 `0.0.0.0:8109`：

```http
GET /health
POST /task2/run
```

`POST /task2/run` 请求体固定为 `{}`，可携带 `Idempotency-Key`。同一进程同时只允许运行
一个任务；已有任务执行时返回 `409 TASK_IN_PROGRESS`。

## 2. 巡检流程

巡检点顺序为：

```text
H1_F_L_INSPECT -> H1_F_R_INSPECT -> H1_B_L_INSPECT -> H1_B_R_INSPECT
-> H2_F_L_INSPECT -> H2_F_R_INSPECT -> H2_B_L_INSPECT -> H2_B_R_INSPECT
```

每个点严格串行执行：

```text
START_POSITION
-> /navigation/navigate
-> SHELF_VIEW_UPPER
-> /perception/inspect
-> SHELF_VIEW_LOWER
-> /perception/inspect
```

巡检请求固定为：

```json
{"task_type":"SHORTAGE"}
```

成功响应：

```json
{"findings":["可口可乐罐装"]}
```

每个商品与调用接口时的巡检点、观察位姿绑定。以“商品名、巡检点、观察位姿”作为
唯一发现标识，避免往返巡检时重复累计同一结果。找到两件后立即停止巡检；不足两件时
沿路线反向巡检并持续往返。

## 3. 抓取和放置

Task2 使用 `config/product-hand-options.yaml` 中的 `product_name`、`target_id`、`hands`
选择抓取手。上观察位姿匹配 L1-L2，下观察位姿匹配 L3-L5；多个候选货位只使用它们
共同支持的手。

补货台流程：

```text
START_POSITION
-> /navigation/navigate {"target_id":"replenishment_pickup"}
-> /pose/prepare {"pose_type":"REPLENISHMENT_TABLE_PICK_READY"}
-> 8086 /pick
```

8086 抓取请求：

```json
{
  "task_type": "SHORTAGE",
  "product_name": "可口可乐罐装",
  "hand": "LEFT"
}
```

放置时不计算具体货位或货架层，而是恢复发现该商品时的上下文：

```text
START_POSITION
-> /navigation/navigate {"target_id":"<记录的巡检点>"}
-> /pose/prepare {"pose_type":"<记录的上下观察位姿>"}
-> 8086 /place
```

`/place` 请求结构与 `/pick` 相同。Task2 不调用 `SHELF_PLACE_READY`，也不介入 8086
内部逻辑。

两件商品可使用不同手时，在补货台连续抓取后逐件放置；只能使用同一手时，按“取一件、
放一件、再取下一件”执行。全部放置成功且双手为空后导航到 `task_boundary`。

## 4. 生产地址

|服务|地址|Task2 使用接口|
|---|---|---|
|导航|`http://192.168.3.226:8081`|`/navigation/health`、`/navigation/navigate`|
|姿态|`http://192.168.3.226:8084`|`/pose/health`、`/pose/prepare`|
|感知|`http://127.0.0.1:8083`|`/perception/health`、`/perception/inspect`|
|取放编排|`http://127.0.0.1:8086`|`/health`、`/pick`、`/place`|

导航、姿态、抓取和放置均使用稳定的 `Idempotency-Key`。网络异常最多重试一次；物理
动作两次均无法确定结果时返回 `ACTION_RESULT_UNKNOWN` 并停止后续动作。
