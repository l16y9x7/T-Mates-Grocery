# 巡检主接口

`inspect/main.py` 是巡检算法的统一入口。正式 `SHORTAGE` 请求会对 Task0 baseline
和当前 head-camera RGB-D 分别执行 row detection、SAM3 前排实例筛选和左右有序槽位
匹配，再用 `inspect/shortage_mapping_config.json` 直接得到缺货商品身份。正式
`MISPLACED` 仍使用 `comparison_based + Qwen` 两阶段识别。

## HTTP 接口

```http
POST /perception/inspect
Content-Type: application/json
```

请求体：

```json
{
  "task_type": "SHORTAGE",
  "location_id": "H1_INSPECT",
  "pose_type": "SHELF_VIEW_UPPER",
  "reference_item_area": 12000
}
```

- `task_type` 支持 `SHORTAGE` 和 `MISPLACED`。
- `location_id` 是当前巡检点位 ID，必填，支持
  `H1_INSPECT/H12_INSPECT/H2_INSPECT/H23_INSPECT/H3_INSPECT`。
- `pose_type` 必填，支持 `""`、`SHELF_VIEW_UPPER` 和 `SHELF_VIEW_LOWER`，与
  `location_id` 一起用于查询当前画面候选 SKU。
- `reference_item_area` 可省略。
- HTTP 接口不接收 RGB 或深度字段。初始 RGB-D 根据
  `agent/output/task0/current.json` 读取
  `agent/output/task0/runs/<scan_id>/<location_id>_UPPER|LOWER/`；尚无指针时兼容
  旧的平铺目录。当前 RGB-D 从 head camera 快照接口获取。
- 当前帧会先保存为临时目录中的 `rgb.jpg`、`depth_mm.npy` 和 `meta.json`。可通过
  `INSPECT_TEMP_DIR` 指定临时目录根路径；请求结束后自动清理。正式 SHORTAGE 的
  baseline/current RGB-D 和槽位诊断结果会另外持久化到
  `INSPECT_QWEN_DEBUG_DIR`（未配置时使用 Qwen review 默认 debug 目录）。
- 离线批测和 Python 测试入口继续允许直接传入 NumPy RGB/深度数据，不会访问相机。

`SHORTAGE` 响应统一使用 `findings`；没有缺货时返回
`{"findings": []}`，检测到缺货区域时返回：

```json
{
  "findings": [
    {
      "shortage_product_name": "可口可乐罐装",
      "slot_id": "H2_L01_C01"
    }
  ]
}
```

正式 SAM shortage 流程按物理货位返回结果；同一商品有多个缺货槽位时，每个
`slot_id` 各返回一条，供 Task2 精确选择导航点、左右手和最终放置槽。

`MISPLACED` 使用相同的顶层结构：

```json
{
  "findings": [
    {
      "misplaced_product_name": "可口可乐罐装",
      "gt_product_name": "雪碧罐装"
    }
  ]
}
```

以下 Qwen 候选逻辑只用于 `MISPLACED` 及保留的离线旧流程。对比算法先定位 bbox，
随后 Qwen 审核器使用 `location_id + pose_type` 获取候选商品；
巡检导航点调用 `/sku/get_inspection_candidate_SKU`，具体商品货位继续调用
`/sku/get_candidate_SKU`，再通过 `/sku/get_image` 获取标准图。Qwen 只能返回候选集合中的标准商品名；无法确认、
候选外名称、重复区域以及实际商品与标准商品相同的放错结果都会被拒绝或过滤。

主接口会直接调用公共模块 `perception/row_detection` 的 `detect_rows()` 检测基准图中的
货架行，不经过 HTTP；`place/locate` 复用同一实现约束目标点云所在层：

- 对比算法输出的 bbox 与传给 Qwen 的 `aligned_current` 使用同一个 `1280×720`
  基准坐标系，避免相机位姿变化导致局部裁图偏移。
- 行检测可以多看到 1 个相邻货架行：`SHELF_VIEW_UPPER` 使用画面最上面 2 行，
  `SHELF_VIEW_LOWER` 使用画面最下面 3 行，并重新映射为 SKU 候选第 1/2/3 行。
- 异常 bbox 至少 60% 落在选中窗口的某行内时，将对应 SKU 行作为可靠约束；
  检测行少于预期数、多出超过 1 行，或 bbox 不在窗口内时，退回全视角候选逻辑。
- 正式 `SHORTAGE` 不再调用 Qwen 或 SKU 候选接口；每层、每组 SAM prompt、预期列数
  和从左到右的商品身份均来自 `shortage_mapping_config.json`。
- baseline/current 的前排槽位严格按从左到右的单调顺序匹配；深度差超过 40 mm
  才判作深度后移。所有已匹配槽位同时发生 30—80 mm 的相近正向漂移时按相机整体
  位姿漂移处理，不判缺货。
- `MISPLACED` 拆成两个独立 Qwen 阶段：第一阶段使用当前异常局部图，通过本地视觉
  特征模型从全量 SKU 标准库召回 Top-K，再由 Qwen 识别
  `misplaced_product_name`，不受当前货架面、可见行或异常所在行限制。未配置
  `INSPECT_SKU_RETRIEVAL_MODEL_PATH` 时保留旧的可见行候选流程用于兼容开发环境；
  正式部署应配置本地模型路径。
- 第二阶段把 row detection 裁出的 current 目标行和摆放正确时的 baseline 目标行上下
  拼接，红框标出同一异常位置，只发送对应 SKU 行候选，判断当前行缺失、被替代的
  `gt_product_name`；两阶段分别校验后才合并为一条放错结果。
- `SHORTAGE` 的 SAM3 输入由公共 row detection 裁到对应货架层，并使用前排深度、
  已标注列数和有序槽位共同过滤后排实例。

当前 SKU 候选行数约定为：`pose_type=""` 对应 1 行，`SHELF_VIEW_UPPER` 对应 2 行，
`SHELF_VIEW_LOWER` 对应 3 行。行检测失败或少于该数量不会使巡检接口失败。

MISPLACED 全库特征检索配置：

- `INSPECT_SKU_RETRIEVAL_MODEL_PATH`：提前下载的 Hugging Face 视觉模型本地目录；
- `INSPECT_SKU_RETRIEVAL_TOP_K`：召回数量，默认 `10`；
- `INSPECT_SKU_RETRIEVAL_DEVICE`：`auto`、`cuda`、`mps` 或 `cpu`，默认自动选择；
- `INSPECT_SKU_RETRIEVAL_INDEX_PATH`：离线特征索引路径；标准图或模型路径变化后会自动重建。

推荐模型为 `google/siglip-base-patch16-224`。联网准备机器上执行：

```bash
python perception/sku/prepare_retrieval_model.py --device auto
```

脚本会把模型下载到 `perception/sku/models/`，并为 `images_new` 的全部商品建立本地
特征索引。正式运行只需设置 `INSPECT_SKU_RETRIEVAL_MODEL_PATH` 指向该模型目录，
模型加载使用 `local_files_only=True`，不需要外网。

## 启动

在 `perception` 目录启动统一服务：

```powershell
python main.py
```

接口文档地址为 `http://127.0.0.1:8083/docs`。
