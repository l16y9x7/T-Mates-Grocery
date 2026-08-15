# 巡检主接口

`inspect/main.py` 是巡检算法的统一入口。当前先运行 `comparison_based` 定位异常
区域；内部保留 bbox 和各算法结果，HTTP 接口输出稳定的商品语义结构。后续由 Qwen
结合 `location_id` 对应货架层的候选商品与标准图填写商品名。

## HTTP 接口

```http
POST /perception/inspect
Content-Type: application/json
```

请求体：

```json
{
  "task_type": "SHORTAGE",
  "location_id": "H1_F",
  "pose_type": "SHELF_VIEW_UPPER",
  "baseline_image_base64": "<满货基准图的 base64 或 data URL>",
  "current_image_base64": "<当前巡检图的 base64 或 data URL>",
  "reference_item_area": 12000
}
```

- `task_type` 支持 `SHORTAGE` 和 `MISPLACED`。
- `location_id` 是当前巡检点位 ID，必填；后续用于查询该位置允许出现的商品。
- `pose_type` 必填，支持 `""`、`SHELF_VIEW_UPPER` 和 `SHELF_VIEW_LOWER`，与
  `location_id` 一起用于查询当前画面候选 SKU。
- `reference_item_area` 可省略。
- 图片输入和现有对比算法阈值参数保持不变。
- 本接口不接收深度：它只确定异常货架和商品语义。正常/当前两组深度留给后续
  `/perception/place/locate`，用于把原商品 mask 三维重投影到当前头部相机视角。

`SHORTAGE` 响应统一使用 `findings`；没有缺货时返回
`{"findings": []}`，检测到缺货区域时返回：

```json
{
  "findings": [
    {"shortage_product_name": "可口可乐罐装"}
  ]
}
```

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

对比算法先定位 bbox，随后 Qwen 审核器使用 `location_id + pose_type` 获取候选商品，
再通过 `/sku/get_image` 获取标准图。Qwen 只能返回候选集合中的标准商品名；无法确认、
候选外名称、重复区域以及实际商品与标准商品相同的放错结果都会被拒绝或过滤。

主接口会直接调用公共模块 `perception/row_detection` 的 `detect_rows()` 检测基准图中的
货架行，不经过 HTTP；`place/locate` 复用同一实现约束目标点云所在层：

- 对比算法输出的 bbox 与传给 Qwen 的 `aligned_current` 使用同一个 `1280×720`
  基准坐标系，避免相机位姿变化导致局部裁图偏移。
- 行检测可以多看到 1 个相邻货架行：`SHELF_VIEW_UPPER` 使用画面最上面 2 行，
  `SHELF_VIEW_LOWER` 使用画面最下面 3 行，并重新映射为 SKU 候选第 1/2/3 行。
- 异常 bbox 至少 60% 落在选中窗口的某行内时，将对应 SKU 行作为可靠约束；
  检测行少于预期数、多出超过 1 行，或 bbox 不在窗口内时，退回全视角候选逻辑。
- `SHORTAGE` 使用缺货前 baseline/reference 图（样例中的 `_1.jpg`）按异常 bbox 裁出
  Qwen 主图，并只下载、发送异常所在行的候选 SKU 标准图。
- `MISPLACED` 拆成两个独立 Qwen 阶段：第一阶段使用当前异常局部图，通过本地视觉
  特征模型从全量 SKU 标准库召回 Top-K，再由 Qwen 识别
  `misplaced_product_name`，不受当前货架面、可见行或异常所在行限制。未配置
  `INSPECT_SKU_RETRIEVAL_MODEL_PATH` 时保留旧的可见行候选流程用于兼容开发环境；
  正式部署应配置本地模型路径。
- 第二阶段把 row detection 裁出的 current 目标行和摆放正确时的 baseline 目标行上下
  拼接，红框标出同一异常位置，只发送对应 SKU 行候选，判断当前行缺失、被替代的
  `gt_product_name`；两阶段分别校验后才合并为一条放错结果。
- `SHORTAGE` reference 局部图的纵向范围限制在对应货架行附近，减少上下层商品干扰。

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
