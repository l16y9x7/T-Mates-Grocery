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

`SHORTAGE` 响应为数组；没有缺货时返回 `[]`，检测到两个缺货区域时返回两个元素：

```json
[
  {"shortage_product_name": "可口可乐罐装"}
]
```

`MISPLACED` 同样返回数组：

```json
[
  {
    "misplaced_product_name": "可口可乐罐装",
    "gt_product_name": "雪碧罐装"
  }
]
```

对比算法先定位 bbox，随后 Qwen 审核器使用 `location_id + pose_type` 获取候选商品，
再通过 `/sku/get_image` 获取标准图。Qwen 只能返回候选集合中的标准商品名；无法确认、
候选外名称、重复区域以及实际商品与标准商品相同的放错结果都会被拒绝或过滤。

## 启动

在 `perception` 目录启动统一服务：

```powershell
python main.py
```

接口文档地址为 `http://127.0.0.1:8083/docs`。
