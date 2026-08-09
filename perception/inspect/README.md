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
  "baseline_image_base64": "<满货基准图的 base64 或 data URL>",
  "current_image_base64": "<当前巡检图的 base64 或 data URL>",
  "reference_item_area": 12000
}
```

- `task_type` 支持 `SHORTAGE` 和 `MISPLACED`。
- `location_id` 是当前巡检点位 ID，必填；后续用于查询该位置允许出现的商品。
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

当前版本尚未接入 Qwen 商品语义识别，因此数组元素数量由对比算法检测到的 bbox 数量
决定，商品名暂时返回空字符串。bbox 仍保存在内部 `InspectResponse` 中，接入 Qwen 后
可直接使用变化区域、周围商品以及 `location_id` 对应候选商品完成识别，而不需要修改
HTTP 输出结构。

## 启动

在 `perception` 目录启动统一服务：

```powershell
python main.py
```

接口文档地址为 `http://127.0.0.1:8083/docs`。
