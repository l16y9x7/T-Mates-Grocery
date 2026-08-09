# 巡检主接口

`inspect/main.py` 是巡检算法的统一入口。当前只运行 `comparison_based`；响应同时保留
各算法的原始结果和空间融合结果，后续可以继续增加视觉大模型或目标检测算法。

## HTTP 接口

```http
POST /perception/inspect
Content-Type: application/json
```

请求体：

```json
{
  "task_type": "SHORTAGE",
  "baseline_image_base64": "<满货基准图的 base64 或 data URL>",
  "current_image_base64": "<当前巡检图的 base64 或 data URL>",
  "reference_item_area": 12000
}
```

- `task_type` 支持 `SHORTAGE` 和 `MISPLACED`。
- `reference_item_area` 可省略。
- 两张图会统一到 `1280×720`，所有 bbox 格式均为
  `[x, y, width, height]`。

响应示例：

```json
{
  "task_type": "SHORTAGE",
  "has_anomaly": true,
  "image_size": [1280, 720],
  "bbox_format": ["x", "y", "width", "height"],
  "findings": [
    {
      "bbox": [620, 420, 90, 160],
      "center": [665, 500],
      "sources": ["comparison_based"],
      "votes": 1
    }
  ],
  "algorithms": [
    {
      "name": "comparison_based",
      "success": true,
      "elapsed_ms": 130.5,
      "findings": [
        {
          "bbox": [620, 420, 90, 160],
          "center": [665, 500],
          "contour_area": 10800.0,
          "changed_pixels": 11200,
          "chroma_dominance_ratio": 0.61
        }
      ],
      "error": null,
      "difference_mode": "hybrid",
      "threshold": 42.0,
      "alignment_success": true
    }
  ]
}
```

当前输出是异常区域 bbox，尚未进行“bbox → 货位编号/商品名”映射。接入货架货位标定
表或商品识别算法后，可在 `InspectionPipeline` 的融合结果上完成该映射。

## 启动

在 `perception` 目录启动统一服务：

```powershell
python main.py
```

接口文档地址为 `http://127.0.0.1:8083/docs`。
