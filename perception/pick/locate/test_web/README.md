# Qwen3 / SAM3 Prompt 测试网页

启动：

```powershell
cd perception/pick/locate/test_web
python -m pip install -r requirements.txt
python server.py
```

浏览器打开：`http://127.0.0.1:8082`

页面顶部选择 `perception/test_data/2026-08-04` 中的 RGB 图片：

- 左侧输入 Qwen prompt。模型输出可以是单个目标对象，也可以是多个目标组成的数组：

  ```json
  [
    {"name": "abc", "bbox": [x1, y1, x2, y2]},
    {"name": "abc", "bbox": [x1, y1, x2, y2]}
  ]
  ```

  每次点击会以 `temperature=0.5` 独立采样三次，并把三次的全部目标框叠加在同一张图上。页面支持 `[0,1000]` 归一化坐标和像素坐标两种绘制方式，并分别展示解析后的 JSON 与模型原始输出；即使 JSON 解析失败，原始输出也会保留。

  IoU 会在三组结果之间两两计算。多个同名目标使用“最大 IoU 一对一匹配”，同时显示未匹配目标数量和总体平均 IoU。

  测试出满意的 Prompt 后，可选择对应 SKU 并点击“保存当前 Prompt”。结果按 `SKU名称: Prompt` 保存到 `perception/pick/locate/qwen_prompt_mapping.json`；再次保存同一 SKU 会覆盖旧值。

- Qwen 完成后，右侧自动显示第一个检测结果向外扩张 10% 的 crop；也可以在下拉框中切换三次采样产生的其他检测框。
- 右侧输入 SAM3 prompt 后，仅对当前 Qwen crop 进行分割。页面绘制 crop 内所有实例的 mask、bbox 和置信度，同时返回映射回原图的 `bbox_original_xyxy`。

默认使用：

- Qwen：`http://211.137.21.33:25542/v1/chat/completions`
- SAM3：`http://211.137.21.33:25541/api/v1/segment`

可通过环境变量 `QWEN3_URL`、`QWEN3_MODEL`、`SAM3_URL` 覆盖。
