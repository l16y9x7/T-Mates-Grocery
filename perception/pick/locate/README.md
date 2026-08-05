# Sorting Pick Locate

当前只保留两个接口：

- `GET /video/frame`：返回本地测试目录中最新一张 RGB 图片。
- `POST /visual/pick/locate`：输入商品名和任务类型，调用 SAM3，返回一个 `bbox + mask`。
- `call_qwen3()`：已包含 Qwen3-VL 图片请求逻辑，`prompt` 暂时为空，并返回未经处理的原始 JSON。

启动：

```powershell
python -m pip install -r requirements.txt
python main.py
```

请求：

```json
{
  "product_name": "蒙牛纯牛奶",
  "task_type": "SORTING"
}
```

响应：

```json
{
  "name": "蒙牛纯牛奶",
  "bbox": [598.1, 73.2, 671.8, 249.6],
  "mask": "iVBORw0KGgo..."
}
```

`bbox` 使用 SAM3 的像素坐标 `[x1,y1,x2,y2]`，`mask` 是无 data-URL 前缀的 PNG base64。

需要修改的主要逻辑都在 `main.py`：

- `get_latest_rgb()`：以后替换为实际视频流取帧。
- `call_qwen3()`：实现 Qwen3-VL 请求和返回格式。
- `locate_product()`：加入商品库图片比对等其他筛选逻辑。
- `prompt_mapping.py`：维护中文商品名到 SAM3 英文粗类别的映射。
