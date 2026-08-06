# Sorting Pick Locate

定位服务根据商品名称查询 SKU，读取已标注的 Qwen3/SAM3 配对 Prompt，在当前 RGB 帧上完成粗定位与精细分割。

## 启动

先启动 SKU 查询服务：

```powershell
cd perception/sku
python api.py --host 0.0.0.0 --port 8080
```

再启动定位服务：

```powershell
cd perception/pick/locate
python -m pip install -r requirements.txt
python main.py
```

默认地址：

- SKU：`http://127.0.0.1:8080`
- Locate：`http://127.0.0.1:8081`

可使用 `SKU_API_URL`、`QWEN3_URL`、`QWEN3_MODEL`、`SAM3_URL` 环境变量覆盖。

## 接口

### `GET /video/frame`

返回当前 RGB 图片。本地测试阶段使用 `perception/test_data/2026-08-04` 中最新的 RGB 图片。

### `POST /visual/pick/locate`

请求只包含商品名称：

```json
{
  "name": "蒙牛纯牛奶"
}
```

处理流程：

1. 调用 `GET /sku/search_by_name` 查询完整 SKU 信息。
2. 从 `qwen_sam_prompt_mapping.json` 读取该商品的 Qwen3 与 SAM3 Prompt。
3. Qwen3 以 `temperature=0.5` 独立采样三次。
4. 对跨采样 bbox 聚类，只保留至少由两个不同采样支持且匹配 IoU 严格大于 `0.85` 的目标；同一目标的坐标取支持框平均值。
5. 将 Qwen `[0,1000]` 归一化 bbox 转为原图像素坐标，向外扩张 10% 后裁图。
6. 在每个去重后的 Qwen crop 上调用 SAM3。
7. 将 SAM3 bbox 和 mask 映射回原始 RGB 图片。

成功响应：

```json
{
  "sku_id": "SKU_002",
  "name": "蒙牛纯牛奶",
  "image_name": "record_20260804_150039_346733_rgb.jpg",
  "instances": [
    {
      "bbox": [598.1, 73.2, 671.8, 249.6],
      "mask": "iVBORw0KGgo...",
      "score": 0.93
    }
  ]
}
```

- `bbox` 是原图像素坐标 `[x1, y1, x2, y2]`。
- `mask` 是原图尺寸的单通道 PNG base64，不包含 data-URL 前缀。
- `instances` 可以包含多个目标，每个 bbox 与同一对象的 mask 一一对应。

## Prompt 文件

`qwen_sam_prompt_mapping.json` 使用商品名称作为 key：

```json
{
  "蒙牛纯牛奶": {
    "qwen3_prompt": "...",
    "sam3_prompt": "frontmost milk carton"
  }
}
```

没有配对 Prompt、没有形成 Qwen 跨采样共识或 SAM3 没有实例时，接口返回对应的 `4xx/5xx` 错误，不会返回未确认的 bbox。

## 测试

```powershell
python -m unittest -v test_main.py
```
