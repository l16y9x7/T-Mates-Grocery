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

也可以在现有输入之外上传指定图片，用于固定图片测试：

```json
{
  "name": "蒙牛纯牛奶",
  "image_name": "record_20260804_141434_337936_rgb.jpg",
  "image_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

- `image_base64` 不传或为 `null` 时，继续使用服务器当前 RGB 帧。
- 传入 `image_base64` 时，接口使用上传图片运行推理；图片可以由调用方从任意路径、URL 或其他来源读取，接口不关心来源。支持纯 base64 或 data URL，最大 20 MB。
- `image_name` 用于标识上传图片并原样写入响应，只允许不包含路径的 JPG/PNG 文件名。
- `image_name` 不用于服务器端查找文件；指定它时必须同时提供 `image_base64`。

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

### 单元测试

```powershell
python -m unittest -v test_main.py
```

### 使用标注图片运行真实推理

`test_inference.py` 会按以下顺序查找测试图片：

```text
商品 name
    → GET /sku/search_by_name
    → sku_id
    → perception/test_data/2026-08-04/image_name_mapping.json
    → 对应的 *_rgb.jpg
    → 读取并编码对应 RGB 图片
    → POST /visual/pick/locate（name + image_name + image_base64）
    → Qwen3/SAM3 完整推理
```

`image_name_mapping.json` 和 `2026-08-04` 目录只属于测试脚本；Locate API 本身不依赖这两个路径。

先启动 SKU API 和 Locate API，然后执行：

```powershell
python test_inference.py "蒙牛纯牛奶"
```

如果同一个 SKU 出现在多张测试图片中，脚本会逐张推理，并以图片名作为结果 key。每张成功结果都会叠加半透明 mask、bbox、实例编号和置信度，默认保存到：

```text
perception/test_data/2026-08-04/locate_results/<SKU_ID>/
```

结果 JSON 中的 `result_image` 字段会记录对应图片的绝对路径。JSON 默认打印到终端，也可以保存到文件：

```powershell
python test_inference.py "蒙牛纯牛奶" --output result.json
```

可使用 `--output-dir` 指定结果图片目录：

```powershell
python test_inference.py "蒙牛纯牛奶" --output result.json --output-dir D:/locate-results
```
