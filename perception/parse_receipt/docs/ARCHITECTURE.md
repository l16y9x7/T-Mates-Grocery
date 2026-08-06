# 代码结构和调用链

这份文档只解释当前第一版小票识别代码。它的核心逻辑是：图片进来，
转成 Qwen 接受的多模态消息，模型输出内部 JSON，本地校验后逐条查询
SKU 服务，最终返回标准商品名和货位。

## 一句话版本

```text
图片/PDF -> Qwen3-VL -> 中间 name/specification -> SKU API -> 最终 name/locations
```

## 模块职责

| 文件 | 作用 |
| --- | --- |
| `receipt_recognizer/config.py` | 从环境变量读取 Qwen 地址、模型名、可选 API key 和超时时间。 |
| `receipt_recognizer/api.py` | Qwen OpenAI-compatible HTTP 客户端。这里的 `api.py` 不是对外服务，而是“去调用 Qwen 的客户端”。 |
| `receipt_recognizer/media.py` | 本地读取图片/PDF，修正 EXIF 方向，限制最大边，转成 JPEG base64 data URL。 |
| `receipt_recognizer/prompts.py` | 放系统 Prompt、用户 Prompt 和一次 JSON 纠正 Prompt。 |
| `receipt_recognizer/schema.py` | 校验模型返回 JSON 的字段、类型、状态，并把内部 `line_items` 投影为 `name/specification` 中间结果。 |
| `receipt_recognizer/sku_client.py` | 逐个使用中间结果的 `name` 请求 `/sku/search_by_name`；精确失败时用 `/sku/get_all_names` 选择编辑距离最近的 SKU。 |
| `receipt_recognizer/inventory.py` | 读取库存 CSV，并统计识别出的 `name` 能否和库存 `sku_name` 精确匹配。 |
| `receipt_recognizer/evaluation.py` | 读取已保存的识别 JSON 和库存 CSV，统计商品名库存命中率。当前不评估规格。 |
| `receipt_recognizer/service.py` | 总编排：预处理、调用 Qwen、必要时纠正一次、构造诊断和业务输出。 |
| `receipt_recognizer/cli.py` | 本地命令行 `receipt-recognizer`。适合快速测单张图片或 PDF。 |
| `receipt_recognizer/probe.py` | 只读验证命令 `qwen-probe`，用于检查模型列表、OpenAPI、纯文本和小票请求。 |
| `receipt_recognizer/server.py` | FastAPI HTTP 服务，对外提供 `GET /health` 和 `POST /receipt/parse`。 |

## CLI 调用链

```text
receipt-recognizer receipt.jpg
  -> cli.py 解析命令行参数
  -> Settings.from_env() 读取 Qwen 配置
  -> ReceiptRecognizer.recognize_file()
  -> media.prepare_input()
  -> service.recognize_data_urls()
  -> api.OpenAICompatibleClient.create_chat_completion()
  -> schema.parse_receipt_result()
  -> business_items()
  -> sku_client.lookup_items()
  -> stdout 打印 SKU 的 name/locations 数组
```

CLI 支持 JPG、PNG 和 PDF。PDF 会先在本地临时目录渲染成图片，默认只取第一页。

## HTTP API 调用链

```text
curl -F "file=@receipt.jpg" http://host:port/receipt/parse
  -> server.py 接收上传文件
  -> media.image_bytes_to_data_url()
  -> service.recognize_data_urls()
  -> api.OpenAICompatibleClient.create_chat_completion()
  -> schema.parse_receipt_result()
  -> business_items()
  -> sku_client.lookup_items()
  -> HTTP 响应返回 SKU 的 name/locations 数组，或 SKU_NOT_FOUND 404
```

HTTP API 第一版只接收 JPG/PNG 图片，不接收 PDF。这样部署依赖更少，也更符合后续机器人或其他程序上传图片的方式。

## 模型调用次数

正常情况下只调用一次 Qwen。

如果模型第一次输出不是合法 JSON，或者字段类型/状态不满足本地校验，会追加一次“只修正结构，不改内容”的纠正请求。第二次仍失败就报错，不静默修补。

客户端不再合并相同商品行；模型识别到几条商品明细，就对 SKU 服务
发起几次查询。名称精确未命中时会用 SKU 全量名称列表选择编辑距离最近的商品；距离过大时才返回 `SKU_NOT_FOUND`。

## Base64 在哪里发生

调用方不需要手动传 Base64。

代码会把 JPG/PNG/PDF 转成：

```text
data:image/jpeg;base64,...
```

这是 Qwen OpenAI-compatible 多模态请求里 `image_url.url` 使用的格式。这样做的好处是：不会把本地文件路径传给服务器，服务器收到的是图片内容本身。

## 业务输出和诊断输出

Qwen 中间结果包含 `name/specification`，默认不会直接对外输出。默认业务输出：

```json
[
  {
    "name": "NFC桔汁",
    "locations": ["H1_F_L1_C01"]
  }
]
```

默认输出不包含 `flavor`、`count` 和 `source_text`。当前版本把口味、香型、型号都保留在完整商品名 `name` 里，数量默认按每条商品明细 1 件处理。

带 `--diagnostics` 或 `?diagnostics=true` 时，会额外返回内部诊断信息，包括：

- `receipt_status`
- `line_items`
- `review_items`
- `corrected_once`
- `finish_reason`
- `usage`

## 库存校验和实验评估

库存文件使用 CSV，至少包含：

```text
sku_name
```

如果某次实验目录里有 `*.items.json`，可以运行：

```bash
receipt-evaluate output/某次实验目录 \
  --inventory data/inventory.csv
```

主要看两个指标：

- `name_inventory_exact_rate`：识别出的商品名能和库存 `sku_name` 直接对上的占比。
- 当前库存 CSV 没有规格字段，且测试小票规格是临时填写的，因此规格字段暂不参与评估指标。

这里的“对上”是精确匹配；模糊包含只作为诊断建议，不算通过。
