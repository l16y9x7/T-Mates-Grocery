# 感知模块 SKU 表

本目录保存商品 ID、名称、参考图片、标准位置和当前库存位置。

## 文件说明

- `products.json`：每个 SKU 一条记录。
- `inspection_candidates.json`：按巡检点和货架层维护 Qwen 可见候选；新巡检导航点映射补齐前保持为空。
- `images_new/`：保存商品 JPG 参考图片；`products.json` 中的 `images/...` 保持为 API 资源路径。
- `build_catalog.py`：从标准摆放清单重新生成 `products.json`。
- `extract_images.py`：从标准摆放 DOCX 按商品单元格及裁剪参数提取参考图片。
- `validate_catalog.py`：检查字段、重复编号和位置冲突。

## 当前状态

每个商品包含以下字段：

```json
{
  "sku_id": "SKU_001",
  "name": "NFC桔汁",
  "images": ["images/SKU_001.jpg"],
  "locations": ["H3_L01_C01", "H3_L01_C02"],
  "inventory": ["H3_L01_C01", "H3_L01_C02"]
}
```

一个商品可能出现在多个标准货位，因此 `locations` 始终使用数组且不会随抓取改变。`inventory` 初始复制 `locations`，成功抓取后按 `slot_id` 删除，用于表示当前仍需抓取的物理位置。

层号与列号约定：

- `H1` 到 `H3`：三个独立货架。
- `L01` 到 `L05`：从上到下。
- `C01` 开始：面对当前货架面时从左到右。

## 使用方式

生成或刷新商品库：

```powershell
python build_catalog.py
```

从标准摆放 DOCX 提取全部商品图片并更新 `images` 字段：

```powershell
python extract_images.py
```

`build_catalog.py` 会保留现有 `images` 字段，并把全部 `inventory` 重置为对应的
`locations`；运行中的单槽位补货/扣减应调用库存接口。

修改后运行校验：

```powershell
python validate_catalog.py
```

感知服务启动时加载 `products.json`，并建立以下索引：

```python
products_by_id: dict[str, Product]
products_by_name: dict[str, Product]
products_by_location: dict[str, Product]
```

正式比赛前应冻结 `catalog_version`，并把版本号写入运行日志。

## HTTP 查询接口

启动服务：

```powershell
python api.py --port 8080
```

服务使用 FastAPI。启动后可以通过 `/docs` 查看 Swagger UI，通过 `/openapi.json` 获取接口定义。商品数据在进程启动时加载，修改 `products.json` 后需要重启服务。

| 方法 | 路径 | 查询参数 | 返回值 |
|---|---|---|---|
| `GET` | `/sku/health` | 无 | `{"status": "READY"}` |
| `GET` | `/sku/search_by_SKU` | `sku`，例如 `SKU_088` | 完整 SKU 商品对象 |
| `GET` | `/sku/search_by_name` | `name` | 完整 SKU 商品对象 |
| `GET` | `/sku/search_by_location` | `location` | 完整 SKU 商品对象 |
| `GET` | `/sku/get_image` | `name` | 商品图片相对路径列表 |
| `GET` | `/sku/get_all_names` | 无 | 所有商品名称列表 |
| `GET` | `/sku/get_candidate_SKU` | JSON 请求体：`location_id`、`pose_type` | 按货架层分组的候选 SKU |
| `GET` | `/sku/get_row_layout` | JSON 请求体：`location_id`、`pose_type` | 当前层逐物理列的商品（保留重复 SKU） |
| `GET` | `/sku/get_inspection_candidate_SKU` | JSON 请求体：巡检点 `location_id`、`pose_type` | 按巡检视角和货架层分组的候选 SKU |
| `POST` | `/sku/modify_inventory` | JSON 请求体：`slot_id`、`modification` | 使用 `replendish` 或 `deplete` 幂等修改一个库存位置 |
| `POST` | `/sku/reset_inventory` | 无 | 将全部商品的 `inventory` 重置为 `locations` |
| `GET` | `/images/...` | 无 | 获取图片文件 |
| `GET` | `/docs` | 无 | FastAPI 自动接口文档 |

按 SKU 查询：

```text
GET /sku/search_by_SKU?sku=SKU_080
```

按商品名查询：

```text
GET /sku/search_by_name?name=外星人电解质水白桃口味0糖
```

按货位查询：

```text
GET /sku/search_by_location?location=H2_L03_C03
```

以上三个查询接口均返回完整商品对象：

```json
{
  "sku_id": "SKU_080",
  "name": "外星人电解质水白桃口味0糖",
  "images": ["images/SKU_080.jpg"],
  "locations": ["H2_L03_C03"],
  "inventory": ["H2_L03_C03"]
}
```

成功抓取后扣减对应库存位置：

```http
POST /sku/modify_inventory
Content-Type: application/json

{"slot_id": "H2_L01_C02", "modification": "deplete"}
```

补回库存时将 `modification` 改为 `replendish`。两个操作都是幂等的，响应中的
`modified` 表示本次是否实际改变了库存。

重置全部库存：

```http
POST /sku/reset_inventory
```

按商品名获取图片路径：

```text
GET /sku/get_image?name=外星人电解质水白桃口味0糖
```

```json
["images/SKU_080.jpg"]
```

获取所有商品名称，无需查询参数或请求体：

```text
GET /sku/get_all_names
```

```json
["NFC桔汁", "蒙牛纯牛奶", "纯甄酸奶"]
```

获取当前相机姿态下可能出现的候选 SKU：

```http
GET /sku/get_candidate_SKU
Content-Type: application/json

{
  "location_id": "H2_L04_C05",
  "pose_type": "SHELF_VIEW_UPPER"
}
```

`location_id` 格式为 `H2_L04_C05`：`H1–H3` 是货架编号，
`L01–L05` 从上到下表示货架层，`C01` 开始表示面对货架时从左到右的物理陈列列。

`pose_type` 取值：

- `""`：只返回 `location_id` 所在层；
- `"SHELF_VIEW_UPPER"`：返回 `L1`、`L2`；
- `"SHELF_VIEW_LOWER"`：返回 `L3`、`L4`、`L5`。

需要保留同一 SKU 占据的每一个物理列时，调用 `/sku/get_row_layout`；它与
`get_candidate_SKU` 不同，不会合并同层的重复 SKU。

巡检流程使用独立的视角候选接口。五点布局支持
`H1_INSPECT/H12_INSPECT/H2_INSPECT/H23_INSPECT/H3_INSPECT`；当
`inspection_candidates.json` 未配置某个点时，接口会根据 `products.json` 的物理
货位和五点可见范围自动生成候选：

```http
GET /sku/get_inspection_candidate_SKU
Content-Type: application/json

{
  "location_id": "H23_INSPECT",
  "pose_type": "SHELF_VIEW_LOWER"
}
```

如果提供手工候选，`inspection_candidates.json` 中的 `rows.1` 到 `rows.5` 对应
`L01` 到 `L05`，并覆盖自动推导结果；接口根据 `pose_type` 返回上面两层或下面
三层。修改文件后需要重启 SKU 服务。

响应外层数组按层从上到下排列，每层商品按照货位列号从左到右排列；同一商品占据
多个相邻货位时只返回一次。每项包含商品标准名称、参考图片和货位，可以直接作为
Qwen 的候选输入：

```json
[
  [
    {
      "sku_id": "SKU_001",
      "name": "NFC桔汁",
      "images": ["images/SKU_001.jpg"],
      "locations": ["H3_L01_C01", "H3_L01_C02"]
    }
  ],
  [
    {
      "sku_id": "SKU_014",
      "name": "品客薯片烧烤牛排味",
      "images": ["images/SKU_014.jpg"],
      "locations": ["H3_L02_C01"]
    }
  ]
]
```

商品不存在时返回 `404`：

```json
{"error_code": "SKU_NOT_FOUND"}
```

运行测试：

```powershell
python -m unittest -v test_api.py
```
