# 感知模块 SKU 表

本目录只保存商品感知所需的最小信息：商品 ID、名称、参考图片和标准位置。

## 文件说明

- `products.json`：每个 SKU 一条记录。
- `images/`：保存商品参考图片；`images` 字段使用相对此目录的路径。
- `build_catalog.py`：从标准摆放清单重新生成 `products.json`。
- `extract_images.py`：从标准摆放 DOCX 按商品单元格及裁剪参数提取参考图片。
- `validate_catalog.py`：检查字段、重复编号和位置冲突。

## 当前状态

每个商品只允许以下四个字段：

```json
{
  "sku_id": "SKU_001",
  "name": "NFC桔汁",
  "images": ["images/SKU_001.jpg"],
  "locations": ["H1_F_L1_C01"]
}
```

一个商品可能出现在多个标准货位，因此 `locations` 始终使用数组。尺寸、重量、物理属性、姿态和抓放方案不在本目录保存。

层号与列号约定：

- `L1` 到 `L5`：从上到下。
- `C01` 开始：面对当前货架面时从左到右。
- `F`/`B`：货架正面/反面。

## 使用方式

生成或刷新商品库：

```powershell
python build_catalog.py
```

从标准摆放 DOCX 提取全部商品图片并更新 `images` 字段：

```powershell
python extract_images.py
```

`build_catalog.py` 会保留现有 `images` 字段，不会在刷新货位时清空图片。

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
python api.py --host 0.0.0.0 --port 8080
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
| `GET` | `/images/...` | 无 | 获取图片文件 |
| `GET` | `/docs` | 无 | FastAPI 自动接口文档 |

按 SKU 查询：

```text
GET /sku/search_by_SKU?sku=SKU_088
```

按商品名查询：

```text
GET /sku/search_by_name?name=外星人电解质水青柠口味0糖
```

按货位查询：

```text
GET /sku/search_by_location?location=H2_F_L4_C05
```

以上三个查询接口均返回完整商品对象：

```json
{
  "sku_id": "SKU_088",
  "name": "外星人电解质水青柠口味0糖",
  "images": ["images/SKU_088.jpg"],
  "locations": ["H2_F_L4_C05"]
}
```

按商品名获取图片路径：

```text
GET /sku/get_image?name=外星人电解质水青柠口味0糖
```

```json
["images/SKU_088.jpg"]
```

获取所有商品名称，无需查询参数或请求体：

```text
GET /sku/get_all_names
```

```json
["NFC桔汁", "蒙牛纯牛奶", "纯甄酸奶"]
```

商品不存在时返回 `404`：

```json
{"error_code": "SKU_NOT_FOUND"}
```

运行测试：

```powershell
python -m unittest -v test_api.py
```
