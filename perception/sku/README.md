# 感知模块 SKU 表

本目录只保存商品感知所需的最小信息：商品 ID、名称、参考图片和标准位置。

## 文件说明

- `products.json`：每个 SKU 一条记录。
- `images/`：保存商品参考图片；`images` 字段使用相对此目录的路径。
- `build_catalog.py`：从标准摆放清单重新生成 `products.json`。
- `validate_catalog.py`：检查字段、重复编号和位置冲突。

## 当前状态

每个商品只允许以下四个字段：

```json
{
  "sku_id": "SKU_001",
  "name": "NFC桔汁",
  "images": [],
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

修改后运行校验：

```powershell
python validate_catalog.py
```

感知服务启动时加载 `products.json`，建议建立以下索引：

```python
products_by_id: dict[str, Product]
products_by_name: dict[str, Product]
product_by_location: dict[str, Product]
```

正式比赛前应冻结 `catalog_version`，并把版本号写入运行日志。

## HTTP 查询接口

启动服务：

```powershell
python -m pip install -r requirements.txt
python api.py --host 0.0.0.0 --port 8080
```

服务使用 FastAPI。启动后可以通过 `/docs` 查看 Swagger UI，通过 `/openapi.json` 获取接口定义。商品数据在进程启动时加载，修改 `products.json` 后需要重启服务。

| 方法 | 路径 | 查询参数 | 用途 |
|---|---|---|---|
| `GET` | `/health` | 无 | 健康检查 |
| `GET` | `/sku/locations` | `name` | 根据商品名查询全部标准位置 |
| `GET` | `/sku/images` | `name` | 根据商品名查询图片 URL |
| `GET` | `/sku/name` | `location` | 根据位置查询商品名 |
| `GET` | `/images/...` | 无 | 获取图片文件 |
| `GET` | `/docs` | 无 | FastAPI 自动接口文档 |

示例响应：

```json
{"name": "NFC桔汁", "locations": ["H1_F_L1_C01"]}
```

```json
{"location": "H1_F_L1_C01", "name": "NFC桔汁"}
```

商品不存在时返回 `404`：

```json
{"error_code": "SKU_NOT_FOUND"}
```

运行测试：

```powershell
python -m unittest -v test_api.py
```
