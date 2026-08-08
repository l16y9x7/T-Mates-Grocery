# Pick Check

`POST /perception/pick/check`

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐罐装",
  "hand": "left"
}
```

接口根据 `hand` 获取对应腕部相机图片，并通过 `product_name`
从 SKU 服务读取第一张标准参考图。两张图与 `prompt.txt` 一起传给 Qwen3。

响应：

```json
{"pick_status": "Success"}
```

或：

```json
{"pick_status": "Fail"}
```

正式服务从 `perception/main.py` 启动。
