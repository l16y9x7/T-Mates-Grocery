# Place Check

`POST /perception/place/check`

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐罐装",
  "hand": "left"
}
```

- `SORTING`：判断目标商品是否放在面前的交付台上。
- `SHORTAGE` / `MISPLACED`：判断目标商品是否放在面前的货架上，并且摆列整齐。

接口会将标准 SKU 参考图和对应 hand 的腕部相机图一起传给 Qwen3。

响应：

```json
{"place_status": "Success"}
```

或：

```json
{"place_status": "Fail"}
```
