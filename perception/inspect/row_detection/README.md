# 货架行检测

这个模块用红色货架横条划分商品行，不依赖训练模型：

1. 把输入统一到 `1280x720`；
2. 在 HSV 中提取两段红色色域；
3. 横向闭运算连接断开的横条，开运算滤掉短小红色物体；
4. 按每个 y 坐标的红色覆盖率和最长连续段筛选货架横条；
5. 当透视较大、横条明显倾斜时，用近水平 Hough 线段作兜底；
6. 按横条中心线从上到下划分商品行。

`pose_type` 是行数和行位置的强约束：

- `SHELF_VIEW_UPPER`：返回最上面两层；
- `SHELF_VIEW_LOWER`：把最后一条横条至图像底部作为候选层，返回最下面三层。

因此 LOWER 视角中，即使最下面一层的下边红条没有进入画面，也能通过图像底边补齐。

批量验证现有样例：

```powershell
python inspect/row_detection/cli.py `
  test_data/inspect_shortage_paired `
  test_data/inspect_misplaced_paired `
  --pattern "*.jpg" `
  --output-dir test_data/row_detection_results `
  --pose-type SHELF_VIEW_LOWER
```

每张图片都会保存原图、红色 mask、横向形态学 mask、标注图和 JSON。

代码调用：

```python
import sys
sys.path.insert(0, "inspect")

from row_detection import detect_rows

result = detect_rows("shelf.jpg")
print(result.rails)
print(result.rows)
row = result.row_for_bbox([500, 250, 100, 180])
```
