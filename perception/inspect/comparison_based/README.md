# 基于前后图对比的缺货检测

该模块无需训练模型，处理链路为：尺寸统一 → ORB/RANSAC 图像配准 → 灰度化与
CLAHE → 绝对差分 → OTSU 二值化 → 开/闭运算 → 轮廓面积判定。

相比最小示例多出的图像配准用于抵消机器人重复拍照时的轻微位移和透视变化；若相机
完全固定，可用 `--no-registration` 关闭。

## Python 调用

```python
import sys

# `inspect` 与 Python 标准库同名，因此把子目录加入模块搜索路径后导入。
sys.path.insert(0, "inspect")
from comparison_based import ComparisonConfig, detect_shortage

config = ComparisonConfig(
    # 建议用一个商品在基准图中的分割面积；判定阈值默认为它的 80%。
    reference_item_area=12000,
)
result = detect_shortage("shelf_full.jpg", "shelf_check.jpg", config)

print(result.has_shortage)
for region in result.shortages:
    print(region.bbox, region.area_ratio_to_reference)
```

若暂时没有单品面积标定，可不传 `reference_item_area`，算法将使用图像面积的 0.15%
作为最低候选面积。正式部署时应针对固定机位标定单品面积或直接调整
`min_contour_area_ratio`。默认还会丢弃小于最大候选 20% 的零碎噪声；需要同时检测
尺寸差异很大的多种商品时，可设置 `min_area_relative_to_largest=0` 关闭该过滤。

## 命令行与调试图

在 `perception` 目录运行：

```powershell
python inspect/comparison_based/cli.py `
  test_data/inspect_shortage_paired/1_1.jpg `
  test_data/inspect_shortage_paired/1_2.jpg `
  --output result.jpg --debug-dir debug
```

标准输出为 JSON，坐标格式是 `(x, y, width, height)`，坐标系与 `_1` 基准图一致。
调试目录包含配准后的实拍图、灰度差分图和形态学处理后的二值掩膜。

算法假定 `_1` 是满货基准、`_2` 是取货后的稳定画面，且两次之间没有补货、人员遮挡
或大幅度改变陈列。绝对差分本身检测的是“变化”；如果场景中同时发生这些变化，它们
也可能成为候选，应在机器人离开画面后拍摄，或通过上层传入货架 ROI 再调用本模块。

## 测试

```powershell
python -m unittest discover -s inspect/comparison_based/tests -v
```
