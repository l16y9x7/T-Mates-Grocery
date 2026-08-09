# 基于前后图对比的缺货检测

该模块无需训练模型，处理链路为：尺寸统一 → ORB/RANSAC 图像配准 → 灰度化与
CLAHE → 亮度差分与 Lab 色度差分融合 → OTSU 二值化 → 开/闭运算 → 轮廓面积判定。

两张输入图默认先通过 `cv2.resize` 统一到 `1280×720`，因此输出 bbox 也使用
`1280×720` 坐标系。若需要保留原始分辨率，可设置 `target_size=None`，命令行则使用
`--keep-input-size`。

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
`min_contour_area_ratio`。默认还会丢弃小于最大候选 30% 的零碎噪声；需要同时检测
尺寸差异很大的多种商品时，可设置 `min_area_relative_to_largest=0` 关闭该过滤。

## 命令行与调试图

在 `perception` 目录运行：

```powershell
python inspect/comparison_based/cli.py `
  test_data/inspect_shortage_paired/1_1.jpg `
  test_data/inspect_shortage_paired/1_2.jpg `
  --output result.jpg --debug-dir debug
```

标准输出为 JSON，坐标格式是 `(x, y, width, height)`，坐标系为标准化后的
`1280×720` 基准图。
`--debug-dir` 会保存：

```text
01_baseline.jpg              原始基准图
02_aligned_current.jpg       配准到基准坐标系的当前图
03_difference.png            亮度与色度融合差分
03a_luminance_difference.png CLAHE 亮度差分
03b_chroma_difference.png    Lab a/b 色度差分
04_difference_heatmap.jpg    差分热力图
05_binary_mask.png           OTSU + 形态学后的掩膜
06_baseline_bboxes.jpg       基准图上的差异 bbox
07_current_bboxes.jpg        当前图上的差异 bbox
08_difference_bboxes.jpg     热力图上的差异 bbox
09_comparison_bboxes.jpg     基准图和当前图左右对比
result.json                  阈值、配准信息及所有 bbox 坐标
```

检测到差异时 bbox 使用红框，并在图中写出 `x/y/w/h`；没有差异时结果图会显示绿色
`NO CHANGE REGION`。Python 调用也可以使用
`result.save_debug("debug", "shelf_full.jpg")` 保存同样的产物。

放错检测建议增加 `--task-type misplaced`。该模式单独对 Lab 色度差分执行自适应阈值，
并要求候选区域至少 35% 的变化像素以色度差异为主，可过滤相机残余位移产生的灰度
边缘框；缺货模式则融合亮度和色度差分。

算法假定 `_1` 是满货基准、`_2` 是取货后的稳定画面，且两次之间没有补货、人员遮挡
或大幅度改变陈列。绝对差分本身检测的是“变化”；如果场景中同时发生这些变化，它们
也可能成为候选，应在机器人离开画面后拍摄，或通过上层传入货架 ROI 再调用本模块。

## 测试

```powershell
python -m unittest discover -s inspect/comparison_based/tests -v
```

## 使用 Qwen3-VL 比较成对图片

`qwen_compare_pairs.py` 会自动查找目录中的 `*_1.jpg` 和 `*_2.jpg`，将每组两张图
同时发送给 Qwen3-VL，并保存结构化差异、Qwen 原始回复和 bbox 标注图。

```powershell
python inspect/comparison_based/qwen_compare_pairs.py `
  test_data/inspect_misplaced_paired `
  --output-dir test_data/inspect_misplaced_paired/qwen_results
```

只处理指定 pair：

```powershell
python inspect/comparison_based/qwen_compare_pairs.py `
  test_data/inspect_misplaced_paired --pair 4
```

脚本默认使用 `config.py` 中的 `QWEN3_URL` 和 `QWEN3_MODEL`，也可以用 `--url`、
`--model` 和 `--timeout` 覆盖。若服务要求鉴权，通过环境变量 `QWEN_API_KEY` 提供。
每组输出包含 `result.json`、`qwen_raw.txt`、前后 bbox 图和左右对比图；总结果写入
`summary.json`。
