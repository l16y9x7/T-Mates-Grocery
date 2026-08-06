# 最前方 Mask 判断方案

## 当前实现：重叠 bbox 链过滤

Locate API 在把所有 SAM3 bbox 和 mask 映射回原图后，会对结果执行一次全局过滤。该步骤可以处理同一个 Qwen crop 内的重复实例，也可以处理来自不同 Qwen crop、映射后互相重叠的重复实例。

两个 bbox 的重叠程度定义为：

```text
overlap_ratio = intersection_area / min(bbox_area_1, bbox_area_2)
```

默认 `overlap_ratio >= 0.2` 时建立一条重叠边。使用连通分量合并整条链，因此 A 与 B 重叠、B 与 C 重叠时，即使 A 与 C 不直接重叠，也会把 A、B、C 作为同一组，最终只保留一个实例。小于 20% 的轻微擦边不会合并，避免误删并排商品。

每条重叠链按以下顺序选择最前方实例：

1. 统计每个 mask 中灰度值不小于 128 的前景像素数量。
2. 如果最大 mask 面积至少是第二名的 2 倍，直接保留最大 mask。
3. 否则计算 `mask_density = mask_area / bbox_area`，保留密度最大的实例。
4. 密度相同时，依次比较 mask 面积、SAM3 score、bbox 面积。

可通过环境变量调节：

```text
SAM_BBOX_OVERLAP_MIN_RATIO=0.2
SAM_FRONT_AREA_DOMINANCE_RATIO=2.0
```

这里使用“交集占较小框比例”而不是 IoU，是因为同一商品的重复检测框可能一大一小或互相嵌套，此时 IoU 可能偏低，但较小框实际上大部分位于较大框内。

## 目标

当 SAM3 返回多个候选 mask 时，可以结合 **mask 点密度** 和 **bbox 大小**，估计哪个实例更靠近相机、处于最前方。

该方法属于启发式判断：通常越靠近相机的目标在图像中占据的面积越大，并且有效 mask 在 bbox 内的覆盖更充分。

## 指标定义

对于每个 SAM3 实例：

```text
mask_area = mask 中前景像素的数量
bbox_area = max(0, x2 - x1) × max(0, y2 - y1)
image_area = 图像宽度 × 图像高度
```

计算两个归一化指标：

```text
mask_density = mask_area / bbox_area
bbox_size = bbox_area / image_area
```

- `mask_density` 越大，表示 bbox 内被目标 mask 覆盖得越充分。
- `bbox_size` 越大，表示目标在画面中的占比越大，通常也更靠近相机。

## 推荐评分

优先使用乘法评分：

```text
front_score = mask_density × bbox_size
```

代入后等价于：

```text
front_score = mask_area / image_area
```

这实际上衡量了目标 mask 在整张图像中的占比。`front_score` 最大的候选实例可作为“最前方目标”。

如果希望单独调节两个指标的影响，可以使用加权评分：

```text
front_score = 0.4 × mask_density + 0.6 × bbox_size
```

推荐让 `bbox_size` 权重略高，因为透视关系通常首先体现在目标的画面尺寸上。权重需要根据测试数据调整。

## Mask 面积悬殊时的直接判断

完成类别、置信度和噪声过滤后，按照 `mask_area` 从大到小排序。如果最大 mask 的面积至少是第二大 mask 的 **2 倍**，可以跳过密度和 bbox 综合评分，直接将最大 mask 判断为最前方目标：

```text
largest_mask_area >= 2 × second_largest_mask_area
```

只有一个有效候选 mask 时，也直接选择该 mask。该规则只应在同一 SKU 或同一 SAM3 prompt 的候选实例之间使用，避免因商品真实尺寸差异造成误判。

## 建议流程

1. 只比较属于同一 SKU 或同一 SAM3 prompt 的候选实例。
2. 过滤置信度低于阈值的实例，例如 `score < 0.5`。
3. 过滤面积过小的噪声 mask。
4. 如果只有一个有效 mask，直接选择；如果最大 mask 面积至少是第二大的 2 倍，直接选择最大 mask。
5. 其余情况下，为每个实例计算 `mask_density`、`bbox_size` 和 `front_score`。
6. 按 `front_score` 从高到低排序，选择第一名。
7. 如果前两名分数非常接近，应保留“不确定”状态，而不是强行判断。

示例不确定条件：

```text
(score_1 - score_2) / max(score_1, 1e-6) < 0.1
```

## 注意事项

- 大而扁平的背景物体可能获得较高分数，因此最好先用 SKU、类别或 Qwen 检测框限制候选范围。
- 目标被遮挡时，`mask_density` 可能降低；只使用密度容易把完整但较远的目标误认为最前方。
- 如果候选 mask 来自 Qwen crop，应统一使用 crop 尺寸计算 `image_area`；若映射回原图比较，则应统一使用原图尺寸。
- 多个目标真实尺寸差异较大时，bbox 大小不能直接等价于距离。该方案更适合同类、尺寸接近的商品。
- 如果后续能获得深度图，应优先使用 mask 区域内的中位深度判断前后关系，本方案作为无深度信息时的备选。
