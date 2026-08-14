# Place Locate 设计草稿

目标接口：`POST /perception/place/locate`

本文档描述一个基于 RGB-D 场景配准的初版方案。它适用于货架整体保持静止、
相机观察位姿发生变化，但目标商品从标准场景中消失的补货场景。

## 结论

方案是合理的，但这里的“转移矩阵”必须是两个相机坐标系之间的 `4×4 SE(3)`
刚体变换，不能直接使用 inspection 中用于二维图像对齐的 `3×3 homography`。

设：

- `T_ref_object`：正常场景中，目标商品在基准相机坐标系下的 6D 位姿；
- `T_cur_ref`：把基准相机坐标系中的三维点变换到当前相机坐标系；
- `T_robot_cur`：当前相机坐标系到机器人基座坐标系的外参。

则：

```text
T_cur_object   = T_cur_ref @ T_ref_object
T_robot_object = T_robot_cur @ T_cur_ref @ T_ref_object
```

矩阵采用列向量约定：`p_target = T_target_source @ p_source`。所有矩阵均为
row-major 的 `4×4` 数组，平移单位统一为毫米。

如果现有 6D 位姿是机械臂 TCP 位姿而不是商品位姿，需要单独明确其语义；不要把
`T_ref_object` 和 `T_ref_tcp` 混用。机器人真正执行放置时通常还需要一个固定的
`T_object_tcp_place` 或预放置偏移。

## 数据准备

每个标准货位至少保存：

```text
reference_scenes/<location_id>/
├── rgb.jpg
├── depth_mm.npy
├── camera_info.json
├── target_mask.png
├── target_pose_camera.json
└── metadata.json
```

- `rgb.jpg` 和 `depth_mm.npy` 必须时间同步并完成深度到 RGB 的对齐。
- `camera_info.json` 保存彩色相机内参和畸变参数。
- `target_mask.png` 是正常场景中目标商品的 mask。
- `target_pose_camera.json` 保存 `T_ref_object`。
- `metadata.json` 至少保存 `product_name`、`location_id`、相机 frame 和单位。

运行时还需要当前 RGB、对齐深度、相机内参，以及当前相机到机器人基座的外参。

## 推荐流程

1. 根据 `product_name + location_id` 加载正常场景 RGB-D、商品 mask 和
   `T_ref_object`。
2. 从正常场景和当前缺货场景生成点云。
3. 排除正常场景的目标商品 mask、当前场景的运动物体和深度无效区域，只保留货架、
   轨道和稳定背景。
4. 使用机器人相机外参、RGB 特征匹配加 PnP，或两者结合，生成配准初值。
5. 在静态货架点云上做带 RANSAC 的刚体配准和 point-to-plane ICP，估计
   `T_cur_ref`。
6. 计算 `T_cur_object = T_cur_ref @ T_ref_object`。
7. 将正常场景目标 mask 内的 RGB-D 点变换到当前相机坐标系，再投影到当前 RGB，
   生成虚拟 `target_mask`。
8. 使用当前深度检查目标体积与现有商品、货架边缘是否碰撞。
9. 将目标位姿转换到机器人基座坐标系后返回。

inspection 的 ORB + homography 可以作为 RGB 特征初值或调试图，但不能替代三维
刚体配准，因为 homography 不包含可靠的三维平移、尺度和离平面旋转。

## Mask 定义

建议明确区分两个 mask：

- `reference_mask`：正常场景中原商品真实分割得到的 mask；
- `target_mask`：把正常场景商品点云变换并投影到当前视角后得到的虚拟目标 mask。

缺货场景中目标商品不存在，因此 `target_mask` 不是 SAM3 在当前 RGB 上直接分割的
结果。它表示“商品按标准位姿放回后，在当前相机中的预期投影”。

重投影时需要做 z-buffer 和当前深度检查。若目标投影被其它物体遮挡，Debug 响应可
同时返回完整投影 mask 和当前视角可见 mask。

## 接口草案

请求：

```json
{
  "product_name": "奥利奥冰淇淋抹茶味",
  "location_id": "H1_F_L2_C11",
  "hand": "left",
  "image_name": null,
  "image_base64": null,
  "depth_npy_base64": null
}
```

- 生产请求不传图片时，从相机服务获取同步 RGB-D 和 camera info。
- 本地测试可以同时上传 RGB 和 NPY 深度。
- `location_id` 不应省略，因为部分 SKU 在多个货位出现，仅凭商品名无法唯一确定
  放置位置。

成功响应：

```json
{
  "product_name": "奥利奥冰淇淋抹茶味",
  "location_id": "H1_F_L2_C11",
  "strategy": "REFERENCE_POSE_TRANSFER",
  "target_pose": {
    "frame_id": "robot_base",
    "unit": "millimeter",
    "matrix": [
      [1, 0, 0, 420],
      [0, 1, 0, -85],
      [0, 0, 1, 930],
      [0, 0, 0, 1]
    ]
  },
  "target_mask": "<base64 PNG>",
  "registration": {
    "current_from_reference": [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
    "rmse_mm": 3.2,
    "inlier_ratio": 0.86,
    "correspondence_count": 1240
  },
  "confidence": 0.94,
  "image_path": "C:/.../current_rgb.jpg"
}
```

Debug 接口还应返回：

- 正常场景 RGB、当前 RGB；
- 静态背景有效 mask；
- 初始配准和 ICP 后的 `T_cur_ref`；
- 变换后的点云投影；
- 完整 `target_mask`、可见 `target_mask`；
- 配准 RMSE、inlier ratio 和失败原因。

## 配准质量门槛

以下数值需要用实际数据标定，初版可以从这些范围开始：

- 静态背景有效三维对应点不少于 100；
- ICP inlier ratio 不低于 0.6；
- 点到平面 RMSE 不高于 10 mm；
- 旋转矩阵行列式接近 1；
- 转换后的目标位姿仍位于目标货架层范围内；
- 目标体积与当前点云没有显著碰撞。

任何关键校验失败时都应返回定位失败，不能继续使用低质量矩阵驱动机械臂。

## 两类放置方式

### 放在前面

优先使用本方案，把正常场景中的前排商品标准位姿转移到当前视角。后方商品、左右
商品和货架轨道都只作为静态场景配准与碰撞检查的辅助，不要求必须成功分割某一个
特定参照物。

### 放在上面

可以继续使用 Pick Locate 找到下方支撑商品，再结合 mask 内深度拟合顶部平面。
如果该货位也有可靠的正常场景记录，仍可用位姿转移结果作为先验和交叉校验。

## 当前实现

`pose_transfer.py` 已提供：

- 4×4 刚体矩阵校验；
- 坐标系变换求逆与组合；
- 已知三维对应点的 SVD 刚体配准；
- `T_cur_ref @ T_ref_object` 位姿转移。

`registration.py` 已提供：

- ORB + BFMatcher + ratio test 的 RGB 特征匹配；
- current 到 reference 的 RANSAC homography 初值；
- 根据对齐后差异图剔除变化区域中的关键点；
- 静态关键点二次 homography 和空间覆盖率统计；
- RGB 关键点的对齐深度采样与三维反投影；
- 三维对应点的 RANSAC + SVD 刚体配准；
- reference mask 点云向当前相机的完整/可见 mask 重投影；
- 重投影目标的 z-buffer `expected_depth_mm`，mask 外深度为 0；
- 标准商品位姿向当前相机和机器人坐标系的转移。

运行核心测试：

```powershell
python -m unittest place.locate.test_pose_transfer place.locate.test_registration -v
```

在 inspection 的 MISPLACED paired RGB 上运行：

```powershell
python -m place.locate.run_paired_registration
```

输出默认保存在：

```text
test_data/inspect_misplaced_paired/place_locate_registration/
```

当前这 6 组 paired 数据只有 RGB，没有 `depth_mm.npy` 和 camera info，因此这里只验证
RGB 静态特征配准，并在结果中明确写入
`se3_status=skipped_missing_depth_and_intrinsics`。本次 6 组全部成功，最终静态内点为
772–2046，二维重投影 RMSE 为 1.13–1.34 px。

SHORTAGE paired 可以使用：

```powershell
python -m place.locate.run_paired_registration `
  --data-root test_data/inspect_shortage_paired `
  --output-root test_data/inspect_shortage_paired/place_locate_registration
```

4 组 SHORTAGE RGB 配准全部成功，最终静态内点为 652–910，二维重投影 RMSE 为
1.12–1.50 px。Pair 4 的 reference 为 `1440×1080`、current 为 `2560×1920`；实现会
在各自原始像素坐标中匹配，再把 current warp 到 reference 尺寸，不要求两张 RGB
分辨率相同。进入 RGB-D 阶段后，每张深度仍必须分别与自己的 RGB 对齐，并使用各自
对应的相机内参。

真正上线前仍需补充 point-to-plane ICP、相机服务同步 RGB-D/camera info 读取、参考场景
管理、目标 mask 标注和三维碰撞检测。当前的 3D RANSAC + SVD 已可在取得同步深度与
内参后直接产生 `T_cur_ref`，但不能用本数据集的 RGB-only 结果宣称 6D 已通过实测。
