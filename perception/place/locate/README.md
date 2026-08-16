# Place Locate

目标接口：`POST /perception/place/locate`

接口已挂载到统一的 `8083` 感知服务。它适用于货架整体保持静止、
相机观察位姿发生变化，但目标商品从标准场景中消失的补货场景。

## 当前接口契约

`/perception/inspect` 负责判断异常货架与商品名称；本接口固定从
`agent/output/task0` 读取正常场景 RGB-D，请求只上传当前场景 RGB-D。接口在 Task0
原图中生成目标商品 bbox/mask，并计算从 Task0 reference 相机到当前相机的 `4×4`
刚体变换；商品 6D 位姿由下游位姿估计接口负责计算。

```json
{
  "task_type": "SHORTAGE",
  "product_name": "可口可乐罐装",
  "location_id": "H1_F_L2_C01",
  "current_image_base64": "<当前场景 RGB>",
  "current_depth_image_base64": "<当前场景 NPY/PNG/RAW 深度>",
  "pose_type": "SHELF_VIEW_UPPER",
  "current_image_name": "current_rgb.jpg",
  "region_index": 1
}
```

`location_id` 可以是商品货位（如 `H1_B_L2_C01`）或巡检导航点（如
`H1_B_L_INSPECT`）。商品货位会根据 `agent/config/product-hand-options.yaml` 中的
`target_id` 映射到左右巡检点，再与视角组成目录名：

```text
agent/output/task0/H1_B_L_INSPECT_LOWER/
├── rgb.jpg
├── depth_mm.npy
└── meta.json
```

`SHELF_VIEW_UPPER` 对应 `_UPPER`，`SHELF_VIEW_LOWER` 对应 `_LOWER`。省略
`pose_type` 时，商品层 L1/L2 自动使用 UPPER，L3/L4/L5 自动使用 LOWER；若直接传
巡检导航点则必须提供 `pose_type`。可用 `INITIAL_SCAN_ROOT` 覆盖 task0 根目录，使用
`PRODUCT_HAND_OPTIONS_PATH` 覆盖货位映射文件。

Task0 和当前缺货检测固定使用 `head_color_optical_frame`，请求不再接收相机内参。
代码内置的 `1280×720` 标定为：

```text
K = [[910.744324, 0,          650.132690],
     [0,          910.395020, 381.874634],
     [0,          0,          1]]
D = [0, 0, 0, 0, 0]
distortion_model = plumb_bob
```

输入 RGB 若经过整体缩放，代码会按实际宽高同步缩放 `fx/fy/cx/cy`。深度必须已对齐
到各自 RGB，数值通过 `depth_unit_mm` 统一换算为毫米。

正式响应全部使用 Task0 reference 原图坐标系，并返回 reference 到 current 的变换：

```json
{
  "product_name": "可口可乐罐装",
  "bbox": [310, 220, 430, 650],
  "mask": "<Task0 原图同尺寸的 PNG base64>",
  "image_path": "agent/output/task0/H1_F_L_INSPECT_UPPER/rgb.jpg",
  "rotate_matrix": [
    [1, 0, 0, 25],
    [0, 1, 0, -28],
    [0, 0, 1, 5],
    [0, 0, 0, 1]
  ]
}
```

`bbox` 为 Task0 原图像素坐标 `[x1,y1,x2,y2]`，由最终全图商品 mask 计算，不再
归一化到 `[1,1000]`。`mask` 与 `image_path` 对应的 Task0 RGB 完全同尺寸、同坐标系。
`rotate_matrix` 虽沿用接口约定名称，实际是包含旋转和平移的完整 `4×4 SE(3)`：

```text
point_current = rotate_matrix @ point_reference
```

下游位姿估计先利用 Task0 RGB、bbox 和 mask 得到 `T_reference_object`，再计算
`T_current_object = rotate_matrix @ T_reference_object`。Place Locate 不接收
`reference_pose`，也不计算或返回商品 `target_pose`。

`pose_type` 与 inspection 含义一致，支持 `""`、`SHELF_VIEW_UPPER` 和
`SHELF_VIEW_LOWER`。接口复用顶层 `perception/row_detection`，将 reference mask
限制到目标商品所在货架层；无法可靠检测行时自动回退到未裁层的 change mask。

`POST /perception/place/locate/debug` 使用相同请求，并额外返回缺货差异 bbox、
reference mask 来源、SAM3 crop/bbox、匹配的 `row_index/row_bbox`、RGB-D 配准质量
指标，以及实际使用的 `inspection_target_id/baseline_path`。存在多个异常区域时，
`region_index` 按从上到下、同行从左到右的 1-based 顺序选择；也可以传
`reference_bbox=[x,y,width,height]` 精确指定 inspection 检出的基准图区域。

`POST /perception/place/locate/reference-mask/debug` 只执行到 Task0 reference mask
生成完成，因此请求不需要 `reference_pose`。它返回 Task0 RGB、原始 change mask、
选中的缺货 component、货架行、SAM3 crop/bbox/score 和最终 reference mask，适合在
test_web 的 `/qwen-review` 中结合已有 shortage 批测结果核验这一阶段，不会被后续重投影
或 6D 位姿校验阻断。网页直接复用已得到的商品名、reference bbox 和 region mask，
不再提供单独的上传页面。

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

初始扫描由 task0 按巡检点和上下视角保存：

```text
agent/output/task0/<inspection_target_id>_<UPPER|LOWER>/
├── rgb.jpg
├── depth_mm.npy
└── meta.json
```

- `rgb.jpg` 和 `depth_mm.npy` 必须时间同步并完成深度到 RGB 的对齐；加载时会校验
  `meta.json` 的尺寸、单位和 `aligned_to=rgb`。
- 相机内参固定使用代码内置的头部相机标定。
- 运行时还需要当前 RGB 和与其对齐的毫米深度。

## 推荐流程

1. 根据 `product_name + location_id` 加载正常场景 RGB-D。
2. 从正常场景和当前缺货场景生成点云。
3. 排除正常场景的目标商品 mask、当前场景的运动物体和深度无效区域，只保留货架、
   轨道和稳定背景。
4. 使用机器人相机外参、RGB 特征匹配加 PnP，或两者结合，生成配准初值。
5. 在静态货架点云上做带 RANSAC 的刚体配准和 point-to-plane ICP，估计
   `T_cur_ref`。
6. 在 Task0 原图中生成商品的全分辨率 mask，并计算其像素 bbox。
7. 返回 Task0 RGB 路径、bbox、mask 和 `T_cur_ref`。
8. 下游位姿估计在 Task0 原图上计算 `T_ref_object`，再使用
   `T_cur_object = T_cur_ref @ T_ref_object`。

inspection 的 ORB + homography 可以作为 RGB 特征初值或调试图，但不能替代三维
刚体配准，因为 homography 不包含可靠的三维平移、尺度和离平面旋转。

## Mask 定义

正式响应只返回 `reference_mask`：即 Task0 正常场景中实际商品的全分辨率二值 mask。
该 mask 不会投影到当前缺货图；它必须与 `image_path` 指向的 Task0 RGB 保持同尺寸、
同坐标系，供下游位姿估计直接使用。

## Debug 响应

正式请求和响应见文档开头的“当前接口契约”。Debug 接口额外返回缺货差异 region、
货架行约束、SAM3 选择信息，以及 RGB 重投影 RMSE、三维 RMSE、对应点数量、inlier
ratio 和同一份 `T_cur_ref`。

## 配准质量门槛

当前 HTTP 层先执行以下保守门槛，后续需要用实际数据继续标定：

- 三维 RANSAC 有效对应点不少于 12；
- 三维 inlier ratio 不低于 0.5；
- 三维 RMSE 不高于 20 mm；
- RGB 重投影 RMSE 不高于 4 px；
- 旋转矩阵行列式接近 1；
- `T_cur_ref` 必须通过完整 `4×4 SE(3)` 合法性校验。

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
- `T_cur_ref @ T_ref_object` 位姿转移工具（供下游位姿估计复用，Place Locate 正式流程
  不再调用）。

`registration.py` 已提供：

- ORB + BFMatcher + ratio test 的 RGB 特征匹配；
- current 到 reference 的 RANSAC homography 初值；
- 根据对齐后差异图剔除变化区域中的关键点；
- 静态关键点二次 homography 和空间覆盖率统计；
- RGB 关键点的对齐深度采样与三维反投影；
- 三维对应点的 RANSAC + SVD 刚体配准；
- reference mask 点云向当前相机的重投影工具（保留为独立几何能力，不属于当前正式
  Place Locate 输出）。

`main.py` 已提供并挂载：

- `POST /perception/place/locate` 正式接口；
- `POST /perception/place/locate/debug` 诊断接口；
- RGB 与 NPY/PNG/TIFF/RAW 深度解码及对齐尺寸校验；
- 内置 `head_color_optical_frame` 的 `1280×720` 标定，并按输入 RGB 尺寸缩放；
- 将 current 配准到 Task0 reference 后，在 reference 坐标系提取唯一缺货 bbox；
- SHORTAGE 使用已有商品 `sam3_prompt` 在 Task0 原图 bbox 附近分割实际商品，按与
  缺货 component 的重叠选择实例，并映射成与 Task0 depth 对齐的全图 reference mask；
- MISPLACED 保留原有的差异 component 深度细化；
- 返回原图 bbox/mask/path 和 `current_from_reference`，不接收或转换商品 6D pose。

运行核心测试：

```powershell
python -m unittest place.locate.test_pose_transfer place.locate.test_registration \
  place.locate.test_reference_mask place.locate.test_main -v
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
