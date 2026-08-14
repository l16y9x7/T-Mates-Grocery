# Sorting Pick Locate

定位服务根据商品名称查询 SKU，读取已标注的 Qwen3/SAM3 配对 Prompt，在当前 RGB 帧上完成粗定位与精细分割。

## 启动

先启动 SKU 查询服务：

```powershell
cd perception/sku
python api.py --port 8080
```

再启动定位服务：

```powershell
cd perception
python -m pip install -r requirements.txt
python main.py
```

默认地址：

- SKU：`http://127.0.0.1:25540`
- Locate：`http://127.0.0.1:8083`

可使用 `SKU_API_URL`、`QWEN3_URL`、`QWEN3_MODEL`、`SAM3_URL` 环境变量覆盖。

未随请求上传图片时，只从根目录 `config.py` 配置的相机服务获取当前 RGB，不读取本地测试图片。`SORTING` 按 `hand` 使用 `left_wrist`/`right_wrist`；`SHORTAGE` 和 `MISPLACED` 固定使用 `head`。其中 `SHORTAGE` 不请求深度，`MISPLACED` 使用与 RGB 对齐的 `head` 深度。非本机 IP 统一由 `CAMERA_SERVICE_HOST` 配置；仍可通过 `CAMERA_SERVICE_URL`、`CAMERA_SNAPSHOT_URL` 和 `CAMERA_SNAPSHOT_TIMEOUT_SECONDS` 覆盖地址与超时，通过 `CAMERA_SNAPSHOT_CACHE_DIR` 指定快照缓存目录。

## 接口

### `GET /video/frame`

返回当前 RGB 图片。默认 `task_type=SORTING`，按 `hand` 返回对应腕部相机；传入 `task_type=SHORTAGE` 或 `task_type=MISPLACED` 时返回头部相机。服务会验证响应是有效 JPG/PNG；连接失败、非 2xx、空响应、图片无效或缓存失败时返回 HTTP 400，不读取本地图片。

### `POST /perception/pick/locate`

请求包含商品名称和左右手信息：

```json
{
  "task_type": "SORTING",
  "product_name": "蒙牛纯牛奶",
  "hand": "left"
}
```

也可以在现有输入之外上传指定图片，用于固定图片测试：

```json
{
  "task_type": "SORTING",
  "product_name": "蒙牛纯牛奶",
  "hand": "left",
  "image_name": "record_20260804_141434_337936_rgb.jpg",
  "image_base64": "/9j/4AAQSkZJRgABAQ..."
}
```

- `image_base64` 不传或为 `null` 时，继续使用服务器当前 RGB 帧。
- 传入 `image_base64` 时，接口使用上传图片运行推理；图片可以由调用方从任意路径、URL 或其他来源读取，接口不关心来源。支持纯 base64 或 data URL，最大 20 MB。
- `image_name` 用于标识上传图片并原样写入响应，只允许不包含路径的 JPG/PNG 文件名。
- `image_name` 不用于服务器端查找文件；指定它时必须同时提供 `image_base64`。
- `level` 默认可省略；仅当 `SORTING` 的商品名和 `hand` 命中 `hard_case_config.json` 中的顺序定位特例时必须提供。

处理流程：

1. 调用 `GET /sku/search_by_name` 查询完整 SKU 信息。
2. 根据 `task_type` 从对应 JSON 读取该商品的 Qwen3 与 SAM3 Prompt：SORTING 使用 `qwen_sam_prompt_mapping.json`，SHORTAGE 使用 `qwen_sam_prompt_mapping_shortage.json`，MISPLACED 使用 `qwen_sam_prompt_mapping_misplaced.json`。
3. 仅将 RGB 拉伸到 `1280×720` 推理画布；原始 RGB 与 depth 均保持原尺寸。
4. Qwen3 在该 `1280×720` RGB 上以 `temperature=0.5` 独立采样三次。
5. 对跨采样 bbox 聚类，只保留至少由两个不同采样支持且匹配 IoU 严格大于 `0.85` 的目标；同一目标的坐标取支持框平均值。
6. 将 Qwen `[0,1000]` bbox 转为 `1280×720` 推理画布像素坐标，向外扩张 10% 后裁图，并在每个去重后的 crop 上调用 SAM3。
7. 将 SAM3 bbox 和 mask 按 X/Y 比例映射回原始 RGB 图片；响应的 `image_size` 是原图尺寸，`inference_image_size` 是 `[1280, 720]`。
8. 普通 case 在存在有效深度且有多个 SAM 实例时，先计算每个 mask 内的深度中位数，并保留距最近实例 30 mm 内的第一排；没有有效深度时保持原候选。
9. 对第一排候选构建重叠链，每条链只保留按 mask 面积与密度判断最靠前的一个实例，然后执行最终 PICK。

### SORTING hard case 顺序定位

以下同品牌易混淆商品只在 `SORTING` 下启用顺序定位：脉动、外星人电解质水、
草原红太阳烧烤料/烧烤酱、镇江香醋/蒸鱼豉油/薄盐生抽。其他 SKU 以及
`SHORTAGE`、`MISPLACED` 保持上述原流程不变。

是否进入 hard case 由 `perception/hard_case_config.json` 中的
`商品名 + level + hand` 精确组合决定。请求传入的 `level` 用于从 SKU `locations`
中定位对应层并查询从左到右的标准 SKU 顺序，不再自动选择最高层。未命中的组合按
普通 case 运行。左手相机从标准顺序左端开始对应，右手相机从右端开始对应。

SAM3 实例优先按 Qwen 陈列堆来源组成陈列列；Qwen 只返回一个合并区域时，使用过滤后的
第一排 SAM 实例作为可见列。系统优先检测原图中的红色货架前沿，按每个 bbox 底边到
透视前沿线的距离保留第一排；检测不到红线时才回退到瓶底高度规则，框高和 mask 面积
不作为硬淘汰条件。hard case 会先筛第一排，再对第一排中的重叠 SAM mask 去重，避免
后排或局部 mask 把相邻商品传递性合并。Debug 响应的 `hard_case` 给出目标 location、顺序和
陈列组映射；最终 `instances` 带有 `mapped_product_name`、
`hard_case_group_index`、`is_selected`。正式接口只返回 `is_selected=true` 的目标实例。
不要求当前图片检测到标准库中的全部品牌列：左手检测结果对应标准顺序最左侧的可见
前缀，右手检测结果对应标准顺序最右侧的可见后缀。只有目标 SKU 不在当前可见列范围
内时才拒识。

成功响应：

```json
{
  "product_name": "蒙牛纯牛奶",
  "bbox": [467, 102, 525, 347],
  "mask": "iVBORw0KGgo...",
  "image_path": "C:/data/locate/monitor_images/62af...jpg"
}
```

- `bbox` 是 `[x1, y1, x2, y2]`，坐标归一化到闭区间 `[1,1000]`。
- `mask` 是原图尺寸的单通道 PNG base64，不包含 data-URL 前缀。
- `image_path` 是服务端持久化原图的本地绝对路径，供同一文件系统上的监控程序直接读取。上传图片按内容哈希存储，接口返回后不会随临时目录删除；存储目录可通过 `LOCATE_MONITOR_IMAGE_DIR` 调整。
- 过滤后仍有多个实例时，正式单实例接口返回 bbox 中心点距离原图中心最近的一个。
- bbox 交集默认覆盖较小框至少 20% 才组成重叠链；链内最大 mask 达到第二名 2 倍时直接保留最大 mask，否则保留 `mask前景像素数 / bbox面积` 最大者。可通过 `SAM_BBOX_OVERLAP_MIN_RATIO` 和 `SAM_FRONT_AREA_DOMINANCE_RATIO` 调节阈值。
- 重叠链过滤后，若最小 mask 面积不超过第二小 mask 的 50%，会再删除这个最小面积离群项一次；通过 `SAM_SMALLEST_MASK_MAX_RATIO` 调节阈值。

### `POST /perception/pick/locate/debug`

测试专用接口，输入与正式接口相同，但额外返回 `image_base64`、`image_media_type`、`sku_id`、`image_name`、`image_path`、`image_size`、共识后的 `qwen_bboxes` 和全部 `instances`。其中 `image_base64` 是本次推理实际使用的原图，调用方不需要访问 Locate 服务所在机器的本地文件。

若原图已经取得，但 Qwen3/SAM3 推理失败，Debug 接口仍返回 HTTP 200 和该原图，并通过 `error`、`error_status_code` 记录原始错误；此时 `qwen_bboxes`、`instances` 可以为空。正式接口仍按原始状态码返回错误，不改变生产调用语义。`test_inference.py` 使用该接口记录 Qwen bbox，并分别绘制 Qwen 图和 SAM3 bbox/mask 图。

## Prompt 文件

| task_type | Prompt JSON |
|---|---|
| `SORTING` | `qwen_sam_prompt_mapping.json`（保留现有配置） |
| `SHORTAGE` | `qwen_sam_prompt_mapping_shortage.json` |
| `MISPLACED` | `qwen_sam_prompt_mapping_misplaced.json` |

`qwen_sam_prompt_mapping.json` 使用商品名称作为 key：

```json
{
  "蒙牛纯牛奶": {
    "qwen3_prompt": "...",
    "sam3_prompt": "frontmost milk carton"
  }
}
```

没有配对 Prompt、没有形成 Qwen 跨采样共识或 SAM3 没有实例时，正式接口返回对应的 `4xx/5xx` 错误，不会返回未确认的 bbox；Debug 接口按上一节约定返回原图与错误信息。

## 测试

### 单元测试

```powershell
python -m unittest -v test_main.py
```

### 使用标注图片运行真实推理

#### 正式接口测试

`request_formal_api.py` 的命令行接收 `task_type`、`product_name`、`level`、`hand` 四个必填输入：

```powershell
python test_formal_api.py SORTING "可口可乐" L1 left
```

脚本会用 `product_name` 请求 SKU API，再通过 `image_name_mapping.json` 和 SKU ID 自动找到 `2026-08-04` 下的所有对应本地图片。随后由脚本内部补充 `image_name` 和 `image_base64`，逐张调用正式 `/perception/pick/locate`，并校验响应只包含 `product_name`、`bbox`、`mask`、`image_path`，bbox 坐标均在 `[1,1000]` 内。

可选保存测试结果：

```powershell
python test_formal_api.py SORTING "可口可乐" L1 left --output formal_result.json
```

#### Debug 推理与结果图

`test_inference.py` 使用与正式接口相同的四个必填输入 `task_type`、`product_name`、`level`、`hand`，并按以下顺序查找测试图片：

```text
product_name
    → GET /sku/search_by_name
    → sku_id
    → perception/test_data/2026-08-04/image_name_mapping.json
    → 对应的 *_rgb.jpg
    → 读取并编码对应 RGB 图片
    → POST /perception/pick/locate/debug（product_name + level + hand + image_base64）
    → Qwen3/SAM3 完整推理
```

`image_name_mapping.json` 和 `2026-08-04` 目录只属于测试脚本；Locate API 本身不依赖这两个路径。测试脚本也是独立的 HTTP 客户端，不导入或调用本地 `main.py`。

Debug API 的离线图片请求可额外提供 `depth_image_name`、`depth_image_base64` 和可选的
`depth_is_bigendian`。深度文件支持二维数值型 NPY、16 位单通道 PNG/TIFF 或无头 16UC1 RAW/BIN，且
尺寸必须与 RGB 一致。未提供离线深度数据时，上传 RGB 的流程继续使用无深度回退。

对于竖直堆叠、抓取时需要拿顶部单件的特定 SORTING SKU，服务不读取深度，而是在 SAM3
最高分前 `0.10` 的候选中选择 bbox 中心最高的实例。当前包括得宝纸巾、海氏海诺创口贴、
德佑湿巾、心相印纸巾、农心碗面、妙洁海绵百洁布、三种康师傅桶面、纯棉酒店大毛巾、
京东京造毛巾、小苏打、心相印厨房纸巾和拖鞋。阈值可通过
`PICK_UPPER_CONFIDENCE_SCORE_MARGIN` 调整。

`中盐精制盐` 同样不读取深度，但不使用上方高置信度规则，而是保留 SAM3 原始候选并
选择实际前景像素数最多的 mask；置信度与 bbox 面积只在 mask 面积相同时用于打破平局。

### 2026-08-13 record 批量 RGB+depth 测试

`test/batch_record_inference.py` 默认读取
`test_data/2026-08-13/sorting_pick_locate_batch.json`，通过当前 8083 Debug API 对映射中的每个
record/商品组合运行 `SORTING` 检测。脚本从 `robot_state.json` 读取左右腕，上传同目录的
`rgb.jpg` 和 `depth_mm.npy`，并把标注图保存为 record 目录下的 `{product_name}.jpg`。
同名 JSON 保存精简响应；全局进度保存在 `sorting_pick_locate_batch_results.json`。

```powershell
# 只校验 52 个 record、商品名、层级、手腕和输入文件
python pick\locate\test\batch_record_inference.py --dry-run

# 断点续跑：跳过成功项，失败项会重新检测
python pick\locate\test\batch_record_inference.py

# 从头覆盖全部结果
python pick\locate\test\batch_record_inference.py --overwrite
```

连续 3 个系统连接错误时脚本会自动中止，避免在推理服务不可达时批量生成无效结果。

默认请求 `127.0.0.1` 上的两个服务：

```text
SKU API:    http://127.0.0.1:25540
Locate API: http://127.0.0.1:8083
```

确认远端服务已启动后执行：

```powershell
python test_inference.py SORTING "蒙牛纯牛奶" left
```

如果同一个 SKU 出现在多张测试图片中，脚本会逐张推理，并以图片名作为结果 key。每张成功结果会保存两张图：`*_qwen.png` 绘制共识去重后的 Qwen bbox，`*_locate.png` 绘制 SAM3 半透明 mask、bbox、实例编号和置信度。默认保存到：

```text
perception/test_data/2026-08-04/locate_results/<SKU_ID>/
```

结果 JSON 会保留 `qwen_bboxes`，并通过 `qwen_result_image` 和 `result_image` 分别记录 Qwen 图与 SAM3 图的绝对路径。JSON 默认打印到终端，也可以保存到文件：

```powershell
python test_inference.py SORTING "蒙牛纯牛奶" left --output result.json
```

可使用 `--output-dir` 指定结果图片目录：

```powershell
python test_inference.py SORTING "蒙牛纯牛奶" left --output result.json --output-dir D:/locate-results
```

如端口或地址调整，可通过 `SKU_API_URL` 和 `LOCATE_API_URL` 环境变量覆盖；超时时间可通过 `SKU_REQUEST_TIMEOUT_SECONDS` 和 `LOCATE_REQUEST_TIMEOUT_SECONDS` 覆盖。
