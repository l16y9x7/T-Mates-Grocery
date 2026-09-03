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

未随请求上传图片时，只从根目录 `config.py` 配置的相机服务获取当前 RGB，不读取本地测试图片。`SORTING` 按 `hand` 使用 `left_wrist`/`right_wrist`；`SHORTAGE` 和 `MISPLACED` 固定使用 `head`。当前整个 Pick/Locate 流程均不请求或使用深度。非本机 IP 统一由 `CAMERA_SERVICE_HOST` 配置；仍可通过 `CAMERA_SERVICE_URL`、`CAMERA_SNAPSHOT_URL` 和 `CAMERA_SNAPSHOT_TIMEOUT_SECONDS` 覆盖地址与超时，通过 `CAMERA_SNAPSHOT_CACHE_DIR` 指定快照缓存目录。

## 接口

### `GET /video/frame`

返回当前 RGB 图片。默认 `task_type=SORTING`，按 `hand` 返回对应腕部相机；传入 `task_type=SHORTAGE` 或 `task_type=MISPLACED` 时返回头部相机。服务会验证响应是有效 JPG/PNG；连接失败、非 2xx、空响应、图片无效或缓存失败时返回 HTTP 400，不读取本地图片。

### `POST /perception/pick/locate`

请求包含商品名称和左右手信息。Task1 的正式 SORTING 抓取还会传递精确货位和实际导航点：

```json
{
  "task_type": "SORTING",
  "product_name": "外星人电解质水白桃口味0糖",
  "hand": "right",
  "level": "L3",
  "slot_id": "H2_L03_C03",
  "target_id": "H12_INSPECT"
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
- 普通 `SORTING` 可传 `slot_id` 启用库存定位。Locate 从 SKU 的 `inventory` 中取得目标所在货架层的剩余位置，按列号从左到右映射检测 bbox；正式响应会原样返回所选 `slot_id`。成功抓取后的库存删除由外层 agent 调用 SKU 服务完成。
- 电解质和非猫薄荷脉动的 hard case 必须同时提供 `slot_id` 和 `target_id`；服务会校验商品、层、货位、手和导航点一致。
- `multi_row_products.json` 配置可能存在前后多排陈列的商品；`*` 表示所有普通 SORTING 商品默认参与多排排号，因此也覆盖后续新增 SKU。hard case 仍使用独立的顺序映射逻辑。
- Locate 本身不保存抓取历史。外部 agent 在确认抓取成功后，将之前正式响应的归一化 bbox 放入 `previous_picked_bboxes`；多排排号会据此避开已经抓取过的位置。

处理流程：

1. 调用 `GET /sku/search_by_name` 查询完整 SKU 信息。
2. 根据 `task_type` 从对应 JSON 读取该商品的 Qwen3 与 SAM3 Prompt：SORTING 使用 `qwen_sam_prompt_mapping.json`，SHORTAGE 使用 `qwen_sam_prompt_mapping_shortage.json`，MISPLACED 使用 `qwen_sam_prompt_mapping_misplaced.json`。
3. 仅将 RGB 拉伸到 `1280×720` 推理画布；原始 RGB 与 depth 均保持原尺寸。
4. Qwen3 在该 `1280×720` RGB 上以 `temperature=0.5` 独立采样三次。
5. 对跨采样 bbox 聚类，只保留至少由两个不同采样支持且匹配 IoU 严格大于 `0.85` 的目标；同一目标的坐标取支持框平均值。
6. 将 Qwen `[0,1000]` bbox 转为 `1280×720` 推理画布像素坐标，向外扩张 10% 后裁图，并在每个去重后的 crop 上调用 SAM3。
7. 将 SAM3 bbox 和 mask 按 X/Y 比例映射回原始 RGB 图片；响应的 `image_size` 是原图尺寸，`inference_image_size` 是 `[1280, 720]`。
8. 多排商品逐列提取 mask 下轮廓，计算下轮廓点到拟合红色货架前沿的有符号垂距，并以垂距中位数归一化后按距离升序分排：相邻候选的归一化距离差大于 `0.05` 时开始下一排。传入 `slot_id` 时，先取最前排去重候选；少于该商品同层 `inventory` 数量才依次从后一排补齐。若最终检测数仍少于库存数，检测框整体位于画面左半边时取库存排序后的后 `detected` 个槽位，位于右半边时取前 `detected` 个槽位，再按 bbox 从左到右映射并选择目标槽位。库存模式不执行最小 mask 离群删除。未传 `slot_id` 时沿用只保留最前排及中心选择的兼容逻辑。当前 Pick/Locate 不请求、解码或使用深度。
9. 对第一排候选构建重叠链，每条链只保留按 mask 面积与密度判断最靠前的一个实例，然后执行最终 PICK。

### SORTING hard case 顺序定位

以下同品牌易混淆商品只在 `SORTING` 下启用顺序定位：非猫薄荷脉动、外星人电解质水、
舒克牙膏柠檬百香果、草原红太阳烧烤料/烧烤酱、镇江香醋/蒸鱼豉油/薄盐生抽。脉动猫薄荷瓶以及其他 SKU、
`SHORTAGE`、`MISPLACED` 保持上述原流程不变。

电解质与非猫薄荷脉动的目标物理槽位及腕部视角由
`perception/hard_case_view_layout.json` 配置。服务按 `slot_id` 确认目标列，再按
`target_id + hand + level + group` 取得图像从左到右的可见槽序；右手在连接点不再被
推断为整排右侧后缀。检测列数与配置不一致、目标槽不在该视角内，或商品/层/槽不一致时
直接拒绝定位。`hard_case_config.json` 继续声明合法商品、层和手，并保留舒克牙膏等旧特例。

SAM3 实例优先按 Qwen 陈列堆来源组成陈列列；Qwen 只返回一个合并区域时，使用过滤后的
第一排 SAM 实例作为可见列。hard case 不使用深度做跨列筛选；系统拟合原图中红色货架
前沿的上边缘，逐列计算 mask 下轮廓到前沿线的有符号垂距做几何初筛；商品底边位于
红线上方的基础容差为自身垂线方向高度的 25%。
基础筛选按 bbox 重叠链估算后最多只剩一列时，才在归一化距离连续的前提下渐进放宽，最大
为 35%；已有两列或更多时不放宽。检测不到红线时才回退到瓶底高度规则，框高
和 mask 面积不作为硬淘汰条件。以上比例可通过
`HARD_CASE_FRONT_UPPER_TOLERANCE_RATIO`、
`HARD_CASE_FRONT_MAX_UPPER_TOLERANCE_RATIO` 和
`HARD_CASE_FRONT_DISTANCE_GAP_RATIO` 调整。hard case 会先筛第一排，再对第一排中的重叠 SAM mask 去重，避免
后排或局部 mask 把相邻商品传递性合并。Debug 响应的 `hard_case` 给出目标 location、
`target_slot_id`、`target_id`、`visible_slot_order` 和陈列组映射；最终 `instances` 带有
`mapped_product_name`、`mapped_slot_id`、`hard_case_group_index`、`is_selected`。
正式接口只返回 `is_selected=true` 的目标实例。

若某个商品、层数和相机方向对应的实体货架存在标准库没有记录的重复列，可在
`perception/hard_case_layout_overrides.json` 中按
`商品名 + level + hand` 配置从相机保证的货架边缘开始的
实际可见顺序：左手使用 `visible_order_from_left`，右手使用
`visible_order_from_right`。重复的实体列必须在数组中重复填写。
只有三个条件全部命中时才使用覆盖顺序，与输入图片文件无关；该旧覆盖机制只用于尚未
迁移到精确 slot/view 配置的 hard case。

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

测试专用接口，输入与正式接口相同，但额外返回 `image_base64`、`image_media_type`、`sku_id`、`image_name`、`image_path`、`image_size`、共识后的 `qwen_bboxes` 和全部 `instances`。多排商品的实例包含 `display_row_index`（`1` 为最前排）、`display_position_in_row`（该排从左到右的序号）、`display_row_source`、`shelf_front_distance_ratio` 和 `history_overlap_count`。其中 `image_base64` 是本次推理实际使用的原图，调用方不需要访问 Locate 服务所在机器的本地文件。

Debug 接口还支持仅供 mock 测试使用的 `mock_inventory` 字符串数组。提供该字段时只支持
`SORTING`，服务从仓库内 `sku/products.json` 读取商品静态信息，用该数组覆盖本次请求的
库存，并完全跳过 25540 的商品及货架行查询；`slot_id` 必须属于该数组。正式
`POST /perception/pick/locate` 的请求模型不包含此字段，生产库存行为不变。

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

为兼容已有调用，Debug API 暂时仍接受 `depth_image_name`、`depth_image_base64` 和
`depth_is_bigendian` 字段，但 Pick/Locate 当前直接忽略这些字段，不解码也不参与候选选择。

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
同名 JSON 保存精简响应；全局进度保存在 `sorting_pick_locate_batch_results.json`。默认并发数为
4，可通过 `--workers` 或环境变量 `LOCATE_BATCH_WORKERS` 调整；汇总文件始终由主线程串行更新。

```powershell
# 只校验 52 个 record、商品名、层级、手腕和输入文件
python pick\locate\test\batch_record_inference.py --dry-run

# 断点续跑：跳过成功项，失败项会重新检测
python pick\locate\test\batch_record_inference.py

# 从头覆盖全部结果
python pick\locate\test\batch_record_inference.py --overwrite

# 使用 2 并发运行（默认是 4）
python pick\locate\test\batch_record_inference.py --workers 2
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
