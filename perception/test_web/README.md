# Locate Debug / Prompt 管理网页

启动：

```powershell
cd perception/test_web
python -m pip install -r requirements.txt
python server.py
```

浏览器打开：`http://127.0.0.1:8082`。

首页顶部会读取 `test_data/2026-08-13/sorting_pick_locate_batch_results.json`，按 record 和
商品浏览批量 Sorting Pick/Locate 结果。页面并排展示原始 `rgb.jpg`、由
`depth_mm.npy` 动态生成的彩色深度预览，以及 `{product_name}.jpg` 检测结果；深度预览中
近处为暖色、远处为冷色、无效深度为黑色。

“重跑当前项（--overwrite）”只会覆盖当前下拉框选中的 `record + product_name`，由 8082
后端在后台执行带 `--overwrite --record ... --product-name ...` 的批测命令。页面每 2 秒读取
一次状态，运行期间会禁用按钮以避免重复任务，完成后自动重新载入图片和总汇总；其他检测项
不会重跑。初次打开首页只展示现有结果，不会自动启动推理。

可用页面：

- `/`：Locate Debug 与 Prompt 管理；
- `/qwen-debug`：粘贴单张图片测试 Qwen3 / SAM3；
- `/qwen-review`：浏览 inspection 的 shortage / misplaced pair，按 bbox 独立测试 Qwen；
- `/qwen-infer`：旧地址兼容入口，会打开同一个 Qwen Review 页面。

## Inspection Qwen Review

页面顶部的“初始扫描货架行检查”读取 `../agent/output/task0` 中全部 Upper/Lower
扫描图，调用共享 `row_detection`，并排显示原图和带 `ROW/RAIL` 的标注图。检测产物
缓存在 `test_data/initial_scan_row_detection`；源 `rgb.jpg` 更新或视角不一致时会自动
重新生成。

“自采缺货批测结果”读取
`test_data/2026-08-16-self-collect-shortage-grouped/shortage_inspection_batch_results.json`，
按巡检分组和 record 展示当前 RGB、对齐后的 bbox/mask overlay、组合 mask，以及每个
region 的独立 mask、bbox 和识别商品名。批测命令：

```powershell
cd perception
python inspect/batch_shortage.py
```

默认同时处理 4 个 record；可用 `--workers 1` 改为串行，或用
`--workers N` 调整并发数。每个 worker 使用独立的 SKU/Qwen HTTP 会话。

Qwen 服务暂不可用时，可以先运行 `python inspect/batch_shortage.py --detection-only`
生成 bbox/mask；服务恢复后直接运行不带该参数的命令，会自动补跑尚未识别的记录。
使用 `--group H1_B_L_INSPECT_LOWER`、`--record record_...` 或 `--limit 10` 可缩小范围，
`--overwrite` 会重新生成已完成结果。

`/qwen-review` 读取 `test_data/inspect_*_paired/qwen_prompt_samples`，页面上方显示
解析结果、原始输出及可编辑 Prompt，下方按 `[IMAGE N]` 顺序展示 bbox 扩展图和该
region 实际发送的候选 SKU 标准图。页面同时展示 baseline、原始 current、算法对齐后的
current，以及货架行/横条/region 归属标注图；这些整图只用于核对，不会发送给 Qwen。

点击“运行当前区域”后，结果卡会同时显示本页审核端到端、后端至解析完成和
Qwen 请求三个耗时。其中本页审核端到端从点击运行计时到结果返回，包含
本地候选图组装、网络、Qwen、结果解析和保存。该页面复用预生成 bbox 和候选图，
因此此耗时不包含前置差分、货架行检测和实时 SKU 查询，页面上会明确标注这一口径。

若要测量“缺货检测 → 商品身份”的真正完整链路，点击“运行完整巡检链路”。
test_web 会从当前 pair 的 baseline/current 重新开始，直接调用
`inspect/main.py` 的 Python 入口，实际执行差分配准、bbox、货架行映射、SKU 候选和 Qwen，
并显示完整链路端到端、后端至结果就绪、inspect 主链路及返回异常数。
该入口使用运行时公共 Prompt，与当前 region 的可编辑 Prompt 测试互不影响。

页面产物与 `/perception/inspect` 使用同一条 Python 调用链：comparison 输出 bbox，
`row_detection.detect_rows()` 完成可靠行匹配；SHORTAGE 从缺货前 `_1` reference 裁图，
MISPLACED 第一阶段使用 `aligned_current`，然后按阶段执行候选过滤。
代码或样例变化后重新生成：

```powershell
cd perception
python inspect/comparison_based/qwen_review/generate_sample_artifacts.py `
  --sku-base-url http://127.0.0.1:25540
```

生成器会调用 `/sku/get_candidate_SKU` 取得按行分组的候选，并通过
`/sku/get_image` 下载标准图；因此运行前需先启动 SKU 服务。默认地址就是
`http://127.0.0.1:25540`。下视角若在画面中检测到 4 行，会把最下面 3 行
映射到 SKU 第 1/2/3 行，SHORTAGE Prompt 只保留 bbox 对应 SKU 行的候选。

MISPLACED 在网页中提供“审核阶段”切换：

- “放错商品”显示当前异常物体图：纵向扩展到目标层上下边界，横向在异常
  bbox 左右各扩半个 bbox 宽度；候选仍使用全部可见行，不按目标行限制。
- “应放商品”只显示 baseline 标准放置图中红框标注的完整一层；纵向按
  目标层高度裁剪，横向保留整行。红框保持异常 bbox 的横向范围，并纵向
  扩展到该层上下边界；候选只包含目标 SKU 行。

每个 bbox、每个审核阶段独立请求 Qwen。点击“保存此样例 Prompt”会写入该阶段目录的
`prompt_override.txt`，不修改公共 Prompt；最近一次推理保存在同一阶段目录的
`qwen_infer_result.json`，两个阶段互不覆盖。

若重新生成后某个旧 override 的 `[IMAGE N]` 数量、候选商品名称或
`CANDIDATE N` 顺序已不符合当前阶段的实际候选输入，文件会保留，但页面会自动使用
最新生成 Prompt 并给出提示，避免两个阶段之间发生图文或商品名称错配。

## 数据来源与推理

Locate Debug 首页不读取 `perception/test_data` 下的本地图片，也不在 test_web 内分别调用 Qwen3 和 SAM3。

选择 SKU，设置 `task_type`、`level` 和 `hand` 后，点击“运行 Locate Debug 完整推理”。test_web 后端会代理调用：

```text
POST http://192.168.130.59:8083/perception/pick/locate/debug
```

请求仅包含：

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐",
  "level": "L1",
  "hand": "left"
}
```

Locate 服务自行调用相机快照接口。Debug 响应中的 `image_base64` 作为页面原图：左侧绘制共识后的 Qwen bbox，右侧叠加最终 SAM3 mask、bbox 和 score。

首页“输入图片”是可选项：选择本地 JPG/PNG 后，网页会在同一个请求中发送
`image_name + image_base64`，完整 Locate 流程使用该原始离线图片；点击“清除图片，
使用相机”后恢复腕部相机。未选择文件时请求字段保持原样，不发送任何本地图片。

离线普通 case 还可以同时上传深度数据。支持与 RGB 同尺寸的二维数值型 `.npy` 数组、
16 位单通道 PNG/TIFF，或者无文件头的 16UC1 `.raw`/`.bin`；RAW 默认按 little-endian 解析，也可在网页切换为
big-endian。只上传 RGB 时保留无深度回退；上传深度数据但没有对应 RGB、尺寸不一致或
深度格式或尺寸不正确时，接口返回 HTTP 400。NPY 自带数据类型和字节序，不使用网页的 RAW 字节序选项。

SORTING 是否进入 hard case，按 `perception/hard_case_config.json` 中的
`商品名 + level + hand` 精确组合判断；未命中的组合按普通 case 运行。
右侧最终 SAM3 画布只显示红色货架前沿对应的第一排实例，标签中的
`G序号 + 商品名` 是按标准 location 顺序得到的映射，绿色“目标”框是正式接口最终
返回的实例。左侧 Debug 元数据中的 `hard_case` 可核对请求层对应的 location、排序方向、
标准顺序与每个可见陈列组 bbox。网页不要求显示完整品牌组：左手从标准顺序左端开始
对应，右手从标准顺序右端开始对应；目标不在当前可见列时页面会显示 Locate Debug
错误。

`/qwen-debug` 仍保留独立的 Qwen bbox 和单 crop SAM3 测试，并在页面顶部新增“完整
Locate Debug（普通 case + hard case）”。该区域使用载入图片的原始分辨率，选择标准
商品名、level 和 hand 后调用 `/api/locate-debug`，不使用页面里手工编辑的独立 Prompt，因而
执行的是当前代码保存的 Prompt、Qwen 三次共识、SAM 后处理和 hard-case location 顺序。
首页和 `/qwen-debug` 都会额外显示正式 Locate 最终实例的原图 bbox 与带边距 crop。
hard case 使用 `is_selected`，普通 case 与正式接口一致选择最靠近图像中心的实例。

可通过 `LOCATE_DEBUG_URL` 环境变量覆盖 Debug 接口地址。

## Prompt 管理

网页和 Locate API 会根据 `task_type` 读写不同 Prompt 文件：

- `SORTING`：`qwen_sam_prompt_mapping.json`（现有 Prompt）
- `SHORTAGE`：`qwen_sam_prompt_mapping_shortage.json`
- `MISPLACED`：`qwen_sam_prompt_mapping_misplaced.json`

切换网页顶部的 `task_type` 时，会自动加载对应文件中的 Prompt。

- 选择已有 SKU 时，网页同时加载 `qwen3_prompt` 和 `sam3_prompt`。
- “保存当前 Prompt”只更新对应 SKU 的 `qwen3_prompt`，保留已保存的 `sam3_prompt`。
- “保存 SAM3 Prompt 范式”同时保存当前 SKU 的 Qwen3/SAM3 配对 Prompt。
- 修改 Prompt 后，需要先保存，再运行 Locate Debug，服务才会读取新内容。
