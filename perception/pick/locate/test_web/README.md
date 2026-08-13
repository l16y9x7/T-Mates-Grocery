# Locate Debug / Prompt 管理网页

启动：

```powershell
cd perception/pick/locate/test_web
python -m pip install -r requirements.txt
python server.py
```

浏览器打开：`http://127.0.0.1:8082`。

可用页面：

- `/`：Locate Debug 与 Prompt 管理；
- `/qwen-debug`：粘贴单张图片测试 Qwen3 / SAM3；
- `/qwen-infer`：浏览 inspection 的 shortage / misplaced pair，按 bbox 独立测试 Qwen。

## Inspection Qwen 样例推理

`/qwen-infer` 读取 `test_data/inspect_*_paired/qwen_prompt_samples`，页面上方显示
解析结果、原始输出及可编辑 Prompt，下方按 `[IMAGE N]` 顺序展示 bbox 扩展图和候选
SKU 标准图。原始 baseline/current 只用于核对定位，不会发送给 Qwen。

每个 bbox 独立请求 Qwen。点击“保存此样例 Prompt”会写入该 region 的
`prompt_override.txt`，不修改公共 `shortage_prompt.txt` 或 `misplaced_prompt.txt`；
最近一次推理保存在 `qwen_infer_result.json`，刷新后仍可查看。

## 数据来源与推理

Locate Debug 首页不读取 `perception/test_data` 下的本地图片，也不在 test_web 内分别调用 Qwen3 和 SAM3。

选择 SKU，设置 `task_type` 和 `hand` 后，点击“运行 Locate Debug 完整推理”。test_web 后端会代理调用：

```text
POST http://127.0.0.1:8083/perception/pick/locate/debug
```

请求仅包含：

```json
{
  "task_type": "SORTING",
  "product_name": "可口可乐",
  "hand": "left"
}
```

Locate 服务自行调用相机快照接口。Debug 响应中的 `image_base64` 作为页面原图：左侧绘制共识后的 Qwen bbox，右侧叠加最终 SAM3 mask、bbox 和 score。

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
