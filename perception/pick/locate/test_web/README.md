# Locate Debug / Prompt 管理网页

启动：

```powershell
cd perception/pick/locate/test_web
python -m pip install -r requirements.txt
python server.py
```

浏览器打开：`http://127.0.0.1:8082`。

## 数据来源与推理

网页不再读取 `perception/test_data` 下的本地图片，也不在 test_web 内分别调用 Qwen3 和 SAM3。

选择 SKU，设置 `name` 和 `hand` 后，点击“运行 Locate Debug 完整推理”。test_web 后端会代理调用：

```text
POST http://192.168.130.59:8083/perception/pick/locate/debug
```

请求仅包含：

```json
{
  "name": "SORTING",
  "product_name": "可口可乐",
  "hand": "left"
}
```

Locate 服务自行调用相机快照接口。Debug 响应中的 `image_base64` 作为页面原图：左侧绘制共识后的 Qwen bbox，右侧叠加最终 SAM3 mask、bbox 和 score。

可通过 `LOCATE_DEBUG_URL` 环境变量覆盖 Debug 接口地址。

## Prompt 管理

`qwen_sam_prompt_mapping.json` 是网页和正式 Locate API 的唯一 Prompt 数据源。

- 选择已有 SKU 时，网页同时加载 `qwen3_prompt` 和 `sam3_prompt`。
- “保存当前 Prompt”只更新对应 SKU 的 `qwen3_prompt`，保留已保存的 `sam3_prompt`。
- “保存 SAM3 Prompt 范式”同时保存当前 SKU 的 Qwen3/SAM3 配对 Prompt。
- 修改 Prompt 后，需要先保存，再运行 Locate Debug，服务才会读取新内容。
