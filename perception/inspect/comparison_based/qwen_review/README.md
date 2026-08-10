# Qwen 商品审核

该目录负责在 `comparison_based` 定位异常 bbox 后识别商品语义：

1. 使用 `location_id + pose_type` 请求 `GET /sku/get_candidate_SKU`；
2. 对每个候选商品请求 `GET /sku/get_image`；
3. 下载候选标准图，最长边超过 1024px 时等比缩小后再发送；
4. 在当前图中分别扩展每个 bbox；
5. 每个扩展区域单独请求一次 Qwen3-VL，并校验商品名必须属于候选集合。

两种任务使用独立系统 Prompt：

- `shortage_prompt.txt`：从局部图和候选标准图中识别该位置商品；
- `misplaced_prompt.txt`：尝试识别当前实际商品，并用左右邻居辅助判断标准商品。

不会向 Qwen 发送完整前后图，也不会在 Prompt 中说明 bbox 的生成方式。`SHORTAGE`
局部图横向扩展 30%，纵向上下各扩展 1.5 个 bbox 高度（每侧最多 100px）；
`MISPLACED` 横向扩展一个 bbox 宽度以包含左右邻居，纵向上下各扩展 0.5 个 bbox
高度（每侧最多 80px）。没有 bbox 时不会请求 SKU 或 Qwen 服务。

## 中间结果

主接口默认将每次审核保存到本目录的 `debug/`，也可以通过环境变量
`INSPECT_QWEN_DEBUG_DIR` 修改根目录。每次请求使用时间戳、点位和随机后缀创建独立
目录：

```text
debug/<timestamp>_<location>_<task>_<id>/
├── request.json
├── candidates.json
├── result.json
├── region_01/
│   ├── bbox_expanded.jpg
│   ├── prompt.txt
│   ├── qwen_raw.txt
│   └── parsed_result.json
└── region_02/...
```

`prompt.txt` 按实际消息顺序保存 system/user 文本，并用 `[IMAGE N]` 表示图片位置，
不会写入冗长的 base64。`bbox_expanded.jpg` 与该 REGION 实际发送给 Qwen 的局部图完全
一致。调试目录已加入 `.gitignore`。

## 生成现有 pair 的 Prompt 样例

下面的命令会运行现有缺货和放错 pair，为每个 bbox 保存一张扩展图和一份可读
Prompt，但不会请求 Qwen：

```powershell
python inspect/comparison_based/qwen_review/generate_sample_artifacts.py
```

结果保存在各测试集的 `qwen_prompt_samples/pair_N/` 下。`manifest.json` 记录 bbox，
并将 Prompt 内每个 `[IMAGE N]` 映射到扩展图或候选商品标准图。候选信息和图片直接
读取本地 SKU 目录，其内容与候选 SKU 接口使用的数据一致。

测试：

```powershell
python -m unittest discover -s inspect/comparison_based/qwen_review/tests -v
```
