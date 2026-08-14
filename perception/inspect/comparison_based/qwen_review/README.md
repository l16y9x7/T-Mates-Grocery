# Qwen 商品审核

该目录负责在 `comparison_based` 定位异常 bbox 后识别商品语义：

1. 使用 `location_id + pose_type` 请求 `GET /sku/get_candidate_SKU`；
2. 对每个候选商品请求 `GET /sku/get_image`；
3. 下载候选标准图，最长边超过 1024px 时等比缩小；
4. `SHORTAGE` 在缺货前 baseline/reference 图中扩展 bbox，并只带目标行候选，标准图仍逐张发送；
5. `MISPLACED` 把标准图按候选顺序生成带数字顶栏的拼图，每张最多 5 列 × 4 行；
6. `MISPLACED` 对每个 bbox 发起两个独立请求：当前局部图识别放错商品；第二阶段把
   row detection 裁出的当前目标行与 baseline 正确行上下拼接，再结合目标行候选判断
   当前行缺失、被替代的商品；两个字段分别按各自候选范围校验后再合并。

两种任务使用独立系统 Prompt：

- `shortage_prompt.txt`：从局部图和候选标准图中识别该位置商品；
- `misplaced_prompt.txt`：从当前局部图和全部可见候选识别实际放错商品；
- `misplaced_expected_prompt.txt`：从红框标注的当前/基准整行对比图和按标准货位
  从左到右排列的目标行候选，识别缺失、被替代商品。

不会向 Qwen 发送整张前后图。`SHORTAGE` 使用缺货前 reference（每组 `_1` 图）裁图，
局部图横向扩展 30%，纵向上下各扩展 1.5 个 bbox 高度（每侧最多 100px）；
`MISPLACED` 第一阶段横向扩展一个 bbox 宽度、
纵向上下各扩展 0.5 个 bbox 高度（每侧最多 80px）；第二阶段分别从对齐后的 current
和 baseline 裁出完整目标货架行、在同一 bbox 位置画红框，再上下拼成一张对比图。
没有 bbox 时不会请求 SKU 或 Qwen 服务。

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
│   ├── misplaced_product/
│   │   ├── input.jpg
│   │   ├── prompt.txt
│   │   ├── qwen_raw.txt
│   │   └── parsed_result.json
│   └── expected_product/
│       ├── input.jpg
│       ├── prompt.txt
│       ├── qwen_raw.txt
│       └── parsed_result.json
└── region_02/...
```

`SHORTAGE` 仍使用 region 根目录中的原有单阶段文件。所有 `prompt.txt` 都按实际消息
顺序保存 system/user 文本，并用 `[IMAGE N]` 表示图片位置，不会写入冗长的 base64；
各阶段的 `input.jpg` 与实际发送给 Qwen 的第一张图一致。调试目录已加入 `.gitignore`。

## 生成现有 pair 的 Prompt 样例

下面的命令会运行现有缺货和放错 pair，为每个 bbox 保存阶段输入图和可读 Prompt，
但不会请求 Qwen：

```powershell
python inspect/comparison_based/qwen_review/generate_sample_artifacts.py
```

结果保存在各测试集的 `qwen_prompt_samples/pair_N/` 下。`manifest.json` 记录 bbox，
并将每个阶段 Prompt 内的 `[IMAGE N]` 映射到阶段主图、候选商品标准图或编号拼图。
MISPLACED 的 `candidate_images` 保存逻辑候选顺序，`candidate_sheets` 保存实际发送的
拼图；SHORTAGE 和旧样例没有拼图时继续逐张发送。候选信息和图片从配置的 SKU 接口读取。

测试：

```powershell
python -m unittest discover -s inspect/comparison_based/qwen_review/tests -v
```
