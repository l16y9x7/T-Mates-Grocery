# Qwen3-VL 小票商品识别

这是一个最小、可审计的小票商品识别项目。它读取一张购物小票图片或 PDF，在本地完成方向修正、缩放和 PDF 渲染，然后通过 OpenAI-compatible Chat Completions API 请求 `Qwen3-VL-4B-Instruct`。程序不会把本地文件路径交给服务器，也不会自动保存模型输出。

第一版只负责“小票文字里的商品行”，不识别画面里的实物，也不做品牌、品类或 SKU 标准化。

## 数据流

本地命令行开发链路：

```text
本地 JPG/PNG/PDF
  -> 本地转成 JPEG data URL
  -> SSH 隧道
  -> 跳板机转发
  -> 内网 Qwen3-VL 服务
  -> JSON 返回本地
```

部署到目标服务器后的 HTTP API 链路：

```text
调用方上传 JPG/PNG
  -> POST /receipt/parse
  -> 服务内部转成 JPEG data URL
  -> 部署环境配置的 Qwen3-VL 服务
  -> 商品 JSON 返回调用方
```

PDF 只在系统临时目录中渲染，临时页会自动删除。图片内容仍会作为 HTTP 请求到达模型服务器；正式发送公司数据前，应向服务维护者确认日志和数据保留策略。

对外 HTTP API 接收的是普通图片文件；Base64 data URL 只是服务内部调用 Qwen Chat Completions 时使用的图片格式。调用方不需要手动把图片转 Base64。

代码结构见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)，服务器部署见 [DEPLOYMENT.md](DEPLOYMENT.md)。

## 安装

```bash
cd perception/parse_receipt
conda activate receipt-qwen-vl
python -m pip install -e .
```

环境已使用 Python 3.11 创建。重新创建时可直接使用仓库中的配置：

```bash
conda env create --file environment.yml
```

```bash
python -m pip install -e .
```

HTTP API 使用 FastAPI 提供接口，使用 Pillow 处理上传图片。PDF 识别需要 Poppler 的 `pdftoppm`；第一版接口只接收 JPG/PNG 图片，PDF 仍建议走本地命令行验证。

## 连接模型服务

本地开发时先在单独终端保持 SSH 隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 \
  -o ServerAliveCountMax=3 \
  -L 127.0.0.1:<local-port>:<qwen-internal-host>:<qwen-port> <jump-host>
```

再在项目终端设置：

```bash
export QWEN_BASE_URL='http://127.0.0.1:<local-port>/v1'
export QWEN_MODEL='Qwen3-VL-4B-Instruct'
export QWEN_TIMEOUT_SECONDS='120'
```

只有接口明确返回认证错误时才设置：

```bash
export QWEN_API_KEY='...'
```

真实 key 不要写入 `.env.example`、代码或 Git。

## 逐步验证

先检查模型列表：

```bash
qwen-probe models
```

检查服务器实际声明的 OpenAPI：

```bash
qwen-probe openapi
```

验证纯文字 Chat：

```bash
qwen-probe text
```

最后才发送无敏感小票 PDF 或图片：

```bash
qwen-probe receipt "/绝对路径/测试小票.pdf" --diagnostics
```

## 正式识别

支持 `.jpg`、`.jpeg`、`.png` 和 `.pdf`：

```bash
receipt-recognizer "/绝对路径/测试小票.pdf"
```

标准输出只有业务数组：

```json
[
  {
    "name": "乐事薯片番茄味",
    "specification": "50g/袋"
  }
]
```

查看行级原文、待复核项、token 使用量和是否触发 JSON 纠正：

```bash
receipt-recognizer "/绝对路径/测试小票.pdf" --diagnostics
```

诊断 JSON 写到 `stderr`，业务数组仍写到 `stdout`，便于后续程序稳定读取。

PDF 默认以 180 DPI 渲染，并且在多图接口验证前只读取第一页：

```bash
receipt-recognizer receipt.pdf --pdf-dpi 200
```

模型服务是否支持一次请求多张图片，需要通过无敏感样本实测确认。确认后才使用 `--max-pdf-pages 2` 或更高值。

实验 Qwen 采样温度时可临时指定 `--temperature`。正式部署默认保持
`0.0`，用于获得更稳定的输出：

```bash
receipt-recognizer receipt-images/receipt8.jpg --temperature 0.2
```

## HTTP API

服务端入口在 `receipt_recognizer.server:app`，接口路径为：

```text
POST /receipt/parse
```

请求参数：

```text
file          必填，JPG/PNG 图片文件
diagnostics   可选，true 时返回诊断信息
max_edge      可选，发送给 Qwen 前的最长边，默认 2200
```

本地启动示例：

```bash
export QWEN_BASE_URL='http://<qwen-host>:<qwen-port>/v1'
export QWEN_MODEL='Qwen3-VL-4B-Instruct'
export QWEN_TIMEOUT_SECONDS='120'

uvicorn receipt_recognizer.server:app \
  --host 127.0.0.1 \
  --port 18080
```

调用示例：

```bash
curl -F "file=@receipt-images/receipt1.jpg" \
  "http://127.0.0.1:18080/receipt/parse"
```

返回默认是业务数组：

```json
[
  {
    "name": "好丽友土豆薯条番茄味",
    "specification": "70g"
  }
]
```

需要诊断信息时：

```bash
curl -F "file=@receipt-images/receipt1.jpg" \
  "http://127.0.0.1:18080/receipt/parse?diagnostics=true"
```

部署到目标服务器时，建议让服务维护者确认最终公网入口、URL 前缀和
`QWEN_BASE_URL`。如果 Qwen/vLLM 就在同一台机器上，服务环境通常可配置为：

```bash
export QWEN_BASE_URL='http://127.0.0.1:8102/v1'
```

## 输出约束

- 商品名保留完整票面 SKU 名称，口味、香型、型号都并入 `name`，不再单独输出 `flavor`。
- 规格只保留重量、容量、尺寸或包装关系，例如 `70g/袋`、`60g*10`；只要字符清晰就原样复制。没有规格时为 `null`。
- 不合并商品行；小票上识别到几条商品明细，业务数组就输出几条。
- 第一版不输出数量字段；当前实验默认每条商品明细数量为 1。
- 第一版不输出 `source_text`；库存校验只统计 `name` 能否和库表精确对上。
- 称重商品、小数数量或模糊内容进入 `review_items`，不进入业务数组。
- 非法 JSON 只允许追加一次格式纠正请求；第二次仍失败就报错。
- 不使用未经部署验证的 `response_format` 或服务专有参数。

## PaddleOCR 原始文本实验

PaddleOCR 目前只作为本地实验工具，不接入正式 `POST /receipt/parse`
接口，也不影响 Qwen 的默认识别流程。它的作用是先把小票照片读成
原始 OCR 文本，后续再和 mentor 提供的商品库 `sku_name` 做严格匹配。

OCR 依赖不放进默认服务环境，避免增加服务器部署复杂度。本地需要实验时再安装：

```bash
python -m pip install -e ".[ocr]"
```

如果运行时提示缺少 `paddle`，请按 PaddleOCR 官方说明为当前机器安装
对应版本的 `paddlepaddle`。Apple Silicon Mac 本地 CPU 测试通常可以先试：

```bash
python -m pip install paddlepaddle
python - <<'PY'
import paddle
paddle.utils.run_check()
PY
```

识别一张本地 JPG/PNG：

```bash
receipt-ocr receipt-images/receipt8.jpg
```

输出是 OCR 原始文本证据，不是最终商品 JSON：

```json
{
  "image": "receipt8.jpg",
  "ocr_lines": [
    {
      "text": "Lay's乐事薯片墨",
      "score": 0.98
    },
    {
      "text": "西哥鸡汁番茄味",
      "score": 0.97
    }
  ],
  "full_text": "Lay's乐事薯片墨 西哥鸡汁番茄味"
}
```

保存结果到文件：

```bash
receipt-ocr receipt-images/receipt8.jpg \
  --output output/ocr-baseline/receipt8.ocr.json
```

第一阶段人工重点看：

- 换行商品名是否读全。
- 口味、香型、型号关键字是否丢失。
- 中英文混合商品名是否稳定。
- 规格是否被正确读出，或是否混入商品名。

## 库表校验与实验评估

库存库使用 CSV，至少包含一列：

```text
sku_name
```

复跑多张测试小票时，仓库不额外封装批处理入口，直接用 shell 循环调用单张识别即可。`receipt-images/` 默认不提交到 Git，请替换为本地无敏感测试图片目录：

```bash
mkdir -p output/name-spec-rerun

for image in receipt-images/receipt{12,13,14,15,16,17,18,19,20}.jpg; do
  name="$(basename "$image" .jpg)"
  receipt-recognizer "$image" > "output/name-spec-rerun/${name}.items.json"
done
```

这样每张图会保存一个 `*.items.json`，内容就是最终业务输出，只含 `name` 和 `specification`。

如果已经有 `*.items.json`，可以统计商品名命中率。当前库存 CSV 暂时没有规格字段，所以规格不参与评估指标。库存文件默认按本地数据处理，不提交真实商品库到 Git：

```bash
receipt-evaluate output/某次实验目录 \
  --inventory data/inventory.csv
```

当前统计口径：

- `name_inventory_exact_rate`：识别出的 `name` 与库存 `sku_name` 精确匹配的占比。
- 不统计规格命中率：当前规格字段只是模型读出的票面信息，后续等库存或标注中有可靠规格字段后再评估。
- 模糊包含只作为诊断建议，不计入通过。

当前 9 张清晰小票实验结果：

```text
图片数：9
商品行数：18
商品名命中库存：18/18 = 100%
```

这个结果只代表当前受控样本。其中商品名命中率来自库存 `sku_name` 精确匹配；规格字段本轮不统计准确率。`receipt12` 曾在旧拆分字段版本中把“青柠味”误识别为“柠檬味”，因此后续仍建议对相似口味样本重复跑多次，观察稳定性。

## 测试

```bash
conda run --name receipt-qwen-vl python -m unittest discover -v
```

测试覆盖配置、HTTP 协议、认证头、严格 JSON 校验、完整商品名与规格输出、库存精确匹配、派生状态规范化、行级商品输出、待复核排除、单次纠正逻辑、HTTP API，以及 PNG 到 JPEG 的本地预处理。
