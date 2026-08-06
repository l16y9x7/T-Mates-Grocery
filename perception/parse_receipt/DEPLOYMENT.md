# 部署说明

这份文档给服务器部署同事使用。当前服务的作用是：

```text
输入：一张 JPG/PNG 小票图片
输出：SKU 标准商品名和标准货位；任一名称未命中时返回 404
```

它不是 Qwen/vLLM 本身，而是 Qwen/vLLM 外面的一层 HTTP 包装服务。

## 1. 服务接口

健康检查：

```text
GET /health
```

小票识别：

```text
POST /receipt/parse
```

请求使用 `multipart/form-data` 上传图片：

```bash
curl -F "file=@receipt.jpg" \
  "http://<host>:<port>/receipt/parse"
```

默认成功响应：

```json
[
  {
    "name": "NFC桔汁",
    "locations": ["H1_F_L1_C01"]
  }
]
```

需要诊断信息时：

```bash
curl -F "file=@receipt.jpg" \
  "http://<host>:<port>/receipt/parse?diagnostics=true"
```

## 2. 环境变量

必须确认：

```bash
export QWEN_BASE_URL='http://<qwen-host>:<qwen-port>/v1'
export QWEN_MODEL='Qwen3-VL-4B-Instruct'
export QWEN_TIMEOUT_SECONDS='120'
export SKU_BASE_URL='http://127.0.0.1:8080'
export SKU_TIMEOUT_SECONDS='3'
```

如果 Qwen/vLLM 和本服务部署在同一台机器上，`<qwen-host>` 通常可以
配置为 `127.0.0.1`；如果不在同一台机器上，请把 `QWEN_BASE_URL`
改成部署机器可以访问到的 OpenAI-compatible `/v1` 地址。

只有 Qwen 服务实际要求认证时才设置：

```bash
export QWEN_API_KEY='...'
```

不要把真实 key 写入代码、README、`.env.example` 或 Git。

`SKU_BASE_URL` 指向 `perception/sku/api.py` 提供的 SKU 查询服务。
该服务启动时可以监听 `0.0.0.0:8080`，但客户端请求时应配置为
可访问地址，例如同机部署用 `http://127.0.0.1:8080`。

小票服务会对每个识别出的商品名请求：

```text
GET /sku/locations?name=<商品名>
```

商品不存在时 SKU 服务返回 404，小票服务会停止本次处理并向调用方返回：

```json
{"error_code": "SKU_NOT_FOUND"}
```

一张小票包含多个商品时会逐条查询；任一查询返回 404，就不返回部分货位结果。
如果 SKU 服务本身不可达，才返回 `sku_connection_error`。

## 3. 安装

推荐在已有 Python/conda 环境中安装当前仓库：

```bash
git clone <repo-url>
cd T-Mates-Grocery/perception/parse_receipt
python -m pip install -e .
```

如果需要单独创建 conda 环境：

```bash
conda env create --file environment.yml
conda activate receipt-qwen-vl
python -m pip install -e .
```

`python -m pip install -e .` 会安装 FastAPI、Pillow、python-multipart 和 uvicorn。

## 4. 启动

本地只给本机访问：

```bash
uvicorn receipt_recognizer.server:app \
  --host 127.0.0.1 \
  --port 18080
```

服务器部署给其他程序访问时，通常监听：

```bash
uvicorn receipt_recognizer.server:app \
  --host 0.0.0.0 \
  --port <port>
```

最终 `<port>`、公网入口、URL 前缀和进程管理方式需要由服务器维护者确认。

## 5. 错误响应

失败时服务返回统一结构：

```json
{
  "error": {
    "type": "upstream_response_error",
    "message": "模型 API 返回 HTTP 502: 响应体为空",
    "upstream_status_code": 502
  }
}
```
