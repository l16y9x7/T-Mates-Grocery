# 小票识别服务

这个目录只负责一条运行链路：

```text
POST /perception/parse
  -> GET 机器人头部相机当前帧
  -> Qwen3-VL 输出商品 name + specification
  -> SKU 服务获取全量 name 并按优先级匹配
  -> 返回包含两个标准商品名的对象
```

调用方不上传图片。调用发生时相机看到的画面，就是本次识别的输入。

## 文件

```text
server.py       主服务，包含 Prompt 和全部运行逻辑
test_server.py  不访问真实服务的单元测试
README.md       配置、启动与调用说明
```

## 依赖

```bash
python -m pip install fastapi pillow uvicorn
```

正式接口不接收 multipart 文件，因此不需要 `python-multipart`。PaddleOCR 不在正式链路中。

## 配置

默认头部相机地址为 `http://192.168.1.226:8085/camera/snapshot?camera=head&type=color`。以下环境变量均可覆盖默认配置：

```bash
export RECEIPT_CAMERA_URL='http://<camera-host>:<port>/camera/snapshot?camera=head&type=color'
export CAMERA_TIMEOUT_SECONDS='5'
export QWEN_BASE_URL='http://<qwen-host>:<port>/v1'
export QWEN_MODEL='Qwen3-VL-4B-Instruct'
export QWEN_TIMEOUT_SECONDS='120'
export SKU_BASE_URL='http://127.0.0.1:25540'
export SKU_TIMEOUT_SECONDS='3'
```

## 启动

在本目录运行：

```bash
cd ..
python main.py
```

`/perception/parse` 已注册到 Locate 的 FastAPI app，由同一个进程监听
8083；不要再单独启动一个 8083 端口的 `server:app`。

终端前台运行时，关闭终端或按 `Ctrl+C` 会停止服务。正式部署应由同事配置 systemd、Supervisor 或容器负责常驻和重启。

## 调用

服务器本机：

```bash
curl -X POST http://127.0.0.1:8083/visual/parse
```

Mac 已将服务器 `8083` 转发到本地 `28083` 时：

```bash
curl -X POST http://127.0.0.1:28083/perception/parse
```

请求体为空，不需要提供图片路径。机器人必须先到达小票拍摄位姿。

Qwen 内部输出只允许：

```json
[
  {
    "name": "康师傅香辣牛肉面",
    "specification": "500g"
  }
]
```

正式成功响应是一个对象，只包含两个 SKU 标准商品名：

```json
{
  "product_names": [
    "康师傅香辣牛肉面",
    "NFC桔汁"
  ]
}
```

Qwen 没有识别出两个商品时，接口返回识别错误，不返回残缺结果。

失败响应包含可用于定位上下游故障的结构化信息，并通过 `X-Request-ID` 响应头返回
同一个请求编号。例如相机服务不可达时：

```json
{
  "error": {
    "type": "camera_connection_error",
    "message": "无法连接相机接口：connection refused",
    "stage": "camera_capture",
    "retryable": true,
    "hint": "检查相机服务是否启动，并确认相机主机、端口和网络可达。",
    "request_id": "3fde0122bcb34cb294b65d726816f971",
    "upstream": "http://192.168.1.226:8085/camera/snapshot",
    "elapsed_ms": 3012.4,
    "timeout_seconds": 5.0
  }
}
```

服务端终端会打印对应的单行错误日志，包含请求来源、HTTP 状态、失败阶段、上游状态、
耗时及提示。客户端也可以传入合法的 `X-Request-ID`，便于跨服务追踪。上游 URL 的
认证信息和查询参数不会写入响应或日志。

SKU 匹配顺序：

1. `name` 直接命中完整 SKU name。
2. 如果 specification 不是 `500ml`、`55g`、`2盒` 这类纯数字加单位，则尝试 `name + specification` 精确匹配。
3. 对同样处理后的 `name + specification` 与全量 SKU name 计算编辑距离，选择距离最短者。

全量 SKU name 为空或 SKU 服务不可用时返回对应上游错误。

## 单帧处理

每次请求只 GET 一张头部相机快照，Qwen 请求中只包含一个 `image_url`。接口没有请求体。

## 测试

```bash
python -m unittest -v test_server.py
```

测试使用内存图片和模拟响应，不访问真实相机、Qwen 或 SKU，也不会保存图片文件。
