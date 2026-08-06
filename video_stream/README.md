# 视频流模块（Agent 调度接口）

本目录向 **Agent 调度模块** 暴露相机 HTTP 约定与 Python 客户端。  
真机服务由 TianJi `retail_camera_http_gateway` 提供（默认 `http://127.0.0.1:8085`）。

## 接口一览

| 方法 | 路径 | 参数 | 返回 |
|------|------|------|------|
| `GET` | `/camera/health` | 无 | `{"status":"READY"}` |
| `GET` | `/camera/list` | 无 | 各相机与 color/depth 在线状态 |
| `GET` | `/camera/snapshot` | `camera` + `type` | JPEG 字节 |
| `GET` | `/camera/stream` | `camera` + `type` | MJPEG 流 |

### 1. `GET /camera/health`

无请求体。成功：

```json
{"status": "READY"}
```

`status` 可能为：`READY` / `STARTING` / `ERROR`。至少一路流有新鲜帧时为 `READY`。

### 2. `GET /camera/list`

成功示例：

```json
{
  "cameras": [
    {
      "id": "head",
      "online": true,
      "streams": [
        {
          "type": "color",
          "topic": "/camera/head/color/image_raw",
          "online": true,
          "width": 640,
          "height": 480,
          "encoding": "rgb8",
          "age_sec": 0.05
        }
      ]
    }
  ]
}
```

### 3. `GET /camera/snapshot`

查询参数：

- `camera`：`head` / `left_wrist` / `right_wrist`（别名 `hand_left`、`hand_right`、`left`、`right`）
- `type`：`color`（默认）或 `depth`

成功返回 `image/jpeg`。

### 4. `GET /camera/stream`

参数同 snapshot。返回 `multipart/x-mixed-replace` MJPEG。  
一路连接只拉一种流；要同时看彩色和深度需开两个连接。

### 错误码（JSON 体 `error_code`）

| HTTP | error_code | 含义 |
|------|------------|------|
| 400 | `INVALID_REQUEST` | `type` 非法等 |
| 404 | `CAMERA_NOT_FOUND` / `NOT_FOUND` | 相机或路径不存在 |
| 503 | `STREAM_NOT_READY` | 该路尚未有帧 |
| 503 | `STREAM_STALE` | 帧过期 |

## 调用示例

```bash
curl -s http://127.0.0.1:8085/camera/health
curl -s http://127.0.0.1:8085/camera/list
curl -o head.jpg 'http://127.0.0.1:8085/camera/snapshot?camera=head&type=color'
ffplay 'http://127.0.0.1:8085/camera/stream?camera=head&type=color'
```

Python（Agent 调度侧）：

```python
from video_stream.client import VideoStreamClient

cam = VideoStreamClient("http://127.0.0.1:8085")
assert cam.health() == "READY"
print(cam.list_cameras())
jpeg = cam.snapshot("head", "color")
open("head.jpg", "wb").write(jpeg)

# 连续取帧（MJPEG）
for frame in cam.iter_stream_jpegs("head", "color", max_frames=3):
    ...
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `client.py` | Agent 用的薄 HTTP 客户端 |
| `test_client.py` | 客户端单测（mock HTTP，无需真机） |

## 依赖与真机启动

客户端仅依赖 Python 标准库。真机需先在 TianJi 侧拉起相机网关，例如：

```bash
cd /path/to/TianJi
./scripts/start_tianji_cameras.sh start
./scripts/start_nav_camera.sh
```

详见 TianJi：`docs/导航与相机最小使用文档.md`。
