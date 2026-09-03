# 外部系统模拟台

这个目录是一个独立的、无 npm 依赖的模拟前端，用于按 `agent/doc/外部系统接口设计.md` 调用统一任务服务。

## 后台管理脚本

先确认机器人统一任务服务已经运行在 `8108`，然后在仓库根目录执行：

```bash
external-demo/manage.sh start
```

脚本默认监听 `0.0.0.0:8765`，自动读取本机局域网 IP，并将请求转发到
`http://127.0.0.1:8108`。常用命令：

```bash
external-demo/manage.sh start
external-demo/manage.sh stop
external-demo/manage.sh restart
external-demo/manage.sh status
external-demo/manage.sh logs
```

PID 保存在 `agent/run/external-demo.pid`，日志保存在
`agent/log/process/external-demo-<时间>.log`。

如需明确指定本机局域网地址和 Agent 地址：

```bash
DEMO_PUBLIC_HOST=192.168.200.65 \
ROBOT_TASK_URL=http://127.0.0.1:8108 \
external-demo/manage.sh restart
```

其他局域网电脑访问 `http://192.168.200.65:8765`。

## 前台启动

先确认机器人统一任务服务已经运行在 `8108`，然后在仓库根目录执行：

```bash
python3 external-demo/server.py
```

浏览器打开 <http://127.0.0.1:8765>。

可通过环境变量切换目标：

```bash
ROBOT_TASK_URL=http://192.168.200.66:8108 DEMO_PORT=8765 python3 external-demo/server.py
```

机器人在另一台机器上运行时，将 `DEMO_CALLBACK_URL` 设置为机器人可访问的本机地址，例如 `http://192.168.200.10:8765/api/callback`；同时让网关监听外部网卡：

```bash
DEMO_HOST=0.0.0.0 DEMO_CALLBACK_URL=http://192.168.200.10:8765/api/callback ROBOT_TASK_URL=http://192.168.200.66:8108 python3 external-demo/server.py
```

## 覆盖的流程

- `GET /api/external/v1/health`：展示整体状态、三个任务是否可接收和依赖状态。
- `POST /api/external/v1/tasks/0/runs`：发送理货请求。
- `POST /api/external/v1/task1/orders`：发送两件不同商品的取货订单。
- `POST /api/external/v1/tasks/2/runs`：发送补货请求。
- `GET /api/external/v1/tasks/{task_run_id}/status`：默认实时轮询任务状态。
- `POST /api/callback`：本地状态回调接收端点，保存并显示机器人主动上报事件。

页面会自动添加 `Authorization`（如果浏览器 localStorage 中存在 `external-access-token`）、`Idempotency-Key` 和 `X-Request-Id`，并记录每次 HTTP 调用和返回 JSON。

## 启用回调演示

当前项目配置默认没有回调地址白名单，因此页面默认使用状态查询。若要演示机器人主动回调，在任务服务配置中设置：

```yaml
external:
  callback_url: http://127.0.0.1:8765/api/callback
  callback_allowed_hosts:
    - 127.0.0.1
```

重启 `8108` 服务后，勾选页面的“请求状态回调”即可。若只设置 `callback_url`，不传任务请求中的回调字段，也会使用服务端默认回调地址。

## 说明

`server.py` 只承担开发演示用途：它提供同源静态页面、本地回调接收和到 `8108` 的 HTTP 转发，不修改机器人业务代码，也不替代生产网关。
