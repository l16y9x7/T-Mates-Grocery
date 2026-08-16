# MQTT\-python上报例程说明 V1\.1

- 样例代码

\[mqtt\_py\_reporter\.tar\.gz\]

- 测试密钥

\[dev\_info\_MQTT测试\.zip\]

# 一、文件包结构

```Bash
├── deploy.sh 部署配置服务自启动的脚本
├── dev_info_AABBCCDDEEFF.txt 连接平台的设备密钥参数（根据参赛的机器人，由我们人工生成，发送到参赛方。后缀为MAC地址，方便知道是哪一个设备的）
├── log 
│   └── reporter.log 脚本运行log
├── README.md 使用说明
├── robot_reporter.service 服务文件
└── src
    ├── iot_mqtt_sdk.py 连接平台的SDK
    └── reporter.py 上报脚本

```

- **后续工程化考虑添加虚拟环境和依赖安装，如果离线环境添加离线安装包**

```Shell
# 使用虚拟环境
python -m venv reporter_env
source reporter_env/bin/activate

# 在线安装 添加 requirements.txt
pip install -r requirements.txt

# 离线安装 下载所有依赖到本地目录（不安装）
pip download -r requirements.txt -d ./offline_packages/
# 从本地目录离线安装（--no-index 禁止联网，-f 指定本地包目录）
pip install --no-index --find-links=./offline_packages/ -r requirements.txt
```

- **第一步，在机器人中解压安装包**

进入用户目录`cd ~`，创建目录并解压安装包：

```Shell
mkdir -p mqtt_py_reporter && tar -xzvf mqtt_py_reporter.tar.gz -C mqtt_py_reporter
```

- **第二步，部署配置服务自启动**

进入解压后的目录 `cd mqtt_py_reporter`，为部署脚本添加执行权限并运行：

```Shell
sudo chmod +x ./deploy.sh
sudo ./deploy.sh
```

服务正常启动后，可通过以下命令查看日志：

```Shell
tail -f ./log/reporter.log
```

# 二、deploy\.sh 部署配置服务自启动脚本
（不需要厂家提供，举办方提供）

```Bash
#!/bin/bash

echo ">>> 开始部署 robot_reporter 服务..."

SERVICE_FILE="/home/master/mqtt_py_reporter/robot_reporter.service"

cp "${SERVICE_FILE}" /etc/systemd/system/

# 重新加载 systemd 配置，启用并启动服务
systemctl daemon-reload
systemctl enable robot_reporter.service
systemctl restart robot_reporter.service

echo ">>> 部署完成！"
systemctl status robot_reporter.service --no-pager
```

# 三、dev\_info\_赛队\_AABBCCDDEEFF\.txt 连接平台的设备密钥参数
（不需要厂家提供，举办方提供）

1. device\_name 是根据赛队“赛队名称\-MAC后四位”（命名为组委会命名）

2. 文件名后缀“赛队\_AABBCCDDEEFF”是报名的MAC

3. Url 为连接的平台地址

```Bash
product_key,xxx
device_key,xxx
device_secret,xxx
device_name,赛队-EEFF
MAC,AABBCCDDEEFF
url,dmp-mqtt.cuiot.cn:1883
```

# 四、robot\_reporter\.service 服务文件

```JavaScript
[Unit]
Description=Robot Reporter Agent
Wants=network-online.target
After=booster-daemon.service network-online.target
Requires=booster-daemon.service

[Service]
Type=simple
User=master
Group=master
WorkingDirectory=/home/master/mqtt_py_reporter
Environment="HOME=/home/master"
Environment="ROS_LOG_DIR=/home/master/.ros/log"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStartPre=/bin/sleep 6
ExecStart=/bin/bash -c "python3 /home/master/mqtt_py_reporter/src/reporter.py >> /home/master/mqtt_py_reporter/log/reporter.log 2>&1"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

# 五、iot\_mqtt\_sdk\.py 连接平台的SDK

```Python
import paho.mqtt.client as mqtt
import json
import time
import threading
import hmac
import hashlib
from typing import Dict, Optional, Union, Tuple

import logging
logger = logging.getLogger(__name__)

class IoTMTTTSDK:
    KEEP_ALIVE = 60
    RECONNECT_INTERVAL = 5

    _TOPIC_PROPERTY_SINGLE = "$sys/{product_key}/{device_key}/property/pub"
    _TOPIC_PROPERTY_BATCH = "$sys/{product_key}/{device_key}/property/batch"
    _TOPIC_EVENT_SINGLE = "$sys/{product_key}/{device_key}/event/pub"
    _TOPIC_EVENT_BATCH = "$sys/{product_key}/{device_key}/event/batch"
    _TOPIC_SERVICE_REPLY = "$sys/{product_key}/{device_key}/service/pub_reply"
    _TOPIC_SERVICE_PUB = "$sys/{product_key}/{device_key}/service/pub"

    def __init__(
        self,
        product_key: str,
        device_key: str,
        device_secret: str,
        mqtt_ip: str = "dmp-mqtt.cuiot.cn",
        mqtt_port: int = 1883,
        service_command_callback: Optional[callable] = None,
    ):
        self.product_key = product_key
        self.device_key = device_key
        self.device_secret = device_secret
        self.mqtt_ip = mqtt_ip
        self.mqtt_port = mqtt_port
        self._handle_service_command = service_command_callback

        # MQTT客户端对象
        self._mqtt_client: Optional[mqtt.Client] = None
        # 存储接收的下行消息
        self.received_messages = {}
        self._reconnect_timer: Optional[threading.Timer] = None
        self._should_reconnect = True

    def _generate_mqtt_auth_info(self) -> Tuple[str, str, str]:
        """生成MQTT连接所需的clientId、username、password"""
        device_id = self.device_key
        # 构建消息内容
        message = device_id + self.device_key + self.product_key
        # 构建用户名
        username = f"{self.device_key}|{self.product_key}"
        # 构建客户端ID
        client_id = f"{device_id}|{self.product_key}|0|0|0"
        # 生成HMAC-SHA256密码
        sha256_hmac = hmac.new(
            self.device_secret.encode("utf-8"), message.encode("utf-8"), hashlib.sha256
        )
        password = sha256_hmac.digest().hex()
        return client_id, username, password

    def connect(self, max_retry: int = 10) -> bool:
        """
        连接MQTT服务器
        :param max_retry: 最大重试次数
        :return: 连接成功返回True，失败返回False
        """
        if self._mqtt_client and self._mqtt_client.is_connected():
            logger.info("MQTT客户端已连接，无需重复连接")
            return True

        try:
            # 生成认证信息
            client_id, username, password = self._generate_mqtt_auth_info()
            logger.info(f"生成MQTT认证信息 - ClientID: {client_id}, Username: {username}, Password (HMAC): {password} (32 bytes)")
            
            # 创建MQTT客户端
            self._mqtt_client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id=client_id,
                protocol=mqtt.MQTTv31,
            )
            self._mqtt_client.username_pw_set(username=username, password=password)
            # 注册消息回调
            self._mqtt_client.on_message = self._on_message_callback
            self._mqtt_client.on_disconnect = self._on_disconnect_callback

            # 连接服务器
            logger.info(f"正在连接MQTT服务器 {self.mqtt_ip}:{self.mqtt_port}...")
            self._mqtt_client.connect(
                host=self.mqtt_ip, port=self.mqtt_port, keepalive=self.KEEP_ALIVE
            )
            # 启动网络循环
            self._mqtt_client.loop_start()

            # 等待连接成功
            retry_count = 0
            while not self._mqtt_client.is_connected():
                if retry_count >= max_retry:
                    logger.error(f"MQTT连接重试{max_retry}次后失败")
                    return False
                logger.info(f"等待MQTT连接...（{retry_count+1}/{max_retry}）")
                time.sleep(1)
                retry_count += 1

            # 订阅下行消息Topic
            self._subscribe_downstream_topics()
            logger.info("MQTT连接成功，已订阅所有下行Topic")
            return True

        except Exception as e:
            logger.error(f"MQTT连接失败：{str(e)}")
            return False

    def _subscribe_downstream_topics(self) -> None:
        """订阅下行消息Topic（平台下发的回复/指令）"""
        if not self._mqtt_client:
            raise RuntimeError("MQTT客户端未初始化")

        # 下行Topic列表
        downstream_topics = [
            f"$sys/{self.product_key}/{self.device_key}/property/pub_reply",
            f"$sys/{self.product_key}/{self.device_key}/property/batch_reply",
            f"$sys/{self.product_key}/{self.device_key}/event/pub_reply",
            f"$sys/{self.product_key}/{self.device_key}/event/batch_reply",
            f"$sys/{self.product_key}/{self.device_key}/service/pub",
        ]

        for topic in downstream_topics:
            self._mqtt_client.subscribe(topic, qos=0)
            logger.info(f"已订阅下行Topic：{topic}")

    def _on_disconnect_callback(self, client, userdata, disconnect_flags, reasonCode, properties):
        if reasonCode != 0:
            logger.warning(f"[MQTT] 连接断开，reasonCode={reasonCode}，准备重连...")
            self._schedule_reconnect()
        else:
            logger.info("[MQTT] 正常断开连接")

    def _schedule_reconnect(self):
        """调度重连"""
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
        if self._should_reconnect:
            logger.info(f"[MQTT] {self.RECONNECT_INTERVAL}秒后尝试重连...")
            self._reconnect_timer = threading.Timer(self.RECONNECT_INTERVAL, self._do_reconnect)
            self._reconnect_timer.start()

    def _do_reconnect(self):
        """执行重连"""
        if not self._should_reconnect:
            return
        try:
            if self._mqtt_client:
                logger.info("[MQTT] 尝试重连...")
                self._mqtt_client.reconnect()
                retry_count = 0
                while not self._mqtt_client.is_connected():
                    if retry_count >= 10:
                        logger.error("[MQTT] 重连失败")
                        self._schedule_reconnect()
                        return
                    time.sleep(1)
                    retry_count += 1
                self._subscribe_downstream_topics()
                logger.info("[MQTT] 重连成功")
        except Exception as e:
            logger.error(f"[MQTT] 重连异常：{str(e)}")
            self._schedule_reconnect()

    def _on_message_callback(self, client, userdata, msg):
        """处理接收的MQTT消息"""
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
            self.received_messages[msg.topic] = payload
            logger.info(f"收到下行消息 [Topic: {msg.topic}]：{payload}")

            if msg.topic.endswith("/service/pub"):
                # 处理平台下发的服务调用指令
                self._handle_service_command(payload)

        except Exception as e:
            logger.error(f"解析下行消息失败：{str(e)}，原始消息：{msg.payload}")

    def _publish_message(
        self, topic: str, payload: Dict, qos: int = 1, max_retry: int = 5
    ) -> bool:
        """
        通用消息发布方法
        :param topic: 发布的Topic
        :param payload: 消息体（字典）
        :param qos: QoS等级
        :param max_retry: 最大重试次数
        :return: 发布成功返回True，失败返回False
        """
        if not self._mqtt_client or not self._mqtt_client.is_connected():
            logger.warning("MQTT未连接，请先调用connect()方法")
            return False

        try:
            payload_str = json.dumps(payload)
            result = self._mqtt_client.publish(topic, payload_str, qos=qos)

            # 等待发布成功
            retry_count = 0
            while not result.is_published():
                if retry_count >= max_retry:
                    logger.error(f"消息发布重试{max_retry}次后失败")
                    return False
                time.sleep(1)
                retry_count += 1

            logger.info(f"消息发布成功 [Topic: {topic}]")
            return True

        except Exception as e:
            logger.error(f"消息发布失败：{str(e)}")
            return False

    def report_property_single(self, payload: Dict) -> bool:
        topic = self._TOPIC_PROPERTY_SINGLE.format(
            product_key=self.product_key, device_key=self.device_key
        )
        return self._publish_message(topic, payload)

    def report_property_batch(self, payload: Dict) -> bool:
        topic = self._TOPIC_PROPERTY_BATCH.format(
            product_key=self.product_key, device_key=self.device_key
        )
        return self._publish_message(topic, payload)

    def report_event_single(self, payload: Dict) -> bool:
        topic = self._TOPIC_EVENT_SINGLE.format(
            product_key=self.product_key, device_key=self.device_key
        )
        return self._publish_message(topic, payload)

    def report_event_batch(self, payload: Dict) -> bool:
        topic = self._TOPIC_EVENT_BATCH.format(
            product_key=self.product_key, device_key=self.device_key
        )
        return self._publish_message(topic, payload)

    def send_service_reply(self, payload: Dict) -> bool:
        reply_topic = self._TOPIC_SERVICE_REPLY.format(
            product_key=self.product_key, device_key=self.device_key
        )
        return self._publish_message(reply_topic, payload)

    def disconnect(self) -> None:
        self._should_reconnect = False
        if self._reconnect_timer:
            self._reconnect_timer.cancel()
            self._reconnect_timer = None
        if self._mqtt_client:
            self._mqtt_client.loop_stop()
            self._mqtt_client.disconnect()
            logger.info("MQTT连接已断开")
            self._mqtt_client = None

    def __del__(self):
        self.disconnect()

```

# 六、reporter\.py 上报脚本

```Python
#!/usr/bin/env python3
"""
MQTT 属性上报示例脚本

周期性上报模拟的属性数据到 IoT 平台。
仅演示属性上报功能，不包含事件和服务。

用法:
    python reporter.py
    python reporter.py --dev-info /path/to/dev_info.txt
"""

from __future__ import annotations

import argparse
import glob
import json
import logging
import os
import random
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

from iot_mqtt_sdk import IoTMTTTSDK

@dataclass
class BatteryInfoStruct:
    voltage: int = 0
    current: int = 0
    soc: int = 0
    temperature: int = 0
    status: int = 2

@dataclass
class OdomStruct:
    x: float = 0.0
    y: float = 0.0
    theta: float = 0.0

@dataclass
class PropertyStruct:
    battery_info: Optional[BatteryInfoStruct] = field(default=None)
    ap_info: Optional[str] = field(default=None)
    head_pos: Optional[object] = field(default=None)
    net_info: Optional[str] = field(default=None)
    odom: Optional[OdomStruct] = field(default=None)
    version: Optional[str] = field(default=None)
    imu_state: Optional[object] = field(default=None)
    robot_status: Optional[int] = field(default=None)
    motor_1: Optional[object] = field(default=None)
    motor_2: Optional[object] = field(default=None)
    motor_3: Optional[object] = field(default=None)
    motor_4: Optional[object] = field(default=None)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  [%(levelname)s] %(name)s:%(lineno)d: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("reporter")

# ============================================================
# 全局配置
# ============================================================
VERSION="1.0.0"
REPORT_INTERVAL_SECONDS = 5

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEV_INFO_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))

# 网络/WiFi 信息缓存（每60秒更新一次）
_net_info = ""
_ap_info = ""

def update_net_info():
    global _net_info
    try:
        result = subprocess.run(
            ["ip", "addr"], capture_output=True, text=True, timeout=5
        )
        net_entries = []
        current_iface = None
        mac_addr = None
        ip_addr = None

        for line in result.stdout.splitlines():
            line_stripped = line.strip()
            if not line_stripped:
                continue
            if line_stripped.startswith("inet ") and current_iface:
                ip_addr = line_stripped.split()[1].split("/")[0]
                continue
            parts = line_stripped.split()
            if len(parts) >= 1:
                token = parts[0]
                if token.endswith(":") and len(token) > 1:
                    iface_name = token[:-1]
                    try:
                        int(iface_name)
                        if len(parts) >= 2 and parts[1].endswith(":"):
                            iface_name = parts[1][:-1]
                    except ValueError:
                        pass
                    if current_iface and ip_addr and mac_addr and current_iface != "lo":
                        net_entries.append(f"{current_iface}:{ip_addr}:{mac_addr}")
                    current_iface = iface_name
                    ip_addr = None
                    mac_addr = None
                elif token == "link/ether" and len(parts) >= 2:
                    mac_addr = parts[1].upper().replace(":", "")

        if current_iface and ip_addr and mac_addr and current_iface != "lo":
            net_entries.append(f"{current_iface}:{ip_addr}:{mac_addr}")

        _net_info = ",".join(net_entries)
        logger.info(f"[NET] Updated net_info: {_net_info[:200]}")
    except Exception as e:
        logger.error(f"[NET] Failed to get network info: {e}")
        _net_info = ""

def update_ap_info():
    threading.Thread(target=_update_ap_info_async, daemon=True).start()

def _update_ap_info_async():
    global _ap_info
    try:
        result = subprocess.run(
            ["nmcli", "-t", "-f", "active,ssid,signal,bssid", "dev", "wifi"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if line.startswith("是:") or line.startswith("yes:"):
                parts = line.split(":")
                if len(parts) >= 4:
                    ssid = parts[1].replace("\\:", ":")
                    rssi = parts[2]
                    bssid = "".join(parts[3:]).replace("\\", "").upper()
                    _ap_info = f"{ssid}:{rssi}:{bssid}"
                    logger.info(f"[AP] Updated ap_info: {_ap_info}")
                    return
        _ap_info = ""
        logger.warning("[AP] No connected WiFi AP found")
    except Exception as e:
        logger.error(f"[AP] Failed to get AP info: {e}")
        _ap_info = ""

def _start_net_ap_timers():
    update_net_info()
    update_ap_info()

    def _net_loop():
        while True:
            time.sleep(60)
            update_net_info()

    def _ap_loop():
        while True:
            time.sleep(60)
            update_ap_info()

    threading.Thread(target=_net_loop, daemon=True).start()
    threading.Thread(target=_ap_loop, daemon=True).start()

def _parse_dev_info(path: str) -> dict:
    info = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",", 1)
            if len(parts) == 2:
                key = parts[0].strip()
                value = parts[1].strip()
                info[key] = value
    return info

def find_dev_info(path=None):
    if path:
        if os.path.isfile(path):
            return _parse_dev_info(path)
        raise FileNotFoundError(f"指定的 dev_info 文件不存在: {path}")
    # 按 dev_info_ 前缀查找目录下第一个 .txt 文件
    matches = sorted(glob.glob(os.path.join(_DEV_INFO_DIR, "dev_info_*.txt")))
    if matches:
        found = matches[0]
        logger.info(f"使用设备信息文件: {found}")
        return _parse_dev_info(found)
    raise FileNotFoundError(
        f"未找到 dev_info_*.txt，请通过 --dev-info 参数指定路径"
    )

def generate_simulated_property() -> PropertyStruct:
    prop = PropertyStruct()
    prop.battery_info = BatteryInfoStruct(
        voltage=random.randint(460, 500),
        current=random.randint(-50, 50),
        soc=random.randint(75, 95),
        temperature=random.randint(30, 40),
        status=1,
    )

    prop.odom = OdomStruct(
        x=round(random.uniform(0.0, 10.0), 2),
        y=round(random.uniform(0.0, 10.0), 2),
        theta=round(random.uniform(-3.14, 3.14), 2),
    )
    
    prop.net_info = _net_info
    prop.ap_info = _ap_info
    prop.version = VERSION
    prop.robot_status = 1

    return prop

def build_property_payload(property_data: PropertyStruct) -> dict:
    data_list = []
    data = property_data

    if data.battery_info is not None:
        data_list.append({
            "key": "battery_info",
            "value": {
                "soc": data.battery_info.soc,
                "voltage": data.battery_info.voltage,
                "current": data.battery_info.current,
                "temperature": data.battery_info.temperature,
                "status": data.battery_info.status,
            }
        })
    
    if data.odom is not None:
        data_list.append({
            "key": "odom",
            "value": {
                "x": data.odom.x,
                "y": data.odom.y,
                "theta": data.odom.theta,
            }
        })
        
    if data.net_info is not None and data.net_info != "":
        data_list.append({"key": "net_info", "value": data.net_info})
        
    if data.ap_info is not None and data.ap_info != "":
        data_list.append({"key": "ap_info", "value": data.ap_info})
        
    if data.version is not None and data.version != "":
        data_list.append({"key": "version", "value": data.version})
  
    if data.robot_status is not None:
        data_list.append({"key": "robot_status", "value": data.robot_status})
        
    return {
        "messageId": str(int(time.time() * 1000)),
        "params": {"data": data_list},
    }

def main():
    parser = argparse.ArgumentParser(description="MQTT 属性上报示例")
    parser.add_argument("--dev-info", default=None, help="指定 dev_info.txt 路径")
    args = parser.parse_args()

    try:
        dev_info = find_dev_info(args.dev_info)
    except FileNotFoundError as e:
        logger.error(e)
        sys.exit(1)

    product_key = dev_info["product_key"]
    device_key = dev_info["device_key"]
    device_secret = dev_info["device_secret"]

    url = dev_info.get("url", "dmp-mqtt.cuiot.cn:1883")
    mqtt_ip, mqtt_port_str = url.rsplit(":", 1)
    mqtt_port = int(mqtt_port_str)

    logger.info(f"product_key={product_key}, device_key={device_key}")
    logger.info(f"MQTT {mqtt_ip}:{mqtt_port}, 上报间隔 {REPORT_INTERVAL_SECONDS}s")

    sdk = IoTMTTTSDK(
        product_key=product_key,
        device_key=device_key,
        device_secret=device_secret,
        mqtt_ip=mqtt_ip,
        mqtt_port=mqtt_port,
    )
    if not sdk.connect():
        logger.error("MQTT 连接失败")
        sys.exit(1)

    logger.info("连接成功，开始周期上报属性 (Ctrl+C 停止)")

    # 启动 net_info / ap_info 后台定时更新（每60秒）
    _start_net_ap_timers()

    try:
        while True:
            prop = generate_simulated_property()
            payload = build_property_payload(prop)
            ok = sdk.report_property_batch(payload)
            item_count = len(payload["params"]["data"])
            if ok:
                logger.info(f"上报成功 | messageId={payload['messageId']} | 属性数={item_count}")
            else:
                logger.warning(f"上报失败 | messageId={payload['messageId']}")
            time.sleep(REPORT_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        logger.info("收到停止信号")
    finally:
        sdk.disconnect()
        logger.info("程序退出")

if __name__ == "__main__":
    main()

```

