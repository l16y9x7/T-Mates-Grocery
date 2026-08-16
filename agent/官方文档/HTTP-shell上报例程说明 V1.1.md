# HTTP\-shell上报例程说明 V1\.1

- 样例代码：

\[shell\_http\_reporter\.tar\.gz\]

- 测试密钥：

\[dev\_info\_HTTP测试\.zip\]



# 一、文件包结构

```Shell
├── deploy.sh 部署配置服务自启动的脚本
├── dev_info_AABBCCDDEEFF.txt 连接平台的设备密钥参数（根据参赛的机器人，由我们人工生成，发送到参赛方。后缀为MAC地址，方便知道是哪一个设备的）
├── log
│   └── reporter.log 脚本运行log
├── README.md 使用说明 
├── robot_reporter.service 服务文件
├── start_report.sh 上报脚本
└── version.txt 版本号

```

- **第一步，在机器人中解压安装包**

进入用户目录`cd ~`，创建目录并解压安装包：

```Shell
mkdir -p shell_http_reporter && tar -xzvf shell_http_reporter.tar.gz -C shell_http_reporter
```

- **第二步，部署配置服务自启动**

进入解压后的目录 `cd shell_http_reporter`，为部署脚本添加执行权限并运行：

```Shell
sudo chmod +x ./deploy.sh
sudo ./deploy.sh
```

服务正常启动后，可通过以下命令查看日志：

```Shell
tail -f ./log/reporter.log
```

# 二、deploy\.sh 部署配置服务自启动脚本

```Shell
#!/bin/bash

echo ">>> 开始部署 robot_reporter 服务..."

SERVICE_FILE="/home/master/shell_http_reporter/robot_reporter.service"

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

```Shell
product_key,xxx
device_key,xxx
device_secret,xxx
device_name,赛队-EEFF
MAC,AABBCCDDEEFF
url,https://dmp-https.cuiot.cn:8943 
```

# 四、robot\_reporter\.service 服务文件

```Shell
[Unit]
Description=Robot Reporter Agent
Wants=network-online.target
After=booster-daemon.service network-online.target
Requires=booster-daemon.service

[Service]
Type=simple
User=master
Group=master
WorkingDirectory=/home/master/shell_http_reporter
Environment="HOME=/home/master"
Environment="ROS_LOG_DIR=/home/master/.ros/log"
Environment="PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStartPre=/bin/sleep 6
ExecStart=/bin/bash -c "source /home/master/shell_http_reporter/start_report.sh 5 >> /home/master/shell_http_reporter/log/reporter.log 2>&1"
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

# 五、start\_report\.sh 上报脚本

```Shell
ts=$((ts / 1000000))  # 纳秒转毫秒
AP_INFO="wifi_test:90:112233445566"
```

```Bash
#!/bin/bash
# 周期性属性上报脚本
# 从 dev_info_*.txt 读取设备凭证，获取 token 后按固定间隔上报属性数据到 IoT 平台

SCRIPT_DIR="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")"

dev_files=("$SCRIPT_DIR"/dev_info_*.txt)
DEV_INFO="${dev_files[0]}"

# 读取设备凭证（格式：key,value 每行）
PRODUCT_KEY=$(sed -n 's/^product_key,//p' "$DEV_INFO")
DEVICE_KEY=$(sed -n 's/^device_key,//p' "$DEV_INFO")
DEVICE_SECRET=$(sed -n 's/^device_secret,//p' "$DEV_INFO")
URL=$(sed -n 's/^url,//p' "$DEV_INFO")
: "${URL:=https://dmp-https.cuiot.cn:8943}"
# 从 URL 提取 host:port（去掉协议前缀）
URL_HOST="${URL#*://}"

VERSION=$(cat "$SCRIPT_DIR/version.txt")

# 上报周期，默认 5 秒，可通过命令行参数覆盖
INTERVAL=${1:-5}

# 全局状态（网络/AP每60秒刷新，SOC每周期从ROS2读取）
NET_INFO=""
AP_INFO=""
BATTERY_SOC=0
LAST_NET_UPDATE=0
LAST_AP_UPDATE=0

# 生成鉴权 JSON（HMAC-SHA256 签名，带时间戳防重放）
gen_auth_json() {
    local operator="0"
    local sign_method="Hmacsha256"
    local ts
    ts=$(date +%s%3N)
    local sign_string="${DEVICE_KEY}${DEVICE_KEY}${PRODUCT_KEY}${sign_method}${operator}${ts}"
    local sign
    sign=$(echo -n "$sign_string" | openssl dgst -sha256 -hmac "$DEVICE_SECRET" | awk '{print $2}')
    cat <<EOF
{
    "productKey": "$PRODUCT_KEY",
    "deviceKey": "$DEVICE_KEY",
    "operator": "$operator",
    "deviceId": "$DEVICE_KEY",
    "timestamp": "$ts",
    "signMethod": "$sign_methjod",
    "sign": "$sign"
}
EOF
}

# POST /auth 获取 token，失败返回非0
get_token() {
    local auth_json
    auth_json=$(gen_auth_json)
    local resp
    resp=$(curl -s -X POST "${URL}/auth" \
        -H "Content-Type: application/json" \
        -d "$auth_json")
    local token
    token=$(echo "$resp" | sed -n 's/.*"token"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')
    if [ -z "$token" ]; then
        echo "$resp" >&2
        return 1
    fi
    echo "$token"
}

PUB_URL="${URL}/topic/sys/${PRODUCT_KEY}/${DEVICE_KEY}/property/batch"

# 属性上报，body 以 UTF-8 字节流通过 application/octet-stream 发送
send_data() {
    local body="$1"
    local token="$2"
    printf '%s' "$body" | curl -s -X POST "$PUB_URL" \
        -H "Host: ${URL_HOST}" \
        -H "Content-Type: application/octet-stream" \
        -H "Authorization: ${token}" \
        --data-binary @-
}

# 采集网络信息：解析 ip addr 输出，格式 iface:IP:MAC（排除 lo）
update_net_info() {
    local result entries current_iface mac_addr ip_addr
    result=$(ip addr 2>/dev/null) || { NET_INFO=""; return 1; }
    entries=()
    current_iface=""; mac_addr=""; ip_addr=""

    while IFS= read -r line; do
        line="${line#"${line%%[![:space:]]*}"}"
        [ -z "$line" ] && continue
        if [[ "$line" =~ ^[0-9]+:\ ([^:]+): ]]; then
            local iface="${BASH_REMATCH[1]}"
            if [ -n "$current_iface" ] && [ -n "$ip_addr" ] && [ -n "$mac_addr" ] && [ "$current_iface" != "lo" ]; then
                entries+=("${current_iface}:${ip_addr}:${mac_addr}")
            fi
            current_iface="$iface"; ip_addr=""; mac_addr=""
        elif [[ "$line" =~ ^link/ether\ ([0-9a-fA-F:]+) ]]; then
            mac_addr="${BASH_REMATCH[1]//:/}"; mac_addr="${mac_addr^^}"
        elif [[ "$line" =~ ^inet\ ([0-9.]+)/ ]]; then
            ip_addr="${BASH_REMATCH[1]}"
        fi
    done <<< "$result"

    if [ -n "$current_iface" ] && [ -n "$ip_addr" ] && [ -n "$mac_addr" ] && [ "$current_iface" != "lo" ]; then
        entries+=("${current_iface}:${ip_addr}:${mac_addr}")
    fi

    local IFS=","
    NET_INFO="${entries[*]}"
    LAST_NET_UPDATE=$(date +%s)
}

# 采集 WiFi AP 信息：解析 nmcli 输出，格式 SSID:RSSI:BSSID（BSSID 去空格转大写）
update_ap_info() {
    local result
    result=$(nmcli -t -f active,ssid,signal,bssid dev wifi 2>/dev/null) || { AP_INFO=""; return 1; }
    while IFS= read -r line; do
        if [[ "$line" == yes:* ]] || [[ "$line" == 是:* ]]; then
            local parts ssid rssi bssid
            IFS=':' read -ra parts <<< "$line"
            ssid="${parts[1]//\\:/:}"
            rssi="${parts[2]}"
            bssid="${parts[*]:3}"; bssid="${bssid//\\/}"; bssid="${bssid// /}"; bssid="${bssid^^}"
            AP_INFO="${ssid}:${rssi}:${bssid}"
            LAST_AP_UPDATE=$(date +%s)
            return 0
        fi
    done <<< "$result"
    AP_INFO=""
    LAST_AP_UPDATE=$(date +%s)
}

# # 从 ROS2 /BatterySocTopic 读取电池 SOC
# # Topic 输出 array('B', [byte0, byte1, byte2, byte3, ...])，前4字节为 float32 LE
# # 例如 [51, 51, 119, 66] → 61.8 → 四舍五入 62
# update_battery_soc() {
#     local raw nums oct1 oct2 oct3 oct4
#     raw=$(ros2 topic echo /BatterySocTopic --once --field msg 2>/dev/null) || { BATTERY_SOC=0; return 1; }
#     nums=$(echo "$raw" | sed 's/.*\[\(.*\)\].*/\1/')
#     IFS=', ' read -r b1 b2 b3 b4 rest <<< "$nums" 2>/dev/null
#     if [ -z "$b1" ] || [ -z "$b4" ]; then
#         BATTERY_SOC=0
#         return 1
#     fi
#     oct1=$(printf '%03o' "$b1"); oct2=$(printf '%03o' "$b2")
#     oct3=$(printf '%03o' "$b3"); oct4=$(printf '%03o' "$b4")
#     # 4字节 → float32 LE → 四舍五入取整
#     BATTERY_SOC=$(printf "\\$oct1\\$oct2\\$oct3\\$oct4" \
#         | od -t f4 -An | tr -d ' ' | awk '{printf "%.0f", $1}')
#     : "${BATTERY_SOC:=-1}"
# }

# init_ros2() {
#     source /opt/ros/humble/setup.bash
#     source /opt/booster/BoosterRos2Interface/install/setup.bash 2>/dev/null || true
#     source /opt/booster/BoosterRos2/install/setup.bash 2>/dev/null || true
# }

# 构造上报 body：
# - battery_info 字段（voltage/current/temperature/status）随机
# - soc 来自 ROS2 实时数据
# - net_info / ap_info 来自 60 秒周期采集
# - robot_status 随机
update_body() {
    local msg_id voltage current temperature status robot_status
    msg_id=$(date +%s)
    voltage=$((30000 + RANDOM % 1201))
    current=$((2000 + RANDOM % 200))
    temperature=$((20 + RANDOM % 10))
    status=2 #$((1 + RANDOM % 3))
    robot_status=1 #$((1 + RANDOM % 3))

    cat <<EOF
{
  "messageId": "$msg_id",
  "params": {
    "data": [
      {
        "key": "battery_info",
        "value": {
          "voltage": $voltage,
          "current": $current,
          "soc": $BATTERY_SOC,
          "temperature": $temperature,
          "status": $status
        }
      },
      {
        "key": "ap_info",
        "value": "$AP_INFO"
      },
      {
        "key": "net_info",
        "value": "$NET_INFO"
      },
      {
        "key": "version",
        "value": "$VERSION"
      },
      {
        "key": "robot_status",
        "value": 1
      }
    ]
  }
}
EOF
}

# ---- 主流程 ----
echo "获取 token ..."
TOKEN=$(get_token) || exit 1
echo "周期上报已启动, 间隔 ${INTERVAL}秒, URL: $PUB_URL"
# init_ros2
echo "----------------------------------------"

while true; do
    TS=$(date '+%Y-%m-%d %H:%M:%S.%3N')
    NOW=$(date +%s)

    # 每 60 秒更新网络和 AP 信息
    if [ $((NOW - LAST_NET_UPDATE)) -ge 60 ]; then
        update_net_info
        echo "[$TS] net_info: $NET_INFO"
    fi
    if [ $((NOW - LAST_AP_UPDATE)) -ge 60 ]; then
        update_ap_info
        echo "[$TS] ap_info: $AP_INFO"
    fi

    # 每周期从 ROS2 读取电池 SOC
    # update_battery_soc
    # echo "[$TS] battery_soc: $BATTERY_SOC"

    BODY=$(update_body)
    RESULT=$(send_data "$BODY" "$TOKEN")
    echo "[$TS] $RESULT"
    echo "----------------------------------------"

    sleep "$INTERVAL"
done

```

# 六、version\.txt 版本号

```Shell
1.0.0
```



