#!/usr/bin/env bash
# GenPose2 Gradio UI：start / stop / restart / status
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_JSON="${ROOT_DIR}/config/conf.json"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-genpose2}"
# 若已 export PYTHON 则尊重；否则在 activate 后使用当前 env 的 python
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-18090}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export GRADIO_ANALYTICS_ENABLED="${GRADIO_ANALYTICS_ENABLED:-False}"
export OPENCV_IO_ENABLE_OPENEXR="${OPENCV_IO_ENABLE_OPENEXR:-1}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"

LOG_DIR="${ROOT_DIR}/logs"
PID_DIR="${ROOT_DIR}/.pids"
NAME="ui"
PID_FILE="${PID_DIR}/${NAME}.pid"
LOG_FILE="${LOG_DIR}/${NAME}.log"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-300}"

mkdir -p "${LOG_DIR}" "${PID_DIR}"

usage() {
  cat <<EOF
用法: bash start.sh {start|stop|restart|status}

环境变量（可选）:
  CONDA_ENV_NAME  默认 ${CONDA_ENV_NAME}
  PYTHON          默认：activate 后的 python（或已 export 的路径）
  HOST            默认 ${HOST}
  PORT            默认 ${PORT}
  CUDA_VISIBLE_DEVICES  默认 ${CUDA_VISIBLE_DEVICES}

UI: http://<host>:${PORT}/
配置: config/conf.json
EOF
}

# ---------------------------------------------------------------------------
# conda: 主动 activate genpose2
# ---------------------------------------------------------------------------
find_conda_sh() {
  local candidates=(
    "${CONDA_SH:-}"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "/home/ubuntu/miniconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
  )
  local c
  for c in "${candidates[@]}"; do
    [[ -n "${c}" && -f "${c}" ]] && { echo "${c}"; return 0; }
  done
  if command -v conda >/dev/null 2>&1; then
    local base
    base="$(conda info --base 2>/dev/null || true)"
    if [[ -n "${base}" && -f "${base}/etc/profile.d/conda.sh" ]]; then
      echo "${base}/etc/profile.d/conda.sh"
      return 0
    fi
  fi
  return 1
}

activate_genpose2() {
  local conda_sh
  if ! conda_sh="$(find_conda_sh)"; then
    echo "[error] 找不到 conda.sh，无法 activate ${CONDA_ENV_NAME}"
    echo "        请安装 Miniconda/Anaconda，或设置 CONDA_SH=/path/to/conda.sh"
    return 1
  fi

  # shellcheck source=/dev/null
  source "${conda_sh}"

  if ! conda env list 2>/dev/null | awk '{print $1}' | grep -qx "${CONDA_ENV_NAME}"; then
    echo "[error] conda 环境不存在: ${CONDA_ENV_NAME}"
    echo "        可用环境:"
    conda env list
    return 1
  fi

  set +u
  conda activate "${CONDA_ENV_NAME}"
  set -u

  if [[ -z "${PYTHON:-}" ]]; then
    PYTHON="$(command -v python)"
  fi

  if [[ ! -x "${PYTHON}" ]]; then
    echo "[error] Python 不可执行: ${PYTHON}"
    return 1
  fi

  local env_prefix
  env_prefix="$(conda info --json 2>/dev/null | "${PYTHON}" -c 'import json,sys; print(json.load(sys.stdin).get("active_prefix") or "")' 2>/dev/null || true)"
  echo "[env] conda activate ${CONDA_ENV_NAME}"
  echo "      conda.sh : ${conda_sh}"
  echo "      python   : ${PYTHON}"
  echo "      version  : $("${PYTHON}" -c 'import sys; print(sys.version.split()[0])')"
  if [[ -n "${env_prefix}" ]]; then
    echo "      prefix   : ${env_prefix}"
  fi
  if [[ "${CONDA_DEFAULT_ENV:-}" != "${CONDA_ENV_NAME}" ]]; then
    echo "[warn] CONDA_DEFAULT_ENV=${CONDA_DEFAULT_ENV:-<empty>}，期望 ${CONDA_ENV_NAME}"
  fi
  return 0
}

# ---------------------------------------------------------------------------
# 从 conf.json 打印外部依赖，并提醒服务需已启动
# ---------------------------------------------------------------------------
print_config_dependencies() {
  if [[ ! -f "${CONF_JSON}" ]]; then
    echo "[warn] 配置文件不存在: ${CONF_JSON}"
    return 0
  fi

  echo ""
  echo "== 配置依赖项（来自 config/conf.json）=="
  ROOT_DIR="${ROOT_DIR}" CONF_JSON="${CONF_JSON}" "${PYTHON}" - <<'PY'
import json
import os
import socket
from pathlib import Path
from urllib.parse import urlparse

conf_path = Path(os.environ["CONF_JSON"])
root = Path(os.environ["ROOT_DIR"])
cfg = json.loads(conf_path.read_text(encoding="utf-8"))

def host_port(url: str):
    u = urlparse(url)
    host = u.hostname or ""
    port = u.port
    if port is None:
        port = 443 if u.scheme == "https" else 80
    return host, port

def tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    if not host:
        return False
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

rows = []

sam3 = cfg.get("sam3") or {}
sam3_url = str(sam3.get("api_url") or "").strip()
if sam3_url:
    rows.append(("SAM3 分割 HTTP", sam3_url, "local_service"))

vlm = cfg.get("vlm") or {}
sam3_prompt = vlm.get("sam3_prompt") or {}
reason = vlm.get("reason") or {}
if sam3_prompt.get("api_url"):
    rows.append(
        (
            f"VLM sam3_prompt ({sam3_prompt.get('provider') or '?'} / {sam3_prompt.get('model') or '?'})",
            str(sam3_prompt["api_url"]).strip(),
            "local_service",
        )
    )
if reason.get("api_url"):
    rows.append(
        (
            f"VLM reason ({reason.get('provider') or '?'} / {reason.get('model') or '?'})",
            str(reason["api_url"]).strip(),
            "remote_api",
        )
    )

gp = cfg.get("genpose2") or {}
for key, label in (
    ("score_ckpt", "ScoreNet 权重"),
    ("energy_ckpt", "EnergyNet 权重"),
    ("scale_ckpt", "ScaleNet 权重"),
):
    rel = str(gp.get(key) or "").strip()
    if not rel:
        continue
    p = Path(rel)
    if not p.is_absolute():
        p = (root / p).resolve()
    else:
        p = p.expanduser().resolve()
    rows.append((label, str(p), "local_file"))

print(f"  conf: {conf_path}")
print("")
for name, target, kind in rows:
    if kind == "local_file":
        ok = Path(target).is_file()
        status = "OK 文件存在" if ok else "MISSING 文件不存在"
        print(f"  [{kind}] {name}")
        print(f"           path : {target}")
        print(f"           check: {status}")
    else:
        host, port = host_port(target)
        if kind == "local_service":
            up = tcp_open(host, port) if host else False
            status = f"port {port} {'UP' if up else 'DOWN — 请先启动该服务'}"
        else:
            status = "远程 API（启动 UI 前请确认网络/密钥可用）"
            if host:
                # light touch: DNS/TCP optional
                up = tcp_open(host, port)
                status += f"； TCP {'ok' if up else 'fail'} {host}:{port}"
        print(f"  [{kind}] {name}")
        print(f"           url  : {target}")
        print(f"           check: {status}")
    print("")
PY

  echo "--------------------------------------------------------------"
  echo "[提醒] 请确认以上「依赖服务」均已事先启动，再使用本 UI："
  echo "       1) SAM3 HTTP（conf.sam3.api_url，常见端口 18003）"
  echo "       2) VLM sam3_prompt（conf.vlm.sam3_prompt.api_url，常见本地 8000）"
  echo "       3) VLM reason（conf.vlm.reason.api_url，远程则需 API Key / 网络）"
  echo "       4) 三个 GenPose2 权重文件已就位（results/ckpts/...）"
  echo "       本脚本只启动 Gradio UI，不会代启上述依赖服务。"
  echo "--------------------------------------------------------------"
  echo ""
}

is_running() {
  [[ -f "${PID_FILE}" ]] && kill -0 "$(cat "${PID_FILE}")" 2>/dev/null
}

port_in_use() {
  ss -tln 2>/dev/null | grep -q ":${1} "
}

wait_for_port() {
  local port="$1"
  local timeout="$2"
  local elapsed=0
  echo -n "[wait] UI ready on port ${port}"
  while (( elapsed < timeout )); do
    if port_in_use "${port}"; then
      echo " ok (${elapsed}s)"
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
    echo -n "."
  done
  echo " timeout"
  echo "[warn] 未在 ${timeout}s 内监听 ${port}，请查看日志: ${LOG_FILE}"
  return 1
}

cmd_start() {
  if is_running; then
    echo "[skip] UI already running (pid $(cat "${PID_FILE}"))"
    echo "       url: http://${HOST}:${PORT}/"
    return 0
  fi

  if ! activate_genpose2; then
    exit 1
  fi
  print_config_dependencies

  if port_in_use "${PORT}"; then
    local pids
    pids=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
    echo "[error] port ${PORT} already in use (pid(s): ${pids:-unknown})"
    echo "        bash start.sh stop  或  kill ${pids}"
    exit 1
  fi

  echo "== GenPose2 Gradio UI =="
  echo "  python: ${PYTHON}"
  echo "  url:    http://${HOST}:${PORT}/"
  echo "  conf:   ${CONF_JSON}"

  cd "${ROOT_DIR}"
  : >"${LOG_FILE}"
  # 用 activate 后的 PATH/env 启动，保证与交互式 conda 一致
  nohup "${PYTHON}" run_ui.py --host "${HOST}" --port "${PORT}" >>"${LOG_FILE}" 2>&1 &
  echo $! >"${PID_FILE}"
  echo "[start] pid=$(cat "${PID_FILE}") log=${LOG_FILE}"

  wait_for_port "${PORT}" "${STARTUP_TIMEOUT}" || true
  echo "[ok] UI: http://${HOST}:${PORT}/"
}

cmd_stop() {
  if ! is_running; then
    if port_in_use "${PORT}"; then
      local pids
      pids=$(ss -tlnp 2>/dev/null | grep ":${PORT} " | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u || true)
      if [[ -n "${pids}" ]]; then
        echo "[stop] port ${PORT} residual pid(s): ${pids}"
        kill ${pids} 2>/dev/null || true
        sleep 1
        kill -9 ${pids} 2>/dev/null || true
      fi
    else
      echo "[skip] UI not running"
    fi
    rm -f "${PID_FILE}"
    return 0
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  echo "[stop] UI (pid ${pid})"
  kill "${pid}" 2>/dev/null || true
  for _ in 1 2 3 4 5; do
    if ! kill -0 "${pid}" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] force kill ${pid}"
    kill -9 "${pid}" 2>/dev/null || true
  fi
  rm -f "${PID_FILE}"
  echo "[ok] stopped"
}

cmd_status() {
  if is_running; then
    echo "[status] running pid=$(cat "${PID_FILE}") url=http://${HOST}:${PORT}/"
  elif port_in_use "${PORT}"; then
    echo "[status] port ${PORT} in use, but pid file missing/stale"
  else
    echo "[status] stopped"
  fi
  # status 时也展示依赖快照（不强制 activate 失败则跳过探测细节）
  if [[ -f "${CONF_JSON}" ]]; then
    if activate_genpose2 2>/dev/null; then
      print_config_dependencies
    else
      echo "[info] 未能 activate ${CONDA_ENV_NAME}，跳过依赖探测；配置见 ${CONF_JSON}"
    fi
  fi
}

cmd_restart() {
  cmd_stop
  cmd_start
}

main() {
  local action="${1:-}"
  case "${action}" in
    start) cmd_start ;;
    stop) cmd_stop ;;
    restart) cmd_restart ;;
    status) cmd_status ;;
    -h|--help|help|"") usage; [[ -n "${action}" ]] || exit 1 ;;
    *) echo "未知命令: ${action}"; usage; exit 1 ;;
  esac
}

main "$@"
