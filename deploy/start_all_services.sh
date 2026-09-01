#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/home/nora/tianji"
PROJECT="$ROOT/T-Mates-Grocery"
LOG_DIR="$ROOT/logs"
ROBOT_IP="192.168.200.66"
RESTART_PERCEPTION=0
CONDA_SH="/home/nora/miniconda3/etc/profile.d/conda.sh"

usage() {
  cat <<'EOF'
Usage: start_all_services.sh [--robot-ip ADDRESS] [--restart-perception]

Starts any unavailable T-Mates Grocery service without stopping healthy ones.
--restart-perception stops only the listener on port 8083 before starting
Perception, so a changed camera IP configuration takes effect.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip) ROBOT_IP="${2:?--robot-ip requires an IPv4 address}"; shift ;;
    --robot-ip=*) ROBOT_IP="${1#*=}" ;;
    --restart-perception) RESTART_PERCEPTION=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

python3 -c 'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1])' "$ROBOT_IP" >/dev/null
[[ -r "$CONDA_SH" ]] || { echo "Missing Conda activation script: $CONDA_SH" >&2; exit 1; }
mkdir -p "$LOG_DIR"

healthy() { curl -fsS --connect-timeout 3 --max-time 5 "$1" >/dev/null 2>&1; }

wait_for() {
  local label="$1" url="$2" seconds="$3"
  local end=$((SECONDS + seconds))
  until healthy "$url"; do
    (( SECONDS < end )) || { echo "Timed out waiting for $label: $url" >&2; return 1; }
    sleep 2
  done
  echo "[ready] $label $url"
}

port_pids() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -t -nP -iTCP:"$port" -sTCP:LISTEN 2>/dev/null || true
  else
    fuser -n tcp "$port" 2>/dev/null | tr ' ' '\n' || true
  fi
}

stop_listener() {
  local port="$1" pid
  for pid in $(port_pids "$port"); do
    echo "Stopping listener PID $pid on port $port"
    kill "$pid"
  done
  for _ in {1..15}; do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 1
  done
  echo "Unable to release port $port" >&2
  return 1
}

launch() {
  local label="$1" command="$2" log="$LOG_DIR/$3"
  echo "[start] $label (log: $log)"
  nohup bash -lc "$command" >"$log" 2>&1 < /dev/null &
  sleep 1
}

if healthy http://127.0.0.1:25541/health; then
  echo "[keep] SAM3 is already healthy"
else
  launch SAM3 "source '$CONDA_SH' && conda activate sam3 && cd '$ROOT/serve_sam3' && exec env CUDA_VISIBLE_DEVICES=0 SAM3_CHECKPOINT_PATH='$ROOT/02-weight/sam3/sam3.pt' SAM3_GPU_MEMORY_FRACTION=0.25 python -m uvicorn sam3_server:app --host 0.0.0.0 --port 25541 --workers 1" "sam3-$(date +%Y%m%d-%H%M%S).log"
  wait_for SAM3 http://127.0.0.1:25541/health 120
fi

if healthy http://127.0.0.1:25542/v1/models; then
  echo "[keep] Qwen3-VL is already healthy"
else
  launch Qwen3-VL "source '$CONDA_SH' && conda activate vllm_env && cd '$ROOT/02-weight' && exec vllm serve Qwen3-VL-4B-Instruct --host 0.0.0.0 --gpu-memory-utilization 0.6 --max-model-len 8192 --limit-mm-per-prompt.video 0 --port 25542" "qwen3-vl-$(date +%Y%m%d-%H%M%S).log"
  wait_for Qwen3-VL http://127.0.0.1:25542/v1/models 240
fi

if healthy http://127.0.0.1:8084/manipulation/health; then
  echo "[keep] GenPose2 is already healthy"
else
  launch GenPose2 "source '$CONDA_SH' && conda activate genpose2 && cd '$PROJECT/manipulation/pose_estimation/GenPose2' && exec python -u http_server.py --host 0.0.0.0 --port 8084" "genpose2-$(date +%Y%m%d-%H%M%S).log"
  wait_for GenPose2 http://127.0.0.1:8084/manipulation/health 180
fi

if (( RESTART_PERCEPTION )); then
  stop_listener 8083
fi
if healthy http://127.0.0.1:8083/perception/health; then
  echo "[keep] Perception is already healthy"
else
  launch Perception "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/perception' && exec env CAMERA_SERVICE_HOST='$ROBOT_IP' INFERENCE_SERVICE_HOST='127.0.0.1' python main.py" "perception-$(date +%Y%m%d-%H%M%S).log"
  wait_for Perception http://127.0.0.1:8083/perception/health 90
fi

if healthy http://127.0.0.1:25540/sku/health; then
  echo "[keep] SKU API is already healthy"
else
  launch SKU "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/perception/sku' && exec python api.py" "sku-$(date +%Y%m%d-%H%M%S).log"
  wait_for SKU http://127.0.0.1:25540/sku/health 60
fi

if command -v uv >/dev/null 2>&1; then
  "$PROJECT/agent/scripts/services.sh" start "$ROBOT_IP"
else
  echo "[fallback] uv is unavailable; starting Agent with the verified t_mates environment"
  if healthy http://127.0.0.1:8086/health; then
    echo "[keep] pick-place is already healthy"
  else
    launch pick-place "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/agent' && export PYTHONPATH='$PROJECT/agent/src' ROBOT_IP='$ROBOT_IP' && exec python -m pick_place_service --config '$PROJECT/agent/config/runtime.production.yaml'" "pick-place-$(date +%Y%m%d-%H%M%S).log"
    wait_for "pick-place process" http://127.0.0.1:8086/openapi.json 90
    echo "[warn] pick-place health stays 503 until the robot services are reachable"
  fi
  if healthy http://127.0.0.1:8108/health; then
    echo "[keep] unified task service is already healthy"
  else
    launch Agent "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/agent' && export PYTHONPATH='$PROJECT/agent/src' ROBOT_IP='$ROBOT_IP' && exec python -m task_service --config '$PROJECT/agent/config/runtime.production.yaml'" "agent-$(date +%Y%m%d-%H%M%S).log"
    wait_for "Agent console" http://127.0.0.1:8108/ 90
  fi
fi
"$PROJECT/agent/scripts/health-check.sh" "$ROBOT_IP"

echo "Task console: http://192.168.200.65:8108/"
