#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT="$(cd "$SCRIPT_DIR/.." && pwd)"
ROOT="$(cd "$PROJECT/.." && pwd)"
LOG_DIR="${LOG_DIR:-$ROOT/logs}"
ROBOT_IP="192.168.200.66"
CONDA_SH="${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
RUNTIME_CONFIG_FILE="${RUNTIME_CONFIG_FILE:-$PROJECT/agent/config/runtime.production.yaml}"
SKU_CATALOG_PATH="${SKU_CATALOG_PATH:-$PROJECT/perception/sku/products.json}"
INITIAL_SCAN_ROOT="${INITIAL_SCAN_ROOT:-$PROJECT/agent/output/task0}"
PRODUCT_HAND_OPTIONS_PATH="${PRODUCT_HAND_OPTIONS_PATH:-$PROJECT/agent/config/product-hand-options.yaml}"
INSPECT_SKU_CATALOG_PATH="${INSPECT_SKU_CATALOG_PATH:-$SKU_CATALOG_PATH}"
export RUNTIME_CONFIG_FILE
export INITIAL_SCAN_ROOT
export PRODUCT_HAND_OPTIONS_PATH
export INSPECT_SKU_CATALOG_PATH
export PATH="${HOME}/.local/bin:${PATH}"

usage() {
  cat <<'EOF'
Usage: start_all_services.sh [--robot-ip ADDRESS]

Restarts Perception, SKU, and Agent from the current project directory.
SAM3, Qwen3-VL, and GenPose2 are started only when not already healthy.
Agent services always use restart so code and config changes take effect.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --robot-ip) ROBOT_IP="${2:?--robot-ip requires an IPv4 address}"; shift ;;
    --robot-ip=*) ROBOT_IP="${1#*=}" ;;
    --restart-perception)
      echo "[warn] --restart-perception is deprecated; perception is always restarted" >&2
      ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
  shift
done

python3 -c 'import ipaddress, sys; ipaddress.IPv4Address(sys.argv[1])' "$ROBOT_IP" >/dev/null
[[ -d "$PROJECT/agent" ]] || { echo "Missing agent directory under project root: $PROJECT" >&2; exit 1; }
[[ -r "$CONDA_SH" ]] || { echo "Missing Conda activation script: $CONDA_SH" >&2; exit 1; }
[[ -r "$RUNTIME_CONFIG_FILE" ]] || { echo "Missing runtime config: $RUNTIME_CONFIG_FILE" >&2; exit 1; }
[[ -r "$SKU_CATALOG_PATH" ]] || { echo "Missing SKU catalog: $SKU_CATALOG_PATH" >&2; exit 1; }
[[ -r "$PRODUCT_HAND_OPTIONS_PATH" ]] || { echo "Missing product hand options: $PRODUCT_HAND_OPTIONS_PATH" >&2; exit 1; }
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
    kill "$pid" 2>/dev/null || true
  done
  for _ in {1..15}; do
    [[ -z "$(port_pids "$port")" ]] && return 0
    sleep 1
  done
  for pid in $(port_pids "$port"); do
    echo "Force stopping listener PID $pid on port $port"
    kill -KILL "$pid" 2>/dev/null || true
  done
  for _ in {1..5}; do
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

restart_service() {
  local label="$1" port="$2" command="$3" log_name="$4" health_url="$5" timeout="$6"
  echo "[restart] $label"
  stop_listener "$port"
  launch "$label" "$command" "$log_name"
  wait_for "$label" "$health_url" "$timeout"
}

ensure_service() {
  local label="$1" health_url="$2" port="$3" command="$4" log_name="$5" timeout="$6"
  if healthy "$health_url"; then
    echo "[keep] $label is already healthy"
    return 0
  fi
  echo "[start] $label"
  stop_listener "$port"
  launch "$label" "$command" "$log_name"
  wait_for "$label" "$health_url" "$timeout"
}

ensure_service SAM3 http://127.0.0.1:25541/health 25541 \
  "source '$CONDA_SH' && conda activate sam3 && cd '$ROOT/serve_sam3' && exec env CUDA_VISIBLE_DEVICES=0 SAM3_CHECKPOINT_PATH='$ROOT/02-weight/sam3/sam3.pt' SAM3_GPU_MEMORY_FRACTION=0.25 python -m uvicorn sam3_server:app --host 0.0.0.0 --port 25541 --workers 1" \
  "sam3-$(date +%Y%m%d-%H%M%S).log" 120

ensure_service Qwen3-VL http://127.0.0.1:25542/v1/models 25542 \
  "source '$CONDA_SH' && conda activate vllm_env && cd '$ROOT/02-weight' && exec vllm serve Qwen3-VL-4B-Instruct --host 0.0.0.0 --gpu-memory-utilization 0.6 --max-model-len 8192 --limit-mm-per-prompt.video 0 --port 25542" \
  "qwen3-vl-$(date +%Y%m%d-%H%M%S).log" 240

ensure_service GenPose2 http://127.0.0.1:8084/manipulation/health 8084 \
  "source '$CONDA_SH' && conda activate genpose2 && cd '$PROJECT/manipulation/pose_estimation/GenPose2' && exec python -u http_server.py --host 0.0.0.0 --port 8084" \
  "genpose2-$(date +%Y%m%d-%H%M%S).log" 180

restart_service Perception 8083 \
  "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/perception' && exec env CAMERA_SERVICE_HOST='$ROBOT_IP' INFERENCE_SERVICE_HOST='127.0.0.1' INITIAL_SCAN_ROOT='$INITIAL_SCAN_ROOT' PRODUCT_HAND_OPTIONS_PATH='$PRODUCT_HAND_OPTIONS_PATH' INSPECT_SKU_CATALOG_PATH='$INSPECT_SKU_CATALOG_PATH' python main.py" \
  "perception-$(date +%Y%m%d-%H%M%S).log" \
  http://127.0.0.1:8083/perception/health 90

restart_service SKU 25540 \
  "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/perception/sku' && exec python api.py --catalog '$SKU_CATALOG_PATH'" \
  "sku-$(date +%Y%m%d-%H%M%S).log" \
  http://127.0.0.1:25540/sku/health 60

if command -v uv >/dev/null 2>&1; then
  echo "[restart] Agent (pick-place + unified task)"
  "$PROJECT/agent/scripts/services.sh" restart --robot-ip "$ROBOT_IP"
else
  echo "[fallback] uv is unavailable; restarting Agent with the verified t_mates environment"
  echo "[restart] pick-place"
  stop_listener 8086
  launch pick-place \
    "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/agent' && export PYTHONPATH='$PROJECT/agent/src' ROBOT_IP='$ROBOT_IP' && exec python -m pick_place_service --config '$RUNTIME_CONFIG_FILE'" \
    "pick-place-$(date +%Y%m%d-%H%M%S).log"
  wait_for "pick-place process" http://127.0.0.1:8086/openapi.json 90
  echo "[warn] pick-place health stays 503 until the robot services are reachable"

  echo "[restart] unified task service"
  stop_listener 8108
  launch Agent \
    "source '$CONDA_SH' && conda activate t_mates && cd '$PROJECT/agent' && export PYTHONPATH='$PROJECT/agent/src' ROBOT_IP='$ROBOT_IP' && exec python -m task_service --config '$RUNTIME_CONFIG_FILE'" \
    "agent-$(date +%Y%m%d-%H%M%S).log"
  wait_for "Agent console" http://127.0.0.1:8108/ 90
fi

"$PROJECT/agent/scripts/health-check.sh" "$ROBOT_IP"

echo "Task console: http://127.0.0.1:8108/"
