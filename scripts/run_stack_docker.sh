#!/usr/bin/env bash
# Manage vLLM (Docker) + API stack.
# Usage:
#   ./scripts/run_stack_docker.sh up     # start vLLM container + API
#   ./scripts/run_stack_docker.sh down   # stop both
#   ./scripts/run_stack_docker.sh status # show health
#   ./scripts/run_stack_docker.sh logs   # tail recent logs

set -euo pipefail

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$PROJECT_ROOT"

# Load .env if present
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  # Export variables from .env but avoid nounset errors on expansion
  set +u
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.env"
  set +a
  set -u
fi

# Defaults (overridable via .env or inline)
IMAGE=${VLLM_DOCKER_IMAGE:-"vllm/vllm-openai:latest"}
VLLM_NAME=${VLLM_CONTAINER_NAME:-"vllm-server"}
VLLM_PORT=${VLLM_PORT:-8025}
API_PORT=${API_PORT:-8018}
HF_HOME_DIR=${HF_HOME:-"/mnt/data/cache/hf"}
MODEL=${MODEL_NAME:-"meta-llama/Llama-3.1-8B-Instruct"}
GPU_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_LEN=${MAX_MODEL_LEN:-2048}
HOST_IP=${VLLM_HOST:-"0.0.0.0"}

# Ensure VLLM_BASE_URL is aligned to the chosen port
export VLLM_BASE_URL=${VLLM_BASE_URL:-"http://localhost:${VLLM_PORT}"}

API_LOG="$PROJECT_ROOT/api.log"

cmd=${1:-"up"}

start_vllm() {
  mkdir -p "$HF_HOME_DIR"
  # Stop existing container silently
  docker rm -f "$VLLM_NAME" >/dev/null 2>&1 || true
  echo "🖼️  vLLM image: $IMAGE"
  echo "🔌 vLLM port:  $VLLM_PORT"
  echo "📦 HF cache:   $HF_HOME_DIR"
  echo "🧠 Model:      $MODEL"
  # Pre-pull to surface image tag issues early; fallback to latest if pull fails
  if ! docker pull "$IMAGE" >/dev/null 2>&1; then
    echo "⚠️  Unable to pull $IMAGE. Falling back to vllm/vllm-openai:latest"
    IMAGE="vllm/vllm-openai:latest"
    docker pull "$IMAGE" >/dev/null 2>&1 || true
  fi
  set -x
  docker run --rm -d \
    --name "$VLLM_NAME" \
    --gpus all \
    -p "$VLLM_PORT:8000" \
    -e HF_HOME="$HF_HOME_DIR" \
    -e HF_TOKEN="${HF_TOKEN:-}" \
    -e HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-${HF_TOKEN:-}}" \
    -v "$HF_HOME_DIR":"/root/.cache/huggingface" \
    "$IMAGE" \
    --host "$HOST_IP" \
    --model "$MODEL" \
    --gpu-memory-utilization "$GPU_UTIL" \
    --max-model-len "$MAX_LEN" \
    --disable-log-stats \
    --disable-frontend-multiprocessing \
    --guided-decoding-backend lm-format-enforcer
  set +x
}

health_vllm() {
  curl -sf "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1
}

wait_vllm() {
  echo -n "⏳ Waiting for vLLM on :$VLLM_PORT"
  for _ in {1..60}; do
    if health_vllm; then echo " — ready"; return 0; fi
    echo -n "."; sleep 2
  done
  echo; echo "❌ vLLM did not become healthy in time"; return 1
}

start_api() {
  export VLLM_BASE_URL="http://localhost:${VLLM_PORT}"
  echo "🌐 API will use VLLM_BASE_URL=$VLLM_BASE_URL"
  # Start API in background via nohup
  nohup ./scripts/start_api.sh >> "$API_LOG" 2>&1 &
  echo "📝 API logs: $API_LOG"
}

status() {
  echo "=== vLLM (Docker) ==="
  docker ps --filter "name=$VLLM_NAME" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
  echo -n "Health: "; if health_vllm; then echo "OK"; else echo "UNHEALTHY"; fi
  echo "=== API ==="
  curl -sf "http://localhost:${API_PORT}/health" >/dev/null 2>&1 && echo "Health: OK" || echo "Health: UNHEALTHY"
}

down() {
  echo "🛑 Stopping API (port $API_PORT)"
  pkill -f "uvicorn .*src.server.api:app" >/dev/null 2>&1 || true
  echo "🛑 Stopping vLLM container: $VLLM_NAME"
  docker rm -f "$VLLM_NAME" >/dev/null 2>&1 || true
}

logs() {
  echo "=== vLLM (Docker) ==="
  docker logs --tail 100 "$VLLM_NAME" 2>/dev/null || echo "(no container)"
  echo "=== API (tail) ==="
  tail -n 100 "$API_LOG" 2>/dev/null || echo "(no log)"
}

case "$cmd" in
  up)
    start_vllm
    wait_vllm
    start_api
    status
    ;;
  down)
    down
    ;;
  status)
    status
    ;;
  logs)
    logs
    ;;
  *)
    echo "Usage: $0 {up|down|status|logs}" >&2
    exit 1
    ;;
esac
