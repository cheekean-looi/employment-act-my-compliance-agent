#!/bin/bash
# Start vLLM inference server for Employment Act Malaysia agent
# Supports both base model and LoRA adapter serving

set -e

# Ensure project src is on PYTHONPATH so local shims (e.g., pyairports) are importable
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Load .env so port/host settings (and others) are honored
if [ -f "$PROJECT_ROOT/.env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$PROJECT_ROOT/.env"
  set +a
fi

# Force a safe CUDA allocator config (disable expandable segments)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64,expandable_segments:False"
echo "Allocator config: PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_CUDA_ALLOC_CONF"

# Configuration from environment or defaults
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-1.5B-Instruct"}
ADAPTER_PATH=${ADAPTER_PATH:-""}
HOST=${VLLM_HOST:-"0.0.0.0"}
PORT=${VLLM_PORT:-"8000"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-"0.9"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-"2048"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}
ENFORCE_EAGER=${ENFORCE_EAGER:-"0"}
DISABLE_FRONTEND_MP=${DISABLE_FRONTEND_MP:-"0"}
PORT_AUTOINC=${PORT_AUTOINC:-"0"}

echo "🚀 Starting vLLM server for Employment Act Malaysia agent"
echo "Model: $MODEL_NAME"
echo "Host: $HOST:$PORT"
echo "Max model length: $MAX_MODEL_LEN"

# Check if model exists or needs download
echo "📦 Checking model availability..."

# Ensure vLLM scheduling constraint: max_num_batched_tokens >= max_model_len
if [[ "$MAX_MODEL_LEN" =~ ^[0-9]+$ ]] && [[ "$MAX_NUM_BATCHED_TOKENS" =~ ^[0-9]+$ ]]; then
    if (( MAX_NUM_BATCHED_TOKENS < MAX_MODEL_LEN )); then
        echo "⚙️  Adjusting MAX_NUM_BATCHED_TOKENS from $MAX_NUM_BATCHED_TOKENS to $MAX_MODEL_LEN to satisfy vLLM constraints"
        MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LEN"
    fi
else
    MAX_NUM_BATCHED_TOKENS="$MAX_MODEL_LEN"
fi

# Base vLLM command
VLLM_CMD="vllm serve $MODEL_NAME \
    --host $HOST \
    --port $PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEMORY_UTIL \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --tensor-parallel-size $TENSOR_PARALLEL_SIZE \
    --disable-log-stats \
    --trust-remote-code"

if [ "$ENFORCE_EAGER" = "1" ]; then
    echo "⚙️  Enforcing eager mode for stability"
    VLLM_CMD="$VLLM_CMD --enforce-eager"
fi

if [ "$DISABLE_FRONTEND_MP" = "1" ]; then
    echo "🔧 Disabling frontend multiprocessing to avoid port bind races"
    VLLM_CMD="$VLLM_CMD --disable-frontend-multiprocessing"
fi

# Add LoRA adapter if provided
if [ -n "$ADAPTER_PATH" ] && [ -d "$ADAPTER_PATH" ]; then
    echo "🔧 Using LoRA adapter: $ADAPTER_PATH"
    VLLM_CMD="$VLLM_CMD --enable-lora --lora-modules employment_act=$ADAPTER_PATH"
fi

# Add quantization if specified
if [ -n "$QUANTIZATION" ]; then
    echo "⚡ Using quantization: $QUANTIZATION"
    VLLM_CMD="$VLLM_CMD --quantization $QUANTIZATION"
fi

# Health check endpoint
echo "🏥 Health check will be available at: http://$HOST:$PORT/health"
echo "📊 Metrics will be available at: http://$HOST:$PORT/metrics"

# Preflight: check if port is free; optionally auto-increment to a free one
check_port() {
  local p="$1"
  if command -v ss >/dev/null 2>&1; then
    ss -lnt "sport = :$p" | grep -q ":$p" && return 1 || return 0
  elif command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1 && return 1 || return 0
  else
    return 0
  fi
}

if ! check_port "$PORT"; then
  if [ "$PORT_AUTOINC" = "1" ]; then
    echo "⚠️  Port $PORT in use; searching for a free port..."
    START_PORT="$PORT"
    for try in $(seq 1 20); do
      PORT=$((PORT+1))
      if check_port "$PORT"; then
        echo "✅ Using alternate port: $PORT"
        break
      fi
    done
    if ! check_port "$PORT"; then
      echo "❌ No free port found starting from $START_PORT. Set PORT_AUTOINC=0 and choose a port manually."
      exit 1
    fi
  else
    echo "❌ Port $PORT is in use. Set VLLM_PORT to a free port or run with PORT_AUTOINC=1."
    exit 1
  fi
fi

# Start server
echo "🎯 Starting vLLM server..."
echo "Command: $VLLM_CMD"
echo ""

exec $VLLM_CMD
