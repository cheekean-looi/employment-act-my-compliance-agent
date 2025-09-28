#!/bin/bash
# Start vLLM inference server for Employment Act Malaysia agent
# Supports both base model and LoRA adapter serving

set -e

# Configuration from environment or defaults
MODEL_NAME=${MODEL_NAME:-"Qwen/Qwen2.5-1.5B-Instruct"}
ADAPTER_PATH=${ADAPTER_PATH:-""}
HOST=${VLLM_HOST:-"0.0.0.0"}
PORT=${VLLM_PORT:-"8000"}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-"4096"}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-"0.9"}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-"2048"}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-"1"}

echo "🚀 Starting vLLM server for Employment Act Malaysia agent"
echo "Model: $MODEL_NAME"
echo "Host: $HOST:$PORT"
echo "Max model length: $MAX_MODEL_LEN"

# Check if model exists or needs download
echo "📦 Checking model availability..."

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

# Start server
echo "🎯 Starting vLLM server..."
echo "Command: $VLLM_CMD"
echo ""

exec $VLLM_CMD
