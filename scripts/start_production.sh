#!/bin/bash
"""
Production startup script for Employment Act Malaysia compliance agent.
Starts services in production mode with proper dependency management.
"""

set -euo pipefail

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Source environment
if [ -f "$PROJECT_ROOT/.env" ]; then
    echo "Loading environment from .env"
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | xargs)
else
    echo "Warning: .env file not found, using defaults"
fi

# Set defaults
export API_HOST=${API_HOST:-"0.0.0.0"}
export API_PORT=${API_PORT:-"8001"}
export VLLM_URL=${VLLM_URL:-"http://localhost:8000"}
export WORKERS=${WORKERS:-"3"}
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

echo "Starting Employment Act Malaysia API in production mode..."
echo "API Host: $API_HOST"
echo "API Port: $API_PORT"
echo "vLLM URL: $VLLM_URL"
echo "Workers: $WORKERS"

# Change to project directory
cd "$PROJECT_ROOT"

# Activate conda environment
echo "Activating conda environment..."
source "$PROJECT_ROOT/activate_env.sh"

# Wait for vLLM service if needed
if [ "${WAIT_FOR_VLLM:-true}" = "true" ]; then
    echo "Checking vLLM service availability..."
    max_retries=30
    retry_count=0
    
    while [ $retry_count -lt $max_retries ]; do
        if curl -f -s "$VLLM_URL/health" > /dev/null 2>&1; then
            echo "vLLM service is ready"
            break
        fi
        
        echo "Waiting for vLLM service... ($((retry_count + 1))/$max_retries)"
        sleep 2
        retry_count=$((retry_count + 1))
    done
    
    if [ $retry_count -eq $max_retries ]; then
        echo "ERROR: vLLM service not available after $max_retries attempts"
        echo "Please ensure vLLM is running at $VLLM_URL"
        exit 1
    fi
fi

# Start API server with gunicorn
echo "Starting FastAPI server with gunicorn..."
exec gunicorn \
    --config gunicorn.conf.py \
    --workers $WORKERS \
    --bind "$API_HOST:$API_PORT" \
    --worker-class uvicorn.workers.UvicornWorker \
    --access-logfile - \
    --error-logfile - \
    --log-level info \
    --preload \
    src.server.api:app