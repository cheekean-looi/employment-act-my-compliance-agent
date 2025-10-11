#!/bin/bash
# Start FastAPI server for Employment Act Malaysia agent
# Production-ready configuration with workers and monitoring

set -e

# Configuration from environment or defaults
HOST=${API_HOST:-"0.0.0.0"}
PORT=${API_PORT:-"8000"}
WORKERS=${API_WORKERS:-"1"}
LOG_LEVEL=${LOG_LEVEL:-"info"}
RELOAD=${API_RELOAD:-"false"}

# Paths
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
API_MODULE="src.server.api:app"

echo "🚀 Starting FastAPI server for Employment Act Malaysia agent"
echo "Host: $HOST:$PORT"
echo "Workers: $WORKERS"
echo "Log level: $LOG_LEVEL"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Validate configuration
echo "🔍 Validating configuration..."

# Check required environment variables
REQUIRED_PATHS=("FAISS_INDEX_PATH" "STORE_PATH")
for path_var in "${REQUIRED_PATHS[@]}"; do
    if [ -z "${!path_var}" ]; then
        echo "⚠️  Warning: $path_var not set, using default"
    else
        if [ ! -f "${!path_var}" ]; then
            echo "❌ Error: Required file not found: ${!path_var}"
            echo "💡 Please run data ingestion pipeline first:"
            echo "   python src/ingest/build_index.py"
            exit 1
        fi
    fi
done

# Check vLLM connectivity
VLLM_URL=${VLLM_BASE_URL:-"http://localhost:8000"}
echo "🔗 Checking vLLM connectivity at $VLLM_URL..."

if curl -s -f "$VLLM_URL/health" > /dev/null 2>&1; then
    echo "✅ vLLM server is ready"
else
    echo "⚠️  Warning: vLLM server not ready at $VLLM_URL"
    echo "💡 Start vLLM server first: ./scripts/run_stack_docker.sh up (or run the vLLM container manually)"
    echo "🔄 API will retry vLLM connection on first request"
fi

# Pre-validate API configuration
echo "🧪 Testing API configuration..."
if python -c "
import sys
sys.path.append('src')
from server.deps import validate_config
result = validate_config()
if not result['valid']:
    print('❌ Configuration validation failed:')
    for error in result['errors']:
        print(f'  • {error}')
    sys.exit(1)
print('✅ Configuration valid')
"; then
    echo "Configuration check passed"
else
    echo "❌ Configuration check failed"
    exit 1
fi

# Build uvicorn command
UVICORN_CMD="uvicorn $API_MODULE \
    --host $HOST \
    --port $PORT \
    --workers $WORKERS \
    --log-level $LOG_LEVEL \
    --access-log \
    --loop asyncio"

# Add reload for development
if [ "$RELOAD" = "true" ]; then
    echo "🔄 Development mode: reload enabled"
    UVICORN_CMD="$UVICORN_CMD --reload"
fi

# Production optimizations
if [ "$WORKERS" -gt 1 ]; then
    echo "⚡ Production mode: $WORKERS workers"
    UVICORN_CMD="$UVICORN_CMD --worker-class uvicorn.workers.UvicornWorker"
fi

# Health checks
echo "🏥 Health check will be available at: http://$HOST:$PORT/health"
echo "📊 Metrics will be available at: http://$HOST:$PORT/metrics"
echo "📚 API docs will be available at: http://$HOST:$PORT/docs"

# Export Python path
export PYTHONPATH="$PROJECT_ROOT/src:$PYTHONPATH"

# Start server
echo "🎯 Starting FastAPI server..."
echo "Command: $UVICORN_CMD"
echo ""

exec $UVICORN_CMD
