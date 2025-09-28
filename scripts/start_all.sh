#!/bin/bash
# Start complete Employment Act Malaysia agent stack
# vLLM + FastAPI + Streamlit with process management

set -e

PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PIDS_FILE="$PROJECT_ROOT/.pids"

echo "🚀 Starting Employment Act Malaysia Agent Stack"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Cleanup function
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ -f "$PIDS_FILE" ]; then
        while read -r service_name pid; do
            if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
                echo "Stopping $service_name (PID: $pid)"
                kill "$pid" 2>/dev/null || true
                # Wait for graceful shutdown
                for i in {1..10}; do
                    if ! kill -0 "$pid" 2>/dev/null; then
                        break
                    fi
                    sleep 1
                done
                # Force kill if still running
                if kill -0 "$pid" 2>/dev/null; then
                    echo "Force killing $service_name"
                    kill -9 "$pid" 2>/dev/null || true
                fi
            fi
        done < "$PIDS_FILE"
        rm -f "$PIDS_FILE"
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Remove old PID file
rm -f "$PIDS_FILE"

# Function to start service in background
start_service() {
    local service_name="$1"
    local script_path="$2"
    local wait_time="$3"
    
    echo "🔄 Starting $service_name..."
    
    # Make script executable
    chmod +x "$script_path"
    
    # Start service in background
    "$script_path" &
    local pid=$!
    
    # Record PID
    echo "$service_name $pid" >> "$PIDS_FILE"
    
    echo "✅ $service_name started (PID: $pid)"
    
    # Wait for service to be ready
    if [ "$wait_time" -gt 0 ]; then
        echo "⏳ Waiting ${wait_time}s for $service_name to be ready..."
        sleep "$wait_time"
    fi
}

# Function to check service health
check_service() {
    local service_name="$1"
    local health_url="$2"
    local max_retries="$3"
    
    echo "🏥 Checking $service_name health..."
    
    for i in $(seq 1 "$max_retries"); do
        if curl -s -f "$health_url" > /dev/null 2>&1; then
            echo "✅ $service_name is healthy"
            return 0
        fi
        
        if [ "$i" -lt "$max_retries" ]; then
            echo "⏳ $service_name not ready, retrying in 5s... ($i/$max_retries)"
            sleep 5
        fi
    done
    
    echo "❌ $service_name health check failed after $max_retries attempts"
    return 1
}

# Validate environment and requirements
echo "🔍 Validating environment..."

# Check Python
if ! command -v python &> /dev/null; then
    echo "❌ Python not found. Please install Python 3.8+"
    exit 1
fi

# Check required directories
REQUIRED_DIRS=("src" "scripts")
for dir in "${REQUIRED_DIRS[@]}"; do
    if [ ! -d "$dir" ]; then
        echo "❌ Required directory not found: $dir"
        exit 1
    fi
done

# Set environment variables if not set
export VLLM_HOST=${VLLM_HOST:-"localhost"}
export VLLM_PORT=${VLLM_PORT:-"8000"}
export API_HOST=${API_HOST:-"localhost"}
export API_PORT=${API_PORT:-"8001"}
export STREAMLIT_HOST=${STREAMLIT_HOST:-"localhost"}
export STREAMLIT_PORT=${STREAMLIT_PORT:-"8501"}

# Update cross-service URLs
export VLLM_BASE_URL="http://$VLLM_HOST:$VLLM_PORT"
export API_BASE_URL="http://$API_HOST:$API_PORT"

echo "📋 Service Configuration:"
echo "  vLLM:      http://$VLLM_HOST:$VLLM_PORT"
echo "  API:       http://$API_HOST:$API_PORT"
echo "  UI:        http://$STREAMLIT_HOST:$STREAMLIT_PORT"
echo ""

# Start services in order
echo "🎬 Starting services..."
echo ""

# 1. Start vLLM server
start_service "vLLM" "./scripts/start_vllm.sh" 30

# Check vLLM health
if ! check_service "vLLM" "$VLLM_BASE_URL/health" 6; then
    echo "⚠️  vLLM health check failed, but continuing..."
    echo "💡 API will handle vLLM connection issues gracefully"
fi

# 2. Start FastAPI server
start_service "FastAPI" "./scripts/start_api.sh" 10

# Check API health
if ! check_service "FastAPI" "$API_BASE_URL/health" 3; then
    echo "❌ FastAPI startup failed"
    exit 1
fi

# 3. Start Streamlit UI
start_service "Streamlit" "./scripts/start_ui.sh" 5

# Final health check
echo ""
echo "🏥 Final health checks..."

# Check all services
ALL_HEALTHY=true

if check_service "vLLM" "$VLLM_BASE_URL/health" 1; then
    echo "  ✅ vLLM: Ready"
else
    echo "  ⚠️  vLLM: Not ready (will retry on demand)"
fi

if check_service "FastAPI" "$API_BASE_URL/health" 1; then
    echo "  ✅ API: Ready"
else
    echo "  ❌ API: Not ready"
    ALL_HEALTHY=false
fi

if curl -s -f "http://$STREAMLIT_HOST:$STREAMLIT_PORT" > /dev/null 2>&1; then
    echo "  ✅ UI: Ready"
else
    echo "  ⚠️  UI: Starting up..."
fi

echo ""
if [ "$ALL_HEALTHY" = true ]; then
    echo "🎉 Employment Act Malaysia Agent is ready!"
else
    echo "⚠️  Some services may not be fully ready"
fi

echo ""
echo "🌐 Access the application:"
echo "  • Web UI:     http://$STREAMLIT_HOST:$STREAMLIT_PORT"
echo "  • API Docs:   http://$API_HOST:$API_PORT/docs"
echo "  • API Health: http://$API_HOST:$API_PORT/health"
echo "  • vLLM API:   http://$VLLM_HOST:$VLLM_PORT/docs"
echo ""
echo "📋 Useful commands:"
echo "  • Health check: curl http://$API_HOST:$API_PORT/health"
echo "  • Test query:   curl -X POST http://$API_HOST:$API_PORT/answer -H 'Content-Type: application/json' -d '{\"query\":\"What is sick leave?\"}'"
echo ""
echo "🛑 Press Ctrl+C to stop all services"

# Keep script running and monitor services
while true; do
    sleep 10
    
    # Check if any service died
    if [ -f "$PIDS_FILE" ]; then
        while read -r service_name pid; do
            if [ -n "$pid" ] && ! kill -0 "$pid" 2>/dev/null; then
                echo "❌ Service $service_name (PID: $pid) has stopped unexpectedly"
                cleanup
                exit 1
            fi
        done < "$PIDS_FILE"
    fi
done
