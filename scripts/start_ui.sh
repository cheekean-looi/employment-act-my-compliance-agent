#!/bin/bash
# Start Streamlit UI for Employment Act Malaysia agent
# Production-ready configuration with proper routing

set -e

# Configuration from environment or defaults
HOST=${STREAMLIT_HOST:-"0.0.0.0"}
PORT=${STREAMLIT_PORT:-"8501"}
API_URL=${API_BASE_URL:-"http://localhost:8000"}

# Paths
PROJECT_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
UI_MODULE="src/ui/app.py"

echo "🎨 Starting Streamlit UI for Employment Act Malaysia agent"
echo "Host: $HOST:$PORT"
echo "API URL: $API_URL"
echo "Project root: $PROJECT_ROOT"

# Change to project root
cd "$PROJECT_ROOT"

# Check if UI module exists
if [ ! -f "$UI_MODULE" ]; then
    echo "❌ Error: UI module not found: $UI_MODULE"
    exit 1
fi

# Check API connectivity
echo "🔗 Checking API connectivity at $API_URL..."
if curl -s -f "$API_URL/health" > /dev/null 2>&1; then
    echo "✅ API server is ready"
else
    echo "⚠️  Warning: API server not ready at $API_URL"
    echo "💡 Start API server first: ./scripts/start_api.sh"
    echo "🔄 UI will show connection errors until API is ready"
fi

# Create Streamlit config if needed
STREAMLIT_CONFIG_DIR="$HOME/.streamlit"
STREAMLIT_CONFIG_FILE="$STREAMLIT_CONFIG_DIR/config.toml"

if [ ! -d "$STREAMLIT_CONFIG_DIR" ]; then
    mkdir -p "$STREAMLIT_CONFIG_DIR"
fi

# Write optimized Streamlit configuration
cat > "$STREAMLIT_CONFIG_FILE" << EOF
[server]
headless = true
port = $PORT
address = "$HOST"
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false

[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"

[client]
showErrorDetails = true
EOF

# Create secrets file with API URL
STREAMLIT_SECRETS_FILE="$STREAMLIT_CONFIG_DIR/secrets.toml"
cat > "$STREAMLIT_SECRETS_FILE" << EOF
API_BASE_URL = "$API_URL"
EOF

echo "📝 Streamlit configuration created"

# Set Python path
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# Build streamlit command
STREAMLIT_CMD="streamlit run $UI_MODULE \
    --server.port $PORT \
    --server.address $HOST \
    --server.headless true"

echo "🎯 Starting Streamlit UI..."
echo "🌐 UI will be available at: http://$HOST:$PORT"
echo "Command: $STREAMLIT_CMD"
echo ""

# Start UI
exec $STREAMLIT_CMD