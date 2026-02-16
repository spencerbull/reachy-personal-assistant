#!/bin/bash
# Start Reachy Personal Assistant - Docker Stack
#
# Usage:
#   ./start_docker.sh          # Start all services
#   ./start_docker.sh --build  # Rebuild bot image first
#   ./start_docker.sh --daemon # Also start the Reachy daemon
#
# Prerequisites:
#   1. Copy .env.template to .env and fill in API keys
#   2. NVIDIA Docker runtime installed
#   3. Sufficient GPU memory for models

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/local"

# Parse arguments
BUILD_FLAG=""
START_DAEMON=false

for arg in "$@"; do
    case $arg in
        --build)
            BUILD_FLAG="--build"
            ;;
        --daemon)
            START_DAEMON=true
            ;;
    esac
done

# Check for .env file
if [ ! -f "../.env" ]; then
    echo "Error: .env file not found!"
    echo "Copy .env.template to .env and fill in your API keys:"
    echo "  cp .env.template .env"
    exit 1
fi

# Start daemon if requested
if [ "$START_DAEMON" = true ]; then
    echo "Starting Reachy daemon on host..."
    cd "$SCRIPT_DIR"
    ./start_daemon.sh &
    DAEMON_PID=$!
    echo "Daemon started (PID: $DAEMON_PID)"
    sleep 5  # Give daemon time to start
    cd "$SCRIPT_DIR/local"
fi

# Start Docker stack
echo "Starting Docker services..."
docker compose --env-file ../.env up -d $BUILD_FLAG

echo ""
echo "============================================"
echo "Reachy Personal Assistant Stack Started"
echo "============================================"
echo ""
echo "Services:"
echo "  - agent-llm:   http://localhost:8002 (Qwen3-VL-30B)"
echo "  - routing-llm: http://localhost:8003 (Phi-3-mini)"
echo "  - bot:         http://localhost:7860 (Pipecat)"
echo ""
echo "Logs: docker compose logs -f [service]"
echo "Stop: docker compose down"
echo ""

if [ "$START_DAEMON" = false ]; then
    echo "Note: Reachy daemon must be running on host!"
    echo "Start it with: ./start_daemon.sh"
fi
