#!/bin/bash

# Start the Bot Service
# Usage: ./start_bot.sh [--langgraph|--nat] [additional args]
#
# Options:
#   --langgraph  Use LangGraph backend (default)
#   --nat        Use NAT (NeMo Agent Toolkit) backend

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$SCRIPT_DIR/bot"

LLM_BACKEND="langgraph"

# Parse arguments
REMAINING_ARGS=()
for arg in "$@"; do
    case $arg in
        --langgraph)
            LLM_BACKEND="langgraph"
            ;;
        --nat)
            LLM_BACKEND="nat"
            ;;
        *)
            REMAINING_ARGS+=("$arg")
            ;;
    esac
done

echo "Starting Bot Service with $LLM_BACKEND backend..."
cd "$BOT_DIR"

# Sync uv venv to ensure dependencies are up-to-date
echo "Syncing bot dependencies..."
uv sync

# Run the bot with the synced environment
LLM_BACKEND=$LLM_BACKEND uv run --env-file "$SCRIPT_DIR/.env" python main.py --host 0.0.0.0 "${REMAINING_ARGS[@]}"
