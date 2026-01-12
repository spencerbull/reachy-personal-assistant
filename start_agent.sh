#!/bin/bash

# Start the Agent Service
# Usage: ./start_agent.sh [--langgraph|--nat] [--local] [additional args]
#
# Options:
#   --langgraph  Use LangGraph agent (default)
#   --nat        Use NAT (NeMo Agent Toolkit) agent
#   --local      Use local vLLM models (NAT only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AGENT_BACKEND="langgraph"
USE_LOCAL=false

# Parse arguments
REMAINING_ARGS=()
for arg in "$@"; do
    case $arg in
        --langgraph)
            AGENT_BACKEND="langgraph"
            ;;
        --nat)
            AGENT_BACKEND="nat"
            ;;
        --local)
            USE_LOCAL=true
            ;;
        *)
            REMAINING_ARGS+=("$arg")
            ;;
    esac
done

if [[ "$AGENT_BACKEND" == "langgraph" ]]; then
    echo "Starting LangGraph Agent Service..."
    
    # Sync agent dependencies first
    cd "$SCRIPT_DIR/agent"
    echo "Syncing agent dependencies..."
    uv sync
    
    # Run LangGraph server from project root (where langgraph.json lives)
    # The langgraph.json paths are relative to the config file location
    cd "$SCRIPT_DIR"
    echo "Starting LangGraph server on port 8001..."
    
    # Load environment variables
    if [[ -f "$SCRIPT_DIR/.env" ]]; then
        set -a
        source "$SCRIPT_DIR/.env"
        set +a
    fi
    
    # Run langgraph using the agent's venv but from project root
    "$SCRIPT_DIR/agent/.venv/bin/langgraph" dev --config "$SCRIPT_DIR/langgraph.json" --port 8001 "${REMAINING_ARGS[@]}"
    
elif [[ "$AGENT_BACKEND" == "nat" ]]; then
    echo "Starting NAT (NeMo Agent Toolkit) Service..."
    cd "$SCRIPT_DIR/nat"
    
    # Sync uv venv to ensure dependencies are up-to-date
    echo "Syncing NAT dependencies..."
    uv sync
    
    # Determine config file
    if [[ "$USE_LOCAL" == true ]]; then
        CONFIG_FILE="src/ces_tutorial/config_local.yml"
        echo "Using LOCAL mode with config: $CONFIG_FILE"
        echo "Ensure your local vLLM containers are running (cd local && docker compose up)"
    else
        CONFIG_FILE="src/ces_tutorial/config.yml"
        echo "Using CLOUD mode (NIM) with config: $CONFIG_FILE"
    fi
    
    # Run NAT agent
    uv run --env-file "$SCRIPT_DIR/.env" nat serve --config_file "$CONFIG_FILE" --port 8001 "${REMAINING_ARGS[@]}"
fi
