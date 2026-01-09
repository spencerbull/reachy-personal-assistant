#!/bin/bash

# Default to NIM (cloud) config
CONFIG_FILE="src/ces_tutorial/config.yml"
MODE="CLOUD"

# Check for --local flag
if [[ "$1" == "--local" ]]; then
    CONFIG_FILE="src/ces_tutorial/config_local.yml"
    MODE="LOCAL"
    echo "Starting NeMo Agent Service in LOCAL mode..."
    echo "Using config: $CONFIG_FILE"
    echo "Ensure your local vLLM containers are running (cd local && docker compose up)"
else
    echo "Starting NeMo Agent Service in CLOUD mode (NIM)..."
    echo "Using config: $CONFIG_FILE"
fi

# Run the agent using uv
uv run --env-file ../.env nat serve --config_file "$CONFIG_FILE" --port 8001 
