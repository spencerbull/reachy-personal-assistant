#!/bin/bash

# Start the Reachy Mini Daemon
# Usage: ./start_daemon.sh [additional args]
#
# This runs the Reachy Mini simulator daemon for development

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BOT_DIR="$SCRIPT_DIR/bot"

echo "Starting Reachy Mini Daemon (Simulation Mode)..."
cd "$BOT_DIR"

# Sync uv venv to ensure dependencies are up-to-date
echo "Syncing bot dependencies..."
uv sync

# Run the daemon
uv run -m reachy_mini.daemon.app.main --no-localhost-only "$@"
