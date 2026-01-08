#!/bin/bash

echo "Starting Reachy Mini Daemon (Simulation Mode)..."

cd bot
uv run -m reachy_mini.daemon.app.main --no-localhost-only "$@"
