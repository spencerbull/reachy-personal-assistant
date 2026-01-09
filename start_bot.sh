#!/bin/bash

echo "Starting Bot Service..."
cd bot
uv run --env-file ../.env python main.py --host 0.0.0.0 "$@"
