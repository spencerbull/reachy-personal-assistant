#!/bin/bash

echo "Starting Local vLLM Containers..."
cd local
docker compose up -d

echo ""
echo "Containers started. Use 'docker compose logs -f' in 'local' directory to monitor."
