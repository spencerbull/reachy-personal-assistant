#!/bin/bash

# Test Local vLLM Models
# Usage: ./test_local_models.sh

set -e

# Configuration
AGENT_PORT=8002
AGENT_MODEL="Qwen/Qwen3-VL-8B-Instruct"

ROUTER_PORT=8003
ROUTER_MODEL="microsoft/Phi-3-mini-128k-instruct"

IMAGE_PORT=8004
IMAGE_MODEL="nvidia/NVIDIA-Nemotron-Nano-12B-v2-VL-NVFP4-QAD"

echo "=================================================="
echo "Testing Local vLLM Models"
echo "=================================================="

# Function to test a model
test_model() {
    local port=$1
    local model=$2
    local name=$3
    
    echo ""
    echo "--------------------------------------------------"
    echo "Testing $name ($model) on port $port..."
    echo "--------------------------------------------------"
    
    # 1. Check Health
    echo -n "Checking health... "
    if curl -s -f "http://localhost:${port}/health" > /dev/null; then
        echo "OK"
    else
        echo "FAILED (Is container running?)"
        return 1
    fi

    # 2. Test Chat Completion
    echo "Sending chat completion request..."
    response=$(curl -s "http://localhost:${port}/v1/chat/completions" \
      -H "Content-Type: application/json" \
      -d "{
        \"model\": \"$model\",
        \"messages\": [{\"role\": \"user\", \"content\": \"Hello! Respond with: I am alive.\"}],
        \"max_tokens\": 20
      }")
    
    # Extract content (simple grep/sed to avoid jq dependency if missing, but printing full json is fine too)
    echo "Response:"
    echo "$response"
    
    # Check if response contains "content" (basic validation)
    if echo "$response" | grep -q "\"content\""; then
        echo "✅ $name seems Operational"
    else
        echo "❌ $name returned unexpected response (Empty content?)"
    fi
}

# Run tests
test_model $AGENT_PORT "$AGENT_MODEL" "Agent LLM"
test_model $ROUTER_PORT "$ROUTER_MODEL" "Router LLM"
test_model $IMAGE_PORT "$IMAGE_MODEL" "Image LLM"

echo ""
echo "=================================================="
echo "Tests Completed"
echo "=================================================="
