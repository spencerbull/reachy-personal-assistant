#!/bin/bash

# Wrapper to run the agent script from nat directory
# Usage: ./start_agent.sh [--local]

cd nat
./run_agent.sh "$@"
