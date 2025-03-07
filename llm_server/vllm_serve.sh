#!/bin/bash

# Load environment variables from .env file
if [ -f ./.env ]; then
    export $(grep -v '^#' ./.env | xargs)
fi

# If MODEL_PATH is not set in .env, use default
if [ -z "$MODEL_PATH" ]; then
    MODEL_PATH=zifeng-ai/leads-mistral-7b-v1
    echo "MODEL_PATH not found in .env, using default: $MODEL_PATH"
else
    echo "Using MODEL_PATH from .env: $MODEL_PATH"
fi

# Kill any existing process on the port
PORT=${PORT:-13141}
kill -9 $(lsof -t -i:$PORT) 2>/dev/null || true

# Set CUDA device
CUDA_DEVICE=${CUDA_DEVICE:-0}
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE

# Start vLLM server
CONFIG_PATH=${CONFIG_PATH:-"llm_server/vllm_config.yaml"}
nohup vllm serve $MODEL_PATH --config $CONFIG_PATH > vllm.log 2>&1 &
