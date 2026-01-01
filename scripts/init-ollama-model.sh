#!/bin/sh
# Initialize Ollama model if it doesn't exist
# This script checks for the required model and pulls it if missing

set -e

# Model name from environment variable, default to qwen2.5:7b
MODEL_NAME="${MODEL_NAME:-qwen2.5:7b}"
# Ollama host - use service name in Docker, localhost for local
OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
export OLLAMA_HOST

echo "Checking for Ollama model: ${MODEL_NAME} (Ollama at ${OLLAMA_HOST})"

# Wait for Ollama API to be ready
echo "Waiting for Ollama API to be ready..."
MAX_RETRIES=30
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if ollama list >/dev/null 2>&1; then
        echo "Ollama API is ready"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
        echo "Error: Ollama API did not become ready after ${MAX_RETRIES} attempts" >&2
        exit 1
    fi
    echo "Waiting for Ollama API... (${RETRY_COUNT}/${MAX_RETRIES})"
    sleep 2
done

# Check if model already exists
echo "Checking if model ${MODEL_NAME} is already installed..."
if ollama list | grep -q "^${MODEL_NAME}"; then
    echo "Model ${MODEL_NAME} is already installed, skipping download"
    exit 0
fi

# Pull the model
echo "Model ${MODEL_NAME} not found. Downloading..."
if ollama pull "${MODEL_NAME}"; then
    echo "Successfully downloaded model ${MODEL_NAME}"
    exit 0
else
    echo "Error: Failed to download model ${MODEL_NAME}" >&2
    exit 1
fi

