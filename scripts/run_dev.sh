#!/bin/bash
# Development server startup script

set -e

echo "Starting Branham Model API (Development Mode)"
echo "=============================================="
echo ""

# Load development config
export CONFIG_ENV=dev

# Check if virtual environment is activated
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
fi

# Check if models are available
if [ ! -d "indices" ]; then
    echo "Warning: indices/ directory not found"
    echo "Run scripts/build_bm25_index.py and scripts/build_faiss_index.py first"
fi

# Start the server with hot reload
echo "Starting server on http://localhost:8000"
echo "API docs available at http://localhost:8000/docs"
echo ""

uv run uvicorn branham_model_api.api.main:app \
    --reload \
    --host 0.0.0.0 \
    --port 8000 \
    --log-level debug

