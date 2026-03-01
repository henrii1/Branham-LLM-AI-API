#!/bin/sh
set -e

echo "=== Branham Model API — Container Startup ==="
echo "PORT=${PORT:-8080}"
echo "HF_HOME=${HF_HOME}"

# Warm the embedding model into memory (eliminates first-request latency).
# --warm-only skips download (model is already baked into the image).
echo ""
echo "--- Warming embedding model ---"
python scripts/prefetch_hf_model.py --warm-only --target-dir "${HF_HOME:-/app/.hf-cache}"
echo "--- Warming complete ---"
echo ""

exec uvicorn branham_model_api.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 30 \
  --log-level info
