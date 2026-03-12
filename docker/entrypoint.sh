#!/bin/sh
set -e

echo "=== Branham Model API — Container Startup ==="
echo "PORT=${PORT:-8080}"
echo "HF_HOME=${HF_HOME}"

exec uvicorn branham_model_api.api.main:app \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --workers 1 \
  --timeout-keep-alive 30 \
  --log-level info
