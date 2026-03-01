FROM python:3.12-slim AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_TORCH_BACKEND=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates curl git libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir uv

COPY pyproject.toml README.md ./
COPY src ./src

# Resolve fresh with CPU-only PyTorch (UV_TORCH_BACKEND=cpu is set above).
# This skips ~4 GB of nvidia-*/triton packages.
# We intentionally omit uv.lock so uv re-resolves for CPU backend.
RUN uv lock && uv sync --frozen --no-dev

# Prefetch embedding model into the image (avoids runtime download).
# HF_HOME inside the image so the model weights are part of the layer.
ENV HF_HOME=/app/.hf-cache
COPY config/default.yaml ./config/default.yaml
COPY scripts/prefetch_hf_model.py ./scripts/prefetch_hf_model.py

RUN /app/.venv/bin/python scripts/prefetch_hf_model.py \
      --all --target-dir /app/.hf-cache

# ── Runtime stage ──────────────────────────────────────────────────
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    HF_HOME=/app/.hf-cache \
    HF_OFFLINE_ONLY=1 \
    PORT=8080

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Installed venv from builder
COPY --from=builder /app/.venv /app/.venv

# HF model cache from builder
COPY --from=builder /app/.hf-cache /app/.hf-cache

# Application code
COPY src ./src
COPY config ./config
COPY scripts/prefetch_hf_model.py ./scripts/prefetch_hf_model.py

# Static data artifacts (indices + text store + reference data)
COPY data/indices/bm25.index          ./data/indices/bm25.index
COPY data/indices/bm25_doc_map.jsonl  ./data/indices/bm25_doc_map.jsonl
COPY data/indices/bm25_meta.json      ./data/indices/bm25_meta.json
COPY data/indices/bm25_vocab.json     ./data/indices/bm25_vocab.json
COPY data/indices/faiss.index         ./data/indices/faiss.index
COPY data/indices/faiss_id_map.jsonl  ./data/indices/faiss_id_map.jsonl
COPY data/indices/faiss_meta.json     ./data/indices/faiss_meta.json
COPY data/processed/chunks.sqlite     ./data/processed/chunks.sqlite
COPY data/reference/biography.txt     ./data/reference/biography.txt

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
