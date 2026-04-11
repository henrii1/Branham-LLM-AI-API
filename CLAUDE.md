# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Production-grade Python API serving a RAG pipeline + multilingual generator with strict grounding for the William Branham sermon corpus. This is the only AI endpoint called by the frontend (Next.js + Supabase). English-only queries are supported; non-English queries are politely declined.

## Commands

```bash
# Install dependencies
uv sync

# Run dev server (hot reload, port 8000)
./scripts/run_dev.sh
# or manually:
uv run uvicorn branham_model_api.api.main:app --reload --host 0.0.0.0 --port 8000

# Run all tests
uv run pytest

# Run a single test file
uv run pytest tests/test_chat_sse.py -v

# Run a single test by name
uv run pytest tests/test_chat_sse.py -k "test_name" -v

# Format
uv run black .

# Lint
uv run ruff check . --fix

# Type check
uv run mypy .

# Full flow debug (non-stream, writes artifacts to data/logs/chat_flow/)
KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/test_chat_full_flow.py

# Build indices
uv run python scripts/build_bm25_index.py
uv run python scripts/build_faiss_index.py

# Prefetch and warm models (for container builds)
uv run python scripts/prefetch_hf_model.py --all --warm --skip-reranker --target-dir ./.hf-cache

# Docker
docker build -t branham-api:latest .
docker run -p 8080:8080 -e OPENROUTER_API_KEY_A=<key> branham-api:latest
```

## Architecture

### Request Flow

`POST /api/chat` (SSE streaming, bearer token auth) ->
1. Language gate (deterministic English-only check) ->
2. RAG pipeline: parallel BM25 + dense retrieval -> RRF fusion -> sermon collation ->
3. Prompt construction with RAG context ->
4. LLM generation via LiteLLM (streaming, multi-provider) ->
5. Tool loop (1-5 rounds: db_search, biography, internet_search) ->
6. Post-check (refusal validation, markdown normalization) ->
7. SSE stream: `start` -> `rag` -> `delta`... -> `final` -> `done`

### Key Modules

- **`src/branham_model_api/api/`** — FastAPI app. `main.py` creates the app with lifespan that preloads `ChatRuntime`. `routes/chat.py` is the main endpoint (~52KB, handles all streaming logic).
- **`src/branham_model_api/core/pipeline/`** — RAG orchestration. `rag_pipeline.py` drives retrieval; `fusion.py` does RRF merge + dedup + sermon collation; `signals.py` computes retrieval quality signals for conditional reranking.
- **`src/branham_model_api/core/tools/`** — Agentic tool system. `registry.py` enforces per-request call limits (3 db_search, 2 biography, 2 internet_search). `loop_runner.py` executes tool-calling rounds. Tools are created via `factory.py`.
- **`src/branham_model_api/generation/litellm_client.py`** — Multi-provider LLM wrapper (OpenRouter, DeepSeek, etc.) with streaming, retry, and API key rotation (A-E keys).
- **`src/branham_model_api/retrieval/`** — `dense/` (HF or vLLM embedder + FAISS index), `bm25/` (rank-bm25 index), `reranker/` (optional, disabled by default), `store/chunk_store.py` (SQLite text store).
- **`src/branham_model_api/core/prompts/templates.py`** — System and user prompt builders. Intent detection affects prompt addenda.
- **`src/branham_model_api/config.py`** — Dataclass-based config loaded from `config/default.yaml` with env var overrides.

### Data Layout

- `data/indices/` — BM25 + FAISS indices (prebuilt, loaded at startup)
- `data/processed/chunks.sqlite` — Canonical chunk text store (~207k chunks)
- `data/reference/biography.txt` — William Branham biography (used by biography tool)
- `config/default.yaml` — All application settings (retrieval params, LLM provider, tool limits, etc.)

### Design Constraints

- No framework-heavy RAG orchestration (no LlamaIndex/LangChain). Composition over orchestration.
- Streaming-first: all responses (answer, refusal, error) stream via SSE. No non-stream path.
- Sermon references use `date_id` + paragraph ranges as canonical locators (not char/token offsets).
- Models/embedding/reranker/generator must be configurable via config, not hard-coded.
- Supports Apple Silicon MPS (dev), CPU (production), CUDA (training).

### Environment Variables

- `CHAT_API_BEARER_KEY` — Overrides bearer key from config
- `LLM_MODEL` — Overrides LLM model selection
- `OPENROUTER_API_KEY_A` through `_E` — API key rotation
- `DEEPSEEK_API_KEY` — DeepSeek provider key
- `HF_HOME` — HuggingFace model cache (default: `.hf-cache`)
- `HF_OFFLINE_ONLY` — Force offline model loading

### Full Design Spec

See `.cursor/rules/design_spec.md` for the complete V1 architecture specification including retrieval signal logic, tool budgets, and prompt structure.
