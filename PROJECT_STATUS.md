# Project Status

## V1 Architecture (January 2026)

The project is being built as a **RAG-first system** with the following V1 architecture:

| Component   | Model                        | Serving       | Required |
|-------------|------------------------------|---------------|----------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          | Yes |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          | No (disabled by default) |
| Generation  | External API (configurable)  | LiteLLM       | Yes |

See `.cursor/rules/design_spec.md` for complete architecture details.
See `config/default.yaml` for all configuration options.

---

## ✅ Completed

### Repository Structure
- ✅ Complete directory structure (24 directories)
- ✅ All Python package `__init__.py` files
- ✅ Configuration system (default/dev/prod YAML files)
- ✅ FastAPI application skeleton
- ✅ Request/Response schemas (Pydantic models)
- ✅ API routes (`/chat`, `/health`)

### Utilities & Helpers
- ✅ Device selection utilities (MPS/CUDA/CPU detection)
- ✅ System prompt template
- ✅ Development server script (`scripts/run_dev.sh`)

### Documentation
- ✅ `.cursor/rules/design_spec.md` - Complete V1 architecture specification
- ✅ `DATA_FORMAT.md` - Data format specification
- ✅ `BM25_INDEX.md` - BM25 build specification
- ✅ `DENSE_RETRIEVAL.md` - Dense retrieval specification (updated for Qwen3-Embedding)
- ✅ `TRAINING_GUIDE.md` - Training workflow documentation (future)

### Data Pipeline (Stages 1-3)
- ✅ **Stage 0**: PDF download script with retry logic
- ✅ **Stage 1**: Canonical paragraphs in SQLite (207,061 paragraphs from 1,203 sermons)
- ✅ **Stage 2**: Deterministic chunks built in SQLite
- ✅ **Stage 3**: BM25 index built (`data/indices/bm25.index`)

### Stage 4 (Complete)
- ✅ Dense embedding + FAISS infrastructure implemented
- ✅ FAISS index built with Qwen3-Embedding-0.6B

---

## ✅ Recently Completed

### RAG Pipeline Core (2026-01-27/28)
- ✅ `retrieval/store/chunk_store.py` — SQLite chunk lookup
- ✅ `core/pipeline/signals.py` — Retrieval signals (flatness, overlap, quote intent)
- ✅ `core/pipeline/fusion.py` — RRF fusion, composite scoring, exact match fallback
- ✅ `core/pipeline/expansion.py` — ±1 expansion with sermon order preservation
- ✅ `core/pipeline/rag_pipeline.py` — Main orchestrator
- ✅ `config.py` — Config loader (reads from `config/default.yaml`)
- ✅ Reranker integration (configurable: never/conditional/always, default: never)
- ✅ Test harness with 20 queries (`scripts/test_rag_pipeline.py`)

---

## 🚧 In Progress

### LiteLLM Generation Integration
Building the generation layer with external API support.

---

## 📋 Next Steps (V1)

### Phase 1: Generation Integration
1. `generation/litellm_client.py` — LiteLLM wrapper for external APIs
2. `generation/api_keys.py` — API key rotation for rate limiting
3. `core/prompts/templates.py` — Prompt building with context

### Phase 2: API Integration
1. Wire up `/chat` endpoint with full pipeline
2. Implement streaming (SSE)
3. `core/pipeline/postcheck.py` — Reference validation (optional)

### Phase 3: Tools (Optional)
1. Sermon Lookup Tool
2. Biography Tool
3. Serper Tool

---

## ⏳ Future (NOT V1)

- Self-hosted generation model via vLLM
- LoRA/QLoRA fine-tuning
- Caching layers
- Multi-GPU inference

---

## 📊 Current State

```
✅ Project scaffolding: 100%
✅ Configuration: 100%
✅ API skeleton: 100%
✅ Documentation: 100%
✅ Data ingestion (Stages 1-4): 100%
✅ Index building (BM25 + FAISS with metadata): 100%
✅ Pipeline design: 100%
✅ RAG Pipeline implementation: 100%
✅ Config loader: 100%
⏳ Generation integration: 0%
⏳ End-to-end testing: 0%
```

---

## 📚 Key Files to Reference

- `.cursor/rules/design_spec.md` - Complete V1 implementation guide
- `config/default.yaml` - All configuration options
- `src/branham_model_api/config.py` - Config loader
- `src/branham_model_api/core/pipeline/` - RAG pipeline components
- `scripts/test_rag_pipeline.py` - Pipeline test harness
- `datasets/docs/DATA_FORMAT.md` - Data specifications
- `WORKING_PROGRESS.md` - Detailed progress log

---

**RAG pipeline complete. LiteLLM integration next. 🚀**
