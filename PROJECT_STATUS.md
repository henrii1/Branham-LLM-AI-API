# Project Status

## V1 Architecture (January 2026)

The project is being built as a **RAG-first system** with the following V1 architecture:

| Component   | Model                        | Serving       |
|-------------|------------------------------|---------------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          |
| Generation  | External API (configurable)  | LiteLLM       |

See `.cursor/rules/design_spec.md` for complete architecture details.

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

## 🚧 In Progress

### RAG Pipeline Implementation
Building the API flow from user query to response.

---

## 📋 Next Steps (V1)

### Phase 1: RAG Pipeline Core
1. `retrieval/store/chunk_store.py` — SQLite chunk lookup (by chunk_id, date_id)
2. `core/pipeline/signals.py` — Retrieval signals (flatness, overlap, top score)
3. `core/pipeline/fusion.py` — Merge BM25 + dense, dedup by chunk_id
4. `core/pipeline/rerank.py` — Conditional reranker (signal-triggered)
5. `core/pipeline/expansion.py` — ±1 expansion + post-expansion dedup

### Phase 2: Pipeline Orchestration
1. `core/pipeline/rag_pipeline.py` — Main orchestrator (ties all steps)
2. `core/pipeline/postcheck.py` — Reference validation, format compliance
3. `core/prompts/templates.py` — Prompt building with context

### Phase 3: Generation Integration
1. `generation/litellm_client.py` — LiteLLM wrapper for external APIs
2. `generation/api_keys.py` — API key rotation for rate limiting
3. Tool implementations (Sermon Lookup, Biography, Serper)

### Phase 4: API Integration
1. Wire up `/chat` endpoint with full pipeline
2. Implement streaming (SSE)
3. Implement health checks for all services

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
✅ Index building (BM25 + FAISS): 100%
✅ Pipeline design: 100%
⏳ RAG Pipeline implementation: 15%
⏳ Generation integration: 0%
⏳ End-to-end testing: 0%
```

---

## 📚 Key Files to Reference

- `.cursor/rules/design_spec.md` - Complete V1 implementation guide
- `datasets/docs/DATA_FORMAT.md` - Data specifications
- `datasets/docs/DENSE_RETRIEVAL.md` - Dense retrieval (Qwen3-Embedding)
- `datasets/docs/BM25_INDEX.md` - BM25 specifications
- `WORKING_PROGRESS.md` - Detailed progress log
- `config/default.yaml` - Configuration reference

---

**Foundation is solid. Stage 4 rebuild in progress. 🚀**
