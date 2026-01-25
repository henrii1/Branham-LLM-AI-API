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

### Stage 4 (Partial - Needs Rebuild)
- ✅ Dense embedding + FAISS infrastructure implemented
- 🔄 **REBUILD REQUIRED**: Change embedding model from jina-v3 to Qwen3-Embedding-0.6B

---

## 🚧 In Progress

### Stage 4 Rebuild
- [ ] Update embedder for `Qwen/Qwen3-Embedding-0.6B`
- [ ] Rebuild FAISS index with new embeddings
- [ ] Validate with benchmark script

---

## 📋 Next Steps (V1)

### Phase 1: Complete Retrieval Infrastructure
1. Rebuild Stage 4 (FAISS index) with Qwen3-Embedding
2. Implement vLLM serving for embedding model (online query embedding)
3. Implement vLLM serving for reranker (conditional)
4. Implement chunk store utilities

### Phase 2: RAG Pipeline
1. Implement fusion logic (BM25 + dense)
2. Implement retrieval signals (flatness, overlap, confidence)
3. Implement date_id collation (sermon-level grouping)
4. Implement context expansion (±1/±2 chunks)
5. Implement post-check enforcement

### Phase 3: Generation Integration
1. Implement LiteLLM client
2. Implement API key rotation for rate limiting
3. Implement prompt building
4. Implement Serper tool (optional, gated)

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
✅ Data ingestion (Stages 1-3): 100%
🔄 Dense retrieval (Stage 4): 80% (needs rebuild with Qwen model)
⏳ vLLM serving setup: 0%
⏳ RAG Pipeline: 10%
⏳ LiteLLM integration: 0%
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
