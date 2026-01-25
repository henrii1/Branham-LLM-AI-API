# Working Progress

Succinct log of development progress.

---

## 2025-01-08

### Environment Setup
- ✅ Installed `uv` via Homebrew
- ✅ Initialized project with `uv init` (Python 3.12.12)
- ✅ Added core dependencies: FastAPI, PyTorch 2.9.1, Transformers 4.57.3, PEFT, Accelerate
- ✅ Added retrieval stack: FAISS-CPU, rank-bm25, aiosqlite
- ✅ Added tokenization: sentencepiece (Llama/Qwen), tiktoken
- ✅ Added training tools: TRL, WandB, TensorBoard, evaluate, spacy
- ✅ Added dev tools: pytest, black, ruff, mypy
- ✅ Configured optional extras: optimum, redis, postgres
- ✅ Total packages: 127 installed
- ✅ Verified device support: MPS detected (Apple Silicon)

### Project Structure
- ✅ Created complete directory tree per `.cursorrules` Section 12
- ✅ 24 directories, 40+ files created
- ✅ All Python packages initialized with `__init__.py`
- ✅ Config system: `default.yaml`, `dev.yaml`, `prod.yaml`

### API Layer
- ✅ FastAPI application skeleton (`src/branham_model_api/api/main.py`)
- ✅ Routes: `/chat`, `/health`
- ✅ Pydantic schemas: `ChatRequest`, `ChatResponse`, `Reference`
- ✅ Middleware structure
- ✅ Verified: API imports successfully

### Core Utilities
- ✅ Device selection: `get_device()`, `get_dtype()` - supports MPS/CUDA/CPU
- ✅ System prompt template (Section 7 compliant)

### Documentation
- ✅ `.cursorrules` - Complete architecture & rules (469 lines)
- ✅ `DATA_FORMAT.md` - Data format specification
- ✅ `TRAINING_GUIDE.md` - Training workflows
- ✅ `REPOSITORY_TREE.md` - Full structure documentation
- ✅ `PROJECT_STATUS.md` - Current status & roadmap
- ✅ `SETUP_COMMANDS.md` - Command reference
- ✅ `README.md` - Updated with setup instructions

### Test Structure
- ✅ Test stubs: `test_chunking.py`, `test_fusion_dedup.py`, `test_postcheck.py`

### Configuration
- ✅ `.gitignore` - Comprehensive ignore patterns
- ✅ `.python-version` - Python 3.12
- ✅ `pyproject.toml` - Dependencies, build config, tool settings
- ✅ `uv.lock` - Locked versions (3684 lines)
- ✅ `.env.example` template (blocked by gitignore)

---

## 2025-01-09

### Dataset Acquisition
- ✅ Created PDF download script (`datasets/ingest/download_pdfs.py`) with retry logic, rate limiting, and resume capability
- ✅ Downloaded English sermon corpus from branham.org (1947-1965) - test verified, ready for full download

---

## 2025-01-10

### PDF Parsing & Paragraph Extraction
- ✅ Created `parse_to_paragraphs.py` with PyMuPDF for clean text extraction, VGR paragraph number detection, page header/footer removal, copyright filtering
- ✅ Processed 1,203 sermons → 207,061 paragraphs into `chunks.sqlite` | No duplicates, no unwanted text, avg 172 para/sermon | Date range: 47-0412 to 65-1212

---

## 2026-01-17

### Stages 1–4 (DATA_FORMAT.md) — short summary
- ✅ **Stage 1**: Canonical paragraphs in SQLite (`data/processed/chunks.sqlite`)
- ✅ **Stage 2**: Deterministic chunks built in SQLite (`chunks` table)
- ✅ **Stage 3**: BM25 build pipeline implemented (`scripts/build_bm25_index.py` → `data/indices/bm25.index`)
- ✅ **Stage 4**: Dense embedding + FAISS build pipeline implemented (`scripts/build_faiss_index.py`)
  - Built with `jinaai/jina-embeddings-v3` (512 dims, MRL truncation)
  - Output: `faiss.index`, `faiss_id_map.jsonl`, `faiss_meta.json`

---

## 2026-01-25

### V1 Architecture Revision
- ✅ Updated design spec: `.cursorrules` → `.cursor/rules/design_spec.md`
- ✅ Revised architecture for V1:
  - **Embedding**: `Qwen/Qwen3-Embedding-0.6B` via vLLM (replacing jina-embeddings-v3)
  - **Reranker**: `Qwen/Qwen3-Reranker-0.6B` via vLLM (conditional invocation)
  - **Generation**: External API via LiteLLM (V1 does NOT self-host generator)
  - **Retrieval flow**: Added sermon-level (date_id) collation before generation
  - **Tool usage**: Serper (web search) with strict constraints
- ✅ Updated `DENSE_RETRIEVAL.md` with new embedding model and vLLM serving config
- ✅ Updated `DATA_FORMAT.md` Stage 4 section
- 🔄 **Stage 4 REBUILD REQUIRED**: Embedding model changed from jina-v3 to Qwen3-Embedding

### Stage 4 Rebuild Plan
- [ ] Update `scripts/build_faiss_index.py` for Qwen3-Embedding-0.6B
- [ ] Update `src/branham_model_api/retrieval/dense/embedder.py` for Qwen model
- [ ] Rebuild FAISS index with new embeddings
- [ ] Validate with `scripts/bench_dense_faiss.py`
- [ ] Update `faiss_meta.json` to reflect new model

---

## Status Summary

**Completed**: 
- Environment setup, project scaffolding, API skeleton, configuration, documentation
- PDF download, paragraph extraction to SQLite
- BM25 index build (Stage 3) ✓
- Initial FAISS index build (Stage 4 with jina-v3) — **NEEDS REBUILD**

**In Progress**:
- V1 architecture revision (design docs updated)
- Stage 4 rebuild with Qwen3-Embedding-0.6B

**Next Steps**:
1. Rebuild Stage 4 (FAISS index) with `Qwen/Qwen3-Embedding-0.6B`
2. Implement vLLM serving for embedding model (online query embedding)
3. Implement vLLM serving for reranker (conditional)
4. Implement LiteLLM integration for generation
5. Implement sermon-level (date_id) collation in retrieval pipeline
6. Implement Serper tool integration (optional, gated)

**Future (NOT V1)**:
- Self-hosted generation model via vLLM
- LoRA/QLoRA fine-tuning (domain adaptation + instruction tuning)
- Caching layers
- Multi-GPU inference

---

## Architecture Reference (V1)

| Component   | Model                        | Serving       |
|-------------|------------------------------|---------------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          |
| Generation  | External API (configurable)  | LiteLLM       |

See `.cursor/rules/design_spec.md` for full architecture details.
