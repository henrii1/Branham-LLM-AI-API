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

## 2026-01-27

### Pipeline Design Refinements

Based on benchmark results (BM25: ~200ms, FAISS: ~4ms + 180ms embedding), revised the retrieval pipeline:

**Removed Early BM25 Guard**:
- BM25 is not fast enough (~200ms) for an "early guard" pattern
- Refusal decision now happens **post-fusion** based on combined retrieval quality
- Saves LLM cost when evidence is genuinely missing, but doesn't add latency to every request

**Updated Retrieval Parameters**:
- `topN = 25` for both BM25 and dense (was 20)
- Parallel retrieval: ~200ms total (max of both)

**Refined Reranker Trigger Signals**:
- Score flatness (std < 0.08) → flat distribution = ambiguous
- BM25/Dense overlap (< 3 in top-10) → retrievers disagree
- Top score threshold (< 0.65) → weak best match
- Quote intent detection → precision matters
- Removed: score gap between #1 and #2 (not useful since we add both anyway)

**Reranker's Primary Value**:
- Improves **sermon ranking accuracy** by re-scoring chunks
- Sermons ranked by best chunk score → truncate to 8 sermons
- Without reranking on ambiguous queries, might keep wrong sermons

**Collation & Expansion**:
- Cap at **8 unique sermons** (date_ids)
- Expansion: **always ±1** (fixed, not variable)
- Sermon Lookup Tool handles additional context needs during generation

**Dedup Flow**:
1. Merge BM25 + Dense → dedup by chunk_id
2. (Optional) Rerank
3. Collate by date_id, rank sermons by best chunk score
4. Keep top 8 sermons
5. Expand ±1 adjacent chunks
6. **Dedup again** (adjacent chunks may already be retrieved)

**Branch-based Inference Backend**:
- `develop` branch → Direct HuggingFace (simpler, no vLLM overhead)
- `main` branch + CUDA → vLLM (production-optimized)
- `main` branch + no CUDA → HuggingFace (fallback)
- Created `utils/git_branch.py` with detection utilities
- Added `inference_backend` config option for overrides

**Updated Files**:
- `.cursor/rules/design_spec.md` - Sections 3.4, 3.5, 6.3, 10.2, 11, 14.3, 18
- `config/default.yaml` - retrieval parameters, inference_backend
- `src/branham_model_api/utils/git_branch.py` - NEW
- `src/branham_model_api/utils/__init__.py` - exports
- `WORKING_PROGRESS.md` - this entry

---

---

## 2026-01-27 (continued)

### RAG Pipeline Implementation

Built the complete retrieval pipeline from user query to context-ready sermons:

**Files Created:**

1. `retrieval/store/chunk_store.py`
   - SQLite-backed chunk lookup
   - `get_chunk()`, `get_chunks()`, `get_chunks_by_sermon()`
   - `get_adjacent_chunks()` for expansion
   - `get_sermon()` metadata lookup

2. `core/pipeline/signals.py`
   - Retrieval signal computation
   - `detect_quote_intent()` - regex patterns for quote-seeking queries
   - `compute_dense_score_std()` - score flatness
   - `compute_overlap()` - BM25/dense agreement
   - `RetrievalSignals.should_rerank()` - trigger decision

3. `core/pipeline/fusion.py`
   - `merge_bm25_dense()` - merge + dedup by chunk_id
   - `apply_rerank_scores()` - apply reranker scores
   - `collate_by_sermon()` - group by date_id, rank sermons, cap at 8
   - `FusedHit`, `SermonGroup` dataclasses

4. `core/pipeline/expansion.py`
   - `expand_chunks()` - fetch ±1 adjacent chunks
   - `expand_and_group()` - expand + group by sermon
   - `format_sermon_context()` - format for prompt
   - `ExpandedChunk`, `ExpandedSermon` dataclasses

5. `core/pipeline/rag_pipeline.py`
   - `RAGPipeline` - main orchestrator
   - `RetrievalConfig` - all configurable thresholds
   - `RetrievalResult` - complete pipeline output
   - `EmbedderProtocol`, `RerankerProtocol` - backend-agnostic interfaces
   - `create_rag_pipeline()` - factory with all components loaded

**Pipeline Flow:**
```
Query → Normalize → [BM25 | Dense] parallel → Signals →
[Reranker if triggered] → Fusion/Dedup → Collate (8 sermons) →
Expand ±1 → Dedup → RetrievalResult
```

**Key Design Decisions:**
- No truncation until sermon collation (keep all chunks from top 8 sermons)
- Reranker improves sermon ranking accuracy
- Expansion always ±1 (Sermon Lookup Tool handles more)
- Refusal based on dense score (semantic similarity)

### Pipeline Testing & Calibration

Tested with 20 queries covering general, quote-seeking, and out-of-domain queries.

**Issues Found & Fixed:**
1. **Score scale mismatch**: BM25 (3-17) vs Dense (0.4-0.8)
   - Fix: Implemented Reciprocal Rank Fusion (RRF) instead of max()
   - RRF normalizes by rank position: score = Σ(1/(k+rank))

2. **Refusal not working**: Off-topic queries passed because BM25 always finds keyword matches
   - Fix: Refusal based on `dense_top_score` (semantic similarity)
   - Threshold: 0.55 (below = off-topic)

3. **Reranker triggers too strict**: Original thresholds triggered on every query
   - Old: std<0.08, overlap<3, top<0.65
   - New: std<0.015, overlap<1, top<0.55

**Test Results (20 queries):**
- Refused: 2 (pasta, quantum physics - correctly off-topic)
- Quote intent detected: 8 (correctly identified)
- Reranker would trigger: ~15 (quote intent + low overlap + flat scores)

**Calibrated Thresholds:**
```yaml
score_std_threshold: 0.015  # Very flat distribution
overlap_threshold: 1        # No BM25/dense agreement
top_score_threshold: 0.55   # Weak semantic match
min_dense_score: 0.55       # Off-topic refusal
```

---

## Status Summary

**Completed**: 
- Environment setup, project scaffolding, API skeleton, configuration, documentation
- PDF download, paragraph extraction to SQLite
- BM25 index build (Stage 3) ✓
- FAISS index build (Stage 4) ✓
- Pipeline design finalized (post-benchmark refinements)

**In Progress**:
- Building the API flow (RAG pipeline implementation)

**Next Steps** (API Flow Implementation):
1. `retrieval/store/chunk_store.py` — SQLite chunk store (lookup by chunk_id, date_id)
2. `core/pipeline/signals.py` — Compute retrieval signals (flatness, overlap, top score)
3. `core/pipeline/fusion.py` — Merge BM25 + dense, dedup by chunk_id
4. `core/pipeline/rerank.py` — Conditional reranker invocation
5. `core/pipeline/expansion.py` — ±1 expansion + dedup
6. `core/pipeline/postcheck.py` — Reference validation, format compliance
7. `core/pipeline/rag_pipeline.py` — Main orchestrator
8. `generation/litellm_client.py` — LiteLLM wrapper
9. `generation/api_keys.py` — API key rotation
10. Wire up `/chat` endpoint with full pipeline

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
