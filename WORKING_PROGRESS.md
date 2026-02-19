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

## 2026-01-28

### Metadata Indexing & Retrieval Improvements

**Problem**: Queries by date_id (e.g., "65-0711") or sermon title didn't rank those sermons first because metadata wasn't indexed.

**Solution**: Patched chunks table and updated index builders.

**Changes to chunks.sqlite:**
- Added `sermon_title` column (populated from sermons table)
- Added `text_with_metadata` column with format:
  ```
  [Sermon: FAITH IS THE SUBSTANCE | ID: 47-0412 | ¶1-5]
  {actual text}
  ```
- Both BM25 and FAISS now index `text_with_metadata` by default
- Script: `scripts/patch_chunks_with_metadata.py`

**Composite Sermon Scoring:**
- Sermons ranked by `composite = 0.5 * norm_chunk_count + 0.5 * norm_best_score`
- Normalization: each value divided by max across all sermons
- Sermons with more retrieved chunks rank higher (breadth of relevance)

**Exact Match Fallback:**
- If query contains date_id pattern (e.g., "65-0711"), add that sermon as 9th slot
- If query matches sermon title via ILIKE, add that sermon as 9th slot
- Deduped if already in top 8

**Expansion Bug Fix:**
- Fixed: `group_expanded_by_sermon()` was re-sorting by best_score only
- Now preserves composite ranking via `sermon_order` parameter

**Expansion Config:**
- `expansion_delta` now configurable (0 = disabled, 1 = ±1 chunks)
- Set to 0 for smaller context; Sermon Lookup Tool handles additional context

**Files Modified:**
- `datasets/docs/DATA_FORMAT.md` - chunks schema
- `src/branham_model_api/retrieval/bm25/index.py` - `use_metadata` param
- `scripts/build_bm25_index.py` - `--no-metadata` flag
- `scripts/build_faiss_index.py` - `--no-metadata` flag
- `src/branham_model_api/core/pipeline/fusion.py` - composite scoring, exact match
- `src/branham_model_api/core/pipeline/expansion.py` - `sermon_order` param
- `src/branham_model_api/core/pipeline/rag_pipeline.py` - exact match integration
- `config/default.yaml` - updated thresholds

---

## 2026-01-28

### Reranker Mode Configuration

**Problem**: Reranker added ~100x latency (~10-12s/query vs ~100ms) with marginal quality improvement.

**Solution**: Made reranker configurable via `config/default.yaml`:
```yaml
reranker:
  enabled: never  # Options: always, conditional, never
```

| Mode | Behavior | Default |
|------|----------|---------|
| `never` | Skip reranking (fastest) | **Yes** |
| `conditional` | Trigger based on signals | No |
| `always` | Always rerank | No |

**Changes:**
- `config/default.yaml` - Added `reranker.enabled: never` as default
- `rag_pipeline.py` - Changed `reranker_enabled: bool` → `reranker_mode: str`
- `test_rag_pipeline.py` - Added `--reranker` flag for conditional mode
- `prefetch_hf_model.py` - Added `--skip-reranker` flag

### Exact Match Improvements

**Fixed**: Multiple exact match sermons now all promoted (not just first)
**Fixed**: Logging clarified ("already in top-8" vs "no retrieved chunks")

### Test Results (20 queries, metadata indices)

| Mode | Total Time | Accuracy |
|------|------------|----------|
| `never` | ~9 seconds | Correct #1 sermon for all title/date_id queries |
| `conditional` | ~3+ minutes | Same accuracy |

**Conclusion**: Metadata-embedded indices make reranker unnecessary for typical queries.

---

## 2026-01-29

### Language Detection & BM25 Skip for Non-English Queries

**Problem**: BM25 is keyword-based (English corpus). Non-English queries returned irrelevant BM25 hits that polluted fusion results:
- Spanish "Explica las siete edades": BM25 matched random substrings
- French "Parlez-moi de la colonne de feu": BM25 returned noise
- Chinese/Korean/Russian: BM25=0 (correctly no matches)

**Solution**: Detect query language using `langid` library and skip BM25 for non-English queries.

**Library Choice**:
| Library | Latency | Accuracy | Python 3.12 |
|---------|---------|----------|-------------|
| FastText | ~1ms | 176 langs | ✗ (numpy 2.0 issue) |
| **langid** | ~0.15ms | 97 langs | ✓ |

**Implementation**:
- Added `detect_query_language()` in `rag_pipeline.py` using `langid.classify()`
- Returns `"en"` for English (use BM25+Dense), `"non-en"` for others (Dense only)
- Cold start: ~846ms (model load) - warmed at container startup
- Per-query: ~0.15ms (negligible)

**Test Results (15 multilingual queries)**:
| Language | Queries | BM25 | Refusals |
|----------|---------|------|----------|
| Spanish | 3 | Skipped | 1 (off-topic) |
| French | 3 | Skipped | 1 (off-topic) |
| German | 2 | Skipped | 0 |
| Portuguese | 2 | Skipped | 0 |
| Chinese | 2 | Skipped | 0 |
| Korean | 1 | Skipped | 0 |
| Russian | 1 | Skipped | 0 |
| English | 1 | Used | 0 |

**Refusal verification**: French off-topic query ("Comment cuisiner des pâtes?" - How to cook pasta) correctly refused with `dense_top=0.4801 < 0.55`.

**Files Modified**:
- `src/branham_model_api/core/pipeline/rag_pipeline.py` - `detect_query_language()`, BM25 skip logic
- `scripts/prefetch_hf_model.py` - `warm_langid()` for container startup
- `scripts/test_rag_pipeline.py` - 15 multilingual test queries
- `pyproject.toml` - Added `langid` dependency
- `.cursor/rules/design_spec.md` - Section 3.6 Language Detection

---

## Status Summary

**Completed**: 
- Environment setup, project scaffolding, API skeleton, configuration, documentation
- PDF download, paragraph extraction to SQLite
- BM25 index build (Stage 3) ✓
- FAISS index build (Stage 4) with metadata ✓
- RAG pipeline core components ✓
  - `chunk_store.py` — SQLite lookup
  - `signals.py` — Retrieval signals (flatness, overlap, quote intent)
  - `fusion.py` — RRF fusion, composite scoring, exact match fallback
  - `expansion.py` — ±1 expansion with sermon order preservation
  - `rag_pipeline.py` — Main orchestrator
- Reranker integration (configurable, default disabled) ✓
- **Language detection** (langid) — BM25 skip for non-English queries ✓
- **Multilingual test harness** (15 queries: ES, FR, DE, PT, ZH, KO, RU, EN) ✓
- Model prefetch + warmup script ✓

**In Progress**:
- LiteLLM generation integration

**Next Steps**:
1. `generation/litellm_client.py` — LiteLLM wrapper
2. `generation/api_keys.py` — API key rotation
3. Wire up `/chat` endpoint with full pipeline
4. `postcheck.py` — Reference validation (optional)

**Future (NOT V1)**:
- Self-hosted generation model via vLLM
- LoRA/QLoRA fine-tuning (domain adaptation + instruction tuning)
- Caching layers
- Multi-GPU inference

---

## Architecture Reference (V1)

| Component   | Model                        | Serving       | Required |
|-------------|------------------------------|---------------|----------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          | Yes |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          | No (default: disabled) |
| Generation  | External API (configurable)  | LiteLLM       | Yes |
| Lang Detect | `langid` (bundled model)     | In-process    | Yes |

See `.cursor/rules/design_spec.md` for full architecture details.

---

## 2026-02-11

### Final Build Plan (Execution Checklist)

This checklist is the working sequence for completing V1 API delivery.

#### Phase A — Config + Contracts
- [x] Add generator config keys in `config/default.yaml`:
  - `models.llm_model: openrouter/deepseek/deepseek-chat`
  - optional prompt/template overrides for system behavior
- [x] Update config loader (`src/branham_model_api/config.py`) to expose generation settings.
- [x] Implement env override: `LLM_MODEL` supersedes YAML value at runtime.
- [x] Align API schemas for final modes (`answer|refusal|error`) and summary fields.
- [x] Keep retrieval expansion as hyper-parameter (default lean context; DB tool for deep fetch).

#### Phase B — Generation Module (LiteLLM + Key Rotation)
- [x] Create `src/branham_model_api/generation/litellm_client.py`.
- [x] Create `src/branham_model_api/generation/api_keys.py`:
  - lookup keys `OPENROUTER_API_KEY_A` ... `OPENROUTER_API_KEY_E`
  - equal-probability key selection
  - retry once with a different key for rate-limit errors only
- [x] Wire OpenRouter auth/header handling through LiteLLM.
- [ ] Add health validation for generation availability (key/model configured).

#### Phase C — Prompting + Context Assembly
- [x] Create `src/branham_model_api/core/prompts/templates.py`.
- [x] Build sermon-group context template:
  - per-sermon metadata block (`sermon_title`, `date_id`)
  - chunk lines with paragraph start/end and chunk ids
  - resilient to `expansion.depth` on/off
- [x] Enforce reference style in prompt:
  - inline canonical references (no paragraph suffix letters)
  - consolidated references block
  - add final note pointing users to The Table app
- [x] Retrieval input strategy:
  - use `query + conversation_summary` for retrieval embedding
  - include short chat history in LLM prompt context for dialogue continuity

#### Phase D — Tooling (Looped, Stateless API)
- [x] Implement tool registry and loop runner (server-internal only).
- [x] Add DB Search tool (max 2 calls/request):
  - single paragraph
  - contiguous paragraph range
  - batched ranges in same sermon
  - batched ranges across multiple sermons
  - return grouped per sermon for prompt injection
- [x] Add Biography tool (max 1 call/request) backed by a local biography text file.
- [x] Add Serper tool (max 2 calls/request), gated and labeled as unverified external data.
- [x] Append tool outputs iteratively into model context until model returns final assistant answer.

#### Phase E — API Route + Streaming
- [x] Implement `/api/chat` full orchestrator:
  - retrieval -> refusal gate -> generation/tool loop -> final response
- [x] SSE for all responses (including refusal and early-stop paths).
- [x] Emit final non-stream payload fields where needed (optional conversation summary, optional external info).
- [x] Keep API stateless across requests.

#### Phase F — Post-check + Policy Enforcement
- [x] Implement `postcheck.py` for:
  - reference presence and canonical format
  - evidence alignment
  - refusal fallback when checks fail
- [x] Bible exception behavior:
  - allow answer for Bible-focused queries even when retrieval confidence is weaker
  - keep answer style grounded and clearly cited
- [x] External-data policy:
  - include `Unverified / External Information` section only when Serper tool is used

#### Phase G — Tests + Developer Tooling
- [x] Keep `scripts/test_rag_pipeline.py` retrieval-only.
- [x] Create new full-flow test script (non-stream) for generation + tools.
- [x] Add API integration test for SSE behavior (including refusal path).
- [x] Validate multilingual behavior, references formatting, and tool-call limits.

#### Phase H — Docs + Deployment Scope
- [x] Update `README.md` and `PROJECT_STATUS.md` with finalized generation/tool flow.
- [x] Ensure docs specify minimal V1 container contents only:
  - required runtime code/config/data indices
  - exclude training and non-runtime assets from container image
- [x] Verify rule/doc consistency with `.cursor/rules/design_spec.md`.

#### Phase I — Latency Bottleneck Elimination (Pre-Deployment)
- [x] Run end-to-end latency profiling on local HTTP SSE path:
  - TTFT (time to first `delta`)
  - total time to `final`
  - retrieval-only latency (cold/warm)
- [x] Remove blocking latency before first streamed token in normal answer/refusal flow.
  - **Streaming-first architecture:** every LLM call is now streamed; text tokens forwarded immediately.
  - **Eliminated double LLM call:** old flow called LLM non-streaming (tool loop) then re-streamed.
  - TTFT for LLM-powered answers: **10–18s** (down from 33–42s, ~60% improvement).
- [x] Profile and reduce Python-side bottlenecks in retrieval and pre-generation orchestration.
  - **Parallel BM25+Dense retrieval** via `ThreadPoolExecutor` (~8ms saved).
  - **Configurable langid** (`retrieval.language_detection.mode: never`): saves ~800ms cold start.
  - Retrieval warm path: 48–95ms, negligible.
- [x] Define acceptance latency targets (local + Cloud Run) before rollout.
  - Early retrieval refusal: < 100ms ✓
  - LLM-powered answer/refusal: < 20s local ✓ (provider TTFT + tool iterations)
  - Remaining bottleneck: tool-call iterations (~5–10s each) and provider first-token latency.
- [x] Track fixes in `LATENCY_IMPROVEMENT_PLAN.md`.

#### Final Step — Cloud Run Deployment
- [ ] Deploy to Google Cloud Run (final rollout step).
- [ ] Set Cloud Run `max instances` to `5` (cost-aware baseline; tune later).
- [ ] Add detailed Cloud Run parameters (concurrency/min instances/timeouts) in follow-up deployment notes.

---

## 2026-02-19

### English-only language gate (current product decision)
- ✅ Non-English queries are declined in the user’s language and streamed via SSE (no retrieval, no tools).
- ✅ Deterministic detection:
  - `user_language` hint (if provided) is authoritative.
  - Otherwise, non-ASCII alphabetic characters trigger non-English handling.
  - ASCII-only non-English queries require the frontend to provide `user_language`.
- ✅ `scripts/test_chat_full_flow.py` matches the API gate behavior for non-stream debugging.
