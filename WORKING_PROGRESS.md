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
- ✅ **Stage 4**: Dense embedding + FAISS build pipeline implemented (`scripts/build_faiss_index.py` → `data/indices/faiss.index` + `faiss_id_map.jsonl` + `faiss_meta.json`)

---

## Status Summary

**Completed**: Environment setup, project scaffolding, API skeleton, configuration, documentation, PDF download, paragraph extraction to SQLite
**Next**: Build chunks → Build BM25 index → Build FAISS index → Training data preparation
**Progress**: Foundation complete, canonical paragraph database ready for chunking pipeline

