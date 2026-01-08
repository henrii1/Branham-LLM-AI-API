# Repository Structure

Complete directory layout as per `.cursorrules` Section 12.

## Project Root

```
Branham-LLM-AI-API/
├── .cursorrules                  # Development rules & architecture guide
├── .gitignore                    # Git ignore patterns
├── .python-version               # Python version (3.12)
├── pyproject.toml                # Project dependencies & configuration
├── uv.lock                       # Locked dependency versions
├── README.md                     # Project documentation
├── LICENSE                       # MIT License
├── REPOSITORY.md                 # This file
├── DEPENDENCIES.md               # Complete dependency list
├── SETUP_COMPLETE.md             # Setup guide
│
├── config/                       # Configuration files
│   ├── default.yaml              # Default configuration
│   ├── dev.yaml                  # Development overrides
│   └── prod.yaml                 # Production overrides
│
├── src/branham_model_api/        # Main Python package
│   ├── __init__.py
│   │
│   ├── api/                      # FastAPI application
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI app entry point
│   │   ├── routes/               # API endpoints
│   │   │   ├── __init__.py
│   │   │   ├── chat.py           # POST /chat endpoint
│   │   │   └── health.py         # GET /health endpoint
│   │   ├── schemas/              # Request/Response schemas
│   │   │   ├── __init__.py
│   │   │   ├── request.py        # ChatRequest model
│   │   │   └── response.py       # ChatResponse model
│   │   └── middleware/           # Middleware components
│   │       ├── __init__.py
│   │       └── logging.py        # Logging middleware (TODO)
│   │
│   ├── core/                     # Core RAG pipeline
│   │   ├── __init__.py
│   │   ├── pipeline/             # Pipeline components
│   │   │   ├── __init__.py
│   │   │   ├── rag_pipeline.py   # Main RAG orchestrator (TODO)
│   │   │   ├── fusion.py         # BM25+Dense fusion (TODO)
│   │   │   ├── rerank.py         # Conditional reranker (TODO)
│   │   │   ├── expansion.py      # Context expansion (TODO)
│   │   │   ├── postcheck.py      # Post-generation checks (TODO)
│   │   │   └── signals.py        # Retrieval signals (TODO)
│   │   ├── prompts/              # Prompt templates
│   │   │   ├── __init__.py
│   │   │   ├── system_prompt.txt # System prompt template
│   │   │   └── templates.py      # Prompt builder (TODO)
│   │   └── refs/                 # Reference formatting
│   │       ├── __init__.py
│   │       └── format_refs.py    # Reference renderer (TODO)
│   │
│   ├── retrieval/                # Retrieval subsystem
│   │   ├── __init__.py
│   │   ├── bm25/                 # BM25 sparse retrieval
│   │   │   ├── __init__.py
│   │   │   ├── index.py          # BM25 indexer (TODO)
│   │   │   └── query.py          # BM25 querier (TODO)
│   │   ├── dense/                # Dense retrieval
│   │   │   ├── __init__.py
│   │   │   ├── embedder.py       # Embedding generator (TODO)
│   │   │   ├── index_faiss.py    # FAISS index builder (TODO)
│   │   │   └── query.py          # FAISS querier (TODO)
│   │   └── store/                # Text chunk storage
│   │       ├── __init__.py
│   │       └── chunk_store.py    # Chunk store interface (TODO)
│   │
│   ├── models/                   # Model loading & inference
│   │   ├── __init__.py
│   │   ├── generator/            # LLM generation
│   │   │   ├── __init__.py
│   │   │   ├── load.py           # Model loader (TODO)
│   │   │   └── infer.py          # Inference engine (TODO)
│   │   └── reranker/             # Reranking model
│   │       ├── __init__.py
│   │       ├── load.py           # Reranker loader (TODO)
│   │       └── infer.py          # Reranker inference (TODO)
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── device.py             # Device selection (MPS/CUDA/CPU)
│       ├── batching.py           # Batching utilities (TODO)
│       └── timing.py             # Performance timing (TODO)
│
├── datasets/                     # Dataset preparation
│   ├── __init__.py
│   ├── ingest/                   # Data ingestion
│   │   ├── __init__.py
│   │   ├── parse_sermons.py      # Sermon parser (TODO)
│   │   ├── normalize.py          # Text normalization (TODO)
│   │   └── build_chunks.py       # Chunking logic (TODO)
│   ├── export/                   # Data export
│   │   ├── __init__.py
│   │   ├── to_jsonl.py           # JSONL exporter (TODO)
│   │   └── upload_dataset.py     # HF dataset uploader (TODO)
│   └── docs/                     # Documentation
│       └── DATA_FORMAT.md        # Data format specification
│
├── training/                     # Model training
│   ├── __init__.py
│   ├── continued_pretrain/       # Continued pretraining
│   │   ├── __init__.py
│   │   ├── train_lora.py         # LoRA training script (TODO)
│   │   └── accelerate_config.yaml # Accelerate config (TODO)
│   ├── instruction_tune/         # Instruction tuning
│   │   ├── __init__.py
│   │   ├── build_qa.py           # Q/A dataset builder (TODO)
│   │   └── train_qa_lora.py      # Q/A LoRA training (TODO)
│   ├── eval/                     # Evaluation
│   │   ├── __init__.py
│   │   ├── retrieval_eval.py     # Retrieval metrics (TODO)
│   │   └── generation_eval.py    # Generation metrics (TODO)
│   └── docs/                     # Training documentation
│       └── TRAINING_GUIDE.md     # Training guide
│
├── scripts/                      # Utility scripts
│   ├── build_bm25_index.py       # Build BM25 index (TODO)
│   ├── build_faiss_index.py      # Build FAISS index (TODO)
│   └── run_dev.sh                # Development server script
│
├── tests/                        # Test suite
│   ├── __init__.py
│   ├── test_chunking.py          # Chunking tests
│   ├── test_fusion_dedup.py      # Fusion/dedup tests
│   └── test_postcheck.py         # Post-check tests
│
├── data/                         # Data directory (gitignored)
│   ├── raw/                      # Raw sermon files
│   ├── processed/                # Processed chunks
│   └── chunks.db                 # SQLite chunk store
│
├── indices/                      # Index files (gitignored)
│   ├── bm25.pkl                  # BM25 index
│   └── faiss.index               # FAISS index
│
└── models/                       # Model files (gitignored)
    ├── cache/                    # HF model cache
    └── adapters/                 # LoRA adapters
        ├── continued_pretrain/
        └── qa_instruction/
```

## Module Responsibilities

### API Layer (`src/branham_model_api/api/`)
- FastAPI application setup
- Request/response validation
- Endpoint handlers
- Middleware (logging, CORS, etc.)

### Core Pipeline (`src/branham_model_api/core/`)
- RAG pipeline orchestration
- Retrieval fusion & deduplication
- Conditional reranking
- Context expansion (±1/±2 chunks)
- Post-generation enforcement
- Prompt template management
- Reference formatting

### Retrieval (`src/branham_model_api/retrieval/`)
- BM25 sparse retrieval
- Dense retrieval (FAISS)
- Chunk storage (SQLite/Redis/PostgreSQL)
- Hybrid search coordination

### Models (`src/branham_model_api/models/`)
- Generator model loading & inference
- Reranker model loading & inference
- LoRA adapter management
- Device-aware loading

### Utilities (`src/branham_model_api/utils/`)
- Device selection (MPS/CUDA/CPU)
- Batching utilities
- Performance timing
- Common helpers

### Datasets (`datasets/`)
- Sermon parsing & normalization
- Chunking implementation (Section 5.1)
- JSONL export for training
- Dataset upload to Hugging Face

### Training (`training/`)
- Continued pretraining LoRA
- Instruction tuning LoRA
- Evaluation scripts
- Multi-GPU support via Accelerate

### Scripts (`scripts/`)
- Index building (BM25, FAISS)
- Development server
- Utility scripts

### Tests (`tests/`)
- Unit tests for all components
- Integration tests for pipeline
- Compliance tests for rules

## Implementation Status

✅ **Complete**:
- Project structure
- Configuration files
- API skeleton (schemas, routes)
- Device utilities
- System prompt template
- Documentation (DATA_FORMAT, TRAINING_GUIDE)
- Test stubs

🚧 **TODO**:
- All pipeline components
- Retrieval implementations
- Model loaders
- Dataset preparation
- Training scripts
- Index builders
- Remaining tests

## Next Steps

1. Implement chunking logic (`datasets/ingest/build_chunks.py`)
2. Create dataset preparation pipeline
3. Build indices (`scripts/build_bm25_index.py`, `scripts/build_faiss_index.py`)
4. Implement retrieval components
5. Build RAG pipeline
6. Add model loaders
7. Implement training scripts
8. Write comprehensive tests

See `.cursorrules` for detailed implementation requirements!

