# Branham-LLM-AI-API
The AI API for William Branham Sermons

## Development Setup

This project uses [uv](https://github.com/astral-sh/uv) for dependency management and Python environment management.

### Prerequisites

- Python 3.12+ (managed by uv)
- uv package manager

### Installing uv

If you don't have uv installed, you can install it via:

**Homebrew (macOS):**
```bash
brew install uv
```

**Official installer:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Setup

1. **Run the setup script:**
   ```bash
   ./setup.sh
   ```

   Or manually:
   ```bash
   # Install Python 3.12
   uv python install 3.12
   
   # Install all dependencies
   uv sync
   ```

2. **Activate the virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

   Or use uv directly (no activation needed):
   ```bash
   uv run python -m src.api.main
   ```

### Project Structure

See `.cursor/rules/design_spec.md` for the complete V1 architecture and repository layout.

### V1 Architecture Summary

| Component   | Model                        | Serving       | Required |
|-------------|------------------------------|---------------|----------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          | Yes |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          | No (disabled by default) |
| Generation  | External API (configurable)  | LiteLLM       | Yes |
| Lang Gate   | Deterministic (see below)    | In-process    | Yes |

**Language Support (current):** English-only.

- If a user query is not English, the API streams a short, polite decline in the user’s language and asks them to re-ask in English.
- Non-English requests skip retrieval and skip tools.
- Detection is deterministic:
  - If the client provides `user_language` and it is not English, it is treated as non-English.
  - Otherwise, non-ASCII alphabetic characters (CJK / accented Latin / Cyrillic, etc.) trigger the non-English path.
  - ASCII-only non-English queries may require the client to provide `user_language`.

### Implemented Flow Status

- `/api/chat` streams via SSE for all outcomes (`answer`, `refusal`, `error`).
- Early retrieval refusal is streamed (no non-stream bypass).
- Internal tool loop is implemented with per-request limits:
  - `db_search`: 3 (hard), system prompt targets <=1 batched
  - `biography_search`: 2 (hard), system prompt targets 1
  - `internet_search`: 2 (hard), system prompt targets 1
  - Total per request: 3 (hard), system prompt targets <=2
- Post-check behavior (current):
  - No refusal gating and no markdown normalization.
  - Only appends/removes the `## Unverified / External Information` section when `internet_search` was used.
  - Retrieval-time refusal remains the primary refusal mechanism.

### Request Contract

Canonical input uses `conversation_id` (backend accepts `session_id` as legacy alias).
See `API_INPUT_FORMAT.md` for full request payload contract and ordering rules.

### Configuration

All settings are in `config/default.yaml`. Key retrieval settings:

```yaml
retrieval:
  reranker:
    enabled: never  # Options: always, conditional, never
  collation:
    max_sermons: 8
  refusal:
    min_dense_score: 0.55
```

### Production Deployment

For container startup, prefetch models AND warm them to eliminate first-request latency:

```bash
# Download models + warm (load into memory)
uv run python scripts/prefetch_hf_model.py --all --warm --skip-reranker --target-dir ./.hf-cache

# Warm only (if models already downloaded)
uv run python scripts/prefetch_hf_model.py --warm-only --skip-reranker --target-dir ./.hf-cache
```

Warmup loads:
- **Embedding model**: ~3-5 seconds (eliminates first-query latency)

### Container Image Scope (V1)

Production image should include only runtime-critical assets:
- `src/branham_model_api/`
- `config/`
- `data/indices/` (`bm25.index`, `faiss.index`, `faiss_id_map.jsonl`, `faiss_meta.json`)
- `data/processed/chunks.sqlite`
- `data/reference/biography.txt`

Do not include non-runtime assets in container image:
- `training/`
- `datasets/ingest/` pipelines and large local processing artifacts
- local debug logs, notebooks, and dev-only files

### Manual Full-Flow Debug (Non-Stream)

Use:
```bash
KMP_DUPLICATE_LIB_OK=TRUE uv run python "scripts/test_chat_full_flow.py"
```

Artifacts are overwritten per run and written to:
- `data/logs/chat_flow/latest_run.json`
- `data/logs/chat_flow/latest_answer.md`
- `data/logs/chat_flow/latest_rag_context.txt`
- `data/logs/chat_flow/latest_llm_messages.json`
- `data/logs/chat_flow/latest_llm_traces.json`

### Dependencies

All dependencies are defined in `pyproject.toml`. Key packages include:

- **API Framework:** FastAPI, Uvicorn, Gunicorn
- **ML Stack:** PyTorch, Transformers, PEFT, Accelerate
- **Retrieval:** FAISS, rank-bm25
- **Storage:** SQLite (async), optional Redis/PostgreSQL support

Optional dependencies can be installed via:
```bash
uv sync --extra gpu          # FAISS GPU support
uv sync --extra optimum      # Optimum optimizations
uv sync --extra flash-attn   # Flash attention
uv sync --extra redis        # Redis support
uv sync --extra postgres     # PostgreSQL support
```

### Development

Run tests:
```bash
uv run pytest
```

Format code:
```bash
uv run black .
uv run ruff check .
```

Type checking:
```bash
uv run mypy .
```
