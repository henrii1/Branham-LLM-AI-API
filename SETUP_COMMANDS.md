# Setup Commands

## Environment Setup

```bash
# Install uv (if not installed)
brew install uv

# Initialize project
uv init --name branham-model-api --python 3.12

# Install dependencies
uv sync

# Install optional extras
uv sync --extra optimum     # Optimum optimizations
uv sync --extra redis        # Redis support
uv sync --extra postgres     # PostgreSQL support
uv sync --all-extras         # All extras
```

### SQLite Setup

**No additional installation required:**
- `sqlite3` is built into Python 3.12+
- `aiosqlite` is included in main dependencies (async support)
- `sqlite-utils` is included in dev dependencies (CLI tools)

SQLite database file will be stored in `data/processed/chunks.sqlite` and deployed with the API.

## Activate Environment

```bash
# Option 1: Activate virtual environment
source .venv/bin/activate

# Option 2: Use uv directly (no activation needed)
uv run python <script>
```

## Development

```bash
# Start dev server
./scripts/run_dev.sh

# Or manually
uv run uvicorn branham_model_api.api.main:app --reload --host 0.0.0.0 --port 8000

# API docs
open http://localhost:8000/docs
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=branham_model_api

# Run specific test file
uv run pytest tests/test_chunking.py
```

## Code Quality

```bash
# Format code
uv run black .

# Lint code
uv run ruff check .

# Fix linting issues
uv run ruff check --fix .

# Type check
uv run mypy .
```

## Dependency Management

```bash
# Add dependency
uv add package-name

# Add dev dependency
uv add --dev package-name

# Update all dependencies
uv lock --upgrade

# Show dependency tree
uv tree

# Show installed packages
uv pip list
```

## Training

```bash
# Download spacy model (for sentence tokenization)
uv run python -m spacy download en_core_web_sm

# Continued pretraining
uv run accelerate launch training/continued_pretrain/train_lora.py

# Instruction tuning
uv run accelerate launch training/instruction_tune/train_qa_lora.py

# Multi-GPU training
uv run accelerate launch --multi_gpu --num_processes=2 training/continued_pretrain/train_lora.py
```

## Index Building

```bash
# Build BM25 index
uv run python scripts/build_bm25_index.py

# Build FAISS index
uv run python scripts/build_faiss_index.py
```

## Utilities

```bash
# Check device support
uv run python -c "from branham_model_api.utils.device import get_device; print(get_device())"

# Verify imports
uv run python -c "from branham_model_api.api.main import app; print('✓ OK')"

# Check Python version
uv run python --version
```

