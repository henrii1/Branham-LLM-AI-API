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

See `.cursorrules` for the complete architecture and repository layout.

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
