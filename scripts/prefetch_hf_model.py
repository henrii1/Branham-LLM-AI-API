#!/usr/bin/env python3
"""
Prefetch and warm model artifacts for production deployment.

Goal:
- Download HF model weights/tokenizer/config files to disk (avoid runtime network dependency)
- Load models into memory (eliminate first-request latency)
- Pre-warm lightweight models (langid) to avoid cold-start latency

Usage (all production models - download + warm):
  uv run python scripts/prefetch_hf_model.py --all --warm --target-dir ./.hf-cache

Usage (download only, no warm):
  uv run python scripts/prefetch_hf_model.py --all --target-dir ./.hf-cache

Usage (warm only, assumes models already downloaded):
  uv run python scripts/prefetch_hf_model.py --warm-only

Notes:
- For container builds, set HF_HOME to a path inside the image and prefetch there.
- Use --warm or --warm-only to actually load models into memory (recommended for production).
- langid model is bundled with the package but needs warming (~844ms cold start).
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Production models for Branham Model API
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Required: dense retrieval
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"    # Optional: only if reranker.enabled != "never"


def _load_default_yaml() -> dict:
    """Load config/default.yaml if available, else empty dict."""
    try:
        import yaml
    except ImportError:
        return {}
    config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"
    if not config_path.exists():
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _read_reranker_mode_from_default_config() -> str | None:
    """Read retrieval.reranker.enabled from config/default.yaml."""
    raw = _load_default_yaml()
    mode = raw.get("retrieval", {}).get("reranker", {}).get("enabled")
    if isinstance(mode, str) and mode.strip().lower() in {"always", "conditional", "never"}:
        return mode.strip().lower()
    return None


def _read_langid_mode_from_default_config() -> str:
    """Read retrieval.language_detection.mode from config/default.yaml."""
    raw = _load_default_yaml()
    mode = raw.get("retrieval", {}).get("language_detection", {}).get("mode", "never")
    return str(mode).strip().lower() if mode else "never"


def warm_langid() -> None:
    """
    Pre-warm langid language detection model.
    
    langid has ~844ms cold-start latency on first call (model loading).
    Subsequent calls are ~0.12ms. This warms the model on container startup.
    """
    import time
    print("\nWarming langid language detector...")
    start = time.perf_counter()
    
    try:
        import langid
        # Trigger model loading with a simple classification
        lang, score = langid.classify("Hello world")
        elapsed_ms = (time.perf_counter() - start) * 1000
        print(f"  ✓ langid warmed in {elapsed_ms:.0f}ms (detected: {lang})")
    except ImportError:
        print("  ✗ langid not installed, skipping")
    except Exception as e:
        print(f"  ✗ langid warm failed: {e}")


def warm_embedder(model_id: str = EMBEDDING_MODEL) -> None:
    """
    Load embedding model into memory and run a test embedding.
    
    This eliminates the ~3-5 second first-request latency by loading the model
    before any user requests arrive. Uses the project's DenseEmbedder.
    """
    import time
    print(f"\nWarming embedder: {model_id}")
    start = time.perf_counter()
    
    try:
        from branham_model_api.retrieval.dense import DenseEmbedder, EmbedderConfig
        
        load_start = time.perf_counter()
        config = EmbedderConfig(
            model_id=model_id,
            query_instruction_template="Instruct: {task}\nQuery:{query}",
            query_task_description="Given a question about William Branham's teachings or sermons, retrieve relevant sermon passages that answer the query",
        )
        embedder = DenseEmbedder(config)
        load_ms = (time.perf_counter() - load_start) * 1000
        print(f"  Model loaded in {load_ms:.0f}ms (device: {embedder.device})")
        
        # Run a test embedding to fully warm the model
        embed_start = time.perf_counter()
        _ = embedder.embed_queries(["What did Brother Branham teach about faith?"])
        embed_ms = (time.perf_counter() - embed_start) * 1000
        print(f"  Test embedding in {embed_ms:.0f}ms")
        
        total_ms = (time.perf_counter() - start) * 1000
        print(f"  ✓ Embedder warmed in {total_ms:.0f}ms total")
        
        return embedder
        
    except ImportError as e:
        print(f"  ✗ branham_model_api not installed: {e}")
    except Exception as e:
        print(f"  ✗ Embedder warm failed: {e}")
    
    return None


def warm_reranker(model_id: str = RERANKER_MODEL) -> None:
    """
    Load reranker model into memory and run a test rerank.
    Uses the project's Reranker class.
    """
    import time
    print(f"\nWarming reranker: {model_id}")
    start = time.perf_counter()
    
    try:
        from branham_model_api.retrieval.reranker import Reranker, RerankerConfig
        
        load_start = time.perf_counter()
        config = RerankerConfig(model_id=model_id)
        reranker = Reranker(config)
        load_ms = (time.perf_counter() - load_start) * 1000
        print(f"  Model loaded in {load_ms:.0f}ms (device: {reranker.device})")
        
        # Run a test rerank
        rank_start = time.perf_counter()
        _ = reranker.rerank("What is faith?", ["Faith is the substance of things hoped for."])
        rank_ms = (time.perf_counter() - rank_start) * 1000
        print(f"  Test rerank in {rank_ms:.0f}ms")
        
        total_ms = (time.perf_counter() - start) * 1000
        print(f"  ✓ Reranker warmed in {total_ms:.0f}ms total")
        
        return reranker
        
    except ImportError as e:
        print(f"  ✗ branham_model_api not installed: {e}")
    except Exception as e:
        print(f"  ✗ Reranker warm failed: {e}")
    
    return None


def prefetch_model(
    model_id: str,
    target_dir: Path,
    revision: str | None = None,
    token: str | None = None,
) -> str:
    """Download a model snapshot and return the local path."""
    print(f"\nPrefetching: {model_id}")
    print(f"  Revision: {revision or '<default>'}")

    local_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=token,
        local_files_only=False,
    )

    print(f"  ✓ Downloaded to: {local_path}")
    return local_path


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Prefetch Hugging Face model snapshots to disk."
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="HF repo id (e.g., Qwen/Qwen3-Embedding-0.6B)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Prefetch all production models (embedding + reranker unless --skip-reranker)",
    )
    parser.add_argument(
        "--skip-reranker",
        action="store_true",
        help="Skip reranker model download (use when reranker.enabled=never in config)",
    )
    parser.add_argument(
        "--force-reranker",
        action="store_true",
        help="Force reranker download/warm even if config/default.yaml has reranker.enabled=never",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional git revision/tag/commit. If omitted, uses default (usually main).",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        required=True,
        help="Directory to store the HF cache (will be used as HF_HOME).",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional HF token (or set HF_TOKEN env var).",
    )
    parser.add_argument(
        "--warm",
        action="store_true",
        help="After download, load models into memory (embedder + langid, reranker if not skipped)",
    )
    parser.add_argument(
        "--warm-only",
        action="store_true",
        help="Skip download, only warm models (assumes already downloaded)",
    )
    args = parser.parse_args()

    if not args.model_id and not args.all and not args.warm_only:
        parser.error("Either --model-id, --all, or --warm-only is required")

    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all downloads go into this directory (portable for container layers).
    os.environ["HF_HOME"] = str(target_dir)

    print("=" * 70)
    print("HF Model Prefetch")
    print("=" * 70)
    print(f"HF_HOME: {target_dir}")

    # Determine what to download
    models_to_fetch = []
    should_warm = args.warm or args.warm_only
    reranker_mode = _read_reranker_mode_from_default_config()
    effective_skip_reranker = args.skip_reranker
    if (
        args.all
        and not args.force_reranker
        and not args.skip_reranker
        and reranker_mode == "never"
    ):
        effective_skip_reranker = True
        print("Auto-skip reranker: config/default.yaml sets retrieval.reranker.enabled=never")
    
    if not args.warm_only:  # Skip download if --warm-only
        if args.all:
            models_to_fetch = [EMBEDDING_MODEL]
            if not effective_skip_reranker:
                models_to_fetch.append(RERANKER_MODEL)
            else:
                print("Skipping reranker download (--skip-reranker or config auto-skip)")
            print(f"Prefetching {len(models_to_fetch)} production model(s)")
        elif args.model_id:
            models_to_fetch = [args.model_id]

        for model_id in models_to_fetch:
            prefetch_model(
                model_id=model_id,
                target_dir=target_dir,
                revision=args.revision,
                token=args.token,
            )

    langid_mode = _read_langid_mode_from_default_config()
    effective_skip_langid = langid_mode == "never"
    if effective_skip_langid:
        print("Auto-skip langid: config/default.yaml sets retrieval.language_detection.mode=never")

    # Warm models if requested
    if should_warm:
        print("\n" + "-" * 70)
        print("WARMING MODELS (loading into memory)")
        print("-" * 70)
        
        # Always warm embedder
        warm_embedder(EMBEDDING_MODEL)
        
        # Warm reranker only if not skipped
        if not effective_skip_reranker:
            warm_reranker(RERANKER_MODEL)
        else:
            print("\nSkipping reranker warm (--skip-reranker or config auto-skip)")
        
        # Warm langid only if mode != never
        if not effective_skip_langid:
            warm_langid()
        else:
            print("\nSkipping langid warm (config: language_detection.mode=never)")

    print("\n" + "=" * 70)
    if args.warm_only:
        print("✓ All models warmed")
    elif should_warm:
        print(f"✓ All models prefetched to {target_dir} AND warmed")
    else:
        print(f"✓ All models prefetched to: {target_dir}")
        print("  Note: Use --warm to also load models into memory")
    print("=" * 70)


if __name__ == "__main__":
    main()
