#!/usr/bin/env python3
"""
Prefetch Hugging Face model artifacts to disk.

Goal:
- Ensure production/deploy images have all required weights/tokenizer/config files on disk
  for fastest startup and to avoid runtime network dependency.

Usage (single model):
  uv run python scripts/prefetch_hf_model.py --model-id Qwen/Qwen3-Embedding-0.6B --target-dir ./.hf-cache

Usage (all production models):
  uv run python scripts/prefetch_hf_model.py --all --target-dir ./.hf-cache

Notes:
- This downloads a full snapshot (all files) for safety and reproducibility.
- For container builds, set HF_HOME to a path inside the image and prefetch there.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


# Production models for Branham Model API
EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"  # Required: dense retrieval
RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"    # Optional: only if reranker.enabled != "never"


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
    args = parser.parse_args()

    if not args.model_id and not args.all:
        parser.error("Either --model-id or --all is required")

    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all downloads go into this directory (portable for container layers).
    os.environ["HF_HOME"] = str(target_dir)

    print("=" * 70)
    print("HF Model Prefetch")
    print("=" * 70)
    print(f"HF_HOME: {target_dir}")

    models_to_fetch = []
    if args.all:
        models_to_fetch = [EMBEDDING_MODEL]
        if not args.skip_reranker:
            models_to_fetch.append(RERANKER_MODEL)
        else:
            print("Skipping reranker (--skip-reranker flag)")
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

    print("\n" + "=" * 70)
    print(f"✓ All models prefetched to: {target_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
