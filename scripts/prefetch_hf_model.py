#!/usr/bin/env python3
"""
Prefetch Hugging Face model artifacts to disk.

Goal:
- Ensure production/deploy images have all required weights/tokenizer/config files on disk
  for fastest startup and to avoid runtime network dependency.

Usage:
  uv run python scripts/prefetch_hf_model.py --model-id jinaai/jina-embeddings-v3 --target-dir ./.hf-cache

Notes:
- This downloads a full snapshot (all files) for safety and reproducibility.
- For container builds, set HF_HOME to a path inside the image and prefetch there.
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Prefetch a Hugging Face model snapshot to disk.")
    parser.add_argument("--model-id", type=str, required=True, help="HF repo id (e.g., jinaai/jina-embeddings-v3)")
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

    target_dir = Path(args.target_dir).expanduser().resolve()
    target_dir.mkdir(parents=True, exist_ok=True)

    # Ensure all downloads go into this directory (portable for container layers).
    os.environ["HF_HOME"] = str(target_dir)

    print(f"Prefetching: {args.model_id}")
    print(f"Revision: {args.revision or '<default>'}")
    print(f"HF_HOME: {target_dir}")
    print("=" * 70)

    local_path = snapshot_download(
        repo_id=str(args.model_id),
        revision=str(args.revision) if args.revision else None,
        token=str(args.token) if args.token else None,
        local_files_only=False,
    )

    print("✓ Snapshot downloaded")
    print(f"Local snapshot path: {local_path}")


if __name__ == "__main__":
    main()

