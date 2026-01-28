#!/usr/bin/env python3
"""
Stage 3: Build BM25 index from chunks (chunks.sqlite -> bm25.index).

Spec references:
- `datasets/docs/DATA_FORMAT.md` (Stage 3)
- `datasets/docs/BM25_INDEX.md` (preprocessing + artifacts)

Usage (with uv):
  uv run python scripts/build_bm25_index.py
  uv run python scripts/build_bm25_index.py --db-path data/processed/chunks.sqlite --out-dir data/indices
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from branham_model_api.retrieval.bm25.index import (
    Bm25Index,
    Bm25PreprocessConfig,
    build_bm25_index_from_sqlite,
    write_bm25_meta,
)


def write_doc_map(path: Path, doc_id_to_chunk_id: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for doc_id, chunk_id in enumerate(doc_id_to_chunk_id):
            f.write(json.dumps({"doc_id": doc_id, "chunk_id": chunk_id}, ensure_ascii=False) + "\n")


def write_vocab_stats(path: Path, index: Bm25Index, *, top_k: int = 200) -> None:
    """
    Optional debugging aid: dump vocabulary stats (top terms by df).
    """
    vocab = []
    for term, (doc_ids, _tfs) in index.postings.items():
        vocab.append(
            {
                "term": term,
                "df": len(doc_ids),
                "idf": index.idf.get(term, 0.0),
            }
        )
    vocab.sort(key=lambda x: x["df"], reverse=True)
    out = {
        "doc_count": len(index.doc_id_to_chunk_id),
        "vocab_size": len(vocab),
        "top_terms_by_df": vocab[:top_k],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 3: Build BM25 index from chunks.sqlite")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Input chunks.sqlite (default: data/processed/chunks.sqlite)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Output directory (default: data/indices)",
    )
    parser.add_argument("--k1", type=float, default=1.2, help="BM25 k1 (default: 1.2)")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b (default: 0.75)")
    parser.add_argument(
        "--remove-stopwords",
        dest="remove_stopwords",
        action="store_true",
        help="Remove English stopwords (default: ON). This flag is kept for compatibility.",
    )
    parser.add_argument(
        "--keep-stopwords",
        dest="remove_stopwords",
        action="store_false",
        help="Keep stopwords (disables stopword removal). Useful for quote-intent evaluation.",
    )
    parser.set_defaults(remove_stopwords=True)
    parser.add_argument(
        "--no-keep-apostrophes",
        action="store_true",
        help="Treat apostrophes as punctuation (default keeps apostrophes in tokens)",
    )
    parser.add_argument(
        "--no-strip-paragraph-markers",
        action="store_true",
        help="Do NOT strip ¶N markers (default strips them so they don't dominate scoring)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only index the first N chunks (debug/testing)",
    )
    parser.add_argument(
        "--write-vocab-json",
        action="store_true",
        help="Write bm25_vocab.json (debugging aid; optional)",
    )
    parser.add_argument(
        "--no-metadata",
        action="store_true",
        help="Use raw text instead of text_with_metadata (not recommended).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    db_path = Path(args.db_path) if args.db_path else (repo_root / "data" / "processed" / "chunks.sqlite")
    out_dir = Path(args.out_dir) if args.out_dir else (repo_root / "data" / "indices")

    if not db_path.exists():
        print(f"Error: chunks.sqlite not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    preprocess_cfg = Bm25PreprocessConfig(
        keep_apostrophes=not args.no_keep_apostrophes,
        strip_paragraph_markers=not args.no_strip_paragraph_markers,
        remove_stopwords=bool(args.remove_stopwords),
    )

    use_metadata = not bool(args.no_metadata)
    
    print(f"Reading chunks from: {db_path}")
    print(f"Writing BM25 artifacts to: {out_dir}")
    print(f"BM25 params: k1={args.k1}, b={args.b}")
    print(f"Preprocess: {preprocess_cfg.to_dict()}")
    print(f"Text source: {'text_with_metadata (includes sermon title, date_id, ¶markers)' if use_metadata else 'raw text'}")
    if args.limit is not None:
        print(f"LIMIT: indexing first {args.limit} chunks only")
    print("=" * 70)

    index, meta = build_bm25_index_from_sqlite(
        db_path,
        k1=float(args.k1),
        b=float(args.b),
        preprocess_cfg=preprocess_cfg,
        limit=args.limit,
        use_metadata=use_metadata,
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "bm25.index"
    meta_path = out_dir / "bm25_meta.json"
    doc_map_path = out_dir / "bm25_doc_map.jsonl"
    vocab_path = out_dir / "bm25_vocab.json"

    index.save(index_path)
    write_bm25_meta(meta_path, meta)
    write_doc_map(doc_map_path, index.doc_id_to_chunk_id)
    if args.write_vocab_json:
        write_vocab_stats(vocab_path, index)

    print("✓ BM25 index written")
    print(f"  - {index_path}")
    print(f"  - {meta_path}")
    print(f"  - {doc_map_path}")
    if args.write_vocab_json:
        print(f"  - {vocab_path}")


if __name__ == "__main__":
    main()

