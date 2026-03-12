#!/usr/bin/env python3
"""
Stage 4: Build embeddings + FAISS index from chunks (chunks.sqlite -> faiss.index).

Spec references:
- `datasets/docs/DATA_FORMAT.md` (Stage 4)
- `datasets/docs/DENSE_RETRIEVAL.md` (embedding + FAISS contracts)

Usage (from repo root):
  uv run python scripts/build_faiss_index.py
  uv run python scripts/build_faiss_index.py --db-path data/processed/chunks.sqlite --out-dir data/indices
"""

from __future__ import annotations

import os
import platform

# macOS note:
# Some combinations of FAISS + other native deps can trigger an OpenMP runtime duplication error.
# See `scripts/experiment_embedding_models.py` for details.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import hashlib
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import numpy as np

from branham_model_api.retrieval.dense.embedder import DenseEmbedder, EmbedderConfig
from branham_model_api.retrieval.dense.index_faiss import (
    FaissBuildConfig,
    build_faiss_index,
    faiss_index_params,
    faiss_index_type_name,
    save_faiss_index,
)


def iter_chunks_from_sqlite(
    db_path: Path,
    *,
    limit: Optional[int] = None,
    use_metadata: bool = True,
) -> Iterator[tuple[str, str]]:
    """
    Iterate over chunks from SQLite database.
    
    Args:
        db_path: Path to chunks.sqlite
        limit: Optional limit on number of chunks
        use_metadata: If True, use text_with_metadata column (includes sermon title, date_id, paragraph markers).
                      If False, use raw text column.
    """
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()
        
        # Use text_with_metadata for better retrieval (includes sermon title, date_id, ¶markers)
        text_col = "text_with_metadata" if use_metadata else "text"
        sql = f"""
        SELECT chunk_id, {text_col}
        FROM chunks
        ORDER BY date_id ASC, chunk_index ASC, chunk_id ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        for chunk_id, text in cur.fetchall():
            yield str(chunk_id), str(text or "")
    finally:
        conn.close()


def write_faiss_id_map(path: Path, chunk_ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for faiss_id, chunk_id in enumerate(chunk_ids):
            f.write(json.dumps({"faiss_id": faiss_id, "chunk_id": chunk_id}, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Stage 4: Build embeddings + FAISS index from chunks.sqlite")
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
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Embedding model id (HF). Required.",
    )
    parser.add_argument("--dim", type=int, default=1024, help="Embedding dim (MRL truncation target).")
    parser.add_argument("--dtype", type=str, default="fp16", help="fp16/bf16/fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=32768)
    parser.add_argument(
        "--pooling",
        type=str,
        default="mean",
        choices=["mean", "cls", "pooler", "last_token"],
        help="Pooling strategy. Use 'last_token' for Qwen3-Embedding (default: mean).",
    )
    parser.add_argument(
        "--padding-side",
        type=str,
        default=None,
        choices=["left", "right"],
        help="Tokenizer padding side. Use 'left' for Qwen3-Embedding. Auto-set if pooling=last_token.",
    )
    parser.add_argument(
        "--query-instruction-template",
        type=str,
        default=None,
        help="Instruction template for queries (e.g., 'Instruct: {task}\\nQuery:{query}'). "
             "Used by Qwen3-Embedding. Not applied during doc embedding.",
    )
    parser.add_argument(
        "--query-task-description",
        type=str,
        default="Given a question about William Branham's teachings or sermons, retrieve relevant sermon passages that answer the query",
        help="Task description for instruction template (replaces {task}).",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable unit-length normalization (not recommended).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    parser.set_defaults(trust_remote_code=True)

    parser.add_argument(
        "--query-prefix",
        type=str,
        default="",
        help="Prefix prepended to query texts (e.g., 'query: ' for E5).",
    )
    parser.add_argument(
        "--doc-prefix",
        type=str,
        default="",
        help="Prefix prepended to document texts (e.g., 'passage: ' for E5).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device preference: auto|mps|cuda|cpu",
    )

    parser.add_argument("--index-type", type=str, default="flatip", choices=["flatip", "hnsw", "ivf_pq"])
    parser.add_argument("--hnsw-m", type=int, default=32)
    parser.add_argument("--hnsw-ef-construction", type=int, default=200)
    parser.add_argument("--hnsw-ef-search", type=int, default=64)
    parser.add_argument("--ivf-nlist", type=int, default=1024)
    parser.add_argument("--pq-m", type=int, default=16)
    parser.add_argument("--nprobe", type=int, default=10)

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only index the first N chunks (debug/testing)",
    )
    parser.add_argument(
        "--write-embeddings-npy",
        action="store_true",
        help="Optional: write embeddings.npy for debugging (large). Not used at serving time.",
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

    model_id = args.model_id
    if not model_id:
        print(
            "Error: embedding model id not provided. Pass --model-id (e.g., Qwen/Qwen3-Embedding-0.6B).",
            file=sys.stderr,
        )
        sys.exit(2)

    trust_remote_code = True
    if bool(args.no_trust_remote_code):
        trust_remote_code = False
    elif bool(args.trust_remote_code):
        trust_remote_code = True

    embed_cfg = EmbedderConfig(
        model_id=str(model_id),
        dim=int(args.dim) if args.dim and args.dim > 0 else None,
        dtype=str(args.dtype),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        pooling=str(args.pooling),  # type: ignore[arg-type]
        normalize=not bool(args.no_normalize),
        trust_remote_code=trust_remote_code,
        query_prefix=str(args.query_prefix or ""),
        doc_prefix=str(args.doc_prefix or ""),
        device_preference=str(args.device),
        padding_side=args.padding_side,  # type: ignore[arg-type]
        query_instruction_template=args.query_instruction_template,
        query_task_description=str(args.query_task_description),
    )
    faiss_cfg = FaissBuildConfig(
        index_type=str(args.index_type),  # type: ignore[arg-type]
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
        hnsw_ef_search=int(args.hnsw_ef_search),
        ivf_nlist=int(args.ivf_nlist),
        pq_m=int(args.pq_m),
        nprobe=int(args.nprobe),
    )

    print(f"Reading chunks from: {db_path}")
    print(f"Writing FAISS artifacts to: {out_dir}")
    print(f"Embedder: model_id={embed_cfg.model_id}")
    print(f"Embed params: dim={embed_cfg.dim}, dtype={embed_cfg.dtype}, batch_size={embed_cfg.batch_size}, max_length={embed_cfg.max_length}")
    print(f"Embed params: pooling={embed_cfg.pooling}, normalize={embed_cfg.normalize}, trust_remote_code={embed_cfg.trust_remote_code}")
    if embed_cfg.query_prefix or embed_cfg.doc_prefix:
        print(f"Prefixes: query_prefix={embed_cfg.query_prefix!r}, doc_prefix={embed_cfg.doc_prefix!r}")
    print(f"FAISS: index_type={faiss_cfg.index_type} params={faiss_index_params(faiss_cfg)}")
    if args.limit is not None:
        print(f"LIMIT: indexing first {args.limit} chunks only")
    print("=" * 70)

    # Read chunks deterministically and hash raw corpus.
    use_metadata = not bool(args.no_metadata)
    corpus_hasher = hashlib.sha256()
    corpus_hash_input = "chunk_id + NUL + text_with_metadata" if use_metadata else "chunk_id + NUL + raw_text"
    chunk_ids: list[str] = []
    texts: list[str] = []
    print(f"Text source: {'text_with_metadata (includes sermon title, date_id, ¶markers)' if use_metadata else 'raw text'}")
    for chunk_id, text in iter_chunks_from_sqlite(db_path, limit=args.limit, use_metadata=use_metadata):
        corpus_hasher.update(chunk_id.encode("utf-8"))
        corpus_hasher.update(b"\x00")
        corpus_hasher.update(text.encode("utf-8"))
        corpus_hasher.update(b"\n")
        chunk_ids.append(chunk_id)
        texts.append(text)

    if not chunk_ids:
        raise RuntimeError(f"No chunks found in DB: {db_path}")

    embedder = DenseEmbedder(embed_cfg)
    tok_ml = embedder.tokenizer_max_length()
    if tok_ml is not None and embed_cfg.max_length > tok_ml:
        print(f"[WARN] embed max_length={embed_cfg.max_length} exceeds tokenizer/model max={tok_ml}. Consider lowering.")

    vectors = embedder.embed_documents(texts)
    if vectors.shape[0] != len(chunk_ids):
        raise RuntimeError(f"Embedding count mismatch: vectors={vectors.shape[0]} chunks={len(chunk_ids)}")

    # Build FAISS index.
    index = build_faiss_index(vectors, faiss_cfg)

    out_dir.mkdir(parents=True, exist_ok=True)
    index_path = out_dir / "faiss.index"
    id_map_path = out_dir / "faiss_id_map.jsonl"
    meta_path = out_dir / "faiss_meta.json"

    save_faiss_index(index, str(index_path))
    write_faiss_id_map(id_map_path, chunk_ids)

    meta: dict[str, Any] = {
        "model_id": embed_cfg.model_id,
        "dim": int(vectors.shape[1]),
        "dtype": embed_cfg.dtype,
        "normalization": "unit_length" if embed_cfg.normalize else "none",
        "pooling": embed_cfg.pooling,
        "padding_side": embed_cfg.padding_side or ("left" if embed_cfg.pooling == "last_token" else "right"),
        "max_length": int(embed_cfg.max_length),
        "query_prefix": embed_cfg.query_prefix,
        "doc_prefix": embed_cfg.doc_prefix,
        "query_instruction_template": embed_cfg.query_instruction_template,
        "query_task_description": embed_cfg.query_task_description,
        "index_type": faiss_index_type_name(index),
        "index_params": faiss_index_params(faiss_cfg),
        "corpus_hash": f"sha256:{corpus_hasher.hexdigest()}",
        "corpus_hash_input": corpus_hash_input,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "total_chunks": int(len(chunk_ids)),
    }
    write_json(meta_path, meta)

    if args.write_embeddings_npy:
        npy_path = out_dir / "embeddings.npy"
        np.save(npy_path, vectors)
        print(f"✓ embeddings saved (debug): {npy_path}")

    print("✓ FAISS index written")
    print(f"  - {index_path}")
    print(f"  - {id_map_path}")
    print(f"  - {meta_path}")


if __name__ == "__main__":
    main()

