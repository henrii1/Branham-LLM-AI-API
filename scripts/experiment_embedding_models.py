#!/usr/bin/env python3
"""
Lightweight embedding-model experiment runner.

Goal:
- Compare candidate multilingual embedding models for this project.
- Surface max sequence length, embedding throughput, and a small intrinsic retrieval proxy.

Notes:
- This is NOT a full IR benchmark (we don't yet have curated query->relevant labels).
- The proxy metrics are still useful for "does the model behave sanely?" and for relative comparisons.
"""

from __future__ import annotations

import os
import platform

# macOS note:
# Some combinations of FAISS + other native deps can trigger:
#   "OMP: Error #15: Initializing libomp.dylib, but found libomp.dylib already initialized."
# This is a known OpenMP runtime duplication issue on macOS.
# Workaround: allow duplicate OpenMP runtime and keep threads low for stability.
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from branham_model_api.retrieval.dense.embedder import DenseEmbedder, EmbedderConfig
from branham_model_api.retrieval.dense.index_faiss import FaissBuildConfig, build_faiss_index
from branham_model_api.retrieval.dense.query import faiss_search


def _date_id_from_chunk_id(chunk_id: str) -> str:
    # chunk_id format is locked: {date_id}_chunk_{index}
    return chunk_id.split("_chunk_", 1)[0]


def _first_words(text: str, n: int = 40) -> str:
    toks = [t for t in (text or "").split() if t.strip()]
    if not toks:
        return ""
    return " ".join(toks[:n])


def _iter_chunks(db_path: Path, *, limit: Optional[int]) -> list[tuple[str, str]]:
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()
        sql = """
        SELECT chunk_id, text
        FROM chunks
        ORDER BY date_id ASC, chunk_index ASC, chunk_id ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)
        return [(str(cid), str(txt or "")) for (cid, txt) in cur.fetchall()]
    finally:
        conn.close()


def _default_prefixes_for_model(model_id: str) -> tuple[str, str]:
    mid = model_id.lower()
    # E5 family typically expects explicit query/passage prefixes.
    if "e5" in mid:
        return ("query: ", "passage: ")
    return ("", "")


def _default_pooling_for_model(model_id: str) -> str:
    mid = model_id.lower()
    # Many BGE baselines use CLS pooling; keep configurable if needed.
    if "bge" in mid:
        return "cls"
    return "mean"


@dataclass(frozen=True)
class ModelRun:
    model_id: str
    dim: int | None
    max_length: int
    dtype: str
    batch_size: int
    pooling: str
    trust_remote_code: bool
    device: str
    query_prefix: str
    doc_prefix: str


def _evaluate_proxy(
    *,
    model: DenseEmbedder,
    chunk_ids: list[str],
    texts: list[str],
    top_k: int,
) -> dict[str, Any]:
    # Build an ANN index for the proxy evaluation to keep runtime predictable.
    doc_vecs = model.embed_documents(texts)
    index = build_faiss_index(
        doc_vecs,
        FaissBuildConfig(index_type="hnsw", hnsw_m=32, hnsw_ef_construction=200, hnsw_ef_search=64),
    )

    queries = [_first_words(t, 40) for t in texts]
    q_vecs = model.embed_queries(queries)
    hits = faiss_search(index, q_vecs, top_n=top_k)

    self_at_1 = 0
    self_at_k = 0
    same_sermon_at_k = 0
    score_gap_sum = 0.0
    score_gap_cnt = 0

    for i, hlist in enumerate(hits):
        if not hlist:
            continue
        faiss_ids = [h.faiss_id for h in hlist]
        if faiss_ids[0] == i:
            self_at_1 += 1
        if i in faiss_ids:
            self_at_k += 1

        did = _date_id_from_chunk_id(chunk_ids[i])
        got_same = any(_date_id_from_chunk_id(chunk_ids[fid]) == did for fid in faiss_ids)
        if got_same:
            same_sermon_at_k += 1

        if len(hlist) >= 2:
            score_gap_sum += float(hlist[0].score - hlist[1].score)
            score_gap_cnt += 1

    n = len(chunk_ids)
    return {
        "n": n,
        "top_k": top_k,
        "self_recall_at_1": self_at_1 / n if n else 0.0,
        "self_recall_at_k": self_at_k / n if n else 0.0,
        "same_sermon_recall_at_k": same_sermon_at_k / n if n else 0.0,
        "avg_score_gap_top1_top2": (score_gap_sum / score_gap_cnt) if score_gap_cnt else None,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Experiment: compare embedding models for dense retrieval")
    parser.add_argument(
        "--db-path",
        type=str,
        default=None,
        help="Input chunks.sqlite (default: data/processed/chunks.sqlite)",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="jinaai/jina-embeddings-v4,jinaai/jina-embeddings-v3,BAAI/bge-m3,intfloat/multilingual-e5-large",
        help="Comma-separated HF model ids to test.",
    )
    parser.add_argument("--limit", type=int, default=2000, help="Number of chunks to sample (deterministic prefix).")
    parser.add_argument("--top-k", type=int, default=10, help="Top-k for proxy retrieval metric.")
    parser.add_argument("--dtype", type=str, default="fp16", help="fp16/bf16/fp32")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--device", type=str, default="auto", help="auto|mps|cuda|cpu")
    parser.add_argument(
        "--dim",
        type=int,
        default=512,
        help="Embedding dim (used for jina v3 MRL-style truncation; others default to full dim unless --force-dim).",
    )
    parser.add_argument(
        "--force-dim",
        action="store_true",
        help="Apply --dim truncation to ALL models (not recommended unless the model supports MRL).",
    )
    parser.add_argument("--trust-remote-code", action="store_true", help="Enable trust_remote_code.")
    parser.add_argument("--no-trust-remote-code", action="store_true", help="Disable trust_remote_code.")
    parser.set_defaults(trust_remote_code=True)
    parser.add_argument(
        "--out-json",
        type=str,
        default=None,
        help="Optional path to write JSON results (default: data/experiments/embedding_benchmark.json).",
    )

    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    db_path = Path(args.db_path) if args.db_path else (repo_root / "data" / "processed" / "chunks.sqlite")
    if not db_path.exists():
        raise SystemExit(f"chunks.sqlite not found: {db_path}")

    out_path = Path(args.out_json) if args.out_json else (repo_root / "data" / "experiments" / "embedding_benchmark.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model_ids = [m.strip() for m in str(args.models).split(",") if m.strip()]
    rows = _iter_chunks(db_path, limit=int(args.limit) if args.limit else None)
    if not rows:
        raise SystemExit("No chunks found in chunks.sqlite (is Stage 2 complete?)")

    chunk_ids = [cid for cid, _ in rows]
    texts = [txt for _, txt in rows]

    trust_remote_code = True
    if bool(args.no_trust_remote_code):
        trust_remote_code = False
    elif bool(args.trust_remote_code):
        trust_remote_code = True

    results: dict[str, Any] = {
        "db_path": str(db_path),
        "sample_chunks": int(len(chunk_ids)),
        "top_k": int(args.top_k),
        "runs": [],
    }

    for model_id in model_ids:
        qp, dp = _default_prefixes_for_model(model_id)
        pooling = _default_pooling_for_model(model_id)
        dim: int | None = None
        if args.force_dim:
            dim = int(args.dim)
        else:
            if "jina-embeddings-v3" in model_id.lower():
                dim = int(args.dim)

        run = ModelRun(
            model_id=model_id,
            dim=dim,
            max_length=int(args.max_length),
            dtype=str(args.dtype),
            batch_size=int(args.batch_size),
            pooling=pooling,
            trust_remote_code=trust_remote_code,
            device=str(args.device),
            query_prefix=qp,
            doc_prefix=dp,
        )

        embed_cfg = EmbedderConfig(
            model_id=run.model_id,
            dim=run.dim,
            dtype=run.dtype,
            batch_size=run.batch_size,
            max_length=run.max_length,
            pooling=run.pooling,  # type: ignore[arg-type]
            normalize=True,
            trust_remote_code=run.trust_remote_code,
            query_prefix=run.query_prefix,
            doc_prefix=run.doc_prefix,
            device_preference=run.device,
        )

        try:
            t0 = time.perf_counter()
            embedder = DenseEmbedder(embed_cfg)
            tok_ml = embedder.tokenizer_max_length()
            # Warmup: embed a tiny batch to trigger kernel compilation/caching.
            _ = embedder.embed_documents(texts[: min(8, len(texts))])
            t1 = time.perf_counter()

            t2 = time.perf_counter()
            _ = embedder.embed_documents(texts)
            t3 = time.perf_counter()

            proxy = _evaluate_proxy(model=embedder, chunk_ids=chunk_ids, texts=texts, top_k=int(args.top_k))

            item = {
                "model_id": run.model_id,
                "tokenizer_max_length": tok_ml,
                "effective_max_length": embed_cfg.max_length,
                "dim": run.dim,  # truncation target (if any)
                "dtype": run.dtype,
                "batch_size": run.batch_size,
                "pooling": run.pooling,
                "query_prefix": run.query_prefix,
                "doc_prefix": run.doc_prefix,
                "trust_remote_code": run.trust_remote_code,
                "device": embedder.device.type,
                "timing_seconds": {
                    "init_and_warmup": (t1 - t0),
                    "embed_all_docs": (t3 - t2),
                },
                "throughput_docs_per_second": (len(texts) / (t3 - t2)) if (t3 - t2) > 0 else None,
                "proxy": proxy,
            }
            results["runs"].append(item)

            print("=" * 70)
            print(f"{run.model_id}")
            print(f"- tokenizer_max_length: {tok_ml}")
            print(f"- config max_length: {embed_cfg.max_length}")
            print(f"- pooling: {run.pooling} | prefixes: query={run.query_prefix!r} doc={run.doc_prefix!r}")
            print(
                f"- throughput (docs/s): {item['throughput_docs_per_second']:.2f}"
                if item["throughput_docs_per_second"]
                else "- throughput: <n/a>"
            )
            print(
                f"- proxy: self@1={proxy['self_recall_at_1']:.3f} self@{args.top_k}={proxy['self_recall_at_k']:.3f} same_sermon@{args.top_k}={proxy['same_sermon_recall_at_k']:.3f}"
            )
        except Exception as e:
            item = {
                "model_id": run.model_id,
                "error": str(e),
                "tokenizer_max_length": None,
                "effective_max_length": embed_cfg.max_length,
                "dim": run.dim,
                "dtype": run.dtype,
                "batch_size": run.batch_size,
                "pooling": run.pooling,
                "query_prefix": run.query_prefix,
                "doc_prefix": run.doc_prefix,
                "trust_remote_code": run.trust_remote_code,
                "device": run.device,
            }
            results["runs"].append(item)
            print("=" * 70)
            print(f"{run.model_id}")
            print(f"- ERROR: {e}")

    out_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print("=" * 70)
    print(f"✓ Results written to: {out_path}")


if __name__ == "__main__":
    main()

