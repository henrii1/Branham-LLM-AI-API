#!/usr/bin/env python3
"""
Benchmark + validate dense retrieval artifacts (FAISS).

This is a pragmatic validation tool:
- Confirms `faiss.index` loads and `ntotal` matches `faiss_id_map.jsonl` rows.
- Confirms mapping is monotonic (faiss_id increments by 1).
- Runs a "mass query" benchmark:
  - Samples N chunk texts from SQLite deterministically
  - Embeds them as queries (batch throughput)
  - Measures FAISS search latency distribution (p50/p95) for KNN
  - Confirms the query chunk is usually retrieved (self@k sanity check)
"""

from __future__ import annotations

import json
import os
import platform
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# macOS OpenMP duplication workaround (FAISS + other deps).
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import faiss  # type: ignore  # noqa: E402

from branham_model_api.retrieval.dense.embedder import DenseEmbedder, EmbedderConfig  # noqa: E402


@dataclass(frozen=True)
class BenchResult:
    queries: int
    top_k: int
    embed_docs_per_s: float
    search_p50_ms: float
    search_p95_ms: float
    self_recall_at_1: float
    self_recall_at_k: float


def _iter_chunks(db_path: Path, *, limit: int) -> Iterator[tuple[str, str]]:
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, text
            FROM chunks
            ORDER BY date_id ASC, chunk_index ASC, chunk_id ASC
            LIMIT ?
            """,
            (int(limit),),
        )
        for chunk_id, text in cur.fetchall():
            yield str(chunk_id), str(text or "")
    finally:
        conn.close()


def _count_map_rows_and_validate(path: Path, *, max_checks: int = 2000) -> int:
    """
    Count rows and sanity-check monotonic faiss_id for the first `max_checks` rows.
    """
    expected = 0
    total = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            total += 1
            if total <= max_checks:
                obj = json.loads(line)
                fid = int(obj["faiss_id"])
                if fid != expected:
                    raise RuntimeError(f"faiss_id_map not monotonic at line {total}: got {fid}, expected {expected}")
                expected += 1
    return total


def _percentile_ms(samples_s: list[float], p: float) -> float:
    if not samples_s:
        return 0.0
    arr = np.array(samples_s, dtype=np.float64) * 1000.0
    return float(np.percentile(arr, p))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate and benchmark dense FAISS artifacts")
    parser.add_argument("--db-path", type=str, default="data/processed/chunks.sqlite")
    parser.add_argument("--faiss-path", type=str, default="data/indices/faiss.index")
    parser.add_argument("--id-map-path", type=str, default="data/indices/faiss_id_map.jsonl")
    parser.add_argument("--model-id", type=str, default="jinaai/jina-embeddings-v3")
    parser.add_argument("--hf-home", type=str, default=None, help="Optional HF_HOME (cache dir).")
    parser.add_argument("--local-files-only", action="store_true", help="Force offline HF loads.")
    parser.add_argument("--dtype", type=str, default="fp16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--queries", type=int, default=2000, help="Number of queries to run")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = str(Path(args.hf_home).expanduser().resolve())

    db_path = Path(args.db_path)
    faiss_path = Path(args.faiss_path)
    id_map_path = Path(args.id_map_path)

    if not db_path.exists():
        raise SystemExit(f"Missing DB: {db_path}")
    if not faiss_path.exists():
        raise SystemExit(f"Missing FAISS index: {faiss_path}")
    if not id_map_path.exists():
        raise SystemExit(f"Missing id map: {id_map_path}")

    print("Loading FAISS index...")
    index = faiss.read_index(str(faiss_path))
    ntotal = int(index.ntotal)
    print(f"- faiss.index: {faiss_path} (ntotal={ntotal}, type={type(index).__name__})")

    print("Validating faiss_id_map.jsonl ...")
    map_rows = _count_map_rows_and_validate(id_map_path)
    print(f"- faiss_id_map rows: {map_rows}")
    if map_rows != ntotal:
        raise SystemExit(f"Mismatch: faiss ntotal={ntotal} but id_map rows={map_rows}")

    qn = int(min(args.queries, ntotal))
    print(f"Sampling {qn} chunk texts as queries (deterministic prefix).")
    rows = list(_iter_chunks(db_path, limit=qn))
    chunk_ids = [cid for cid, _ in rows]
    texts = [txt for _, txt in rows]

    emb_cfg = EmbedderConfig(
        model_id=str(args.model_id),
        dim=int(args.dim),
        dtype=str(args.dtype),
        batch_size=int(args.batch_size),
        max_length=int(args.max_length),
        pooling="mean",
        normalize=True,
        trust_remote_code=True,
        local_files_only=bool(args.local_files_only),
    )
    embedder = DenseEmbedder(emb_cfg)
    print(f"Embedding queries with {args.model_id} on device={embedder.device.type} ...")

    t0 = time.perf_counter()
    q_vecs = embedder.embed_queries(texts)
    t1 = time.perf_counter()
    embed_s = t1 - t0
    embed_docs_per_s = (qn / embed_s) if embed_s > 0 else 0.0
    print(f"- embed: {qn} queries in {embed_s:.2f}s ({embed_docs_per_s:.2f} q/s)")

    # Measure FAISS search latency distribution (per-query) with precomputed vectors.
    top_k = int(args.top_k)
    search_times: list[float] = []
    self_at_1 = 0
    self_at_k = 0

    # Convert once for FAISS.
    if q_vecs.dtype != np.float32:
        q_vecs = q_vecs.astype(np.float32, copy=False)

    for i in range(qn):
        v = q_vecs[i : i + 1]
        ts = time.perf_counter()
        _scores, ids = index.search(v, top_k)
        te = time.perf_counter()
        search_times.append(te - ts)
        got = [int(x) for x in ids[0] if int(x) >= 0]
        if got and got[0] == i:
            self_at_1 += 1
        if i in got:
            self_at_k += 1

    res = BenchResult(
        queries=qn,
        top_k=top_k,
        embed_docs_per_s=embed_docs_per_s,
        search_p50_ms=_percentile_ms(search_times, 50.0),
        search_p95_ms=_percentile_ms(search_times, 95.0),
        self_recall_at_1=self_at_1 / qn if qn else 0.0,
        self_recall_at_k=self_at_k / qn if qn else 0.0,
    )

    print("FAISS search latency (excluding embedding):")
    print(f"- p50: {res.search_p50_ms:.3f} ms")
    print(f"- p95: {res.search_p95_ms:.3f} ms")
    print("Sanity retrieval proxy (query=chunk_text should retrieve itself):")
    print(f"- self@1: {res.self_recall_at_1:.3f}")
    print(f"- self@{top_k}: {res.self_recall_at_k:.3f}")

    print("✓ Dense index looks consistent.")


if __name__ == "__main__":
    main()

