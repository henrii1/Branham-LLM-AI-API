#!/usr/bin/env python3
"""
Comprehensive retrieval benchmark comparing Qwen, Jina, and BM25.

This script:
1. Loads the current FAISS index (Qwen)
2. Loads embedders for both Qwen and Jina
3. Measures latency for query embedding, FAISS search, BM25 search
4. Measures self-retrieval accuracy
5. Outputs comparison results

Usage:
    uv run python scripts/bench_retrieval_comparison.py
    uv run python scripts/bench_retrieval_comparison.py --queries 200 --top-k 10
"""

from __future__ import annotations

import json
import os
import pickle
import platform
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

# macOS OpenMP workaround
if platform.system() == "Darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import faiss  # type: ignore  # noqa: E402

from branham_model_api.retrieval.dense.embedder import DenseEmbedder, EmbedderConfig  # noqa: E402
from branham_model_api.retrieval.bm25 import Bm25Index, bm25_search  # noqa: E402


@dataclass
class LatencyStats:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass
class EmbedderBenchResult:
    model_id: str
    single_query_latency: LatencyStats
    batch_throughput_qps: float
    self_recall_at_1: float
    self_recall_at_k: float


@dataclass
class SearchBenchResult:
    name: str
    latency: LatencyStats
    self_recall_at_1: float
    self_recall_at_k: float


def compute_latency_stats(times_s: list[float]) -> LatencyStats:
    """Convert seconds to ms and compute stats."""
    arr = np.array(times_s) * 1000
    return LatencyStats(
        mean_ms=float(np.mean(arr)),
        p50_ms=float(np.percentile(arr, 50)),
        p95_ms=float(np.percentile(arr, 95)),
        p99_ms=float(np.percentile(arr, 99)),
    )


def sample_chunks(db_path: Path, limit: int) -> tuple[list[str], list[str]]:
    """Sample chunks deterministically for benchmark."""
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT chunk_id, text FROM chunks
            ORDER BY date_id ASC, chunk_index ASC, chunk_id ASC
            LIMIT ?
            """,
            (limit,),
        )
        rows = cur.fetchall()
        return [r[0] for r in rows], [r[1] for r in rows]
    finally:
        conn.close()


def benchmark_embedder(
    embedder: DenseEmbedder,
    texts: list[str],
    index: Optional[faiss.Index],
    top_k: int,
    *,
    n_single_queries: int = 100,
    skip_retrieval: bool = False,
) -> EmbedderBenchResult:
    """
    Benchmark an embedder for latency and accuracy.
    
    Args:
        skip_retrieval: If True, skip self-retrieval test (use when embedder dim != index dim)
    """
    model_id = embedder.cfg.model_id

    # Warmup
    _ = embedder.embed_queries(texts[:3])

    # Single query latency
    single_times = []
    for i in range(min(n_single_queries, len(texts))):
        t0 = time.perf_counter()
        _ = embedder.embed_queries([texts[i]])
        single_times.append(time.perf_counter() - t0)

    single_latency = compute_latency_stats(single_times)

    # Batch throughput
    t0 = time.perf_counter()
    all_vecs = embedder.embed_documents(texts)  # Use embed_documents for self-retrieval
    batch_time = time.perf_counter() - t0
    throughput = len(texts) / batch_time if batch_time > 0 else 0

    # Self-retrieval accuracy (only if index is compatible)
    self_at_1 = 0.0
    self_at_k = 0.0
    
    if not skip_retrieval and index is not None:
        all_vecs = all_vecs.astype(np.float32)
        self_1 = 0
        self_k = 0
        for i in range(len(texts)):
            _, ids = index.search(all_vecs[i : i + 1], top_k)
            got = [int(x) for x in ids[0] if x >= 0]
            if got and got[0] == i:
                self_1 += 1
            if i in got:
                self_k += 1
        n = len(texts)
        self_at_1 = self_1 / n if n > 0 else 0
        self_at_k = self_k / n if n > 0 else 0

    return EmbedderBenchResult(
        model_id=model_id,
        single_query_latency=single_latency,
        batch_throughput_qps=throughput,
        self_recall_at_1=self_at_1,
        self_recall_at_k=self_at_k,
    )


def benchmark_faiss_search(
    index: faiss.Index,
    vectors: np.ndarray,
    top_k: int,
) -> SearchBenchResult:
    """Benchmark FAISS search latency."""
    vectors = vectors.astype(np.float32)
    times = []
    self_at_1 = 0
    self_at_k = 0

    for i in range(len(vectors)):
        t0 = time.perf_counter()
        _, ids = index.search(vectors[i : i + 1], top_k)
        times.append(time.perf_counter() - t0)

        got = [int(x) for x in ids[0] if x >= 0]
        if got and got[0] == i:
            self_at_1 += 1
        if i in got:
            self_at_k += 1

    n = len(vectors)
    return SearchBenchResult(
        name="FAISS",
        latency=compute_latency_stats(times),
        self_recall_at_1=self_at_1 / n if n > 0 else 0,
        self_recall_at_k=self_at_k / n if n > 0 else 0,
    )


def benchmark_bm25_search(
    bm25_index: Bm25Index,
    texts: list[str],
    chunk_ids: list[str],
    top_k: int,
) -> SearchBenchResult:
    """Benchmark BM25 search latency."""
    times = []
    self_at_1 = 0
    self_at_k = 0

    for i, text in enumerate(texts):
        t0 = time.perf_counter()
        hits = bm25_search(bm25_index, text, top_n=top_k)
        times.append(time.perf_counter() - t0)

        hit_ids = [h.chunk_id for h in hits]
        if hit_ids and hit_ids[0] == chunk_ids[i]:
            self_at_1 += 1
        if chunk_ids[i] in hit_ids:
            self_at_k += 1

    n = len(texts)
    return SearchBenchResult(
        name="BM25",
        latency=compute_latency_stats(times),
        self_recall_at_1=self_at_1 / n if n > 0 else 0,
        self_recall_at_k=self_at_k / n if n > 0 else 0,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark retrieval: Qwen vs Jina vs BM25")
    parser.add_argument("--db-path", type=str, default="data/processed/chunks.sqlite")
    parser.add_argument("--faiss-path", type=str, default="data/indices/faiss.index")
    parser.add_argument("--faiss-meta-path", type=str, default="data/indices/faiss_meta.json")
    parser.add_argument("--bm25-path", type=str, default="data/indices/bm25.index")
    parser.add_argument("--queries", type=int, default=200, help="Number of test queries")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K for retrieval")
    parser.add_argument("--skip-jina", action="store_true", help="Skip Jina benchmark")
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    db_path = repo_root / args.db_path
    faiss_path = repo_root / args.faiss_path
    faiss_meta_path = repo_root / args.faiss_meta_path
    bm25_path = repo_root / args.bm25_path

    print("=" * 70)
    print("RETRIEVAL BENCHMARK: Qwen vs Jina vs BM25")
    print("=" * 70)

    # Load FAISS index and meta
    print("\n[1/5] Loading FAISS index...")
    with open(faiss_meta_path) as f:
        faiss_meta = json.load(f)
    index = faiss.read_index(str(faiss_path))
    print(f"  Model: {faiss_meta['model_id']}")
    print(f"  Dim: {faiss_meta['dim']}, Type: {faiss_meta['index_type']}")
    print(f"  Total chunks: {faiss_meta['total_chunks']}")

    # Load BM25 index
    print("\n[2/5] Loading BM25 index...")
    with open(bm25_path, "rb") as f:
        bm25_index: Bm25Index = pickle.load(f)
    print(f"  Docs: {len(bm25_index.doc_id_to_chunk_id)}")

    # Sample test data
    print(f"\n[3/5] Sampling {args.queries} test chunks...")
    chunk_ids, texts = sample_chunks(db_path, args.queries)
    print(f"  Sampled {len(texts)} chunks")

    # Create Qwen embedder
    print("\n[4/5] Initializing embedders...")
    print("  Loading Qwen/Qwen3-Embedding-0.6B...")
    qwen_cfg = EmbedderConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        pooling="last_token",
        max_length=32768,
        dim=1024,
        batch_size=16,
        # Note: For self-retrieval benchmark, we don't use instruction template
        # since documents were embedded without it
    )
    qwen_embedder = DenseEmbedder(qwen_cfg)
    print(f"    Device: {qwen_embedder.device.type}")

    # Create Jina embedder (optional)
    jina_embedder: Optional[DenseEmbedder] = None
    if not args.skip_jina:
        print("  Loading jinaai/jina-embeddings-v3...")
        try:
            jina_cfg = EmbedderConfig(
                model_id="jinaai/jina-embeddings-v3",
                pooling="mean",
                max_length=8192,
                dim=512,  # MRL truncation
                batch_size=16,
            )
            jina_embedder = DenseEmbedder(jina_cfg)
            print(f"    Device: {jina_embedder.device.type}")
        except Exception as e:
            print(f"    Failed to load Jina: {e}")
            jina_embedder = None

    # Run benchmarks
    print(f"\n[5/5] Running benchmarks (top_k={args.top_k})...")
    print("-" * 70)

    # Qwen benchmark
    print("\nBenchmarking Qwen...")
    qwen_result = benchmark_embedder(qwen_embedder, texts, index, args.top_k)

    # Get vectors for FAISS search benchmark
    qwen_vecs = qwen_embedder.embed_documents(texts)
    faiss_result = benchmark_faiss_search(index, qwen_vecs, args.top_k)

    # BM25 benchmark
    print("Benchmarking BM25...")
    bm25_result = benchmark_bm25_search(bm25_index, texts, chunk_ids, args.top_k)

    # Jina benchmark (optional)
    jina_result: Optional[EmbedderBenchResult] = None
    if jina_embedder is not None:
        print("Benchmarking Jina...")
        # Note: Jina vectors (512-dim) won't match Qwen index (1024-dim)
        # We measure embedding latency only, skip self-retrieval
        jina_result = benchmark_embedder(
            jina_embedder, texts, index, args.top_k, skip_retrieval=True
        )

    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'QUERY EMBEDDING LATENCY':-^70}")
    print(f"{'Model':<35} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 70)
    print(
        f"{'Qwen/Qwen3-Embedding-0.6B':<35} "
        f"{qwen_result.single_query_latency.mean_ms:>7.1f}ms "
        f"{qwen_result.single_query_latency.p50_ms:>7.1f}ms "
        f"{qwen_result.single_query_latency.p95_ms:>7.1f}ms "
        f"{qwen_result.single_query_latency.p99_ms:>7.1f}ms"
    )
    if jina_result:
        print(
            f"{'jinaai/jina-embeddings-v3':<35} "
            f"{jina_result.single_query_latency.mean_ms:>7.1f}ms "
            f"{jina_result.single_query_latency.p50_ms:>7.1f}ms "
            f"{jina_result.single_query_latency.p95_ms:>7.1f}ms "
            f"{jina_result.single_query_latency.p99_ms:>7.1f}ms"
        )

    print(f"\n{'BATCH EMBEDDING THROUGHPUT':-^70}")
    print(f"{'Model':<35} {'Queries/sec':>15}")
    print("-" * 70)
    print(f"{'Qwen/Qwen3-Embedding-0.6B':<35} {qwen_result.batch_throughput_qps:>14.1f}")
    if jina_result:
        print(f"{'jinaai/jina-embeddings-v3':<35} {jina_result.batch_throughput_qps:>14.1f}")

    print(f"\n{'SEARCH LATENCY':-^70}")
    print(f"{'Method':<35} {'Mean':>8} {'p50':>8} {'p95':>8} {'p99':>8}")
    print("-" * 70)
    print(
        f"{'FAISS (Qwen index)':<35} "
        f"{faiss_result.latency.mean_ms:>7.2f}ms "
        f"{faiss_result.latency.p50_ms:>7.2f}ms "
        f"{faiss_result.latency.p95_ms:>7.2f}ms "
        f"{faiss_result.latency.p99_ms:>7.2f}ms"
    )
    print(
        f"{'BM25':<35} "
        f"{bm25_result.latency.mean_ms:>7.2f}ms "
        f"{bm25_result.latency.p50_ms:>7.2f}ms "
        f"{bm25_result.latency.p95_ms:>7.2f}ms "
        f"{bm25_result.latency.p99_ms:>7.2f}ms"
    )

    print(f"\n{'SELF-RETRIEVAL ACCURACY (chunk retrieves itself)':-^70}")
    print(f"{'Method':<35} {'self@1':>10} {'self@{}'.format(args.top_k):>10}")
    print("-" * 70)
    print(
        f"{'Dense (Qwen FAISS)':<35} "
        f"{qwen_result.self_recall_at_1:>9.1%} "
        f"{qwen_result.self_recall_at_k:>9.1%}"
    )
    print(
        f"{'BM25':<35} "
        f"{bm25_result.self_recall_at_1:>9.1%} "
        f"{bm25_result.self_recall_at_k:>9.1%}"
    )

    # End-to-end latency summary
    print(f"\n{'END-TO-END LATENCY (p95)':-^70}")
    embed_p95 = qwen_result.single_query_latency.p95_ms
    faiss_p95 = faiss_result.latency.p95_ms
    bm25_p95 = bm25_result.latency.p95_ms
    hybrid_p95 = embed_p95 + max(faiss_p95, bm25_p95)  # Parallel search

    print(f"  Query embedding:     {embed_p95:>7.1f}ms")
    print(f"  FAISS search:        {faiss_p95:>7.2f}ms")
    print(f"  BM25 search:         {bm25_p95:>7.2f}ms")
    print(f"  Hybrid (parallel):   {hybrid_p95:>7.1f}ms")

    print("\n" + "=" * 70)
    print("BENCHMARK COMPLETE")
    print("=" * 70)

    # Save results to JSON
    results = {
        "config": {
            "queries": args.queries,
            "top_k": args.top_k,
            "faiss_model": faiss_meta["model_id"],
            "faiss_dim": faiss_meta["dim"],
            "total_chunks": faiss_meta["total_chunks"],
        },
        "qwen": {
            "model_id": qwen_result.model_id,
            "single_query_latency_ms": {
                "mean": qwen_result.single_query_latency.mean_ms,
                "p50": qwen_result.single_query_latency.p50_ms,
                "p95": qwen_result.single_query_latency.p95_ms,
                "p99": qwen_result.single_query_latency.p99_ms,
            },
            "batch_throughput_qps": qwen_result.batch_throughput_qps,
            "self_recall_at_1": qwen_result.self_recall_at_1,
            "self_recall_at_k": qwen_result.self_recall_at_k,
        },
        "faiss_search": {
            "latency_ms": {
                "mean": faiss_result.latency.mean_ms,
                "p50": faiss_result.latency.p50_ms,
                "p95": faiss_result.latency.p95_ms,
                "p99": faiss_result.latency.p99_ms,
            },
        },
        "bm25": {
            "latency_ms": {
                "mean": bm25_result.latency.mean_ms,
                "p50": bm25_result.latency.p50_ms,
                "p95": bm25_result.latency.p95_ms,
                "p99": bm25_result.latency.p99_ms,
            },
            "self_recall_at_1": bm25_result.self_recall_at_1,
            "self_recall_at_k": bm25_result.self_recall_at_k,
        },
    }

    if jina_result:
        results["jina"] = {
            "model_id": jina_result.model_id,
            "single_query_latency_ms": {
                "mean": jina_result.single_query_latency.mean_ms,
                "p50": jina_result.single_query_latency.p50_ms,
                "p95": jina_result.single_query_latency.p95_ms,
                "p99": jina_result.single_query_latency.p99_ms,
            },
            "batch_throughput_qps": jina_result.batch_throughput_qps,
        }

    out_path = repo_root / "data" / "experiments" / "retrieval_benchmark.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
