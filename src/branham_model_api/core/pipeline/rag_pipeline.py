"""
RAG Pipeline orchestrator.

Orchestrates the retrieval flow from user query to context-ready sermons:

1. Query preprocessing (normalize, detect intent)
2. Parallel retrieval (BM25 + Dense with topN=25 each)
3. Compute retrieval signals
4. Conditional reranker (if signals indicate ambiguity)
5. Fusion + dedup by chunk_id
6. Collate by date_id, rank sermons by best chunk score, cap at 8
7. Expansion ±1
8. Dedup after expansion

This module handles everything UP TO the generation LLM.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, Sequence

import faiss
import numpy as np

logger = logging.getLogger(__name__)

from branham_model_api.core.pipeline.expansion import (
    ExpandedSermon,
    expand_and_group,
)
from branham_model_api.core.pipeline.fusion import (
    FusedHit,
    SermonGroup,
    apply_rerank_scores,
    collate_by_sermon,
    merge_bm25_dense,
)
from branham_model_api.core.pipeline.signals import (
    RetrievalSignals,
    compute_signals,
    detect_quote_intent,
)
from branham_model_api.retrieval.bm25.index import Bm25Index
from branham_model_api.retrieval.bm25.query import Bm25Hit, bm25_search
from branham_model_api.retrieval.dense.query import DenseHit, faiss_search
from branham_model_api.retrieval.store.chunk_store import ChunkStore


# -----------------------------------------------------------------------------
# Protocols for embedder and reranker (allows both HF and vLLM implementations)
# -----------------------------------------------------------------------------


class EmbedderProtocol(Protocol):
    """Protocol for query embedding."""

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        """Embed query texts, returning normalized vectors."""
        ...


class RerankerProtocol(Protocol):
    """Protocol for reranking."""

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Returns list of (original_index, score) tuples, sorted by score descending.
        """
        ...


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------


@dataclass
class RetrievalConfig:
    """Configuration for retrieval pipeline."""

    # Retrieval parameters
    bm25_top_n: int = 25
    dense_top_n: int = 25

    # Reranker trigger thresholds (adjusted based on observed data)
    # Note: Original thresholds (0.08, 3, 0.65) were too strict - triggered on every query
    reranker_enabled: bool = True
    score_std_threshold: float = 0.015  # Very flat distribution (typical is 0.01-0.05)
    overlap_threshold: int = 1  # No agreement at all (typical is 0-3)
    top_score_threshold: float = 0.55  # Weak semantic match

    # Collation
    max_sermons: int = 8

    # Expansion
    expansion_delta: int = 1

    # Refusal threshold - based on DENSE score (semantic similarity)
    # Dense scores: 0.7+ strong match, 0.5-0.7 moderate, <0.5 weak/off-topic
    min_dense_score: float = 0.50  # Below this = likely off-topic


@dataclass
class RetrievalResult:
    """Result from the retrieval pipeline (before LLM generation)."""

    # Query info
    query: str
    query_normalized: str
    quote_intent: bool

    # Retrieval signals
    signals: RetrievalSignals
    reranker_triggered: bool

    # Results
    sermon_groups: list[SermonGroup]  # After collation, before expansion
    expanded_sermons: list[ExpandedSermon]  # After expansion

    # Metadata
    bm25_hit_count: int
    dense_hit_count: int
    fused_hit_count: int
    total_chunks: int  # After expansion

    # Refusal
    should_refuse: bool
    refuse_reason: str | None = None


# -----------------------------------------------------------------------------
# Query preprocessing
# -----------------------------------------------------------------------------


def normalize_query(query: str) -> str:
    """
    Normalize query for retrieval.

    - Strip whitespace
    - Normalize unicode
    - Collapse multiple spaces
    """
    import unicodedata

    query = query.strip()
    query = unicodedata.normalize("NFKC", query)
    query = re.sub(r"\s+", " ", query)
    return query


# -----------------------------------------------------------------------------
# FAISS ID map loading
# -----------------------------------------------------------------------------


def load_faiss_id_map(path: str | Path) -> dict[int, str]:
    """
    Load FAISS ID to chunk_id mapping from JSONL file.

    Each line: {"faiss_id": 123, "chunk_id": "47-0412M_chunk_3"}
    """
    mapping: dict[int, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                record = json.loads(line)
                mapping[record["faiss_id"]] = record["chunk_id"]
    return mapping


# -----------------------------------------------------------------------------
# RAG Pipeline
# -----------------------------------------------------------------------------


class RAGPipeline:
    """
    RAG Pipeline orchestrator.

    Handles retrieval from query to context-ready sermons.
    Does NOT handle LLM generation (that's in the generation module).
    """

    def __init__(
        self,
        *,
        bm25_index: Bm25Index,
        faiss_index: faiss.Index,
        faiss_id_map: dict[int, str],
        chunk_store: ChunkStore,
        embedder: EmbedderProtocol,
        reranker: RerankerProtocol | None = None,
        config: RetrievalConfig | None = None,
    ):
        self.bm25_index = bm25_index
        self.faiss_index = faiss_index
        self.faiss_id_map = faiss_id_map
        self.chunk_store = chunk_store
        self.embedder = embedder
        self.reranker = reranker
        self.config = config or RetrievalConfig()

    def _run_bm25(self, query: str) -> list[Bm25Hit]:
        """Run BM25 retrieval."""
        start = time.perf_counter()
        hits = bm25_search(
            self.bm25_index,
            query,
            top_n=self.config.bm25_top_n,
        )
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"BM25: {len(hits)} hits in {elapsed:.1f}ms")
        if hits:
            logger.debug(f"  Top BM25: {hits[0].chunk_id} score={hits[0].score:.4f}")
        return hits

    def _run_dense(self, query: str) -> list[DenseHit]:
        """Run dense retrieval (embed + FAISS search)."""
        # Embed query
        start = time.perf_counter()
        query_vec = self.embedder.embed_queries([query])  # [1, dim]
        embed_elapsed = (time.perf_counter() - start) * 1000
        
        # Search FAISS
        start = time.perf_counter()
        results = faiss_search(
            self.faiss_index,
            query_vec,
            top_n=self.config.dense_top_n,
        )
        search_elapsed = (time.perf_counter() - start) * 1000
        
        hits = results[0] if results else []
        logger.info(f"Dense: embed={embed_elapsed:.1f}ms, search={search_elapsed:.1f}ms, {len(hits)} hits")
        if hits:
            logger.debug(f"  Top Dense: faiss_id={hits[0].faiss_id} score={hits[0].score:.4f}")
        return hits

    def _run_reranker(
        self,
        query: str,
        fused_hits: Sequence[FusedHit],
    ) -> dict[str, float]:
        """
        Run reranker on fused hits.

        Returns mapping from chunk_id to reranker score.
        """
        if self.reranker is None:
            return {}

        start = time.perf_counter()
        
        # Get chunk texts
        chunk_ids = [h.chunk_id for h in fused_hits]
        chunks = self.chunk_store.get_chunks(chunk_ids)

        # Extract texts in order
        documents = [chunks[cid].text for cid in chunk_ids if cid in chunks]
        valid_chunk_ids = [cid for cid in chunk_ids if cid in chunks]

        if not documents:
            return {}

        # Rerank
        ranked = self.reranker.rerank(query, documents)

        # Map back to chunk_ids
        scores = {}
        for orig_idx, score in ranked:
            if orig_idx < len(valid_chunk_ids):
                scores[valid_chunk_ids[orig_idx]] = score

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(f"Reranker: {len(documents)} docs in {elapsed:.1f}ms")
        if scores:
            top_id = max(scores, key=scores.get)
            logger.debug(f"  Top reranked: {top_id} score={scores[top_id]:.4f}")
        
        return scores

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Execute the full retrieval pipeline.

        Args:
            query: User query

        Returns:
            RetrievalResult with context-ready sermons
        """
        total_start = time.perf_counter()
        config = self.config
        
        logger.info(f"=== RETRIEVE START: '{query[:80]}{'...' if len(query) > 80 else ''}' ===")

        # 1. Normalize query
        query_normalized = normalize_query(query)
        quote_intent = detect_quote_intent(query_normalized)
        logger.debug(f"Query normalized, quote_intent={quote_intent}")

        # 2. Parallel retrieval (BM25 + Dense)
        bm25_hits = self._run_bm25(query_normalized)
        dense_hits = self._run_dense(query_normalized)

        # 3. Compute retrieval signals
        signals = compute_signals(
            query_normalized,
            bm25_hits,
            dense_hits,
            self.faiss_id_map,
        )

        # 4. Fusion + dedup
        fused_hits = merge_bm25_dense(bm25_hits, dense_hits, self.faiss_id_map)

        # 5. Conditional reranker
        reranker_triggered = False
        if config.reranker_enabled and self.reranker is not None:
            should_rerank = signals.should_rerank(
                score_std_threshold=config.score_std_threshold,
                overlap_threshold=config.overlap_threshold,
                top_score_threshold=config.top_score_threshold,
            )
            if should_rerank:
                reranker_triggered = True
                rerank_scores = self._run_reranker(query_normalized, fused_hits)
                if rerank_scores:
                    fused_hits = apply_rerank_scores(fused_hits, rerank_scores)

        # 6. Collate by sermon, cap at max_sermons
        sermon_groups = collate_by_sermon(fused_hits, max_sermons=config.max_sermons)

        # 7. Check for refusal (based on DENSE score for semantic relevance)
        should_refuse = False
        refuse_reason = None
        dense_top = signals.dense_top_score

        if not fused_hits:
            should_refuse = True
            refuse_reason = "No relevant chunks found"
        elif dense_top < config.min_dense_score:
            should_refuse = True
            refuse_reason = f"Query appears off-topic (dense_score={dense_top:.3f} < {config.min_dense_score})"
            logger.info(f"Refusal: dense_top={dense_top:.4f} < threshold={config.min_dense_score}")

        # 8. Expand chunks ±1 (only if not refusing)
        expanded_sermons: list[ExpandedSermon] = []
        total_chunks = 0

        if not should_refuse:
            # Get all chunks from selected sermons
            all_sermon_chunks: list[FusedHit] = []
            for group in sermon_groups:
                all_sermon_chunks.extend(group.chunks)

            # Expand
            expanded_sermons = expand_and_group(
                all_sermon_chunks,
                self.chunk_store,
                delta=config.expansion_delta,
            )

            # Count total chunks
            total_chunks = sum(len(s.chunks) for s in expanded_sermons)

        total_elapsed = (time.perf_counter() - total_start) * 1000
        
        if should_refuse:
            logger.warning(f"REFUSING: {refuse_reason}")
        else:
            logger.info(
                f"Expansion: {len(expanded_sermons)} sermons, {total_chunks} total chunks"
            )
        
        logger.info(
            f"=== RETRIEVE DONE in {total_elapsed:.1f}ms: "
            f"bm25={len(bm25_hits)}, dense={len(dense_hits)}, fused={len(fused_hits)}, "
            f"reranked={reranker_triggered}, sermons={len(sermon_groups)}, "
            f"refuse={should_refuse} ==="
        )
        
        return RetrievalResult(
            query=query,
            query_normalized=query_normalized,
            quote_intent=quote_intent,
            signals=signals,
            reranker_triggered=reranker_triggered,
            sermon_groups=sermon_groups,
            expanded_sermons=expanded_sermons,
            bm25_hit_count=len(bm25_hits),
            dense_hit_count=len(dense_hits),
            fused_hit_count=len(fused_hits),
            total_chunks=total_chunks,
            should_refuse=should_refuse,
            refuse_reason=refuse_reason,
        )


# -----------------------------------------------------------------------------
# Factory function for creating pipeline with correct backend
# -----------------------------------------------------------------------------


def create_rag_pipeline(
    *,
    bm25_index_path: str | Path,
    faiss_index_path: str | Path,
    faiss_id_map_path: str | Path,
    chunk_store_path: str | Path,
    embedder: EmbedderProtocol,
    reranker: RerankerProtocol | None = None,
    config: RetrievalConfig | None = None,
) -> RAGPipeline:
    """
    Create a RAG pipeline with all components loaded.

    Args:
        bm25_index_path: Path to BM25 index file
        faiss_index_path: Path to FAISS index file
        faiss_id_map_path: Path to FAISS ID map JSONL file
        chunk_store_path: Path to chunks.sqlite
        embedder: Embedder instance (HF or vLLM)
        reranker: Optional reranker instance (HF or vLLM)
        config: Optional retrieval configuration

    Returns:
        Configured RAGPipeline
    """
    # Load BM25 index
    bm25_index = Bm25Index.load(Path(bm25_index_path))

    # Load FAISS index
    faiss_index = faiss.read_index(str(faiss_index_path))

    # Load FAISS ID map
    faiss_id_map = load_faiss_id_map(faiss_id_map_path)

    # Open chunk store
    chunk_store = ChunkStore(chunk_store_path)

    return RAGPipeline(
        bm25_index=bm25_index,
        faiss_index=faiss_index,
        faiss_id_map=faiss_id_map,
        chunk_store=chunk_store,
        embedder=embedder,
        reranker=reranker,
        config=config,
    )
