"""
Retrieval fusion and deduplication.

Merges BM25 and dense retrieval results, deduplicates by chunk_id,
and optionally applies reranker scores.

Flow:
1. Convert both result sets to unified format (chunk_id + score)
2. Deduplicate by chunk_id (keep highest score from either retriever)
3. Optionally apply reranker scores
4. Return merged results
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Sequence

from branham_model_api.retrieval.bm25.query import Bm25Hit
from branham_model_api.retrieval.dense.query import DenseHit

logger = logging.getLogger(__name__)


@dataclass
class FusedHit:
    """A fused retrieval result."""

    chunk_id: str
    score: float  # Unified score (reranker if available, else max(bm25, dense))
    bm25_score: float | None  # Original BM25 score (if from BM25)
    dense_score: float | None  # Original dense score (if from dense)
    rerank_score: float | None  # Reranker score (if reranked)

    @property
    def source(self) -> str:
        """Which retriever(s) found this chunk."""
        sources = []
        if self.bm25_score is not None:
            sources.append("bm25")
        if self.dense_score is not None:
            sources.append("dense")
        return "+".join(sources) if sources else "unknown"


def merge_bm25_dense(
    bm25_hits: Sequence[Bm25Hit],
    dense_hits: Sequence[DenseHit],
    faiss_id_to_chunk_id: dict[int, str],
    *,
    rrf_k: int = 60,
) -> list[FusedHit]:
    """
    Merge BM25 and dense results using Reciprocal Rank Fusion (RRF).

    RRF normalizes scores by rank position, solving the score scale mismatch
    between BM25 (unbounded) and dense (0-1) scores.

    RRF score = sum(1 / (k + rank)) for each retriever
    
    Args:
        bm25_hits: BM25 retrieval results (ordered by score desc)
        dense_hits: Dense (FAISS) retrieval results (ordered by score desc)
        faiss_id_to_chunk_id: Mapping from FAISS ID to chunk_id
        rrf_k: RRF smoothing parameter (default 60, standard value)

    Returns:
        List of FusedHit, sorted by RRF score descending
    """
    # Track scores per chunk
    chunks: dict[str, dict] = {}

    # Add BM25 results with rank-based RRF score
    for rank, hit in enumerate(bm25_hits):
        chunk_id = hit.chunk_id
        rrf_score = 1.0 / (rrf_k + rank + 1)  # rank is 0-indexed
        
        if chunk_id not in chunks:
            chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "rrf_score": rrf_score,
                "bm25_score": hit.score,
                "bm25_rank": rank + 1,
                "dense_score": None,
                "dense_rank": None,
                "rerank_score": None,
            }
        else:
            chunks[chunk_id]["rrf_score"] += rrf_score
            chunks[chunk_id]["bm25_score"] = hit.score
            chunks[chunk_id]["bm25_rank"] = rank + 1

    # Add dense results with rank-based RRF score
    for rank, hit in enumerate(dense_hits):
        chunk_id = faiss_id_to_chunk_id.get(hit.faiss_id)
        if chunk_id is None:
            continue  # Skip unknown FAISS IDs
        
        rrf_score = 1.0 / (rrf_k + rank + 1)
        
        if chunk_id not in chunks:
            chunks[chunk_id] = {
                "chunk_id": chunk_id,
                "rrf_score": rrf_score,
                "bm25_score": None,
                "bm25_rank": None,
                "dense_score": hit.score,
                "dense_rank": rank + 1,
                "rerank_score": None,
            }
        else:
            chunks[chunk_id]["rrf_score"] += rrf_score
            chunks[chunk_id]["dense_score"] = hit.score
            chunks[chunk_id]["dense_rank"] = rank + 1

    # Convert to FusedHit objects
    result = [
        FusedHit(
            chunk_id=c["chunk_id"],
            score=c["rrf_score"],  # RRF score is now the unified score
            bm25_score=c["bm25_score"],
            dense_score=c["dense_score"],
            rerank_score=c["rerank_score"],
        )
        for c in chunks.values()
    ]
    
    # Sort by RRF score descending
    result.sort(key=lambda h: h.score, reverse=True)
    
    # Count sources
    bm25_only = sum(1 for h in result if h.bm25_score is not None and h.dense_score is None)
    dense_only = sum(1 for h in result if h.dense_score is not None and h.bm25_score is None)
    both = sum(1 for h in result if h.bm25_score is not None and h.dense_score is not None)
    logger.info(
        f"Fusion (RRF k={rrf_k}): {len(bm25_hits)} BM25 + {len(dense_hits)} dense → "
        f"{len(result)} unique (bm25_only={bm25_only}, dense_only={dense_only}, both={both})"
    )
    if result:
        top = result[0]
        logger.debug(
            f"Top fused: {top.chunk_id} rrf={top.score:.4f} "
            f"(bm25={top.bm25_score}, dense={top.dense_score})"
        )
    
    return result


def apply_rerank_scores(
    fused_hits: Sequence[FusedHit],
    rerank_scores: dict[str, float],
) -> list[FusedHit]:
    """
    Apply reranker scores to fused hits.

    Reranker score becomes the primary score for ranking.

    Args:
        fused_hits: Previously merged hits
        rerank_scores: Mapping from chunk_id to reranker score

    Returns:
        List of FusedHit with reranker scores, sorted by score descending
    """
    result = []
    for hit in fused_hits:
        rerank_score = rerank_scores.get(hit.chunk_id)
        if rerank_score is not None:
            # Use reranker score as primary
            result.append(FusedHit(
                chunk_id=hit.chunk_id,
                score=rerank_score,
                bm25_score=hit.bm25_score,
                dense_score=hit.dense_score,
                rerank_score=rerank_score,
            ))
        else:
            # Keep original score (shouldn't happen if all chunks were reranked)
            result.append(hit)

    # Re-sort by score descending
    result.sort(key=lambda h: h.score, reverse=True)
    return result


@dataclass
class SermonGroup:
    """A group of chunks from the same sermon."""

    date_id: str
    best_score: float  # Highest chunk score in this sermon
    chunks: list[FusedHit]  # Chunks from this sermon, sorted by score desc


def collate_by_sermon(
    fused_hits: Sequence[FusedHit],
    *,
    max_sermons: int = 8,
) -> list[SermonGroup]:
    """
    Group chunks by sermon (date_id), rank sermons by best chunk score,
    and cap at max_sermons.

    This is where truncation happens - we keep only the top N sermons.

    Args:
        fused_hits: Merged retrieval results
        max_sermons: Maximum number of sermons to keep (default 8)

    Returns:
        List of SermonGroup, sorted by best_score descending,
        capped at max_sermons.
    """
    # Group by date_id (extracted from chunk_id)
    sermons: dict[str, list[FusedHit]] = {}
    for hit in fused_hits:
        # chunk_id format: {date_id}_chunk_{index}
        # e.g., "47-0412M_chunk_3" -> date_id = "47-0412M"
        parts = hit.chunk_id.rsplit("_chunk_", 1)
        if len(parts) != 2:
            continue  # Invalid chunk_id format
        date_id = parts[0]

        if date_id not in sermons:
            sermons[date_id] = []
        sermons[date_id].append(hit)

    # Create SermonGroups with best score
    groups = []
    for date_id, chunks in sermons.items():
        # Sort chunks by score within sermon
        chunks.sort(key=lambda h: h.score, reverse=True)
        best_score = chunks[0].score if chunks else 0.0
        groups.append(SermonGroup(
            date_id=date_id,
            best_score=best_score,
            chunks=chunks,
        ))

    # Sort sermons by best score, cap at max_sermons
    groups.sort(key=lambda g: g.best_score, reverse=True)
    
    selected = groups[:max_sermons]
    total_chunks = sum(len(g.chunks) for g in selected)
    
    logger.info(
        f"Collation: {len(groups)} sermons → {len(selected)} selected (max={max_sermons}), "
        f"{total_chunks} total chunks"
    )
    for i, g in enumerate(selected[:3]):  # Log top 3 sermons
        logger.debug(f"  Sermon {i+1}: {g.date_id} best_score={g.best_score:.4f} chunks={len(g.chunks)}")
    
    return selected


def get_all_chunks_from_sermons(sermon_groups: Sequence[SermonGroup]) -> list[FusedHit]:
    """
    Flatten sermon groups back to a list of chunks.

    Returns all chunks from the selected sermons (no truncation within sermons).
    """
    result = []
    for group in sermon_groups:
        result.extend(group.chunks)
    # Sort by score for consistent ordering
    result.sort(key=lambda h: h.score, reverse=True)
    return result
