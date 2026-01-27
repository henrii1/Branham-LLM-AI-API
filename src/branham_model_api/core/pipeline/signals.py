"""
Retrieval signal computation for conditional reranking.

Signals are computed after BM25 + dense retrieval to determine
whether reranking is needed.

Trigger signals (any one triggers reranker):
- score_std < threshold: Flat score distribution = ambiguous ranking
- overlap < threshold: BM25/dense disagree on top results
- top_score < threshold: Weak best match
- quote_intent: User wants exact quote (precision matters)
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from statistics import stdev
from typing import Sequence

from branham_model_api.retrieval.bm25.query import Bm25Hit
from branham_model_api.retrieval.dense.query import DenseHit

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievalSignals:
    """Computed retrieval signals for reranker trigger decision."""

    # Score distribution signals
    dense_score_std: float  # Std dev of top-K dense scores
    dense_top_score: float  # Best dense score

    # Overlap signal
    bm25_dense_overlap: int  # Count of chunk_ids in both top-10

    # Query intent
    quote_intent: bool  # True if query seeks exact quote

    def should_rerank(
        self,
        *,
        score_std_threshold: float = 0.08,
        overlap_threshold: int = 3,
        top_score_threshold: float = 0.65,
    ) -> bool:
        """
        Determine if reranking should be triggered.

        Any one condition triggers reranking.
        """
        triggers = []
        
        if self.dense_score_std < score_std_threshold:
            triggers.append(f"score_std={self.dense_score_std:.4f} < {score_std_threshold}")
        if self.bm25_dense_overlap < overlap_threshold:
            triggers.append(f"overlap={self.bm25_dense_overlap} < {overlap_threshold}")
        if self.dense_top_score < top_score_threshold:
            triggers.append(f"top_score={self.dense_top_score:.4f} < {top_score_threshold}")
        if self.quote_intent:
            triggers.append("quote_intent=True")
        
        should = len(triggers) > 0
        if should:
            logger.info(f"Reranker TRIGGERED: {', '.join(triggers)}")
        else:
            logger.info(f"Reranker NOT triggered: std={self.dense_score_std:.4f}, overlap={self.bm25_dense_overlap}, top={self.dense_top_score:.4f}, quote={self.quote_intent}")
        
        return should


# Patterns that indicate quote-seeking intent
QUOTE_INTENT_PATTERNS = [
    r"\bexact\s+quote\b",
    r"\bwhere\s+did\s+he\s+say\b",
    r"\bwhich\s+sermon\b",
    r"\bwhat\s+did\s+(?:brother\s+)?branham\s+say\b",
    r"\bquote\s+(?:from|about|on)\b",
    r"\bwhat\s+are\s+his\s+exact\s+words\b",
    r"\b(?:in\s+)?paragraph\s+\d+\b",
    r"\b(?:65|64|63|62|61|60|59|58|57|56|55|54|53|52|51|50|49|48|47)-\d{4}[MAESBX]?\b",  # date_id pattern
]

_QUOTE_INTENT_RE = re.compile(
    "|".join(QUOTE_INTENT_PATTERNS),
    re.IGNORECASE,
)


def detect_quote_intent(query: str) -> bool:
    """
    Detect if query is seeking an exact quote.

    Returns True if any quote-intent pattern matches.
    """
    return bool(_QUOTE_INTENT_RE.search(query))


def compute_dense_score_std(
    dense_hits: Sequence[DenseHit],
    *,
    top_k: int = 10,
) -> float:
    """
    Compute standard deviation of top-K dense scores.

    Low std indicates flat distribution (ambiguous ranking).
    """
    if len(dense_hits) < 2:
        return 0.0

    scores = [h.score for h in dense_hits[:top_k]]
    if len(scores) < 2:
        return 0.0

    return stdev(scores)


def compute_overlap(
    bm25_hits: Sequence[Bm25Hit],
    dense_hits: Sequence[DenseHit],
    faiss_id_to_chunk_id: dict[int, str],
    *,
    top_k: int = 10,
) -> int:
    """
    Compute overlap between BM25 and dense top-K results.

    High overlap = retrievers agree (more confident).
    Low overlap = retrievers disagree (might need reranking).
    """
    bm25_top = {h.chunk_id for h in bm25_hits[:top_k]}
    dense_top = {faiss_id_to_chunk_id[h.faiss_id] for h in dense_hits[:top_k] if h.faiss_id in faiss_id_to_chunk_id}

    return len(bm25_top & dense_top)


def compute_signals(
    query: str,
    bm25_hits: Sequence[Bm25Hit],
    dense_hits: Sequence[DenseHit],
    faiss_id_to_chunk_id: dict[int, str],
    *,
    top_k: int = 10,
) -> RetrievalSignals:
    """
    Compute all retrieval signals.

    Args:
        query: User query text
        bm25_hits: BM25 retrieval results
        dense_hits: Dense (FAISS) retrieval results
        faiss_id_to_chunk_id: Mapping from FAISS ID to chunk_id
        top_k: Number of top results to consider for overlap/std

    Returns:
        RetrievalSignals with all computed values
    """
    # Dense score signals
    dense_score_std = compute_dense_score_std(dense_hits, top_k=top_k)
    dense_top_score = dense_hits[0].score if dense_hits else 0.0

    # Overlap signal
    overlap = compute_overlap(bm25_hits, dense_hits, faiss_id_to_chunk_id, top_k=top_k)

    # Quote intent
    quote_intent = detect_quote_intent(query)

    logger.debug(
        f"Signals computed: dense_std={dense_score_std:.4f}, "
        f"dense_top={dense_top_score:.4f}, overlap={overlap}, quote_intent={quote_intent}"
    )

    return RetrievalSignals(
        dense_score_std=dense_score_std,
        dense_top_score=dense_top_score,
        bm25_dense_overlap=overlap,
        quote_intent=quote_intent,
    )
