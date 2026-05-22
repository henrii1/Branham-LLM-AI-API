"""
RAG Pipeline modules.

Provides the retrieval pipeline from user query to context-ready sermons:
- signals: Compute retrieval signals for conditional reranking
- fusion: Merge BM25 + dense results, deduplicate
- expansion: Expand context by ±1 chunks
- rag_pipeline: Main orchestrator

Usage:
    from branham_model_api.core.pipeline import (
        RAGPipeline,
        RetrievalConfig,
        RetrievalResult,
        create_rag_pipeline,
    )

    pipeline = create_rag_pipeline(
        bm25_index_path="data/indices/bm25.index",
        faiss_index_path="data/indices/faiss.index",
        faiss_id_map_path="data/indices/faiss_id_map.jsonl",
        chunk_store_path="data/processed/chunks.sqlite",
        embedder=embedder,
        reranker=reranker,
    )

    result = pipeline.retrieve("What did Brother Branham say about faith?")

    if result.should_refuse:
        print(result.refuse_reason)
    else:
        for sermon in result.expanded_sermons:
            print(f"Sermon: {sermon.date_id}, Score: {sermon.best_score}")
"""

from .expansion import (
    ExpandedChunk,
    ExpandedSermon,
    expand_and_group,
    expand_chunks,
    format_all_context,
    format_sermon_context,
    group_expanded_by_sermon,
)
from .fusion import (
    FusedHit,
    SermonGroup,
    apply_rerank_scores,
    collate_by_sermon,
    find_exact_match_sermons,
    get_all_chunks_from_sermons,
    merge_bm25_dense,
)
from .rag_pipeline import (
    EmbedderProtocol,
    RAGPipeline,
    RerankerProtocol,
    RetrievalConfig,
    RetrievalResult,
    create_rag_pipeline,
    load_faiss_id_map,
    normalize_query,
)
from .postcheck import (
    PostcheckResult,
    finalize_answer,
    has_bible_reference,
    is_comparison_query,
    has_sermon_reference,
    is_bible_query,
)
from .signals import (
    RetrievalSignals,
    compute_dense_score_std,
    compute_overlap,
    compute_signals,
    detect_quote_intent,
)
from .early_gates import (
    is_specific_fact_query,
    is_unclear_query,
)

__all__ = [
    # Main pipeline
    "RAGPipeline",
    "RetrievalConfig",
    "RetrievalResult",
    "create_rag_pipeline",
    "load_faiss_id_map",
    "normalize_query",
    # Postcheck
    "PostcheckResult",
    "finalize_answer",
    "is_bible_query",
    "is_comparison_query",
    "has_sermon_reference",
    "has_bible_reference",
    # Protocols
    "EmbedderProtocol",
    "RerankerProtocol",
    # Signals
    "RetrievalSignals",
    "compute_signals",
    "compute_dense_score_std",
    "compute_overlap",
    "detect_quote_intent",
    # Fusion
    "FusedHit",
    "SermonGroup",
    "merge_bm25_dense",
    "apply_rerank_scores",
    "collate_by_sermon",
    "find_exact_match_sermons",
    "get_all_chunks_from_sermons",
    # Expansion
    "ExpandedChunk",
    "ExpandedSermon",
    "expand_chunks",
    "expand_and_group",
    "group_expanded_by_sermon",
    "format_sermon_context",
    "format_all_context",
    # Early refusal gates
    "is_specific_fact_query",
    "is_unclear_query",
]
