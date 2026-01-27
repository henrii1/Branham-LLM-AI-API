#!/usr/bin/env python3
"""
Test script for the RAG pipeline.

Tests 20 queries covering:
- General questions
- Quote-seeking queries (should trigger reranker via quote_intent)
- Ambiguous queries (may trigger reranker via signals)
- Out-of-domain queries (should refuse)

Usage:
    uv run python scripts/test_rag_pipeline.py
"""

import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from branham_model_api.core.pipeline import (
    RAGPipeline,
    RetrievalConfig,
    RetrievalResult,
    create_rag_pipeline,
)
from branham_model_api.retrieval.dense import DenseEmbedder, EmbedderConfig
from branham_model_api.retrieval.reranker import Reranker, RerankerConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
# Set our modules to DEBUG for detailed output
logging.getLogger("branham_model_api.core.pipeline").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

# Test queries - 20 total
TEST_QUERIES = [
    # === General sermon questions (5) ===
    "What did Brother Branham teach about faith?",
    "Explain the seven church ages",
    "What is the third pull?",
    "Tell me about the pillar of fire",
    "What did Branham say about healing?",
    
    # === Quote-seeking queries - should trigger reranker (5) ===
    "What exact quote did Brother Branham say about the rapture?",
    "Where did he say 'thus saith the Lord'?",
    "Which sermon talks about the seven seals?",
    "What did Brother Branham say in paragraph 45 of 65-1128M?",
    "Quote from the sermon on faith",
    
    # === Specific sermon references (3) ===
    "What is taught in 65-0711 sermon?",
    "Explain the message from 63-0318",
    "What happened in the 1933 vision?",
    
    # === Theological questions (4) ===
    "What is the serpent's seed doctrine?",
    "Explain predestination according to Branham",
    "What are the seven thunders?",
    "Tell me about the spoken word",
    
    # === Out-of-domain / should refuse (3) ===
    "What is the weather like today?",
    "How do I cook pasta?",
    "Tell me about quantum physics",
]


def run_test(
    pipeline: RAGPipeline,
    query: str,
    query_num: int,
) -> dict:
    """Run a single test query and return summary stats."""
    print(f"\n{'='*80}")
    print(f"QUERY {query_num}: {query}")
    print("="*80)
    
    result = pipeline.retrieve(query)
    
    # Summary
    summary = {
        "query": query,
        "quote_intent": result.quote_intent,
        "reranker_triggered": result.reranker_triggered,
        "should_refuse": result.should_refuse,
        "refuse_reason": result.refuse_reason,
        "bm25_hits": result.bm25_hit_count,
        "dense_hits": result.dense_hit_count,
        "fused_hits": result.fused_hit_count,
        "sermon_count": len(result.expanded_sermons),
        "total_chunks": result.total_chunks,
        "signals": {
            "dense_score_std": result.signals.dense_score_std,
            "dense_top_score": result.signals.dense_top_score,
            "bm25_dense_overlap": result.signals.bm25_dense_overlap,
        },
    }
    
    # Print result summary
    print(f"\nRESULT:")
    print(f"  Quote intent: {result.quote_intent}")
    print(f"  Reranker triggered: {result.reranker_triggered}")
    print(f"  Should refuse: {result.should_refuse}")
    if result.refuse_reason:
        print(f"  Refuse reason: {result.refuse_reason}")
    print(f"  Hits: BM25={result.bm25_hit_count}, Dense={result.dense_hit_count}, Fused={result.fused_hit_count}")
    print(f"  Signals: std={result.signals.dense_score_std:.4f}, top={result.signals.dense_top_score:.4f}, overlap={result.signals.bm25_dense_overlap}")
    
    if result.expanded_sermons:
        print(f"  Sermons: {len(result.expanded_sermons)}, Total chunks: {result.total_chunks}")
        for i, sermon in enumerate(result.expanded_sermons[:3]):
            title_str = f" - {sermon.title}" if sermon.title else ""
            print(f"    {i+1}. {sermon.date_id}{title_str} (score={sermon.best_score:.4f}, chunks={len(sermon.chunks)})")
    
    return summary


def main():
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    indices_dir = data_dir / "indices"
    processed_dir = data_dir / "processed"
    
    print("="*80)
    print("RAG PIPELINE TEST")
    print("="*80)
    print(f"BM25 index: {indices_dir / 'bm25.index'}")
    print(f"FAISS index: {indices_dir / 'faiss.index'}")
    print(f"FAISS ID map: {indices_dir / 'faiss_id_map.jsonl'}")
    print(f"Chunk store: {processed_dir / 'chunks.sqlite'}")
    
    # Create embedder (HuggingFace for development)
    print("\nLoading embedder (Qwen3-Embedding-0.6B)...")
    embedder_config = EmbedderConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        dim=1024,
        dtype="fp16",
        batch_size=1,
        max_length=512,  # Short for queries
        pooling="last_token",
        normalize=True,
        trust_remote_code=True,
        padding_side="left",
        query_instruction_template="Instruct: {task}\nQuery:{query}",
        query_task_description="Given a question about William Branham's teachings or sermons, retrieve relevant sermon passages that answer the query",
    )
    embedder = DenseEmbedder(embedder_config)
    print("Embedder loaded!")
    
    # Create reranker (optional - can be slow, set to None to skip)
    USE_RERANKER = False  # Set to True to test reranker
    reranker = None
    if USE_RERANKER:
        print("\nLoading reranker (Qwen3-Reranker-0.6B)...")
        reranker_config = RerankerConfig(
            model_id="Qwen/Qwen3-Reranker-0.6B",
            max_length=8192,  # Shorter for speed
            dtype="fp16",
            batch_size=4,
        )
        reranker = Reranker(reranker_config)
        print("Reranker loaded!")
    
    # Create pipeline config
    config = RetrievalConfig(
        bm25_top_n=25,
        dense_top_n=25,
        reranker_enabled=USE_RERANKER,
        score_std_threshold=0.015,  # Adjusted: very flat distribution
        overlap_threshold=1,  # Adjusted: no agreement at all
        top_score_threshold=0.55,  # Adjusted: weak semantic match
        max_sermons=8,
        expansion_delta=1,
        min_dense_score=0.55,  # Below this = off-topic
    )
    
    # Create pipeline
    print("\nCreating RAG pipeline...")
    pipeline = create_rag_pipeline(
        bm25_index_path=indices_dir / "bm25.index",
        faiss_index_path=indices_dir / "faiss.index",
        faiss_id_map_path=indices_dir / "faiss_id_map.jsonl",
        chunk_store_path=processed_dir / "chunks.sqlite",
        embedder=embedder,
        reranker=reranker,
        config=config,
    )
    print("Pipeline ready!")
    
    # Run tests
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        try:
            summary = run_test(pipeline, query, i)
            results.append(summary)
        except Exception as e:
            logger.error(f"Query {i} failed: {e}", exc_info=True)
            results.append({"query": query, "error": str(e)})
    
    # Final summary
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    
    total = len(results)
    errors = sum(1 for r in results if "error" in r)
    refused = sum(1 for r in results if r.get("should_refuse", False))
    reranked = sum(1 for r in results if r.get("reranker_triggered", False))
    quote_intents = sum(1 for r in results if r.get("quote_intent", False))
    
    print(f"Total queries: {total}")
    print(f"Errors: {errors}")
    print(f"Refused: {refused}")
    print(f"Reranker triggered: {reranked}")
    print(f"Quote intent detected: {quote_intents}")
    
    # Signal statistics
    valid_results = [r for r in results if "signals" in r]
    if valid_results:
        avg_std = sum(r["signals"]["dense_score_std"] for r in valid_results) / len(valid_results)
        avg_top = sum(r["signals"]["dense_top_score"] for r in valid_results) / len(valid_results)
        avg_overlap = sum(r["signals"]["bm25_dense_overlap"] for r in valid_results) / len(valid_results)
        print(f"\nSignal averages:")
        print(f"  dense_score_std: {avg_std:.4f} (threshold: {config.score_std_threshold})")
        print(f"  dense_top_score: {avg_top:.4f} (threshold: {config.top_score_threshold})")
        print(f"  bm25_dense_overlap: {avg_overlap:.1f} (threshold: {config.overlap_threshold})")
    
    # Identify queries that would trigger reranker
    print("\nQueries that WOULD trigger reranker (if enabled):")
    for r in valid_results:
        triggers = []
        if r["signals"]["dense_score_std"] < config.score_std_threshold:
            triggers.append(f"std={r['signals']['dense_score_std']:.4f}")
        if r["signals"]["bm25_dense_overlap"] < config.overlap_threshold:
            triggers.append(f"overlap={r['signals']['bm25_dense_overlap']}")
        if r["signals"]["dense_top_score"] < config.top_score_threshold:
            triggers.append(f"top={r['signals']['dense_top_score']:.4f}")
        if r["quote_intent"]:
            triggers.append("quote_intent")
        
        if triggers:
            print(f"  - '{r['query'][:50]}...' -> {', '.join(triggers)}")


if __name__ == "__main__":
    main()
