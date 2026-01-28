#!/usr/bin/env python3
"""
Test script for the RAG pipeline.

Tests 20 queries covering:
- General questions
- Quote-seeking queries (should trigger reranker via quote_intent)
- Ambiguous queries (may trigger reranker via signals)
- Specific sermon queries (by date_id or title)
- Out-of-domain queries (should refuse)

Usage:
    uv run python scripts/test_rag_pipeline.py
    uv run python scripts/test_rag_pipeline.py --use-new-indices  # After rebuild
"""

import argparse
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
    
    # === Specific sermon references (should find via exact match) (3) ===
    "What is taught in 65-0711 sermon?",
    "Explain the message from 63-0318",
    "Tell me about the Spoken Word Is The Original Seed sermon",
    
    # === Theological questions (4) ===
    "What is the serpent's seed doctrine?",
    "Explain predestination according to Branham",
    "What are the seven thunders?",
    "What happened in the 1933 vision?",
    
    # === Out-of-domain / should refuse (3) ===
    "What is the weather like today?",
    "How do I cook pasta?",
    "Tell me about quantum physics",
]


def run_test(
    pipeline: RAGPipeline,
    query: str,
    query_num: int,
    faiss_id_map: dict[int, str],
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
        
        # Build a map from date_id to sermon_group for composite scores
        group_map = {g.date_id: g for g in result.sermon_groups}
        
        for i, sermon in enumerate(result.expanded_sermons):  # Show all sermons
            title_str = f" - {sermon.title}" if sermon.title else ""
            group = group_map.get(sermon.date_id)
            if group:
                # Show retrieved chunks (before expansion) and composite score
                print(
                    f"    {i+1}. {sermon.date_id}{title_str}\n"
                    f"       retrieved={group.chunk_count} chunks, expanded={len(sermon.chunks)} chunks\n"
                    f"       best_score={sermon.best_score:.4f}, composite={group.composite_score:.4f}"
                )
            else:
                # Exact match sermon (no composite score - added as bonus)
                print(
                    f"    {i+1}. {sermon.date_id}{title_str} [EXACT MATCH]\n"
                    f"       expanded={len(sermon.chunks)} chunks, best_score={sermon.best_score:.4f}"
                )
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Test RAG pipeline")
    parser.add_argument(
        "--use-new-indices",
        action="store_true",
        help="Use new indices (after rebuild with metadata). Default uses old v1_text_only indices.",
    )
    parser.add_argument(
        "--reranker",
        action="store_true",
        help="Enable reranker (Qwen3-Reranker-0.6B). Slower but potentially better ranking.",
    )
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(__file__).parent.parent / "data"
    indices_dir = data_dir / "indices"
    processed_dir = data_dir / "processed"
    
    # Choose index files based on flag
    if args.use_new_indices:
        bm25_path = indices_dir / "bm25.index"
        faiss_path = indices_dir / "faiss.index"
        faiss_id_map_path = indices_dir / "faiss_id_map.jsonl"
        print("Using NEW indices (with metadata)")
    else:
        bm25_path = indices_dir / "bm25_v1_text_only.index"
        faiss_path = indices_dir / "faiss_v1_text_only.index"
        faiss_id_map_path = indices_dir / "faiss_v1_text_only_id_map.jsonl"
        print("Using OLD indices (v1_text_only, no metadata)")
    
    print("="*80)
    print("RAG PIPELINE TEST")
    print("="*80)
    print(f"BM25 index: {bm25_path}")
    print(f"FAISS index: {faiss_path}")
    print(f"FAISS ID map: {faiss_id_map_path}")
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
    
    # Create reranker (optional - can be slow)
    USE_RERANKER = args.reranker
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
    
    # Create pipeline config from config/default.yaml
    config = RetrievalConfig.from_yaml()
    
    # Override reranker_mode if --reranker flag is set
    if USE_RERANKER:
        config.reranker_mode = "conditional"
    
    print(f"\nConfig loaded from default.yaml:")
    print(f"  reranker_mode: {config.reranker_mode}")
    print(f"  max_sermons: {config.max_sermons}")
    print(f"  min_dense_score: {config.min_dense_score}")
    
    # Create pipeline
    print("\nCreating RAG pipeline...")
    pipeline = create_rag_pipeline(
        bm25_index_path=bm25_path,
        faiss_index_path=faiss_path,
        faiss_id_map_path=faiss_id_map_path,
        chunk_store_path=processed_dir / "chunks.sqlite",
        embedder=embedder,
        reranker=reranker,
        config=config,
    )
    print("Pipeline ready!")
    
    # Load faiss_id_map for chunk_id display
    import json
    faiss_id_map = {}
    with open(faiss_id_map_path, "r") as f:
        for line in f:
            rec = json.loads(line)
            faiss_id_map[rec["faiss_id"]] = rec["chunk_id"]
    
    # Run tests
    results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        try:
            summary = run_test(pipeline, query, i, faiss_id_map)
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
    
    # Composite score explanation
    print("\n" + "="*80)
    print("COMPOSITE SCORE EXPLANATION")
    print("="*80)
    print("""
The composite score balances chunk frequency and retrieval score:

  composite = 0.5 * norm_chunk_count + 0.5 * norm_best_score

Where:
  - norm_chunk_count = chunk_count / max_chunk_count  (0-1 range)
  - norm_best_score = best_score / max_best_score    (0-1 range)
  - max values are computed across ALL sermons in the result set

Example: If max_chunk_count=10 and max_best_score=0.03:
  - Sermon A (8 chunks, score=0.025): 0.5*(8/10) + 0.5*(0.025/0.03) = 0.817
  - Sermon B (2 chunks, score=0.030): 0.5*(2/10) + 0.5*(0.030/0.03) = 0.600
  - Sermon A ranks higher due to more chunks despite lower score

This helps surface sermons with broad topical coverage, not just single high-scoring chunks.
""")
    
    # Specific sermon match analysis
    print("="*80)
    print("SPECIFIC SERMON MATCH ANALYSIS")
    print("="*80)
    
    # Check queries 11-13 which have specific sermon references
    specific_queries = [
        (11, "65-0711"),
        (12, "63-0318"),
        (13, "Spoken Word"),
    ]
    for q_idx, expected in specific_queries:
        r = results[q_idx - 1]  # 0-indexed
        if "error" in r:
            print(f"Q{q_idx}: ERROR - {r['error']}")
            continue
        
        # Get the sermon_count and check if the expected sermon is in the results
        # For now just print what was found
        print(f"Q{q_idx} (expected: {expected}):")
        print(f"  Sermons found: {r.get('sermon_count', 0)}")
        if not r.get("should_refuse", True):
            # The actual sermons are in expanded_sermons but we don't have that in summary
            # Just note that it wasn't refused
            print(f"  Query was processed (not refused)")
        else:
            print(f"  Query was REFUSED: {r.get('refuse_reason', 'unknown')}")


if __name__ == "__main__":
    main()
