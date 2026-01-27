"""
Reranker module for conditional reranking in the RAG pipeline.

Provides both HuggingFace (offline/dev) and vLLM (online/prod) implementations
of Qwen3-Reranker for scoring query-document relevance.

Usage (HuggingFace - dev/offline):
    from branham_model_api.retrieval.reranker import Reranker, RerankerConfig

    cfg = RerankerConfig(cache_dir=".hf-cache")
    reranker = Reranker(cfg)
    ranked = reranker.rerank("What is faith?", ["doc1...", "doc2..."])

Usage (vLLM - production):
    from branham_model_api.retrieval.reranker import VLLMReranker, VLLMRerankerConfig

    cfg = VLLMRerankerConfig()
    reranker = VLLMReranker(cfg)
    ranked = reranker.rerank("What is faith?", ["doc1...", "doc2..."])
"""

from .reranker import Reranker, RerankerConfig
from .vllm_reranker import VLLMReranker, VLLMRerankerConfig, create_qwen_vllm_reranker

__all__ = [
    # HuggingFace implementation (offline/dev)
    "Reranker",
    "RerankerConfig",
    # vLLM implementation (online/prod)
    "VLLMReranker",
    "VLLMRerankerConfig",
    "create_qwen_vllm_reranker",
]
