"""
vLLM-based embedder for online query embedding.

This module provides a vLLM-backed embedder for low-latency query embedding
at serving time. The offline index build uses the standard HuggingFace
DenseEmbedder (see embedder.py), while this module handles online queries.

Key design choices:
- Separate vLLM instances for embedder and reranker (independent scaling)
- Query instruction template applied for Qwen3-Embedding style models
- L2 normalization for cosine similarity (inner product in FAISS)
- Lazy initialization to avoid loading model until first use

Usage:
    from branham_model_api.retrieval.dense.vllm_embedder import VLLMEmbedder, VLLMEmbedderConfig

    cfg = VLLMEmbedderConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        query_instruction_template="Instruct: {task}\nQuery:{query}",
    )
    embedder = VLLMEmbedder(cfg)
    vectors = embedder.embed_queries(["What is faith?", "Explain the seven seals"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class VLLMEmbedderConfig:
    """Configuration for vLLM-based embedder."""

    model_id: str
    # Normalization for cosine similarity (should match index build)
    normalize: bool = True
    # Query instruction template for Qwen3-Embedding style models
    # Use {task} and {query} placeholders
    query_instruction_template: str | None = None
    # Task description for instruction template.
    # Default is tailored for William Branham sermon retrieval.
    query_task_description: str = "Given a question about William Branham's teachings or sermons, retrieve relevant sermon passages that answer the query"
    # Simple prefix (alternative to instruction template, e.g., E5 "query: ")
    query_prefix: str = ""
    # vLLM-specific settings
    gpu_memory_utilization: float = 0.5
    # Maximum model length for Qwen3-Embedding (supports up to 32k).
    # Online queries may include conversation history, so we use full context.
    max_model_len: int = 32768
    # Trust remote code for custom model implementations
    trust_remote_code: bool = True
    # Tensor parallel size (for multi-GPU)
    tensor_parallel_size: int = 1
    # Optional dtype override (None = auto)
    dtype: str | None = None
    # Enforce eager mode (disable CUDA graphs for debugging)
    enforce_eager: bool = False
    # Additional vLLM kwargs
    vllm_kwargs: dict = field(default_factory=dict)


class VLLMEmbedder:
    """
    vLLM-based embedder for online query embedding.

    Uses vLLM's native embed() method for efficient batched inference.
    Designed for low-latency serving; use DenseEmbedder for offline index builds.
    """

    def __init__(self, cfg: VLLMEmbedderConfig):
        self.cfg = cfg
        self._model = None  # Lazy init

    def _ensure_model(self):
        """Lazy-load the vLLM model on first use."""
        if self._model is not None:
            return

        try:
            from vllm import LLM
        except ImportError as e:
            raise ImportError(
                "vLLM is required for online embedding. Install with: pip install vllm>=0.8.5"
            ) from e

        cfg = self.cfg
        kwargs = {
            "model": cfg.model_id,
            "task": "embed",
            "trust_remote_code": cfg.trust_remote_code,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "max_model_len": cfg.max_model_len,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "enforce_eager": cfg.enforce_eager,
            **cfg.vllm_kwargs,
        }
        if cfg.dtype:
            kwargs["dtype"] = cfg.dtype

        self._model = LLM(**kwargs)

    def _format_queries(self, queries: Sequence[str]) -> list[str]:
        """
        Apply instruction template and/or prefix to queries.

        For Qwen3-Embedding: "Instruct: {task}\nQuery:{query}"
        For E5-style models: "query: {query}"
        """
        cfg = self.cfg
        formatted = list(queries)

        # Apply instruction template if configured (Qwen3-Embedding style)
        if cfg.query_instruction_template:
            template = cfg.query_instruction_template.replace("{task}", cfg.query_task_description)
            formatted = [template.replace("{query}", q) for q in formatted]

        # Apply simple prefix if configured (E5 style)
        if cfg.query_prefix:
            formatted = [cfg.query_prefix + q for q in formatted]

        return formatted

    def embed_queries(self, queries: Sequence[str]) -> np.ndarray:
        """
        Embed query texts using vLLM.

        Args:
            queries: List of query strings

        Returns:
            np.ndarray of shape (n_queries, dim) with dtype float32
        """
        if not queries:
            return np.zeros((0, 0), dtype=np.float32)

        self._ensure_model()
        formatted = self._format_queries(queries)

        # vLLM embed() returns list of EmbeddingRequestOutput
        outputs = self._model.embed(formatted)

        # Extract embeddings and stack
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])

        # Normalize for cosine similarity (inner product in FAISS)
        if self.cfg.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    def embed_documents(self, documents: Sequence[str]) -> np.ndarray:
        """
        Embed document texts using vLLM.

        Note: Documents don't get instruction template (only queries do).
        For offline index builds, use DenseEmbedder instead.
        """
        if not documents:
            return np.zeros((0, 0), dtype=np.float32)

        self._ensure_model()

        # Documents don't get instruction template, only optional prefix
        formatted = list(documents)
        doc_prefix = ""  # Documents typically don't need prefix
        if doc_prefix:
            formatted = [doc_prefix + d for d in formatted]

        outputs = self._model.embed(formatted)
        embeddings = torch.tensor([o.outputs.embedding for o in outputs])

        if self.cfg.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings.cpu().numpy().astype(np.float32)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def create_qwen_vllm_embedder(
    model_id: str = "Qwen/Qwen3-Embedding-0.6B",
    gpu_memory_utilization: float = 0.5,
    max_model_len: int = 32768,
    **kwargs,
) -> VLLMEmbedder:
    """
    Factory function to create a Qwen3-Embedding vLLM embedder with sensible defaults.

    This is the recommended way to create an embedder for online query embedding.
    Qwen3-Embedding-0.6B supports up to 32k context length.
    """
    cfg = VLLMEmbedderConfig(
        model_id=model_id,
        query_instruction_template="Instruct: {task}\nQuery:{query}",
        query_task_description="Given a question about William Branham's teachings or sermons, retrieve relevant sermon passages that answer the query",
        normalize=True,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        **kwargs,
    )
    return VLLMEmbedder(cfg)
