"""
Reranker using HuggingFace transformers (offline/dev usage).

This module provides a Qwen3-Reranker implementation for:
- Development on Apple Silicon (MPS)
- Offline evaluation and testing
- Fallback when vLLM is not available

For production online serving, use vllm_reranker.py instead.

Key design:
- Uses AutoModelForCausalLM with yes/no token logits
- Domain-specific instruction for William Branham sermon retrieval
- Cached model loading from HF_HOME/.hf-cache
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from branham_model_api.utils.device import get_device, get_dtype


@dataclass(frozen=True)
class RerankerConfig:
    """Configuration for HuggingFace-based reranker."""

    model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    # Maximum sequence length for reranking (Qwen3-Reranker supports 32k)
    max_length: int = 32768
    # Data type for model weights
    dtype: str = "fp16"  # fp16/bf16/fp32
    # Device preference
    device_preference: str = "auto"
    # Trust remote code for custom model implementations
    trust_remote_code: bool = True
    # Force offline loading (no downloads)
    local_files_only: bool = False
    # Explicit cache directory (or use HF_HOME)
    cache_dir: str | None = None
    # Batch size for reranking
    batch_size: int = 8
    # Domain-specific instruction for William Branham sermons
    instruction: str = "Given a question about William Branham's teachings or sermons, determine if the document contains relevant information to answer the query"


class Reranker:
    """
    HuggingFace-based reranker using Qwen3-Reranker.

    Uses yes/no token logits to compute relevance scores between
    query-document pairs.
    """

    def __init__(self, cfg: RerankerConfig):
        self.cfg = cfg
        self.device = get_device(cfg.device_preference)
        self.dtype = get_dtype(cfg.dtype)

        # Load tokenizer with left padding (required for Qwen)
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            padding_side="left",
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
            cache_dir=cfg.cache_dir,
        )

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=cfg.local_files_only,
            cache_dir=cfg.cache_dir,
        )
        self.model.eval()
        self.model.to(self.device)

        # Apply dtype
        if self.device.type in {"cuda", "mps"}:
            self.model.to(dtype=self.dtype)

        # Get yes/no token IDs for scoring
        self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")

        # Prompt format for Qwen3-Reranker
        self.prefix = (
            "<|im_start|>system\n"
            "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
            "Note that the answer can only be \"yes\" or \"no\"."
            "<|im_end|>\n<|im_start|>user\n"
        )
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

        # Pre-tokenize prefix/suffix
        self.prefix_tokens = self.tokenizer.encode(self.prefix, add_special_tokens=False)
        self.suffix_tokens = self.tokenizer.encode(self.suffix, add_special_tokens=False)

    def _format_pair(self, query: str, document: str, instruction: str | None = None) -> str:
        """Format a query-document pair for reranking."""
        inst = instruction or self.cfg.instruction
        return f"<Instruct>: {inst}\n<Query>: {query}\n<Document>: {document}"

    def _process_inputs(self, pairs: list[str]) -> dict:
        """Tokenize and prepare inputs for the model."""
        # Tokenize without prefix/suffix (we'll add them manually)
        inputs = self.tokenizer(
            pairs,
            padding=False,
            truncation=True,
            return_attention_mask=False,
            max_length=self.cfg.max_length - len(self.prefix_tokens) - len(self.suffix_tokens),
        )

        # Add prefix and suffix tokens
        for i, token_ids in enumerate(inputs["input_ids"]):
            inputs["input_ids"][i] = self.prefix_tokens + token_ids + self.suffix_tokens

        # Pad and convert to tensors
        inputs = self.tokenizer.pad(
            inputs,
            padding=True,
            return_tensors="pt",
            max_length=self.cfg.max_length,
        )

        # Move to device
        return {k: v.to(self.device) for k, v in inputs.items()}

    @torch.no_grad()
    def _compute_scores(self, inputs: dict) -> list[float]:
        """Compute relevance scores from model logits."""
        outputs = self.model(**inputs)
        # Get logits for the last token position
        batch_scores = outputs.logits[:, -1, :]

        # Extract yes/no logits
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]

        # Stack and compute log softmax
        stacked = torch.stack([false_vector, true_vector], dim=1)
        log_probs = torch.nn.functional.log_softmax(stacked, dim=1)

        # Return probability of "yes" (relevance score)
        scores = log_probs[:, 1].exp().tolist()
        return scores

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        *,
        instruction: str | None = None,
    ) -> list[tuple[int, float]]:
        """
        Rerank documents by relevance to query.

        Args:
            query: The search query
            documents: List of document texts to rerank
            instruction: Optional custom instruction (default: sermon retrieval)

        Returns:
            List of (original_index, score) tuples, sorted by score descending
        """
        if not documents:
            return []

        # Format all pairs
        pairs = [self._format_pair(query, doc, instruction) for doc in documents]

        # Process in batches
        all_scores: list[float] = []
        for i in range(0, len(pairs), self.cfg.batch_size):
            batch = pairs[i : i + self.cfg.batch_size]
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores)

        # Return sorted by score descending
        indexed_scores = list(enumerate(all_scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)
        return indexed_scores

    def score_pairs(
        self,
        pairs: Sequence[tuple[str, str]],
        *,
        instruction: str | None = None,
    ) -> list[float]:
        """
        Score a list of (query, document) pairs.

        Args:
            pairs: List of (query, document) tuples
            instruction: Optional custom instruction

        Returns:
            List of relevance scores (0-1)
        """
        if not pairs:
            return []

        # Format all pairs
        formatted = [self._format_pair(q, d, instruction) for q, d in pairs]

        # Process in batches
        all_scores: list[float] = []
        for i in range(0, len(formatted), self.cfg.batch_size):
            batch = formatted[i : i + self.cfg.batch_size]
            inputs = self._process_inputs(batch)
            scores = self._compute_scores(inputs)
            all_scores.extend(scores)

        return all_scores
