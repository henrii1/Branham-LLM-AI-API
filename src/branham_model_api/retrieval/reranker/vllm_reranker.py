"""
vLLM-based reranker for online serving.

This module provides a vLLM-backed reranker for low-latency reranking
at serving time. Uses Qwen3-Reranker with yes/no token logits.

Key design:
- Separate vLLM instance from embedding (independent scaling)
- Prefix caching enabled for efficiency
- Domain-specific instruction for William Branham sermon retrieval
- Lazy initialization to avoid loading model until first use

Usage:
    from branham_model_api.retrieval.reranker.vllm_reranker import VLLMReranker, VLLMRerankerConfig

    cfg = VLLMRerankerConfig(model_id="Qwen/Qwen3-Reranker-0.6B")
    reranker = VLLMReranker(cfg)
    ranked = reranker.rerank("What is faith?", ["doc1...", "doc2..."])
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Sequence


@dataclass(frozen=True)
class VLLMRerankerConfig:
    """Configuration for vLLM-based reranker."""

    model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    # Maximum sequence length (Qwen3-Reranker supports 32k)
    max_model_len: int = 32768
    # GPU memory utilization (lower = more headroom)
    gpu_memory_utilization: float = 0.6
    # Enable prefix caching for efficiency
    enable_prefix_caching: bool = True
    # Tensor parallel size (for multi-GPU)
    tensor_parallel_size: int = 1
    # Trust remote code
    trust_remote_code: bool = True
    # Enforce eager mode (disable CUDA graphs for debugging)
    enforce_eager: bool = False
    # Domain-specific instruction for William Branham sermons
    instruction: str = "Given a question about William Branham's teachings or sermons, determine if the document contains relevant information to answer the query"
    # Additional vLLM kwargs
    vllm_kwargs: dict = field(default_factory=dict)


class VLLMReranker:
    """
    vLLM-based reranker for online serving.

    Uses Qwen3-Reranker with yes/no token logits to compute relevance scores.
    Designed for low-latency production serving.
    """

    def __init__(self, cfg: VLLMRerankerConfig):
        self.cfg = cfg
        self._model = None
        self._tokenizer = None
        self._sampling_params = None
        self._true_token = None
        self._false_token = None
        self._suffix_tokens = None

    def _ensure_model(self):
        """Lazy-load the vLLM model on first use."""
        if self._model is not None:
            return

        try:
            from vllm import LLM, SamplingParams
            from vllm.inputs.data import TokensPrompt
        except ImportError as e:
            raise ImportError(
                "vLLM is required for online reranking. Install with: pip install vllm>=0.8.5"
            ) from e

        from transformers import AutoTokenizer

        cfg = self.cfg

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
        self._tokenizer.padding_side = "left"
        self._tokenizer.pad_token = self._tokenizer.eos_token

        # Get yes/no token IDs
        self._true_token = self._tokenizer("yes", add_special_tokens=False).input_ids[0]
        self._false_token = self._tokenizer("no", add_special_tokens=False).input_ids[0]

        # Suffix for chat format
        suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self._suffix_tokens = self._tokenizer.encode(suffix, add_special_tokens=False)

        # Sampling params for reranking
        self._sampling_params = SamplingParams(
            temperature=0,
            max_tokens=1,
            logprobs=20,
            allowed_token_ids=[self._true_token, self._false_token],
        )

        # Load vLLM model
        kwargs = {
            "model": cfg.model_id,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "max_model_len": cfg.max_model_len,
            "enable_prefix_caching": cfg.enable_prefix_caching,
            "gpu_memory_utilization": cfg.gpu_memory_utilization,
            "trust_remote_code": cfg.trust_remote_code,
            "enforce_eager": cfg.enforce_eager,
            **cfg.vllm_kwargs,
        }
        self._model = LLM(**kwargs)

        # Store TokensPrompt for later use
        self._TokensPrompt = TokensPrompt

    def _format_instruction(self, query: str, document: str, instruction: str | None = None) -> list[dict]:
        """Format query-document pair as chat messages."""
        inst = instruction or self.cfg.instruction
        return [
            {
                "role": "system",
                "content": (
                    "Judge whether the Document meets the requirements based on the Query and the Instruct provided. "
                    "Note that the answer can only be \"yes\" or \"no\"."
                ),
            },
            {
                "role": "user",
                "content": f"<Instruct>: {inst}\n\n<Query>: {query}\n\n<Document>: {document}",
            },
        ]

    def _process_inputs(
        self,
        pairs: Sequence[tuple[str, str]],
        instruction: str | None = None,
    ) -> list:
        """Prepare inputs for vLLM."""
        max_len = self.cfg.max_model_len - len(self._suffix_tokens)

        # Format as chat messages
        messages = [self._format_instruction(q, d, instruction) for q, d in pairs]

        # Apply chat template
        tokenized = self._tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )

        # Truncate and add suffix
        prompts = []
        for tokens in tokenized:
            truncated = tokens[:max_len] + self._suffix_tokens
            prompts.append(self._TokensPrompt(prompt_token_ids=truncated))

        return prompts

    def _compute_scores(self, prompts: list) -> list[float]:
        """Compute relevance scores from vLLM outputs."""
        outputs = self._model.generate(prompts, self._sampling_params, use_tqdm=False)

        scores = []
        for output in outputs:
            final_logits = output.outputs[0].logprobs[-1]

            # Get logprobs for yes/no tokens
            if self._true_token not in final_logits:
                true_logit = -10
            else:
                true_logit = final_logits[self._true_token].logprob

            if self._false_token not in final_logits:
                false_logit = -10
            else:
                false_logit = final_logits[self._false_token].logprob

            # Compute normalized score
            true_score = math.exp(true_logit)
            false_score = math.exp(false_logit)
            score = true_score / (true_score + false_score)
            scores.append(score)

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

        self._ensure_model()

        # Create query-document pairs
        pairs = [(query, doc) for doc in documents]
        prompts = self._process_inputs(pairs, instruction)
        scores = self._compute_scores(prompts)

        # Return sorted by score descending
        indexed_scores = list(enumerate(scores))
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

        self._ensure_model()

        prompts = self._process_inputs(pairs, instruction)
        return self._compute_scores(prompts)

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._model is not None


def create_qwen_vllm_reranker(
    model_id: str = "Qwen/Qwen3-Reranker-0.6B",
    gpu_memory_utilization: float = 0.6,
    **kwargs,
) -> VLLMReranker:
    """
    Factory function to create a Qwen3-Reranker vLLM instance with sensible defaults.

    This is the recommended way to create a reranker for online serving.
    """
    cfg = VLLMRerankerConfig(
        model_id=model_id,
        gpu_memory_utilization=gpu_memory_utilization,
        instruction="Given a question about William Branham's teachings or sermons, determine if the document contains relevant information to answer the query",
        **kwargs,
    )
    return VLLMReranker(cfg)
