"""
Dense embedding utilities (Stage 4).

Key requirements (from `.cursorrules` + datasets/docs/DATA_FORMAT.md + DENSE_RETRIEVAL.md):
- Model choice is config-driven (no hard-coded model IDs).
- English corpus is embedded; queries may be multilingual (prefer multilingual embedders).
- Embeddings are normalized (unit length) and searched with cosine similarity implemented as IP.
- Deterministic behavior where possible (stable chunk ordering; fixed max_length; stable id map).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Iterable, Literal, Sequence

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from branham_model_api.utils.device import get_device, get_dtype


PoolingStrategy = Literal["mean", "cls", "pooler"]


@dataclass(frozen=True)
class EmbedderConfig:
    model_id: str
    dim: int | None = None  # Optional truncation (MRL-friendly models like jina v3)
    dtype: str = "fp16"  # fp16/bf16/fp32
    batch_size: int = 32
    max_length: int = 512
    pooling: PoolingStrategy = "mean"
    normalize: bool = True
    trust_remote_code: bool = True
    # If set, forces HF to load from local cache only (no downloads).
    local_files_only: bool = False
    # Optional explicit cache directory (otherwise HF_HOME/default cache is used).
    cache_dir: str | None = None
    query_prefix: str = ""
    doc_prefix: str = ""
    # For models that require task routing (e.g. jina-embeddings-v4).
    task_label: str | None = None  # e.g., "retrieval"
    query_prompt_name: str | None = "query"
    doc_prompt_name: str | None = "passage"
    device_preference: str = "auto"


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: [B, T, H], attention_mask: [B, T]
    mask = attention_mask.unsqueeze(-1).to(dtype=last_hidden.dtype)  # [B, T, 1]
    summed = (last_hidden * mask).sum(dim=1)  # [B, H]
    denom = mask.sum(dim=1).clamp(min=1.0)  # [B, 1]
    return summed / denom


def _l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(p=2, dim=-1, keepdim=True).clamp(min=eps))


class DenseEmbedder:
    """
    Hugging Face transformer embedder with explicit pooling.

    Notes:
    - Many embedding models publish recommended pooling; we keep pooling configurable.
    - We avoid adding extra frameworks (e.g., sentence-transformers) for simplicity and control.
    """

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg

        self.device = get_device(cfg.device_preference)
        self.dtype = get_dtype(cfg.dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg.model_id,
            trust_remote_code=cfg.trust_remote_code,
            local_files_only=bool(cfg.local_files_only),
            cache_dir=cfg.cache_dir,
        )

        # PyTorch path
        try:
            self.model = AutoModel.from_pretrained(
                cfg.model_id,
                trust_remote_code=cfg.trust_remote_code,
                local_files_only=bool(cfg.local_files_only),
                cache_dir=cfg.cache_dir,
            )
        except ImportError as e:
            msg = str(e)
            if "Torchvision" in msg or "torchvision" in msg:
                raise RuntimeError(
                    f"Embedding model {cfg.model_id!r} requires extra deps (torchvision, usually pillow). "
                    "Install them (e.g., `uv add torchvision pillow && uv sync`) and retry."
                ) from e
            raise
        self.model.eval()
        self.model.to(self.device)

        # Cache forward parameters so we can pass model-specific kwargs safely.
        if self.model is not None:
            try:
                self._forward_params = set(inspect.signature(self.model.forward).parameters.keys())
            except Exception:  # pragma: no cover
                self._forward_params = set()

            # Precision:
            # - CUDA: bf16/fp16 are generally safe and much faster.
            # - MPS: fp16 is usually the safe speed default; bf16 support can vary by stack/version.
            if self.device.type == "cuda":
                self.model.to(dtype=self.dtype)
            elif self.device.type == "mps":
                if self.dtype == torch.bfloat16:
                    # Best-effort: attempt bf16, but fail fast with a clear message if unsupported.
                    try:
                        self.model.to(dtype=self.dtype)
                    except Exception as e:  # pragma: no cover
                        raise RuntimeError(
                            "bf16 requested on MPS but not supported by this runtime. "
                            "Use fp16 for Apple Silicon development, and bf16 on NVIDIA A/H-series."
                        ) from e
                else:
                    self.model.to(dtype=self.dtype)
        else:
            self._forward_params = set()

        # Some embedding models behave better with pad token set; ensure we have one.
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token_id is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token


    def tokenizer_max_length(self) -> int | None:
        """
        Best-effort max sequence length discovery.

        Tokenizers sometimes report a huge sentinel value; if so, fall back to model config.
        """
        encode_text = getattr(self.model, "encode_text", None)
        if callable(encode_text):
            # Some models (e.g. jina v4) advertise max_length via encode_text signature.
            try:
                sig = inspect.signature(encode_text)
                p = sig.parameters.get("max_length")
                if p is not None and isinstance(p.default, int) and 0 < p.default < 100_000:
                    return int(p.default)
            except Exception:  # pragma: no cover
                pass

        ml = getattr(self.tokenizer, "model_max_length", None)
        if isinstance(ml, int) and 1 <= ml <= 1_000_000:
            return ml
        mconf = getattr(self.model, "config", None)
        mpe = getattr(mconf, "max_position_embeddings", None)
        if isinstance(mpe, int) and mpe > 0:
            return mpe
        return None

    def _encode_texts(self, texts: Sequence[str], *, prompt_name: str | None = None) -> np.ndarray:
        cfg = self.cfg
        if not texts:
            return np.zeros((0, cfg.dim or 0), dtype=np.float32)

        # Model-native encoding path (preferred when available).
        # Example: jina-embeddings-v4 exposes encode_text() and handles its own processor + task routing.
        encode_text = getattr(self.model, "encode_text", None)
        if callable(encode_text):
            with torch.no_grad():
                emb = encode_text(
                    texts=list(texts),
                    task=(cfg.task_label or "retrieval"),
                    max_length=int(cfg.max_length),
                    batch_size=int(cfg.batch_size),
                    return_multivector=False,
                    return_numpy=False,
                    truncate_dim=int(cfg.dim) if cfg.dim is not None and cfg.dim > 0 else None,
                    prompt_name=prompt_name,
                )
            if isinstance(emb, list):
                # Defensive: some implementations may return per-item tensors
                emb_t = torch.stack([e for e in emb if isinstance(e, torch.Tensor)], dim=0)
            elif isinstance(emb, torch.Tensor):
                emb_t = emb
            else:  # pragma: no cover
                raise TypeError(f"encode_text returned unexpected type: {type(emb)}")

            if cfg.normalize:
                emb_t = _l2_normalize(emb_t)
            vec = emb_t.to(dtype=torch.float32).cpu().numpy()
            return vec.astype(np.float32, copy=False)

        out: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, len(texts), cfg.batch_size):
                batch = texts[i : i + cfg.batch_size]
                toks = self.tokenizer(
                    list(batch),
                    padding=True,
                    truncation=True,
                    max_length=int(cfg.max_length),
                    return_tensors="pt",
                )
                toks = {k: v.to(self.device) for k, v in toks.items()}

                extra_kwargs: dict[str, object] = {}
                # jina-embeddings-v4 (and similar) require task selection.
                if "task_label" in self._forward_params:
                    extra_kwargs["task_label"] = cfg.task_label or "retrieval"
                elif "task" in self._forward_params:
                    extra_kwargs["task"] = cfg.task_label or "retrieval"

                # Some models accept prompt routing (query vs passage).
                if prompt_name and "prompt_name" in self._forward_params:
                    extra_kwargs["prompt_name"] = prompt_name

                # Prefer a single-vector embedding when the model supports multivector outputs.
                if "return_multivector" in self._forward_params:
                    extra_kwargs["return_multivector"] = False

                # Autocast helps on CUDA; on MPS, float16 can be finicky for some ops.
                # We still respect cfg.dtype but keep computation stable by letting PyTorch pick where needed.
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype,
                    enabled=(self.device.type in {"cuda"} and self.dtype in {torch.float16, torch.bfloat16}),
                ):
                    outputs = self.model(**toks, **extra_kwargs)

                # Some embedding models return embeddings directly (tuple/list/tensor).
                emb: torch.Tensor
                if isinstance(outputs, torch.Tensor):
                    emb = outputs
                elif isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], torch.Tensor):
                    emb = outputs[0]
                else:
                    # Standard transformer outputs: pool from hidden states.
                    if cfg.pooling == "pooler" and hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
                        emb = outputs.pooler_output
                    elif cfg.pooling == "cls":
                        emb = outputs.last_hidden_state[:, 0, :]
                    else:
                        emb = _mean_pool(outputs.last_hidden_state, toks["attention_mask"])

                if cfg.normalize:
                    emb = _l2_normalize(emb)

                emb = emb.to(dtype=torch.float32).cpu()
                vec = emb.numpy()
                if cfg.dim is not None and cfg.dim > 0 and vec.shape[1] >= cfg.dim:
                    vec = vec[:, : cfg.dim]
                out.append(vec)

        return np.vstack(out).astype(np.float32, copy=False)

    def embed_documents(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.cfg.doc_prefix or ""
        if prefix:
            texts = [prefix + t for t in texts]
        return self._encode_texts(texts, prompt_name=self.cfg.doc_prompt_name)

    def embed_queries(self, texts: Sequence[str]) -> np.ndarray:
        prefix = self.cfg.query_prefix or ""
        if prefix:
            texts = [prefix + t for t in texts]
        return self._encode_texts(texts, prompt_name=self.cfg.query_prompt_name)

    def embed_iter(self, texts: Iterable[str], *, kind: Literal["doc", "query"]) -> np.ndarray:
        # Convenience when upstream produces iterables; we materialize because tokenizers need lengths.
        materialized = list(texts)
        if kind == "doc":
            return self.embed_documents(materialized)
        return self.embed_queries(materialized)

