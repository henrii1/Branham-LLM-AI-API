"""
Configuration loader for Branham Model API.

Loads settings from config/default.yaml and provides typed access.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _find_config_file() -> Path:
    """Find config/default.yaml relative to project root."""
    # Try relative to this file
    src_dir = Path(__file__).parent
    project_root = src_dir.parent.parent
    config_path = project_root / "config" / "default.yaml"
    
    if config_path.exists():
        return config_path
    
    # Try current working directory
    cwd_config = Path.cwd() / "config" / "default.yaml"
    if cwd_config.exists():
        return cwd_config
    
    raise FileNotFoundError(
        f"Could not find config/default.yaml. Tried:\n"
        f"  - {config_path}\n"
        f"  - {cwd_config}"
    )


def load_yaml_config(path: Path | None = None) -> dict[str, Any]:
    """Load raw YAML config as dictionary."""
    if path is None:
        path = _find_config_file()
    
    with open(path) as f:
        return yaml.safe_load(f)


@dataclass
class RerankerConfig:
    """Reranker configuration."""
    enabled: str = "never"  # always, conditional, never
    score_std_threshold: float = 0.015
    overlap_threshold: int = 1
    top_score_threshold: float = 0.55
    quote_intent: bool = True


@dataclass
class RetrievalSettings:
    """Retrieval pipeline settings from config."""
    # BM25
    bm25_top_n: int = 25
    
    # Dense
    dense_top_n: int = 25
    
    # Reranker
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    
    # Collation
    max_sermons: int = 8
    
    # Expansion
    expansion_delta: int = 0
    
    # Refusal
    min_dense_score: float = 0.20
    min_bm25_score: float = 0.20
    
    # Language detection: "never" or "auto"
    language_detection_mode: str = "never"


@dataclass
class LLMProviderSettings:
    """Active LLM provider configuration resolved from YAML."""
    provider: str = "deepseek"
    model: str = "deepseek/deepseek-chat"
    base_url: str | None = None
    key_prefix: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.2
    timeout: float = 30.0
    # Optional multi-provider route configs used when provider == "mixed".
    # Example:
    #   {"deepseek": {"model": "...", "base_url": "...", "key_prefix": "DEEPSEEK_API_KEY"},
    #    "openrouter": {"model": "...", "base_url": "...", "key_prefix": "OPENROUTER_API_KEY"}}
    routes: dict[str, dict[str, str | None]] = field(default_factory=dict)

    @property
    def effective_model(self) -> str:
        """Env var LLM_MODEL overrides YAML model."""
        return os.getenv("LLM_MODEL", self.model)


@dataclass
class ModelSettings:
    """Model settings for retrieval/generation."""
    embedding_model_id: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model_id: str = "Qwen/Qwen3-Reranker-0.6B"
    tokenizer_model_id: str | None = None
    llm: LLMProviderSettings = field(default_factory=LLMProviderSettings)

    @property
    def effective_llm_model(self) -> str:
        """Backwards-compatible accessor."""
        return self.llm.effective_model


@dataclass
class AppConfig:
    """Full application configuration."""
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    
    # Index paths
    bm25_path: str = "./data/indices/bm25.index"
    faiss_path: str = "./data/indices/faiss.index"
    chunk_store_path: str = "./data/processed/chunks.sqlite"
    
    @classmethod
    def from_yaml(cls, path: Path | None = None) -> "AppConfig":
        """Load config from YAML file."""
        raw = load_yaml_config(path)
        
        # Parse retrieval settings
        ret = raw.get("retrieval", {})
        reranker_raw = ret.get("reranker", {})
        
        reranker = RerankerConfig(
            enabled=reranker_raw.get("enabled", "never"),
            score_std_threshold=reranker_raw.get("triggers", {}).get("score_std_threshold", 0.015),
            overlap_threshold=reranker_raw.get("triggers", {}).get("overlap_threshold", 1),
            top_score_threshold=reranker_raw.get("triggers", {}).get("top_score_threshold", 0.55),
            quote_intent=reranker_raw.get("triggers", {}).get("quote_intent", True),
        )
        
        retrieval = RetrievalSettings(
            bm25_top_n=ret.get("bm25", {}).get("top_n", 25),
            dense_top_n=ret.get("dense", {}).get("top_n", 25),
            reranker=reranker,
            max_sermons=ret.get("collation", {}).get("max_sermons", 8),
            expansion_delta=ret.get("expansion", {}).get("depth", 0),
            min_dense_score=float(ret.get("refusal", {}).get("min_dense_score", 0.20)),
            min_bm25_score=float(ret.get("refusal", {}).get("min_bm25_score", 0.20)),
            language_detection_mode=ret.get("language_detection", {}).get("mode", "never"),
        )

        models_raw = raw.get("models", {})

        # Parse LLM provider config
        llm_raw = raw.get("llm", {})
        active_provider = llm_raw.get("provider", "deepseek")
        providers_raw = llm_raw.get("providers", {})
        provider_cfg = providers_raw.get(active_provider, {})

        routes: dict[str, dict[str, str | None]] = {}
        if str(active_provider).strip().lower() == "mixed":
            # Mixed mode: use DeepSeek direct + OpenRouter (DeepSeek) together.
            # Key distribution is driven by which env keys are present (4 deepseek + 1 openrouter).
            for rid in ("deepseek", "openrouter"):
                cfg = providers_raw.get(rid, {}) or {}
                routes[rid] = {
                    "model": cfg.get("model"),
                    "base_url": cfg.get("base_url"),
                    "key_prefix": cfg.get("key_prefix"),
                }
            # For backwards-compatible fields, prefer deepseek route as the "primary".
            provider_cfg = providers_raw.get("deepseek", {}) or {}

        llm_settings = LLMProviderSettings(
            provider=active_provider,
            model=provider_cfg.get("model", "deepseek/deepseek-chat"),
            base_url=provider_cfg.get("base_url"),
            key_prefix=provider_cfg.get("key_prefix", "DEEPSEEK_API_KEY"),
            temperature=float(llm_raw.get("temperature", 0.2)),
            timeout=float(llm_raw.get("timeout", 30.0)),
            routes=routes,
        )

        models = ModelSettings(
            embedding_model_id=models_raw.get(
                "embedding_model_id",
                "Qwen/Qwen3-Embedding-0.6B",
            ),
            reranker_model_id=models_raw.get(
                "reranker_model_id",
                "Qwen/Qwen3-Reranker-0.6B",
            ),
            tokenizer_model_id=models_raw.get("tokenizer_model_id"),
            llm=llm_settings,
        )
        
        # Parse index paths
        indices = raw.get("indices", {})
        
        return cls(
            retrieval=retrieval,
            models=models,
            bm25_path=indices.get("bm25_path", "./data/indices/bm25.index"),
            faiss_path=indices.get("faiss_path", "./data/indices/faiss.index"),
            chunk_store_path=indices.get("chunk_store_path", "./data/processed/chunks.sqlite"),
        )


# Singleton instance (lazy loaded)
_config: AppConfig | None = None


def get_config(reload: bool = False) -> AppConfig:
    """Get application config (singleton, loaded from default.yaml)."""
    global _config
    if _config is None or reload:
        _config = AppConfig.from_yaml()
    return _config
