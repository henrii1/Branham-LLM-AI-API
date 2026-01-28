"""
Configuration loader for Branham Model API.

Loads settings from config/default.yaml and provides typed access.
"""

from __future__ import annotations

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
    min_dense_score: float = 0.55


@dataclass
class AppConfig:
    """Full application configuration."""
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)
    
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
            min_dense_score=ret.get("refusal", {}).get("min_dense_score", 0.55),
        )
        
        # Parse index paths
        indices = raw.get("indices", {})
        
        return cls(
            retrieval=retrieval,
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
