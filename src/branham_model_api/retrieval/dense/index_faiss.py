"""
FAISS index build/load helpers for dense retrieval (Stage 4).

Indexing contract:
- Vectors correspond 1:1 with chunk rows in deterministic order.
- Mapping is stored externally as `faiss_id_map.jsonl` (faiss_id -> chunk_id).
- Similarity uses cosine over unit-normalized vectors, implemented as inner product in FAISS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import faiss  # type: ignore
import numpy as np


FaissIndexType = Literal["flatip", "hnsw", "ivf_pq"]


@dataclass(frozen=True)
class FaissBuildConfig:
    index_type: FaissIndexType = "flatip"

    # HNSW
    hnsw_m: int = 32
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64

    # IVF+PQ
    ivf_nlist: int = 1024
    pq_m: int = 16
    nprobe: int = 10


def _validate_vectors(vectors: np.ndarray) -> np.ndarray:
    if vectors.ndim != 2:
        raise ValueError(f"Expected 2D vectors, got shape={vectors.shape}")
    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32, copy=False)
    return vectors


def build_faiss_index(vectors: np.ndarray, cfg: FaissBuildConfig) -> faiss.Index:
    """
    Build a FAISS index. Assumes vectors are already unit-normalized if using cosine/IP.
    """
    vectors = _validate_vectors(vectors)
    n, dim = vectors.shape
    if n == 0:
        raise RuntimeError("Cannot build FAISS index: zero vectors")

    if cfg.index_type == "flatip":
        index = faiss.IndexFlatIP(dim)
        index.add(vectors)
        return index

    if cfg.index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, int(cfg.hnsw_m), faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = int(cfg.hnsw_ef_construction)
        index.hnsw.efSearch = int(cfg.hnsw_ef_search)
        index.add(vectors)
        return index

    if cfg.index_type == "ivf_pq":
        nlist = int(cfg.ivf_nlist)
        pq_m = int(cfg.pq_m)
        if nlist <= 0:
            raise ValueError("ivf_nlist must be > 0")
        if pq_m <= 0:
            raise ValueError("pq_m must be > 0")
        if dim % pq_m != 0:
            raise ValueError(f"PQ requires dim % pq_m == 0 (dim={dim}, pq_m={pq_m})")

        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFPQ(quantizer, dim, nlist, pq_m, 8, faiss.METRIC_INNER_PRODUCT)
        index.nprobe = int(cfg.nprobe)
        index.train(vectors)
        index.add(vectors)
        return index

    raise ValueError(f"Unknown FAISS index_type: {cfg.index_type}")


def faiss_index_type_name(index: faiss.Index) -> str:
    # Human-readable name for meta.json
    return type(index).__name__


def faiss_index_params(cfg: FaissBuildConfig) -> dict[str, Any]:
    if cfg.index_type == "flatip":
        return {}
    if cfg.index_type == "hnsw":
        return {
            "hnsw_m": cfg.hnsw_m,
            "hnsw_ef_construction": cfg.hnsw_ef_construction,
            "hnsw_ef_search": cfg.hnsw_ef_search,
        }
    return {"ivf_nlist": cfg.ivf_nlist, "pq_m": cfg.pq_m, "nprobe": cfg.nprobe}


def save_faiss_index(index: faiss.Index, path: str) -> None:
    faiss.write_index(index, path)


def load_faiss_index(path: str) -> faiss.Index:
    return faiss.read_index(path)

