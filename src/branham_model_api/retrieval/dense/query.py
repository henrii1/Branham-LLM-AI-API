"""
Dense retrieval query utilities.
"""

from __future__ import annotations

from dataclasses import dataclass

import faiss  # type: ignore
import numpy as np


@dataclass(frozen=True)
class DenseHit:
    faiss_id: int
    score: float


def faiss_search(index: faiss.Index, query_vecs: np.ndarray, *, top_n: int = 20) -> list[list[DenseHit]]:
    """
    Search FAISS with inner product. Assumes query vectors are unit-normalized if using cosine.

    Returns hits per query, ordered best-first.
    """
    if query_vecs.ndim == 1:
        query_vecs = query_vecs[None, :]
    if query_vecs.dtype != np.float32:
        query_vecs = query_vecs.astype(np.float32, copy=False)

    scores, ids = index.search(query_vecs, int(top_n))
    out: list[list[DenseHit]] = []
    for q in range(ids.shape[0]):
        hits: list[DenseHit] = []
        for j in range(ids.shape[1]):
            fid = int(ids[q, j])
            if fid < 0:
                continue
            hits.append(DenseHit(faiss_id=fid, score=float(scores[q, j])))
        out.append(hits)
    return out

