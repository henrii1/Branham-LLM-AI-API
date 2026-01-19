"""
BM25 query utilities.

These are designed for:
- Early BM25 guard (fast pre-LLM refusal)
- Lexical retrieval stage (BM25 half of hybrid retrieval)
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Iterable, Optional

from .index import Bm25Index, Bm25PreprocessConfig, tokenize_for_bm25


@dataclass(frozen=True)
class Bm25Hit:
    chunk_id: str
    score: float


def bm25_search(
    index: Bm25Index,
    query: str,
    *,
    top_n: int = 20,
    min_idf: float = 0.0,
    preprocess_cfg: Optional[Bm25PreprocessConfig] = None,
) -> list[Bm25Hit]:
    if preprocess_cfg is None:
        preprocess_cfg = Bm25PreprocessConfig(
            unicode_form=index.preprocess.get("unicode_form", "NFKC"),
            lowercase=bool(index.preprocess.get("lowercase", True)),
            keep_apostrophes=bool(index.preprocess.get("keep_apostrophes", True)),
            strip_paragraph_markers=bool(index.preprocess.get("strip_paragraph_markers", True)),
            remove_stopwords=bool(index.preprocess.get("remove_stopwords", False)),
        )

    q_tokens = tokenize_for_bm25(query, preprocess_cfg)
    if not q_tokens:
        return []

    n_docs = len(index.doc_id_to_chunk_id)
    scores = [0.0] * n_docs

    k1 = index.k1
    b = index.b
    avgdl = index.avg_doc_len if index.avg_doc_len > 0 else 1.0

    # Score only over query terms (typical BM25 usage).
    for term in q_tokens:
        idf = index.idf.get(term)
        if idf is None or idf <= min_idf:
            continue
        posting = index.postings.get(term)
        if not posting:
            continue

        doc_ids, tfs = posting
        for doc_id, tf in zip(doc_ids, tfs, strict=False):
            dl = index.doc_len[doc_id]
            denom = tf + k1 * (1.0 - b + b * (dl / avgdl))
            contrib = idf * (tf * (k1 + 1.0)) / denom
            scores[doc_id] += contrib

    # Extract top-N hits (ignore zero-score docs).
    # N is relatively small (~chunks count), so O(N log k) is fine.
    best = heapq.nlargest(top_n, ((s, i) for i, s in enumerate(scores) if s > 0.0))
    return [Bm25Hit(chunk_id=index.doc_id_to_chunk_id[i], score=s) for s, i in best]


def bm25_guard_passes(
    index: Bm25Index,
    query: str,
    *,
    threshold: float,
    top_n: int = 5,
    min_idf: float = 0.0,
) -> bool:
    """
    Early BM25 guard:
    - Returns True if BM25 finds any "good enough" match.

    Note: Threshold calibration is corpus-dependent. v1 uses raw BM25 scores.
    """
    hits = bm25_search(index, query, top_n=top_n, min_idf=min_idf)
    if not hits:
        return False
    return hits[0].score >= threshold

