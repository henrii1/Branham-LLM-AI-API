"""
BM25 indexing for Branham Model API (Stage 3).

Design goals (from `.cursorrules` + `datasets/docs/DATA_FORMAT.md`):
- Documents are **chunks** (same unit as dense retrieval), keyed by `chunk_id`.
- Deterministic preprocessing + deterministic doc_id assignment (ORDER BY date_id, chunk_index).
- Lightweight on-disk artifact: `bm25.index` + `bm25_meta.json` (+ optional doc map).
- No stemming, no stopword removal by default (quote-intent fidelity).
"""

from __future__ import annotations

import hashlib
import json
import math
import pickle
import re
import sqlite3
import unicodedata
from array import array
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator, Optional


PARA_MARKER_RE = re.compile(r"¶\s*\d+[a-z]?\b", re.IGNORECASE)

# Includes common "smart quotes" and dash variants observed in PDFs / OCR.
FANCY_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u2014": "-",
        "\u2013": "-",
    }
)


@dataclass(frozen=True)
class Bm25PreprocessConfig:
    unicode_form: str = "NFKC"
    lowercase: bool = True
    keep_apostrophes: bool = True
    strip_paragraph_markers: bool = True
    # Default ON (v1): improves speed and reduces index size.
    # Can be disabled via config/CLI if quote-intent evaluation shows regressions.
    remove_stopwords: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "unicode_form": self.unicode_form,
            "lowercase": self.lowercase,
            "keep_apostrophes": self.keep_apostrophes,
            "strip_paragraph_markers": self.strip_paragraph_markers,
            "remove_stopwords": self.remove_stopwords,
            "stemming": False,  # explicitly unsupported in v1
            "tokenizer": "whitespace",
        }


def _load_stopwords() -> set[str]:
    """
    If stopword removal is enabled, we use spaCy's built-in English stopword list.
    """
    try:
        from spacy.lang.en.stop_words import STOP_WORDS  # type: ignore

        return set(STOP_WORDS)
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Stopword removal requested but spaCy stopwords are unavailable. "
            "Install/enable spaCy or disable BM25_REMOVE_STOPWORDS."
        ) from e


def normalize_for_bm25(text: str, cfg: Bm25PreprocessConfig) -> str:
    """
    Deterministic normalization pipeline (see `datasets/docs/BM25_INDEX.md`).
    """
    if not text:
        return ""

    text = unicodedata.normalize(cfg.unicode_form, text)
    text = text.translate(FANCY_TRANSLATION)
    if cfg.lowercase:
        text = text.lower()

    if cfg.strip_paragraph_markers:
        text = PARA_MARKER_RE.sub(" ", text)

    # Replace punctuation with whitespace (optionally keep apostrophes).
    out_chars: list[str] = []
    for ch in text:
        if ch.isalnum():
            out_chars.append(ch)
            continue
        if cfg.keep_apostrophes and ch == "'":
            out_chars.append(ch)
            continue
        # Treat everything else (including hyphens) as whitespace.
        out_chars.append(" ")

    return " ".join("".join(out_chars).split())


def tokenize_for_bm25(text: str, cfg: Bm25PreprocessConfig) -> list[str]:
    norm = normalize_for_bm25(text, cfg)
    if not norm:
        return []
    tokens = norm.split()
    if cfg.remove_stopwords:
        stop = _load_stopwords()
        tokens = [t for t in tokens if t not in stop]
    return tokens


@dataclass
class Bm25Index:
    """
    A lightweight BM25 inverted index optimized for fast query-time scoring.

    Postings representation:
      term -> (doc_ids, tfs)
    where doc_ids is an array('I') and tfs is an array('H').
    """

    version: int
    k1: float
    b: float
    avg_doc_len: float
    doc_len: array  # array('I')
    doc_id_to_chunk_id: list[str]
    postings: dict[str, tuple[array, array]]  # term -> (doc_ids array('I'), tfs array('H'))
    idf: dict[str, float]
    preprocess: dict[str, Any]

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "Bm25Index":
        with path.open("rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, Bm25Index):
            raise TypeError(f"Invalid BM25 index object in {path}")
        return obj


def iter_chunks_from_sqlite(db_path: Path, *, limit: Optional[int] = None) -> Iterator[tuple[str, str]]:
    conn = sqlite3.connect(db_path, timeout=60.0)
    try:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        cur = conn.cursor()

        sql = """
        SELECT chunk_id, text
        FROM chunks
        ORDER BY date_id ASC, chunk_index ASC, chunk_id ASC
        """
        if limit is not None:
            sql += " LIMIT ?"
            cur.execute(sql, (limit,))
        else:
            cur.execute(sql)

        for chunk_id, text in cur.fetchall():
            yield str(chunk_id), str(text or "")
    finally:
        conn.close()


def build_bm25_index_from_sqlite(
    db_path: Path,
    *,
    k1: float = 1.2,
    b: float = 0.75,
    preprocess_cfg: Optional[Bm25PreprocessConfig] = None,
    limit: Optional[int] = None,
) -> tuple[Bm25Index, dict[str, Any]]:
    """
    Build a BM25 index from `chunks.sqlite` (Stage 3).

    Returns:
      (index, meta_dict) where meta_dict is intended for `bm25_meta.json`.
    """
    if preprocess_cfg is None:
        preprocess_cfg = Bm25PreprocessConfig()

    postings: dict[str, tuple[array, array]] = {}
    doc_len = array("I")
    doc_id_to_chunk_id: list[str] = []

    corpus_hasher = hashlib.sha256()
    corpus_hash_input = "chunk_id + NUL + raw_text"

    total_dl = 0
    doc_id = 0

    for chunk_id, text in iter_chunks_from_sqlite(db_path, limit=limit):
        # Corpus hash is over the raw text to detect any corpus drift.
        corpus_hasher.update(chunk_id.encode("utf-8"))
        corpus_hasher.update(b"\x00")
        corpus_hasher.update(text.encode("utf-8"))
        corpus_hasher.update(b"\n")

        tokens = tokenize_for_bm25(text, preprocess_cfg)
        tf = Counter(tokens)

        dl = len(tokens)
        doc_len.append(dl)
        total_dl += dl
        doc_id_to_chunk_id.append(chunk_id)

        for term, freq in tf.items():
            # Store tf in uint16 (more than enough for chunk-sized docs).
            tf_u16 = min(int(freq), 65535)
            if term not in postings:
                postings[term] = (array("I"), array("H"))
            postings[term][0].append(doc_id)
            postings[term][1].append(tf_u16)

        doc_id += 1

    n_docs = len(doc_id_to_chunk_id)
    if n_docs == 0:
        raise RuntimeError(f"No chunks found in DB: {db_path}")

    avgdl = total_dl / float(n_docs) if n_docs else 0.0

    # Compute IDF for each term (standard BM25).
    idf: dict[str, float] = {}
    for term, (doc_ids, _tfs) in postings.items():
        df = len(doc_ids)
        # idf = log((N - df + 0.5) / (df + 0.5) + 1)
        idf_val = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        idf[term] = float(idf_val)

    meta: dict[str, Any] = {
        "params": {"k1": float(k1), "b": float(b)},
        "normalization_rules": preprocess_cfg.to_dict(),
        "corpus_hash": f"sha256:{corpus_hasher.hexdigest()}",
        "corpus_hash_input": corpus_hash_input,
        "build_timestamp": datetime.now(timezone.utc).isoformat(),
        "doc_count": n_docs,
        "avg_doc_len": avgdl,
        "vocab_size": len(idf),
        "index_format": "pickle:Bm25Index(v1)",
    }

    index = Bm25Index(
        version=1,
        k1=float(k1),
        b=float(b),
        avg_doc_len=float(avgdl),
        doc_len=doc_len,
        doc_id_to_chunk_id=doc_id_to_chunk_id,
        postings=postings,
        idf=idf,
        preprocess=preprocess_cfg.to_dict(),
    )
    return index, meta


def write_bm25_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

