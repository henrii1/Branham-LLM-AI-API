# BM25 Index Build

**Stage 3** of the data ingestion pipeline.

**Goal**: Build a fast, lightweight BM25 lexical index over sermon chunks for keyword-based retrieval.

---

## Core Design Constraint

BM25 documents MUST be the exact same unit as dense retrieval:
- document = chunk
- primary key = chunk_id
- text = chunks.text
- metadata stored in SQLite (date_id, paragraph range, chunk_index)

This ensures fusion/dedup by chunk_id is trivial and references remain stable.

---

## Input and Output

### Input

```sql
SELECT chunk_id, text FROM chunks
ORDER BY date_id, chunk_index;
```

### Output Artifacts

- `bm25.index` (binary BM25 index)
- `bm25_meta.json` (parameters, preprocessing rules, corpus hash, version)
- `bm25_doc_map.jsonl` (optional: doc_id -> chunk_id mapping if using integer doc IDs)
- `bm25_vocab.json` (optional: term stats for debugging)

---

## Preprocessing Rules (Deterministic)

Applied to chunk text when building the index:

1. **Unicode normalize**: NFKC
2. **Lowercase**: Convert all text to lowercase
3. **Replace fancy quotes/dashes**: Convert to ASCII equivalents
   - `"` `"` → `"`
   - `'` `'` → `'`
   - `—` `–` → `-`
4. **Normalize punctuation**: Replace most punctuation with whitespace (optionally keep apostrophes)
5. **Collapse whitespace**: Multiple spaces → single space
6. **Tokenize**: Split on whitespace

### Paragraph Markers

If chunk text contains `¶N` markers:
- Strip `¶\d+[a-z]?` tokens during preprocessing
- Do not let paragraph numbers dominate BM25 scoring

### Stopwords and Stemming

- **Default**: NO stemming, NO stopword removal (preserves quote-intent fidelity)
- Optionally allow stopword removal via config (OFF by default)

---

## BM25 Parameters (Config-Driven)

### Required Configuration

```yaml
BM25_K1: 1.2                      # Typical range: 1.2–1.6
BM25_B: 0.75                      # Document length normalization
BM25_TOKENIZER_MODE: "whitespace" # v1 baseline
BM25_NORMALIZE: true              # Apply preprocessing pipeline
```

### Optional Configuration

```yaml
BM25_REMOVE_STOPWORDS: false      # Default: false
BM25_STEM: false                  # Default: false
```

---

## Build Procedure (Deterministic)

### Step 1: Read All Chunks

```sql
SELECT chunk_id, text FROM chunks
ORDER BY date_id, chunk_index;
```

- Order by `(date_id, chunk_index)` for stable doc_id assignment
- Maintain stable ordering for reproducibility

### Step 2: Preprocess and Tokenize

For each chunk:
1. Apply normalization pipeline (Section: Preprocessing Rules)
2. Tokenize by whitespace
3. Produce tokens list per chunk

### Step 3: Build Term Statistics

Compute:
- `df(term)`: document frequency for each term
- `N`: total number of documents
- `idf(term)`: inverse document frequency using BM25 formula

BM25 IDF formula (standard):
```
idf(term) = log((N - df(term) + 0.5) / (df(term) + 0.5) + 1)
```

### Step 4: Build Inverted Index

Structure:
```
term -> postings_list[(doc_id, tf), ...]
```

For each term:
- Store list of documents containing the term
- Store term frequency (tf) in each document

Additional data structures:
- `doc_len[doc_id]`: length of each document (in tokens)
- `avg_doc_len`: average document length across corpus

### Step 5: Persist Index

Write to disk:

#### bm25.index

Single binary file containing:
- Inverted index: `term -> postings_list`
- Document lengths: `doc_len[doc_id]`
- IDF values: `idf[term]`
- Average document length: `avg_doc_len`
- doc_id to chunk_id mapping (if using integer doc IDs)

#### bm25_meta.json

```json
{
  "params": {
    "k1": 1.2,
    "b": 0.75
  },
  "normalization_rules": {
    "unicode_form": "NFKC",
    "lowercase": true,
    "tokenizer": "whitespace",
    "remove_stopwords": false,
    "stemming": false
  },
  "corpus_hash": "sha256:...",
  "build_timestamp": "2024-01-15T10:30:00Z",
  "doc_count": 55000,
  "avg_doc_len": 280.5,
  "vocab_size": 45000
}
```

### Step 6 (Optional): Persist Mapping

If using integer doc IDs internally, write mapping:

#### bm25_doc_map.jsonl

```jsonl
{"doc_id": 0, "chunk_id": "47-0412M_chunk_0"}
{"doc_id": 1, "chunk_id": "47-0412M_chunk_1"}
{"doc_id": 2, "chunk_id": "47-0412M_chunk_2"}
```

If storing chunk_id directly in the index, this file is optional. (use chunk_id directly)

---

## Index Format Choice

### Recommended: Custom BM25 Inverted Index

Build a lightweight custom index with full control:

**Data structures**:
- `doc_len: Dict[doc_id, int]` — token count per document
- `avg_doc_len: float` — average tokens per document
- `inverted_index: Dict[term, List[Tuple[doc_id, tf]]]` — postings lists
- `idf: Dict[term, float]` — IDF scores
- `doc_id_to_chunk_id: Dict[int, str]` — mapping (if using integer doc IDs)

**Serialization**:
- Pickle (simple, fast)
- MessagePack (cross-language, compact)
- Custom binary format (most control)

### Alternative: Lightweight Library

If using a library (e.g., `rank-bm25` in Python):
- Must support deterministic build
- Must persist to single file
- Must key results back to chunk_id
- Must use same chunk units as FAISS

---

## BM25 Scoring Formula (Reference)

For query `Q` and document `D`:

```
score(D, Q) = Σ idf(q_i) · (tf(q_i, D) · (k1 + 1)) / (tf(q_i, D) + k1 · (1 - b + b · |D| / avg_doc_len))
```

Where:
- `q_i`: term in query
- `tf(q_i, D)`: term frequency of q_i in document D
- `|D|`: length of document D (in tokens)
- `k1`: term frequency saturation parameter
- `b`: document length normalization parameter

---

## Storage Summary

### Files Created

```
data/indices/
├── bm25.index            # Binary BM25 index
├── bm25_meta.json        # Build metadata
├── bm25_doc_map.jsonl    # Optional: doc_id mapping
└── bm25_vocab.json       # Optional: vocabulary stats
```

### No Additional Database Required

BM25 uses:
- SQLite `chunks` table as input (source of documents)
- File-based index for serving
- No separate database dependency

---

## Implementation Checklist

- [ ] Read chunks from SQLite ordered by `(date_id, chunk_index)`
- [ ] Implement deterministic text preprocessing pipeline
- [ ] Tokenize by whitespace (v1 baseline)
- [ ] Build term statistics (df, idf)
- [ ] Build inverted index with postings lists
- [ ] Compute and store document lengths
- [ ] Persist `bm25.index` binary file
- [ ] Persist `bm25_meta.json` metadata file
- [ ] Validate index can be loaded and queried
- [ ] Verify results return chunk_id + score

---

**End of specification.**

