# Branham Model API — Dataset, Chunking, Indexing & Training Spec

**Project**: BranhamGPT  
**Scope**: All data required for RAG, training, indexing, and serving

This file is the authoritative contract for how all William Branham sermon data is ingested, normalized, chunked, indexed, and exported for:

- BM25 retrieval
- Dense (FAISS) retrieval
- RAG serving
- Continued pretraining (LoRA / QLoRA)
- Q/A instruction tuning

This document is written so that an AI system or engineer can execute it deterministically end-to-end.

---

## 1) Canonical Identifiers

### 1.1 date_id (LOCKED)

**Format**:

```
dd-mm-yy<T>   when multiple sermons exist that day
dd-mm-yy      when only one sermon exists that day
```

Where `<T>` ∈ `{S, M, A, E, B, X}`:

- `S` = Sunday School
- `M` = Morning
- `A` = Afternoon
- `E` = Evening
- `B` = Breakfast Sermon
- `X` = Extras

**Examples**:
- `47-0412M` = April 12, 1947, morning (multiple sermons that day)
- `64-1226E` = December 26, 1964, evening (multiple sermons that day)
- `47-0412` = April 12, 1947 (single sermon that day)

**date_id is the only canonical sermon locator used for**:
- References
- chunk_id generation
- RAG grounding
- UI rendering

**Important**: Sermon titles are human-readable only and never authoritative.

### 1.2 chunk_id (LOCKED)

**Format**: `{date_id}_chunk_{index}`

**Examples**:
- `47-0412M_chunk_3`
- `64-1226E_chunk_12`
- `47-0412_chunk_1`
- `47-0412M_chunk_0` (first chunk)

The `chunk_index` is sequential within each sermon (0 to N-1).

---

## 2) Storage Philosophy

We minimize databases but use them where speed and integrity matter.

### 2.1 What must be SQLite

SQLite is the canonical store for anything that must be:
- looked up by chunk_id
- joined
- referenced at runtime

**These must be in SQLite**:
- sermons
- paragraphs
- chunks

### 2.2 What must be JSONL

JSONL is used when:
- streaming large datasets
- training
- reproducibility

**These must be JSONL**:
- continued pretraining data
- instruction tuning data
- FAISS id map
- dataset manifests

### 2.3 What must be binary indexes

- BM25 index (`.pkl` or custom binary format)
- FAISS index (`.index` file)

---

## 3) Final Runtime Bundle

This is what the Model API loads at boot:

```
bundle/
├── chunks.sqlite          # Canonical text store (sermons, paragraphs, chunks)
├── sermons.csv            # Optional: lightweight sermon metadata for UI
├── bm25.index             # BM25 index (keyed by chunk_id)
├── faiss.index            # FAISS index (dense vectors)
├── faiss_id_map.jsonl     # Maps FAISS row id → chunk_id
└── manifest.json          # Bundle metadata (version, counts, checksums)
```

**No raw PDFs, no JSONL training files in the runtime bundle.**

---

## 4) SQLite Schema (Canonical)

### 4.1 sermons

Stores sermon metadata.

```sql
CREATE TABLE sermons (
  date_id TEXT PRIMARY KEY,
  title TEXT,
  source TEXT,
  language TEXT DEFAULT 'en'
);
```

### 4.2 paragraphs (raw canonical text)

Each paragraph must correspond to VGR numbering when available.  
If not present, paragraphs are generated deterministically.

```sql
CREATE TABLE paragraphs (
  date_id TEXT,
  paragraph_no INTEGER,
  text TEXT,
  PRIMARY KEY (date_id, paragraph_no)
);
```

### 4.3 chunks (retrieval and grounding unit)

```sql
CREATE TABLE chunks (
  chunk_id TEXT PRIMARY KEY,
  date_id TEXT,
  paragraph_start INTEGER,
  paragraph_end INTEGER,
  chunk_index INTEGER,
  text TEXT,
  word_count INTEGER,
  char_count INTEGER
);

-- Indexes for fast retrieval
CREATE INDEX idx_chunks_date ON chunks(date_id);
CREATE INDEX idx_chunks_range ON chunks(date_id, paragraph_start, paragraph_end);
```

**Field descriptions**:
- `chunk_id`: Unique identifier (e.g., `47-0412M_chunk_3`)
- `date_id`: Sermon identifier (e.g., `47-0412M` or `47-0412`)
- `paragraph_start`: Starting paragraph number (inclusive)
- `paragraph_end`: Ending paragraph number (inclusive)
- `chunk_index`: Position within sermon (0 to N-1, for expansion logic)
- `text`: The actual chunk text
- `word_count`: Word count (for budgeting and analysis)
- `char_count`: Character count (for budgeting and analysis)

---

## 5) Chunking Rules (LOCKED)

We preserve VGR paragraph numbering and pack paragraphs into retrieval-friendly chunks.

### 5.1 Atomic unit

**Paragraphs are the atomic units**.  
Never split a paragraph unless it is extremely long.

### 5.2 Budgeting

Use **word count** (not tokenizer).

```
TARGET_WORDS ≈ 260–320
HARD_MAX ≈ 380
```

Algorithm:
1. Start with first paragraph of sermon
2. Accumulate consecutive paragraphs
3. Add next paragraph if total stays under `HARD_MAX`
4. If adding next paragraph exceeds `HARD_MAX`, finalize current chunk
5. Start new chunk with that paragraph

### 5.3 If a paragraph is too long

If a single paragraph exceeds `HARD_MAX` words:
- Split on **sentence boundaries**
- Assign sub-paragraph identifiers

**Example**:
```
¶23a
¶23b
```

These are stored as separate rows in the `paragraphs` table but still part of the same `date_id`.

### 5.4 Each chunk stores

- `paragraph_start`: First paragraph in chunk
- `paragraph_end`: Last paragraph in chunk
- `chunk_index`: Sequential index (0 to N-1 per sermon)
- `text`: Full chunk text
- `word_count`: Total words
- `char_count`: Total characters

### 5.5 Optional paragraph markers inside chunk text

Chunks **MAY** render paragraphs with markers for LLM clarity:

```
¶2 Then Brother Branham said...
¶3 And the Angel of the Lord appeared...
```

This helps the LLM understand structure, but **references are always driven by metadata** (not by parsing markers).

### 5.6 No overlap

Chunks never overlap. Each paragraph belongs to exactly one chunk.

---

## 6) Data Ingestion Pipeline

This pipeline is parallelizable by sermon.

### Stage 0 — Source intake

**Inputs**:
- `raw_pdfs/<year>/*.pdf`
- or `api_dump/*.json` (when authorized)

**Output**:

```jsonl
ingest_manifest.jsonl
```

Each record:

```json
{
  "source_path": "raw_pdfs/1947/47-0412M.pdf",
  "checksum": "sha256:...",
  "date_id": "47-0412M",
  "title": "Faith Is The Substance"
}
```

---

### Stage 1 — Parse to canonical paragraphs

**For each sermon**:
1. Extract text
2. Extract `date_id`
3. Extract title
4. Detect VGR paragraph numbers
5. If none, paragraphize deterministically

**Write to SQLite**:
- Insert into `sermons` table
- Insert into `paragraphs` table

**No chunking yet.**

**Output**: `chunks.sqlite` (with `sermons` and `paragraphs` tables populated)

---

### Stage 2 — Build chunks

**Input**: Read paragraphs ordered by `(date_id, paragraph_no)`

**For each date_id**:
1. Pack paragraphs using word budget (rules from Section 5)
2. Assign `chunk_index` sequentially (0 to N-1)
3. Generate `chunk_id` = `{date_id}_chunk_{chunk_index}` (no zero-padding)
4. Write rows into `chunks` table

**Output**: `chunks.sqlite` (with `chunks` table populated)

---

### Stage 3 — Build BM25 Index

**Input**:

```sql
SELECT chunk_id, text FROM chunks;
```

**Process**:
- BM25 documents are chunks, keyed by `chunk_id`
- Use lightweight BM25 library (e.g., `rank-bm25` in Python)
- Serialize index

**Output**: `bm25.index` (`.pkl` or custom binary format)

---

### Stage 4 — Build Embeddings and FAISS Index

**Input**:

```sql
SELECT chunk_id, text FROM chunks;
```

**Process**:
1. Generate embeddings using configured embedding model
2. Build FAISS index from embeddings
3. Assign FAISS row id (0 to N-1, sequential)
4. Write mapping: `faiss_id → chunk_id`

**Output**:
- `faiss.index`
- `faiss_id_map.jsonl`

Each row in `faiss_id_map.jsonl`:

```json
{ "faiss_id": 12345, "chunk_id": "47-0412M_chunk_3" }
```

---

## 7) Training Data Formats

### 7.1 Continued Pretraining Dataset (Section 8.2)

Generated from `chunks.sqlite`.

**File**: `continued_pretrain.jsonl`

Each record:

```json
{
  "text": "¶2 Then Brother Branham said... ¶3 And the Angel...",
  "date_id": "47-0412M",
  "language": "en",
  "paragraph_start": 2,
  "paragraph_end": 3
}
```

**Purpose**:
- Train LoRA/QLoRA adapter on sermon corpus
- Internalize sermon tone, cadence, discourse structure
- Multilingual sermon-title familiarity (if multilingual shards included)

**Required fields**:
- `text`: The text content (chunk text)
- `date_id`: Sermon identifier
- `language`: Language code (ISO 639-1)
- `paragraph_start`: Starting paragraph
- `paragraph_end`: Ending paragraph

**Note**: Shard deterministically for reproducibility.

---

### 7.2 Q/A Instruction Tuning Dataset (Section 8.3)

**Generated AFTER RAG system exists.**

This dataset is generated by the RAG system itself.

**Flow**:
1. Use BM25 and FAISS
2. Ask grounded questions
3. Generate answers with citations
4. Store as training data

**File**: `instruction_tune.jsonl`

Each record:

```json
{
  "question": "What did Brother Branham teach about the tares?",
  "answer": "According to [47-0412M: ¶2–¶3], he explained that the tares represent...",
  "context": "¶2 ... ¶3 ...",
  "date_id": "47-0412M",
  "chunk_ids": ["47-0412M_chunk_3", "47-0412M_chunk_4"],
  "mode": "synthesis"
}
```

**Purpose**:
- Teach cite-or-refuse behavior
- Teach quote-intent handling
- Teach multilingual response formatting

**Required fields**:
- `question`: User question
- `answer`: Generated answer with inline citations
- `context`: Retrieved context (chunk text)
- `date_id`: Primary sermon referenced
- `chunk_ids`: Array of chunk_ids used for grounding
- `mode`: `quote` | `synthesis` | `refusal`

---

## 8) File Organization

### 8.1 Development structure

```
data/
├── raw/                           # Raw sermon files (not in git)
│   ├── pdfs/
│   │   ├── 1947/
│   │   │   ├── 47-0412M.pdf
│   │   │   └── ...
│   │   └── ...
│   └── api_dump/
│       └── sermons.json
│
├── processed/                     # Processed data (SQLite, manifests)
│   ├── chunks.sqlite              # Canonical text store
│   ├── ingest_manifest.jsonl     # Source intake log
│   └── stats.json                 # Dataset statistics
│
├── training/                      # Training datasets
│   ├── continued_pretrain.jsonl
│   ├── instruction_tune.jsonl
│   └── splits/
│       ├── train.jsonl
│       ├── val.jsonl
│       └── test.jsonl
│
├── indices/                       # Built indices
│   ├── bm25.index
│   ├── faiss.index
│   └── faiss_id_map.jsonl
│
└── bundle/                        # Runtime bundle (for deployment)
    ├── chunks.sqlite
    ├── sermons.csv
    ├── bm25.index
    ├── faiss.index
    ├── faiss_id_map.jsonl
    └── manifest.json
```

### 8.2 Production runtime bundle

Only the `bundle/` directory is deployed to production.

**SQLite Deployment Note**:
- `chunks.sqlite` is deployed with the API server
- SQLite is file-based and fast for read-heavy workloads
- No cloud database needed (optimized for speed and simplicity)
- Database file should be on fast local storage (SSD)

---

## 9) Why This Architecture Works

1. **Paragraphs are canonical** — VGR numbering is preserved; references are stable
2. **Chunks are retrieval-stable** — No overlap; deterministic boundaries
3. **References are deterministic** — date_id + paragraph range (no offsets)
4. **BM25 and FAISS share chunk_id** — Fusion and deduplication are trivial
5. **Training and retrieval use identical units** — No mismatch between systems
6. **SQLite is the single source of truth** — Fast lookups, joins, integrity
7. **JSONL is only for training and mapping** — Streaming, reproducibility
8. **No tokenizer required for chunking** — Word-based budgeting is model-agnostic
9. **Dense retrieval can be replaced** — Without re-chunking the corpus

---

## 10) Important Notes and Non-Negotiables

### 10.1 Reference system
- **No offsets**: We don't use character/token offsets, only paragraph ranges
- **Canonical references**: date_id is the stable locator, not sermon title
- **Never reference anything except date_id and paragraph range**

### 10.2 Chunking
- **Never allow overlapping chunks**
- **Never split paragraphs unless absolutely necessary** (extremely long)
- **Always use word counts for budgeting**, not tokens

### 10.3 Storage
- **Never store embeddings inside chunk records** (keep SQLite lightweight)
- **Never let titles drive grounding or retrieval** (date_id only)

### 10.4 Training
- **Deterministic sharding**: For reproducibility in training
- **Same units everywhere**: Chunks used for retrieval = chunks used for training

### 10.5 Deduplication
- **Use chunk_id for deduplication** in retrieval fusion

---

## 11) Reference Rendering (Section 11 from .cursorrules)

### 11.1 Inline references

Inside the answer when relevant:

```
According to [47-0412M: ¶2–¶3], Brother Branham explained...
```

### 11.2 References block

At the end of the answer (consolidated):

```
**References:**
- [47-0412M: ¶2–¶3] Faith Is The Substance
- [64-1226E: ¶15–¶17] The Spoken Word Is The Original Seed
```

Each reference includes:
- `date_id`
- paragraph range (`¶start–¶end`)
- optionally: sermon title (for readability)

---

## 12) API Response Format (Section 10 from .cursorrules)

When the Model API returns an answer, it must include:

```json
{
  "answer": "According to [47-0412M: ¶2–¶3], Brother Branham explained...",
  "mode": "synthesis",
  "references": [
    {
      "date_id": "47-0412M",
      "paragraph_start": 2,
      "paragraph_end": 3,
      "chunk_ids": ["47-0412M_chunk_3"]
    }
  ]
}
```

**Fields**:
- `answer`: Generated answer with inline references
- `mode`: `quote` | `synthesis` | `refusal`
- `references`: Array of reference objects
  - `date_id`: Sermon identifier
  - `paragraph_start`: Starting paragraph
  - `paragraph_end`: Ending paragraph
  - `chunk_ids`: Array of chunk_ids used for grounding

---

## Appendix: Quick Reference

### date_id examples
- `47-0412M` (multiple sermons that day)
- `47-0412` (single sermon that day)

### chunk_id examples
- `47-0412M_chunk_3`
- `64-1226E_chunk_12`
- `47-0412_chunk_0` (first chunk)

### Chunking targets
- Target: 260–320 words
- Hard max: 380 words

### Tables
- `sermons`: date_id, title, source, language
- `paragraphs`: date_id, paragraph_no, text
- `chunks`: chunk_id, date_id, paragraph_start, paragraph_end, chunk_index, text, word_count, char_count

### Runtime bundle files
- `chunks.sqlite`
- `bm25.index`
- `faiss.index`
- `faiss_id_map.jsonl`
- `manifest.json`

---

**End of specification.**
