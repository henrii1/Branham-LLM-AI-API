# Dense Retrieval — Embeddings & FAISS Index Build

**Stage 4** of the data ingestion pipeline.

**Goal**:
- Dense retrieval must work for English corpus while accepting multilingual queries.
- Use a multilingual embedder so queries in other languages still retrieve English chunks reasonably well.
- Keep embeddings + FAISS strictly keyed by chunk_id (same unit as BM25) to enable clean fusion/dedup.

---

## 4.1 Recommended Starting Embedding Model (Config-Driven)

### Primary Starting Point (Good Default)

**Embedding model**: `jinaai/jina-embeddings-v3`

**Key capabilities**:
- Multilingual pretraining (broad language coverage)
- Long input support (up to 8192 tokens per model docs/marketplace references)
- Matryoshka Representation Learning (MRL): you can truncate the output embedding dimension while preserving much of the quality

### Dimensionality Choice (For Faster ANN + Smaller Index)

- Default v3 output is 1024 dims, but MRL allows truncation down to smaller dims such as 512.
- **Starting recommendation**: `DIM=512` (trade-off: faster + smaller index; validate on retrieval eval later)

### Notes on Tokenization

- jina-embeddings-v3 is based on an XLM-R style stack per marketplace documentation; XLM-R tokenization uses SentencePiece.
- **Implementation rule**: always load the tokenizer via Hugging Face `AutoTokenizer` for the selected embedding model (do not hardcode tokenizers). The model's tokenizer config will select SentencePiece where appropriate.

---

## 4.2 Alternative Multilingual Embedding Models (Keep in Config)

Keep these as evaluated alternatives (model_id is configurable; select via experiments):

### BAAI/bge-m3
- Popular multilingual embedding baseline; designed for multi-linguality and retrieval versatility.

### intfloat/multilingual-e5-large / multilingual-e5-* variants
- Common multilingual retrieval baselines; widely used in RAG systems. (Discoverable via HF ST listings.)

### Qwen/Qwen3-Embedding-* (0.6B/4B/8B)
- Modern multilingual embedding family; available as HF embedding models.

### Selection Guidance

- Start with `jina-embeddings-v3` at 512 dims because MRL makes dimension reduction straightforward.
- Validate against `bge-m3` and `multilingual-e5-large` on a small internal retrieval eval set (top-k recall on known sermon queries).

---

## 4.3 Embedding Generation Contract (Implementation Rules)

### Input

```sql
SELECT chunk_id, text FROM chunks
```

### Output Artifacts

- `faiss.index` (binary FAISS index)
- `faiss_id_map.jsonl` (row_id -> chunk_id mapping)
- `faiss_meta.json` (model_id, dim, dtype, normalization, build params, corpus hash)
- optional: `embeddings.memmap` (float16 embeddings for rebuild/debug; not required for serving)

### Key Rules

- Use the **SAME chunk records as BM25** (documents = chunks, keyed by chunk_id).
- Store **NO embeddings inside chunks.sqlite**.
- Keep all embedding model choices configurable:
  - `EMBED_MODEL_ID` (e.g., `jinaai/jina-embeddings-v3`)
  - `EMBED_DIM` (e.g., 512 using MRL truncation)
  - `EMBED_DTYPE` (`fp16`/`bf16`/`fp32`; fp16 recommended on GPU)
  - `EMBED_BATCH_SIZE` (tune per hardware)
  - `EMBED_MAX_LENGTH` (respect model limits; v3 supports long inputs per docs, but set an explicit cap)

### Normalization & Similarity (Recommended)

- Normalize embeddings to unit length and use cosine similarity.
- In FAISS, cosine similarity is typically implemented as inner product over normalized vectors.

### MRL Truncation (jina v3)

- Generate full embeddings then truncate to first `EMBED_DIM` dims (e.g., 512).
- This is aligned with MRL usage guidance and preserves performance better than arbitrary projection.

### Tokenizer

- Use `AutoTokenizer.from_pretrained(EMBED_MODEL_ID)` to ensure SentencePiece is used when required by the model. (Do not hardcode.)

---

## 4.4 FAISS Index Type (Start Simple, Keep Configurable)

### Start with a Reliable Baseline

- **IndexFlatIP** (exact search; best quality; slower if very large)

### Then Graduate to ANN When Needed

- **IndexHNSWFlat** (fast, good quality, simple)
- or **IVF+PQ variants** if memory becomes a concern

### Config Flags

- `FAISS_INDEX_TYPE` = `flatip` | `hnsw` | `ivf_pq`
- `FAISS_HNSW_M` = 32 (example starting value)
- `FAISS_IVF_NLIST`, `FAISS_PQ_M`, `FAISS_NPROBE` (only if using IVF/PQ)

### Practical Guidance for 1100 Sermons

- Expect total chunks to be "tens of thousands" depending on chunk size; HNSW is often a good starting ANN option.
- Use `DIM=512` to keep index smaller and searches faster.

---

## 4.5 Multilingual Behavior (Query Language vs English Corpus)

### Intent

- Corpus is English-only in v1.
- Queries may be multilingual.
- Embeddings enable cross-lingual retrieval: user-language query -> English chunk hits.

### Generation Model Compatibility (Llama / Qwen2.5-7B-Instruct)

- The embedder and generator are decoupled.
- Any generator (e.g., Llama or Qwen2.5 Instruct) can consume retrieved English context and, via system prompt + user_language, produce the answer in the user's language.
- Key enforcement remains the same: strict grounding + references; refuse if evidence is insufficient.

---

## 4.6 Build Procedure (Deterministic)

### Step 1: Read All Chunk Rows

- `chunk_id`, `text` ordered by `(date_id, chunk_index)` for stable mapping

### Step 2: Embed Texts in Batches

- Use `torch.no_grad()`
- Move model to `DEVICE` (`cuda`/`mps`/`cpu`) per `DEVICE_PREFERENCE`
- Use `EMBED_DTYPE`
- Use tokenizer from `AutoTokenizer`

### Step 3: Produce Embedding Vectors

- Pooling strategy must be explicitly defined by the embedding model's recommended usage (do not invent pooling).
- Normalize vectors (unit length).
- If model supports MRL truncation, truncate to `EMBED_DIM` (e.g., 512).

### Step 4: Build FAISS Index

- Choose index type per config
- Add vectors in the same order as processed

### Step 5: Write Mapping

- `faiss_id_map.jsonl`
  ```json
  { "faiss_id": <row_index>, "chunk_id": "<chunk_id>" }
  ```

### Step 6: Persist

- `faiss.index`
- `faiss_meta.json` (model_id, dim, dtype, normalization, index params, corpus hash)

### Step 7: Serving-Time Contract

- Dense retrieval returns faiss row ids -> chunk_ids via mapping -> chunk records via `chunks.sqlite`
- Dedup and fusion are chunk_id-based (same as BM25)

---

## 4.7 Notes for Later Optimization (Do Not Block v1)

- If you later adopt a managed vector DB, keep the same embedding contract (chunk_id as primary key).
- If you later want faster embeddings on NVIDIA, consider optional acceleration flags (`optimum`, `torch.compile`, etc.) but gate them behind config.
- Keep reranker separate; dense embeddings should remain "fast path".

---

## Configuration Summary

### Required Environment Variables / Config

```yaml
# Embedding Model
EMBED_MODEL_ID: "jinaai/jina-embeddings-v3"  # or alternative from 4.2
EMBED_DIM: 512                                # MRL truncation target
EMBED_DTYPE: "fp16"                           # fp16/bf16/fp32
EMBED_BATCH_SIZE: 32                          # tune per hardware
EMBED_MAX_LENGTH: 8192                        # respect model limits

# FAISS Index
FAISS_INDEX_TYPE: "flatip"                    # flatip/hnsw/ivf_pq
FAISS_HNSW_M: 32                              # if using HNSW
FAISS_IVF_NLIST: 1024                         # if using IVF
FAISS_PQ_M: 16                                # if using PQ
FAISS_NPROBE: 10                              # if using IVF

# Device
DEVICE_PREFERENCE: "auto"                     # mps/cuda/cpu/auto
```

---

## Output File Formats

### faiss_id_map.jsonl

```jsonl
{"faiss_id": 0, "chunk_id": "47-0412M_chunk_0"}
{"faiss_id": 1, "chunk_id": "47-0412M_chunk_1"}
{"faiss_id": 2, "chunk_id": "47-0412M_chunk_2"}
```

### faiss_meta.json

```json
{
  "model_id": "jinaai/jina-embeddings-v3",
  "dim": 512,
  "dtype": "fp16",
  "normalization": "unit_length",
  "index_type": "IndexFlatIP",
  "index_params": {},
  "corpus_hash": "sha256:...",
  "build_timestamp": "2024-01-15T10:30:00Z",
  "total_chunks": 55000
}
```

---

## Integration with Retrieval Pipeline

### Dense Retrieval Flow

1. User query (any language) → embedder → query vector
2. Query vector → FAISS search → top-K faiss row ids
3. Row ids → `faiss_id_map.jsonl` → chunk_ids
4. Chunk_ids → `chunks.sqlite` → chunk records
5. Chunk records → fusion with BM25 results → dedup by chunk_id

### Key Contract Points

- **Same units**: BM25 and FAISS both use chunks (keyed by chunk_id)
- **Clean fusion**: Deduplication is trivial (same chunk_id)
- **Multilingual queries**: Embedder handles cross-lingual retrieval
- **English corpus**: All chunks are English; generator handles multilingual output

---

**End of specification.**

