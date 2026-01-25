# Dense Retrieval — Embeddings & FAISS Index Build

**Stage 4** of the data ingestion pipeline.

**Goal**:
- Dense retrieval must work for English corpus while accepting multilingual queries.
- Use a multilingual embedder so queries in other languages still retrieve English chunks reasonably well.
- Keep embeddings + FAISS strictly keyed by chunk_id (same unit as BM25) to enable clean fusion/dedup.

---

## 4.1 V1 Embedding Model (LOCKED)

### Primary Model

**Embedding model**: `Qwen/Qwen3-Embedding-0.6B`

**Key capabilities**:
- Strong multilingual pretraining (broad language coverage)
- Optimized for retrieval tasks
- Efficient 0.6B parameter size suitable for low-latency serving
- Good balance of quality and speed for production use

### Serving Strategy (V1)

**Offline (Index Build)**:
- Corpus embeddings are generated during the build step.
- Can use direct HuggingFace model loading or vLLM batch embedding.
- Output: pre-built `faiss.index` deployed with the API.

**Online (Query Embedding)**:
- Query embedding served via **vLLM** with embedding-optimized configuration.
- Tuned for low-latency since only a single query is embedded per request.
- vLLM handles batching, memory management, and efficient inference.

### Dimensionality

- Use the model's native output dimension (typically 1024 for Qwen3-Embedding).
- **Do not truncate** unless memory becomes a constraint; evaluate quality impact first.

### Notes on Tokenization

- Qwen3-Embedding uses Qwen's tokenizer (SentencePiece-based).
- **Implementation rule**: Always load the tokenizer via HuggingFace `AutoTokenizer` for the selected embedding model. Do not hardcode tokenizers.

---

## 4.2 Alternative Embedding Models (Keep in Config)

Keep these as evaluated alternatives (model_id is configurable; select via experiments):

### jinaai/jina-embeddings-v3
- MRL (Matryoshka Representation Learning) support for dimension truncation.
- Long input support (up to 8192 tokens).
- Previously evaluated; good fallback option.

### BAAI/bge-m3
- Popular multilingual embedding baseline.
- Designed for multi-linguality and retrieval versatility.

### intfloat/multilingual-e5-large / multilingual-e5-* variants
- Common multilingual retrieval baselines.
- Widely used in RAG systems.

### Selection Guidance

- **V1 default**: `Qwen/Qwen3-Embedding-0.6B` (balances quality, speed, and vLLM serving compatibility).
- Validate retrieval quality against alternatives on a small internal eval set (top-k recall on known sermon queries).

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
- optional: `embeddings.npy` (float32 embeddings for rebuild/debug; not required for serving)

### Key Rules

- Use the **SAME chunk records as BM25** (documents = chunks, keyed by chunk_id).
- Store **NO embeddings inside chunks.sqlite**.
- Keep all embedding model choices configurable:
  - `EMBED_MODEL_ID` (e.g., `Qwen/Qwen3-Embedding-0.6B`)
  - `EMBED_DIM` (model's native dimension; truncation optional)
  - `EMBED_DTYPE` (`fp16`/`bf16`/`fp32`; fp16 recommended on GPU)
  - `EMBED_BATCH_SIZE` (tune per hardware)
  - `EMBED_MAX_LENGTH` (respect model limits)

### Normalization & Similarity (Required)

- Normalize embeddings to unit length and use cosine similarity.
- In FAISS, cosine similarity is implemented as inner product over normalized vectors.

### Tokenizer

- Use `AutoTokenizer.from_pretrained(EMBED_MODEL_ID)` to load the correct tokenizer.

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

- Expect total chunks to be "tens of thousands" depending on chunk size.
- HNSW is often a good starting ANN option for this corpus size.

---

## 4.5 Multilingual Behavior (Query Language vs English Corpus)

### Intent

- Corpus is English-only in V1.
- Queries may be multilingual.
- Embeddings enable cross-lingual retrieval: user-language query → English chunk hits.

### Generation Model Compatibility

- The embedder and generator are decoupled.
- Generator (via LiteLLM → external API) consumes retrieved English context and produces the answer in the user's language.
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

- Pooling strategy: Use model's recommended approach (typically mean pooling for Qwen3-Embedding).
- Normalize vectors (unit length).

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

- Dense retrieval returns faiss row ids → chunk_ids via mapping → chunk records via `chunks.sqlite`
- Dedup and fusion are chunk_id-based (same as BM25)

---

## 4.7 vLLM Serving Configuration (V1 Online)

### Embedding vLLM Instance

The embedding model is served via vLLM for online query embedding:

```yaml
# vLLM embedding config (conceptual)
model: "Qwen/Qwen3-Embedding-0.6B"
task: "embed"  # or equivalent vLLM embedding mode
dtype: "float16"
max_model_len: 1024  # typical query length; tune as needed
# Low-latency config for single-query embedding
```

### Key Points

- **Single-query optimized**: Queries are embedded one at a time (or small batches).
- **Low latency**: Config tuned for fast response, not throughput.
- **Separate from reranker**: Embedding vLLM instance is separate from reranker vLLM instance.

---

## 4.8 Notes for Later Optimization (Do Not Block V1)

- If you later adopt a managed vector DB, keep the same embedding contract (chunk_id as primary key).
- If you later want faster embeddings on NVIDIA, consider optional acceleration flags (`optimum`, `torch.compile`, etc.) but gate them behind config.
- Keep reranker separate; dense embeddings should remain "fast path".

---

## Configuration Summary

### Required Environment Variables / Config

```yaml
# Embedding Model
EMBED_MODEL_ID: "Qwen/Qwen3-Embedding-0.6B"  # V1 default
EMBED_DIM: null                               # Use model's native dimension
EMBED_DTYPE: "fp16"                           # fp16/bf16/fp32
EMBED_BATCH_SIZE: 32                          # tune per hardware
EMBED_MAX_LENGTH: 1024                         # typical max for queries

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
  "model_id": "Qwen/Qwen3-Embedding-0.6B",
  "dim": 1024,
  "dtype": "fp16",
  "normalization": "unit_length",
  "index_type": "IndexFlatIP",
  "index_params": {},
  "corpus_hash": "sha256:...",
  "build_timestamp": "2026-01-25T10:30:00Z",
  "total_chunks": 55000
}
```

---

## Integration with Retrieval Pipeline

### Dense Retrieval Flow (V1)

1. User query (any language) → vLLM (Qwen3-Embedding-0.6B) → query vector
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

## Reranker (Conditional)

### Model

- `Qwen/Qwen3-Reranker-0.6B` served via separate vLLM instance.

### When Invoked

- Only when retrieval signals indicate ambiguity or low confidence.
- Scope: small top-K window (typically 10–20 candidates).

### Integration

- Reranker scores are used to re-order candidates before fusion/selection.
- Does not replace dense retrieval; it refines the ranking.

See main design spec for full reranker flow.

---

**End of specification.**
