# Branham Model API — System Architecture Diagrams

> Prepared for Global Talent Visa application.
> Diagrams reflect the full V1 production architecture as designed and implemented.

---

## 1. Full System Architecture (End-to-End)

```
 +==============================+
 |        END USER (Browser)    |
 +==============+===============+
                |
                | HTTPS
                v
 +==============================+       +=========================+
 |     Next.js Frontend App     |       |     Supabase Backend    |
 |------------------------------|       |-------------------------|
 | - Chat UI (split layout)     |<----->| - Auth / Sessions       |
 | - Evidence panel (RAG)       |       | - Conversation DB       |
 | - SSE event parser           |       | - User profiles         |
 | - conversation_summary mgmt  |       | - History persistence   |
 +==============+===============+       +=========================+
                |
                | POST /api/chat (SSE)
                | Authorization: Bearer <KEY>
                |
                | Request JSON:
                | {
                |   conversation_id,
                |   query,
                |   user_language?,
                |   conversation_summary?,
                |   history_window?
                | }
                v
 +=====================================================================+
 |                                                                     |
 |                    BRANHAM MODEL API (FastAPI)                       |
 |                    ~~~~~~~~~~~~~~~~~~~~~~~~~~                       |
 |   Python 3.12+ | Uvicorn | SSE Streaming | Bearer Token Auth       |
 |                                                                     |
 |  +---------------------------------------------------------------+  |
 |  |                    REQUEST INGRESS                             |  |
 |  |---------------------------------------------------------------|  |
 |  |  1. Bearer token validation                                   |  |
 |  |  2. Parse request body (conversation_id, query, etc.)         |  |
 |  |  3. Normalize query (whitespace, punctuation)                 |  |
 |  +---------------------------+-----------------------------------+  |
 |                              |                                      |
 |                              v                                      |
 |  +---------------------------------------------------------------+  |
 |  |                  LANGUAGE GATE (Deterministic)                 |  |
 |  |---------------------------------------------------------------|  |
 |  |                                                               |  |
 |  |  user_language provided?                                      |  |
 |  |      |                                                        |  |
 |  |      +-- YES: starts with "en"? -----> ALLOW (English)        |  |
 |  |      |                                                        |  |
 |  |      +-- YES: NOT "en" prefix? ------> BLOCK (non-English)    |  |
 |  |      |                                                        |  |
 |  |      +-- NO: scan query characters                            |  |
 |  |              any non-ASCII alphabetic? --> BLOCK               |  |
 |  |              all ASCII alphabetic? ------> ALLOW               |  |
 |  |                                                               |  |
 |  |  BLOCK path:                                                  |  |
 |  |    - Translate English-only message to user's language (LLM)  |  |
 |  |    - Stream refusal via SSE: start -> delta -> final -> done  |  |
 |  |    - Skip retrieval, skip tools entirely                      |  |
 |  |                                                               |  |
 |  +------------------+--------------------------------------------+  |
 |                     |                                               |
 |                     | ALLOW (English)                               |
 |                     v                                               |
 |  +---------------------------------------------------------------+  |
 |  |              EMIT SSE "start" EVENT                            |  |
 |  |  { conversation_id }                                          |  |
 |  +---------------------------+-----------------------------------+  |
 |                              |                                      |
 |                              v                                      |
 |  +===============================================================+  |
 |  ||                   RAG PIPELINE                               ||  |
 |  ||=============================================================||  |
 |  ||                                                             ||  |
 |  ||  Build retrieval query:                                     ||  |
 |  ||    query + conversation_summary (if provided)               ||  |
 |  ||                                                             ||  |
 |  ||  Regex intent detection:                                    ||  |
 |  ||    - quote_intent (exact quote / "where did he say")        ||  |
 |  ||    - query_length (word count)                              ||  |
 |  ||                                                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |         STEP 1: QUERY EMBEDDING                        |  ||  |
 |  ||  |  Qwen/Qwen3-Embedding-0.6B                             |  ||  |
 |  ||  |  (HuggingFace direct on dev / vLLM on prod+CUDA)       |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 2: PARALLEL RETRIEVAL (target < 150ms)         |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  +-------------------+     +------------------------+  |  ||  |
 |  ||  |  |   BM25 Retrieval  |     |   Dense (FAISS) Retr.  |  |  ||  |
 |  ||  |  |   topN = 25       |     |   topN = 25            |  |  ||  |
 |  ||  |  |   rank-bm25 lib   |     |   FAISS.search()       |  |  ||  |
 |  ||  |  |   k1=1.5, b=0.75  |     |   cosine similarity    |  |  ||  |
 |  ||  |  +---------+---------+     +-----------+------------+  |  ||  |
 |  ||  |            |                           |               |  ||  |
 |  ||  |            +-------------+-------------+               |  ||  |
 |  ||  |                          |                             |  ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||                             |                                ||  |
 |  ||                             v                                ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 3: COMPUTE RETRIEVAL SIGNALS                   |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  dense_score_std   = std(top-10 dense scores)          |  ||  |
 |  ||  |  dense_top_score   = max(dense scores)                 |  ||  |
 |  ||  |  bm25_dense_overlap = |top10_bm25 ∩ top10_dense|       |  ||  |
 |  ||  |  quote_intent      = regex match (bool)                |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 4: CONDITIONAL RERANKER (see detail below)     |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  Config mode: never | conditional | always             |  ||  |
 |  ||  |  Default: "never" (adds ~10s latency when active)      |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 5: RRF FUSION + DEDUP                          |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  Reciprocal Rank Fusion:                               |  ||  |
 |  ||  |    score = Σ 1/(k + rank_i) across retrievers          |  ||  |
 |  ||  |  Dedup by chunk_id                                     |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 6: SERMON COLLATION                            |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  Group chunks by date_id (sermon)                      |  ||  |
 |  ||  |  Composite score per sermon:                           |  ||  |
 |  ||  |    0.5*(chunk_count/max) + 0.5*(best_score/max)        |  ||  |
 |  ||  |  Cap at 8 sermons                                      |  ||  |
 |  ||  |  Exact match fallback: +1 sermon if date_id in query   |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 7: CONTEXT EXPANSION + DEDUP                   |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  Default: ±0 (disabled; DB Search tool handles this)   |  ||  |
 |  ||  |  Optional: ±1 adjacent chunks within same sermon       |  ||  |
 |  ||  |  Dedup after expansion (adjacent may already exist)    |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    STEP 8: POST-FUSION REFUSAL CHECK                   |  ||  |
 |  ||  |                                                        |  ||  |
 |  ||  |  IF no fused chunks --> REFUSE (no LLM call)           |  ||  |
 |  ||  |  IF dense_top < 0.55 AND bm25_top < threshold          |  ||  |
 |  ||  |    --> REFUSE (off-topic, stream refusal via SSE)       |  ||  |
 |  ||  +----------------------------+--------------------------+  ||  |
 |  ||                               |                             ||  |
 |  ||                               | PASS                        ||  |
 |  ||                               v                             ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||  |    EMIT SSE "rag" EVENT (before LLM call)              |  ||  |
 |  ||  |    { retrieval_query, rag_context, retrieval signals }  |  ||  |
 |  ||  +-------------------------------------------------------+  ||  |
 |  ||                                                             ||  |
 |  +===============================================================+  |
 |                              |                                      |
 |                              v                                      |
 |  +===============================================================+  |
 |  ||              PROMPT CONSTRUCTION                             ||  |
 |  ||=============================================================||  |
 |  ||                                                             ||  |
 |  ||  System prompt:                                             ||  |
 |  ||    - Grounding rules (answer only from context)             ||  |
 |  ||    - Refusal instructions                                   ||  |
 |  ||    - Reference formatting (date_id + paragraph ranges)      ||  |
 |  ||    - External info section rules                            ||  |
 |  ||    - Tool definitions: db_search, biography, internet_search||  |
 |  ||                                                             ||  |
 |  ||  User message:                                              ||  |
 |  ||    - Current query                                          ||  |
 |  ||    - conversation_summary (retrieval context)               ||  |
 |  ||    - history_window (dialogue continuity)                   ||  |
 |  ||    - Retrieved RAG context (deduplicated, expanded)         ||  |
 |  ||    - user_language                                          ||  |
 |  ||                                                             ||  |
 |  +===============================================================+  |
 |                              |                                      |
 |                              v                                      |
 |  +===============================================================+  |
 |  ||                 LLM GENERATION + TOOL LOOP                  ||  |
 |  ||=============================================================||  |
 |  ||                                                             ||  |
 |  ||   LiteLLM  ------>  External Provider API                   ||  |
 |  ||   (multi-provider)  (OpenRouter / DeepSeek / etc.)          ||  |
 |  ||                                                             ||  |
 |  ||   API Key Rotation: OPENROUTER_API_KEY_A .. _E              ||  |
 |  ||   Random selection per request, retry on 429                ||  |
 |  ||                                                             ||  |
 |  ||   +-----------------------------------------------------+   ||  |
 |  ||   |             TOOL-CALLING LOOP                        |   ||  |
 |  ||   |   Max rounds: 4 (hard) / 3 (soft)                   |   ||  |
 |  ||   |   Max iterations: 5                                  |   ||  |
 |  ||   |                                                      |   ||  |
 |  ||   |   Per iteration:                                     |   ||  |
 |  ||   |     1. Offer tool defs (if budget remains)           |   ||  |
 |  ||   |     2. Stream LLM response                           |   ||  |
 |  ||   |     3. Tool calls in response?                        |   ||  |
 |  ||   |         NO  --> final answer text --> EXIT LOOP       |   ||  |
 |  ||   |         YES --> execute tools in parallel             |   ||  |
 |  ||   |              --> append results as messages           |   ||  |
 |  ||   |              --> next iteration                       |   ||  |
 |  ||   |                                                      |   ||  |
 |  ||   |   +----------------------------------------------+   |   ||  |
 |  ||   |   |        AVAILABLE TOOLS                       |   |   ||  |
 |  ||   |   |                                              |   |   ||  |
 |  ||   |   |  +----------------+  +------------------+    |   |   ||  |
 |  ||   |   |  | DB SEARCH      |  | BIOGRAPHY        |    |   |   ||  |
 |  ||   |   |  | (Sermon Lookup)|  | (Bio facts)      |    |   |   ||  |
 |  ||   |   |  | Max: 3 calls   |  | Max: 2 calls     |    |   |   ||  |
 |  ||   |   |  | Modes:         |  | Curated data     |    |   |   ||  |
 |  ||   |   |  |  read_sermon   |  | source: bio.txt  |    |   |   ||  |
 |  ||   |   |  |  read_paragraphs|  +------------------+   |   |   ||  |
 |  ||   |   |  |  search_quote  |                          |   |   ||  |
 |  ||   |   |  |  batch_read    |  +------------------+    |   |   ||  |
 |  ||   |   |  +----------------+  | INTERNET SEARCH  |    |   |   ||  |
 |  ||   |   |                      | (Serper, gated)  |    |   |   ||  |
 |  ||   |   |  SQLite chunk store  | Max: 2 calls     |    |   |   ||  |
 |  ||   |   |  (~207k chunks)      | Disabled default |    |   |   ||  |
 |  ||   |   |                      | Results labeled  |    |   |   ||  |
 |  ||   |   |                      | "Unverified"     |    |   |   ||  |
 |  ||   |   |                      +------------------+    |   |   ||  |
 |  ||   |   +----------------------------------------------+   |   ||  |
 |  ||   +-----------------------------------------------------+   ||  |
 |  ||                                                             ||  |
 |  +===============================================================+  |
 |                              |                                      |
 |                              v                                      |
 |  +---------------------------------------------------------------+  |
 |  |              POST-CHECK NORMALIZATION                          |  |
 |  |---------------------------------------------------------------|  |
 |  |  - Append/remove "Unverified / External Information" section  |  |
 |  |  - Markdown normalization                                     |  |
 |  |  - Reference format validation (date_id + paragraph range)    |  |
 |  |  - No post-LLM refusal conversion (streaming-first)          |  |
 |  +---------------------------+-----------------------------------+  |
 |                              |                                      |
 |                              v                                      |
 |  +---------------------------------------------------------------+  |
 |  |              SSE STREAM OUTPUT                                 |  |
 |  |---------------------------------------------------------------|  |
 |  |                                                               |  |
 |  |  "start"  { conversation_id }                                 |  |
 |  |      |                                                        |  |
 |  |      v                                                        |  |
 |  |  "rag"    { retrieval_query, rag_context, retrieval }         |  |
 |  |      |       (emitted before first LLM token)                 |  |
 |  |      v                                                        |  |
 |  |  "delta"  { text } ... { text } ... { text }                  |  |
 |  |      |       (progressive answer chunks)                      |  |
 |  |      v                                                        |  |
 |  |  "final"  { mode, answer, external_info,                      |  |
 |  |      |      conversation_summary, query_summary }             |  |
 |  |      v                                                        |  |
 |  |  "done"   { ok: true }                                        |  |
 |  |                                                               |  |
 |  +---------------------------------------------------------------+  |
 |                                                                     |
 +=====================================================================+
```

---

## 2. Conditional Reranker Decision Flow

```
                    +---------------------------+
                    |   Retrieval Signals from   |
                    |   BM25 + Dense results     |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    |  Config: reranker.enabled  |
                    +--+----------+----------+--+
                       |          |          |
                  "never"    "conditional"  "always"
                       |          |          |
                       v          v          v
                   +------+  +--------+  +------+
                   | SKIP |  | CHECK  |  | RUN  |
                   +------+  | SIGNALS|  +--+---+
                       |     +---+----+     |
                       |         |          |
                       |         v          |
                       |  +-------------------------------+
                       |  | ANY of these true?            |
                       |  |                               |
                       |  | dense_score_std < 0.08        |
                       |  |   (flat scores = ambiguous)   |
                       |  |                               |
                       |  | bm25_dense_overlap < 3        |
                       |  |   (retrievers disagree)       |
                       |  |                               |
                       |  | dense_top_score < 0.65        |
                       |  |   (weak semantic match)       |
                       |  |                               |
                       |  | quote_intent == true           |
                       |  |   (user wants exact quote)    |
                       |  +------+------------------------+
                       |         |
                       |    YES  |  NO
                       |    |    |  |
                       |    v    |  v
                       |  +---+  | +------+
                       |  |RUN|  | | SKIP |
                       |  +-+-+  | +--+---+
                       |    |    |    |
                       |    v    |    |
                       | +------------------+  |
                       | | Qwen3-Reranker   |  |
                       | | -0.6B            |  |
                       | | Re-score top-K   |  |
                       | | candidates       |  |
                       | | (~20-50 chunks)  |  |
                       | | Latency: ~10-12s |  |
                       | +--------+---------+  |
                       |          |            |
                       +----------+------------+
                                  |
                                  v
                    +---------------------------+
                    |  Continue to RRF Fusion   |
                    +---------------------------+
```

---

## 3. Language Gate Decision Flow

```
                        +---------------------+
                        |   Incoming Request   |
                        |   query, user_lang   |
                        +----------+----------+
                                   |
                                   v
                        +---------------------+
                        | user_language field  |
                        | provided?            |
                        +---+-------------+---+
                            |             |
                           YES            NO
                            |             |
                            v             v
                  +----------------+  +----------------------+
                  | Starts with    |  | Scan query chars     |
                  | "en"?          |  | for non-ASCII        |
                  +--+----------+--+  | alphabetic           |
                     |          |     +---+--------------+---+
                    YES         NO        |              |
                     |          |     Non-ASCII      All ASCII
                     |          |     found          chars
                     v          v         |              |
                  +------+  +------+  +------+      +------+
                  |ALLOW |  |BLOCK |  |BLOCK |      |ALLOW |
                  |Eng.  |  |      |  |      |      |Eng.  |
                  +--+---+  +--+---+  +--+---+      +--+---+
                     |         |         |              |
                     |         v         v              |
                     |   +---------------------------+  |
                     |   | NON-ENGLISH REFUSAL PATH  |  |
                     |   |                           |  |
                     |   | 1. Skip retrieval         |  |
                     |   | 2. Skip tools             |  |
                     |   | 3. Translate refusal msg   |  |
                     |   |    to user's language     |  |
                     |   | 4. Stream via SSE:        |  |
                     |   |    start->delta->final->  |  |
                     |   |    done (mode: "refusal") |  |
                     |   +---------------------------+  |
                     |                                  |
                     +----------------------------------+
                     |
                     v
              +-------------+
              | Continue to |
              | RAG Pipeline|
              +-------------+
```

---

## 4. vLLM Serving Architecture (Production Target)

```
 +=====================================================================+
 |                   vLLM DEPLOYMENT (Production)                      |
 |=====================================================================|
 |                                                                     |
 |  Branch-based backend selection:                                    |
 |                                                                     |
 |  +--------------------+     +--------------------+                  |
 |  |  develop branch    |     |  main branch       |                  |
 |  |  (any device)      |     |                    |                  |
 |  |--------------------|     |--------------------|                  |
 |  |  HuggingFace       |     |  CUDA available?   |                  |
 |  |  Transformers      |     |    |           |   |                  |
 |  |  (direct inference)|     |   YES          NO  |                  |
 |  +--------------------+     |    |           |   |                  |
 |                             |    v           v   |                  |
 |                             | +------+  +------+ |                  |
 |                             | | vLLM |  |  HF  | |                  |
 |                             | | Prod |  | Fall | |                  |
 |                             | |      |  | back | |                  |
 |                             | +------+  +------+ |                  |
 |                             +--------------------+                  |
 |                                                                     |
 |  +---------------------------------+  +---------------------------+ |
 |  |  vLLM Instance 1: EMBEDDER     |  | vLLM Instance 2: RERANKER | |
 |  |---------------------------------|  |---------------------------| |
 |  |  Model: Qwen3-Embedding-0.6B   |  | Model: Qwen3-Reranker    | |
 |  |                                 |  |         -0.6B            | |
 |  |  Config: Embedding-optimized    |  | Config: Reranker-         | |
 |  |    - Low-latency single query   |  |   optimized              | |
 |  |    - Online serving mode        |  |   - Small-batch scoring  | |
 |  |    - Mean pooling               |  |   - Latency oriented     | |
 |  |    - Query instruction template |  |   - Prefix caching       | |
 |  |                                 |  |   - Yes/No token logits  | |
 |  |  Input: User query text         |  |                          | |
 |  |  Output: Dense embedding vector |  | Input: (query, doc) pairs| |
 |  |                                 |  | Output: Relevance scores | |
 |  |  GPU Memory: configurable       |  |                          | |
 |  |    (0.5-0.6 utilization)        |  | GPU Memory: configurable | |
 |  |                                 |  |   (0.5-0.6 utilization)  | |
 |  |  Supports:                      |  |                          | |
 |  |    - CUDA graphs                |  | Supports:                | |
 |  |    - Tensor parallelism         |  |   - CUDA graphs          | |
 |  |    - Native quantization        |  |   - Tensor parallelism   | |
 |  |    - Lazy initialization        |  |   - Native quantization  | |
 |  +---------------------------------+  +---------------------------+ |
 |                                                                     |
 |  +---------------------------------------------------------------+  |
 |  |  HuggingFace Direct Backend (Dev / Fallback)                  |  |
 |  |---------------------------------------------------------------|  |
 |  |                                                               |  |
 |  |  Embedder:                      Reranker:                     |  |
 |  |    AutoModel.from_pretrained()    AutoModelForCausalLM        |  |
 |  |    Manual pooling (mean/cls)      .from_pretrained()          |  |
 |  |    Supports MPS (Apple Silicon)   Yes/No token classification |  |
 |  |    Offline mode via               Batch size: 8               |  |
 |  |      snapshot_download()          Supports MPS + CPU          |  |
 |  |                                                               |  |
 |  +---------------------------------------------------------------+  |
 |                                                                     |
 +=====================================================================+

 Model Artifact Flow (Container Startup):

 +----------------+      +------------------+      +-------------------+
 | HuggingFace Hub|----->| prefetch_hf_model|----->| .hf-cache/        |
 | (Remote)       | pull | .py --all --warm | save | (Local volume)    |
 +----------------+      | --skip-reranker  |      |                   |
                          +------------------+      | Qwen3-Embedding/ |
                                                    | Qwen3-Reranker/  |
                          At startup:               | (safetensors)    |
                          No weights baked into     +--------+----------+
                          container image                    |
                                                             | load at boot
                                                             v
                                                    +-------------------+
                                                    | vLLM / HF Runtime |
                                                    | (warm-up + health)|
                                                    +-------------------+
```

---

## 5. Data Flow & Storage Architecture

```
 +=================================================================+
 |                    INDEX + DATA ARTIFACTS                        |
 |=================================================================|
 |                                                                 |
 |  Sermon Corpus (English)                                        |
 |       |                                                         |
 |       | Offline index build (scripts/)                          |
 |       v                                                         |
 |  +-----------------------------------------------------------+  |
 |  |                                                           |  |
 |  |  data/indices/                data/processed/             |  |
 |  |  +-----------------+         +-----------------------+    |  |
 |  |  | bm25.index      |         | chunks.sqlite         |   |  |
 |  |  | (BM25 inverted  |         | (~207k chunks)        |   |  |
 |  |  |  index, keyed   |         |                       |   |  |
 |  |  |  by chunk_id)   |         | Schema per chunk:     |   |  |
 |  |  +-----------------+         |  - chunk_id (unique)   |   |  |
 |  |  | faiss.index     |         |  - date_id (sermon)   |   |  |
 |  |  | (FAISS flat IP  |         |  - paragraph_start    |   |  |
 |  |  |  / HNSW)        |         |  - paragraph_end      |   |  |
 |  |  +-----------------+         |  - chunk_index (pos)   |   |  |
 |  |  | faiss_id_map    |         |  - text (content)     |   |  |
 |  |  |   .jsonl        |         +-----------------------+    |  |
 |  |  +-----------------+                                      |  |
 |  |  | faiss_meta.json |         data/reference/              |  |
 |  |  +-----------------+         +-----------------------+    |  |
 |  |                              | biography.txt         |   |  |
 |  |  Both indices embed          | (curated bio facts)   |   |  |
 |  |  text_with_metadata:         +-----------------------+    |  |
 |  |    sermon title +                                         |  |
 |  |    date_id +                                              |  |
 |  |    paragraph markers                                      |  |
 |  +-----------------------------------------------------------+  |
 |                                                                 |
 +=================================================================+

 Chunking Strategy (LOCKED):

 +-------------------+    +-------------------+    +------------------+
 | Sermon text       |--->| Paragraph-aware   |--->| Chunks (~350     |
 | (full sermon)     |    | token-budget      |    |  tokens each)    |
 |                   |    | chunking          |    |                  |
 |                   |    |                   |    | Cut only at      |
 |                   |    | If paragraph >    |    | paragraph        |
 |                   |    | budget: split on  |    | boundaries       |
 |                   |    | sentence boundary |    |                  |
 +-------------------+    +-------------------+    | No overlap       |
                                                   | (expansion       |
                                                   | handles context) |
                                                   +------------------+
```

---

## 6. SSE Event Timeline (Frontend Perspective)

```
 Client                              Server
   |                                    |
   |  POST /api/chat                    |
   |  Authorization: Bearer <KEY>       |
   |  { conversation_id, query, ... }   |
   |----------------------------------->|
   |                                    |  Validate bearer token
   |                                    |  Language gate check
   |                                    |  Begin RAG pipeline
   |                                    |
   |    event: start                    |
   |    { conversation_id }             |
   |<-----------------------------------|
   |                                    |
   |          (retrieval runs)          |
   |                                    |
   |    event: rag                      |
   |    { retrieval_query,              |
   |      rag_context,     <-- render   |
   |      retrieval }         evidence  |
   |<-----------------------------------| panel immediately
   |                                    |
   |      (LLM generation begins)      |
   |      (tool loop may execute)       |
   |                                    |
   |    event: delta                    |
   |    { text: "Brother Branham..." }  |
   |<-----------------------------------|
   |    event: delta                    |  Append to
   |    { text: "taught about..." }     |  live UI
   |<-----------------------------------|
   |    event: delta                    |
   |    { text: "faith in..." }         |
   |<-----------------------------------|
   |           ...                      |
   |                                    |
   |    event: final                    |
   |    { mode: "answer",              |
   |      answer: "full text",         |
   |      external_info: null,         |
   |      conversation_summary: "..." } |
   |<-----------------------------------| Save summary
   |                                    | for next turn
   |    event: done                     |
   |    { ok: true }                    |
   |<-----------------------------------|
   |                                    |
```

---

## 7. Tool Invocation Priority & Budget

```
 Query arrives
      |
      v
 +--------------------+
 | 1. RAG Retrieval   |  <-- Always first for sermon queries
 |    (default path)  |      BM25 + Dense + Fusion
 +--------+-----------+
          |
          | Context insufficient?
          v
 +--------------------+
 | 2. DB Search Tool  |  <-- Exact quotes, paragraph lookup
 |    (Sermon Lookup) |      Max 3 calls (batch supported)
 |    Modes:          |
 |     read_sermon    |
 |     read_paragraphs|
 |     search_quote   |
 |     batch_read     |
 +--------+-----------+
          |
          | Biographical question?
          v
 +--------------------+
 | 3. Biography Tool  |  <-- Verified facts about Branham
 |    Max 2 calls     |      Source: curated biography.txt
 +--------+-----------+
          |
          | External/current info needed?
          v
 +--------------------+
 | 4. Internet Search  |  <-- LAST RESORT (Serper)
 |    (Serper)         |      Max 2 calls
 |    Disabled default |      Results marked "Unverified"
 +--------------------+

 Budget Enforcement:
 +------------------------------------------+
 |  Soft limit: 3 tool-call rounds           |
 |  Hard limit: 4 tool-call rounds           |
 |  Max iterations: 5                        |
 |  Parallel execution within each round     |
 |  db_search calls in same round = 1 batch  |
 +------------------------------------------+
```

---

## 8. Summary Diagram (Screenshot-Friendly)

```
+===========================================================================+
|                    BRANHAM MODEL API — V1 ARCHITECTURE                     |
+===========================================================================+
|                                                                           |
|   [ End User ]                                                            |
|       |                                                                   |
|       v                                                                   |
|   [ Next.js Frontend ]  <--->  [ Supabase (Auth, Sessions, History) ]     |
|       |                                                                   |
|       | POST /api/chat (SSE, Bearer Auth)                                 |
|       v                                                                   |
|  +------------------------------------------------------------------+     |
|  |                    FastAPI Application                            |     |
|  |                                                                  |     |
|  |   Language Gate -----> Non-English? --> Polite refusal (streamed) |     |
|  |       |                                                          |     |
|  |       | English                                                  |     |
|  |       v                                                          |     |
|  |   RAG Pipeline                                                   |     |
|  |   +----------------------------------------------------------+   |     |
|  |   |                                                          |   |     |
|  |   |  Query Embedding (Qwen3-Embedding-0.6B)                  |   |     |
|  |   |       |                                                  |   |     |
|  |   |       v                                                  |   |     |
|  |   |  BM25 (topN=25) ---|--- Dense/FAISS (topN=25)            |   |     |
|  |   |       |                       |                          |   |     |
|  |   |       +--- Parallel retrieval -+                         |   |     |
|  |   |                   |                                      |   |     |
|  |   |  Retrieval Signals (std, overlap, top score, quote)      |   |     |
|  |   |                   |                                      |   |     |
|  |   |  Conditional Reranker (Qwen3-Reranker-0.6B, off by dflt) |   |     |
|  |   |                   |                                      |   |     |
|  |   |  RRF Fusion + Dedup ---> Sermon Collation (max 8)        |   |     |
|  |   |                   |                                      |   |     |
|  |   |  Refusal Check (score thresholds)                        |   |     |
|  |   +----------------------------------------------------------+   |     |
|  |       |                                                          |     |
|  |       v                                                          |     |
|  |   Prompt Construction (system rules + RAG context + tools)       |     |
|  |       |                                                          |     |
|  |       v                                                          |     |
|  |   LLM Generation via LiteLLM -----> External API Provider        |     |
|  |       |                              (OpenRouter / DeepSeek)      |     |
|  |       |                              API Key Rotation (A-E)       |     |
|  |       v                                                          |     |
|  |   Tool Loop (max 4 rounds)                                       |     |
|  |   +----------------------------------------------------------+   |     |
|  |   | DB Search (sermon lookup, max 3) | Biography (max 2)     |   |     |
|  |   | Internet Search (Serper, max 2, gated, disabled default) |   |     |
|  |   +----------------------------------------------------------+   |     |
|  |       |                                                          |     |
|  |       v                                                          |     |
|  |   Post-check (normalize markdown, validate refs)                 |     |
|  |       |                                                          |     |
|  |       v                                                          |     |
|  |   SSE Stream: start -> rag -> delta... -> final -> done          |     |
|  +------------------------------------------------------------------+     |
|                                                                           |
|  +------------------------------------------------------------------+     |
|  |                    INDEX & DATA LAYER                             |     |
|  |                                                                  |     |
|  |  FAISS Index  |  BM25 Index  |  SQLite Chunks (~207k)  | Bio.txt|     |
|  +------------------------------------------------------------------+     |
|                                                                           |
|  +------------------------------------------------------------------+     |
|  |               MODEL SERVING (Current / Planned)                  |     |
|  |                                                                  |     |
|  |  CURRENT (dev):  HuggingFace Transformers (MPS/CPU)              |     |
|  |  PLANNED (prod): vLLM (CUDA) - Embedder + Reranker instances     |     |
|  |  GENERATION:     LiteLLM -> External APIs (not self-hosted V1)   |     |
|  +------------------------------------------------------------------+     |
|                                                                           |
+===========================================================================+

 Reference System: date_id + paragraph_start/end (canonical sermon locator)
 Sermon format:  dd-mm-yy or dd-mm-yy<T> where T in {S,M,A,E,B,X}
 Answer format:  inline refs [date_id: ¶N-¶M] + consolidated References block
 Streaming:      ALL responses via SSE (answers, refusals, errors)
 Supported:      English only (V1) | MPS dev / CPU prod / CUDA training
```

---

*Document generated for Global Talent Visa application — Branham Model API V1.*