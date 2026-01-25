# Design Spec — Branham Model API (Python / Hugging Face Stack)

Project: **Branham Model API**

Purpose: Build a production-grade Python API that serves a **fast RAG pipeline** + **multilingual generator** with **strict grounding** and **message references**.  
This API is the *only* AI endpoint called by the web app (Next.js + Supabase).

---

## 0) Non-negotiables

- **Python-first** implementation.
- Avoid framework-heavy RAG orchestration layers (e.g., LlamaIndex) unless there is a measured performance win.
- RAG is authoritative for verification and references. Fine-tuning helps style/synthesis, but does **not** replace retrieval for correctness.
- Must support:
  - Apple Silicon **MPS** (development)
  - **High Performant CPU** (production)
  - **Multi-GPU** training (via `accelerate`) — future versions
- Must support **concurrency** (handful of concurrent users) and stream responses.
- **No hard-coded model choices.** Embedding model, reranker, generator, tokenizer must be configurable via env/config.
- **Single markdown file** should be usable as Cursor rules. Keep it explicit and implementation-oriented.

---

## 1) Domain constraints and design implications

### 1.1 Sermon structure constraints
- Sermons often tell long stories across multiple paragraphs.
- Users may remember the teaching conceptually but not exact words; BM25 alone can miss paraphrases.
- Therefore:
  - We need **hybrid retrieval** (BM25 + dense).
  - We need **context reconstruction** around retrieved chunks (bounded neighborhood expansion).

### 1.2 Canonical reference system (LOCKED)
- Each sermon has a unique **date_id** (no duplicates across the corpus).
- date_id format:
  - `dd-mm-yy<T>` when multiple sermons on same day, where `<T>` ∈ `{S, M, A, E, B, X}`:
    - `S` = Sunday School
    - `M` = Morning
    - `A` = Afternoon
    - `E` = Evening
    - `B` = Breakfast Sermon
    - `X` = Extras
  - `dd-mm-yy` when single sermon on that day (no suffix)
- References must always include:
  - `date_id`
  - paragraph number(s) (`paragraph_start`, `paragraph_end`)
  - chunk_id(s) used for grounding
- Sermon titles may appear in other languages (model may infer due to pretraining), but **date_id is the canonical and stable locator**.
- We do not rely on char/token offsets; paragraph range is the location anchor.

### 1.3 Multilingual behavior (v1)
- v1 will index **English corpus** only.
- Users may query in other languages.
- Generator outputs answer in user's language.
- References remain in the canonical format: date_id + paragraph range + chunk_ids.

---

## 2) V1 RAG + Generation Architecture

### 2.1 Overall Flow
1. User query received.
2. Query embedding generated using **Qwen/Qwen3-Embedding-0.6B** served via **vLLM (Embedding config)**.
3. Dense retrieval + BM25 executed over pre-embedded sermon corpus.
4. Retrieval quality evaluated.
5. Conditional reranker invoked using **Qwen/Qwen3-Reranker-0.6B** served via **vLLM (Reranker config)** only when retrieval quality is weak.
6. Top-ranked chunks are **collated/grouped by date_id (sermon)** to improve coherence and reduce cross-sermon noise.
7. A preset number of sermons (by date_id) are selected and their best chunks are packed into prompt context (token-budgeted).
8. Text generation performed via a managed Model API layer (e.g., **LiteLLM**), not direct vendor calls, to enable easy model switching.
9. Optional tool calling (Serper) allowed under strict constraints.
10. Final answer returned, grounded in RAG context.

### 2.2 Behavior goals
- Answer only when the retrieved evidence supports the response.
- If evidence is insufficient/mismatched → return a **fixed refusal** message.
- Responses must include references and should support:
  - **Inline references** inside the answer when relevant.
  - A **References** block at the end (like mature RAG systems).

### 2.3 Latency goals (architecture targets)
- p95 retrieval stage (BM25 + dense + fusion + dedup + expansion): target < 150ms on production hardware.
- p95 TTFT (time-to-first-token): target < 1.2s.
- Reranker is conditional; do not pay its cost on clean queries.
- Streaming required (SSE preferred).

### 2.4 Caching policy (V1)
- **No query/response caching in V1** due to highly diverse queries.
- **App layer** (Next.js backend) handles caching if any.
- Model API should not depend on caching for correctness.
- The Model API may still keep warm state (loaded models, loaded indices, etc.).

---

## 3) Retrieval & Reranking (V1)

### 3.1 Embedding model
- **Model**: `Qwen/Qwen3-Embedding-0.6B`
- **Serving**: vLLM with **embedding-optimized configuration** (low-latency for single query embedding online).
- **Offline**: Corpus is pre-embedded; only query embedding happens at request time.

### 3.2 Reranker model
- **Model**: `Qwen/Qwen3-Reranker-0.6B`
- **Serving**: vLLM with **reranker-optimized configuration** (small-batch/latency oriented).
- **Invocation**: Conditional — triggered only when retrieval signals indicate ambiguity or low confidence.
- **Scope**: Limited to a small top-K window (typically 20–50 candidates).

### 3.3 vLLM Service Separation
- Embedding and reranker are deployed as **separate vLLM instances/services** with **different configurations**:
  - **Embedding vLLM config**: tuned for low-latency single-query embedding.
  - **Reranker vLLM config**: tuned for low-latency scoring of small candidate sets.
- They may be co-located on the same hardware, but their configs remain separate to avoid tuning compromises.

### 3.4 Retrieval flow
1. **Early BM25 guard**: Fast pre-check; if no good matches, return fixed refusal (no model call).
2. **Parallel retrieval**: BM25 + dense retrieval run in parallel (topN each; config-driven).
3. **Compute signals**: dense score flatness, overlap between BM25 and dense top results, confidence heuristics.
4. **Conditional reranker**: Trigger if quote intent OR flatness OR low overlap OR low confidence.
5. **Fusion + dedup**: Merge results, deduplicate by chunk_id.
6. **Collate by date_id**: Group chunks by sermon; enforce fixed cap on number of sermons.
7. **Select top-K chunks**: Typically K in [5..10], config-driven.
8. **Expand context**: Bounded expansion ±1 default; ±2 on triggers.

---

## 4) Text Generation (V1)

### 4.1 Model API via LiteLLM
- Generation is handled via a **third-party Model API**, accessed through **LiteLLM** (or equivalent) for:
  - Provider abstraction (OpenRouter, OpenAI, Anthropic, etc.)
  - Rapid switching between models/providers
  - Consistent request/response logging and policy enforcement
- **Note**: V1 does NOT self-host the generator model. Self-hosted generation is planned for future versions when API costs ≈ hosting costs.

### 4.2 Model preferences
- Preference for Qwen-family or other **SentencePiece-based multilingual models**.
- Model must handle multilingual output well.
- Output tokens are expected to be small; input tokens may be large due to RAG context.

### 4.3 Rate limiting strategy
- Procure multiple higher-tier API keys.
- Store API key identifiers in a lookup table (integer → key-name).
- Randomly select a key per request to distribute load and reduce throttling.
- Must comply with provider terms and operational limits.

---

## 5) Serper (Web Search) Tool Usage

### 5.1 Default state
- **Serper is disabled by default.**

### 5.2 Invocation conditions
Tool invocation allowed only when:
- Query explicitly requests current, real-time, or external information, OR
- Query is biographical/factual and cannot be satisfied by RAG corpus.

### 5.3 Hard constraints
- Maximum **1–2 Serper calls per request**.
- Serper must be invoked rarely (exception path, not normal path).

### 5.4 Output handling when Serper is used
- Answer must include a clearly labeled **"Unverified / External Information"** section.
- External sources must be included as Markdown links.
- Web-derived content must **not** be blended into RAG-grounded claims.

---

## 6) Answer Structure Rules

### 6.1 Primary answer
- Always grounded in sermon corpus RAG context.
- Includes inline references where appropriate (e.g., `[47-0412M: ¶2–¶3]`).
- Ends with a consolidated **References** block.

### 6.2 External information (if any)
- Explicitly separated from RAG-grounded content.
- Clearly labeled as unverified.
- Linked to sources.
- **No mixing** of corpus-grounded assertions with web-derived assertions.

### 6.3 Refusal behavior
Two refusal paths:
1. **Pre-LLM refusal**: Early BM25 guard fails (fast skip).
2. **Post-LLM refusal**: Model output fails evidence alignment / references / formatting checks.

---

## 7) System prompt contract (must be enforced)

System prompt must guide the model to:
- Answer ONLY using the provided sermon context.
- If context does not support the answer, output the fixed refusal message.
- Never answer generic questions unrelated to Branham sermons.
- Always include references:
  - inline references when appropriate (e.g., `[47-0412M: ¶2–¶3]`)
  - plus a consolidated references list at end

**Important**: Server-side post-check must enforce the contract; do not rely on prompt alone.

---

## 8) Technology stack (required)

### 8.1 Language/runtime
- Python **3.12+** (3.13 allowed if compatible with dependencies)
- FastAPI + Uvicorn/Gunicorn
- Streaming: SSE (preferred), WebSocket optional

### 8.2 V1 Core stack
- **vLLM**: Serves embedding model and reranker (separate instances).
- **LiteLLM**: Proxies generation requests to external APIs.
- **FAISS**: Dense retrieval (pre-built index).
- **Custom BM25**: Lexical retrieval (pre-built index).
- **SQLite**: Chunk text store (`chunks.sqlite`).

### 8.3 Hugging Face + PyTorch stack (for index building & future training)
- torch
- transformers
- accelerate (multi-GPU training — future)
- peft (LoRA — future)
- bitsandbytes (QLoRA / quantization on NVIDIA — future)
- datasets
- safetensors

Optional (gated by flags; do not assume availability):
- optimum
- flash-attn / xformers
- torch.compile

---

## 9) Data storage conventions

### 9.1 Text store schema (LOCKED)
Each chunk record:
- `chunk_id` (unique)
- `date_id` (sermon identifier)
- `paragraph_start` (int)
- `paragraph_end` (int)
- `chunk_index` (int, position within date_id; recommended for expansion)
- `text` (string)

No offsets required.

### 9.2 BM25 index mapping and dedup
Preferred approach:
- BM25 indexes the same chunk text used by dense retrieval, keyed by `chunk_id`.
- This makes fusion and dedup trivial: dedup is done by `chunk_id`.

---

## 10) Chunking and context expansion (LOCKED)

### 10.1 Chunking rule (LOCKED)
Goal: narrative-friendly but retrieval-stable chunking.

- Create paragraph-aware token-budget chunks around **~350 tokens**.
- Cut only at paragraph boundaries.
- If a paragraph is too long for the token budget:
  - split that paragraph on **sentence boundaries**
- No overlap (we retrieve multiple chunks and expand as needed).
- Store paragraph ranges per chunk.

### 10.2 Bounded neighborhood expansion (LOCKED)
We reconstruct story continuity without storing separate "parent chunks".

- Default expansion depth: **±1 chunk**
- Allow **±2** chunks when more context is needed.
- Expansion triggers should include:
  - query patterns (regex) suggesting narrative/synthesis (e.g., "tell the story", "explain the context", "what happened when…")
  - retrieval confidence signals (flat scores / low overlap / low confidence)
  - reranker-triggered cases (often indicates ambiguity)

Expansion constraints:
- Must respect strict token budget caps.
- Must deduplicate repeated chunks.
- Prefer continuity within same date_id.

---

## 11) End-to-end pipeline steps (V1)

1. Receive query request.
2. Normalize query (whitespace, punctuation).
3. Regex checks:
   - quote intent (e.g., "exact quote", "where did he say", "which sermon")
   - narrative intent patterns (optional)
4. **Early BM25 guard**:
   - Run a fast BM25 check.
   - If BM25 returns "nothing good" (below configured threshold) → return fixed refusal (no model call).
5. **Embed query** via vLLM (Qwen3-Embedding-0.6B).
6. Run **BM25 + dense retrieval in parallel** (topN each; config-driven).
7. Compute retrieval signals:
   - dense score flatness
   - overlap between BM25 and dense top results
   - confidence heuristics
8. **Conditional reranker** (Qwen3-Reranker-0.6B via vLLM):
   - Trigger if quote intent OR flatness OR low overlap OR low confidence.
9. Fuse + dedup:
   - merge BM25 + dense (+ reranker if present)
   - deduplicate by chunk_id
10. **Collate by date_id**:
    - Group chunks by sermon
    - Select top N sermons (config-driven cap)
11. Select final top-K chunks from selected sermons.
12. Expand context:
    - bounded expansion ±1 default; ±2 on triggers
13. Build prompt with:
    - system prompt rules
    - retrieved context
    - user language
    - formatting requirements (inline refs + refs block)
14. **Generate response** via LiteLLM → external API (multilingual model).
15. Post-check enforcement:
    - references present
    - references valid (date_id exists; chunk_ids exist; paragraph range present)
    - output format compliance
    - evidence alignment (if required checks fail → refusal)
16. Return answer + references payload.

---

## 12) Deployment & Model Artifact Handling (V1)

### 12.1 Container strategy
- **No model weights are baked into container images.**
- Models are fetched and loaded at container startup; containers remain lightweight and stateless.
- vLLM loads standard HF artifacts (tokenizer/config + weights, commonly **safetensors**).

### 12.2 Artifact policy
- **Production**: Keep only the minimal HF snapshot artifacts required to boot reliably (mounted cache volume) + any vLLM runtime cache artifacts created at startup.
- **Development**: May keep both the raw safetensors snapshot and extra vLLM/runtime artifacts to speed iteration.

### 12.3 Warm-up
- Warm-up performed at startup to avoid first-request latency.
- Health check should verify models are loaded and indices are available.

### 12.4 Index artifacts (deployed with API)
- `bm25.index`
- `faiss.index`
- `faiss_id_map.jsonl`
- `faiss_meta.json`
- `chunks.sqlite`

---

## 13) Training plan (Future — NOT V1)

> **Note**: V1 uses external generation APIs via LiteLLM. Self-hosted fine-tuned models are planned for future versions when API costs ≈ hosting costs.

### 13.1 Continued pretraining adapter (future)
- Train a LoRA/QLoRA adapter on the sermon corpus to internalize:
  - sermon tone, cadence
  - recurring motifs and phrasing
  - multilingual sermon-title familiarity (if multilingual shards are included)

### 13.2 Training data format (future)
- JSONL preferred.
- Each record should include at least:
  - `text`
  - `date_id`
  - `language`
  - `paragraph_start`
  - `paragraph_end`
- Shard deterministically for reproducibility.

### 13.3 Q/A instruction adapter (future)
- Separate dataset creation pipeline to create grounded Q/A records.
- Must teach:
  - cite-or-refuse behavior
  - quote-intent behavior
  - multilingual output formatting

---

## 14) Hardware and device support rules

### 14.1 Device selection
- Prefer `mps` if available and enabled (development).
- Else prefer `cuda` (production).
- Else CPU.

### 14.2 Configuration flags (required)
- `DEVICE_PREFERENCE=mps|cuda|cpu|auto`
- `DTYPE=fp16|bf16|fp32`

### 14.3 Multi-GPU (future)
- Training must support multi-GPU via `accelerate` (future versions).
- Keep inference modular so multi-GPU inference can be added later (do not hardcode single-device assumptions).

---

## 15) API contract

### 15.1 POST /chat
Request fields:
- `session_id` (string)
- `query` (string)
- `history_window` (recent turns)


Response fields:
- `answer` (string)
- `mode` (`tool call` | `valid response` | `refusal`)
- `references` (array):
  - `date_id`
  - `paragraph_start`
  - `paragraph_end`
  - `chunk_ids` (array)
- `external_info` (optional object, present only when Serper was used):
  - `disclaimer` (string)
  - `sources` (array of URLs)

### 15.2 GET /health
- readiness check: indices available + vLLM services healthy + LiteLLM configured

---

## 16) Preferred reference rendering

- Inline references in the answer when relevant:
  - Example: `... (see [47-0412M: ¶2–¶3])`
- References block at end (consolidated):
  - Each item includes date_id + paragraph range + optionally title.

---

## 17) Required repository layout (scaffold)

The repository must be organized so each subsystem is clear and independently runnable.

```
model-api/
├── README.md
├── config/
│   ├── default.yaml
│   ├── dev.yaml
│   └── prod.yaml
├── src/
│   └── branham_model_api/
│       ├── api/
│       │   ├── main.py
│       │   ├── routes/
│       │   │   ├── chat.py
│       │   │   └── health.py
│       │   ├── schemas/
│       │   │   ├── request.py
│       │   │   └── response.py
│       │   └── middleware/
│       ├── core/
│       │   ├── pipeline/
│       │   │   ├── rag_pipeline.py
│       │   │   ├── fusion.py
│       │   │   ├── rerank.py
│       │   │   ├── expansion.py
│       │   │   ├── postcheck.py
│       │   │   └── signals.py
│       │   ├── prompts/
│       │   │   ├── system_prompt.txt
│       │   │   └── templates.py
│       │   └── refs/
│       │       └── format_refs.py
│       ├── retrieval/
│       │   ├── bm25/
│       │   │   ├── index.py
│       │   │   └── query.py
│       │   ├── dense/
│       │   │   ├── embedder.py
│       │   │   ├── index_faiss.py
│       │   │   └── query.py
│       │   └── store/
│       │       └── chunk_store.py
│       ├── generation/
│       │   ├── litellm_client.py
│       │   └── api_keys.py
│       └── utils/
│           ├── device.py
│           ├── batching.py
│           └── timing.py
├── datasets/
│   ├── ingest/
│   ├── export/
│   └── docs/
│       ├── DATA_FORMAT.md
│       ├── BM25_INDEX.md
│       └── DENSE_RETRIEVAL.md
├── training/                     # Future — NOT V1
│   ├── continued_pretrain/
│   ├── instruction_tune/
│   ├── eval/
│   └── docs/
│       └── TRAINING_GUIDE.md
├── scripts/
│   ├── build_bm25_index.py
│   ├── build_faiss_index.py
│   └── run_dev.sh
└── tests/
    ├── test_chunking.py
    ├── test_fusion_dedup.py
    └── test_postcheck.py
```

---

## 18) ASCII architecture diagram (V1)

```
+------------------+          +-------------------------+
|     Frontend     |          |     Web App Backend     |
| (Next.js client) |--HTTPS-->| (sessions, history, DB) |
+------------------+          +-----------+-------------+
                                        |
                                        | Single downstream call
                                        v
                         +--------------+--------------+
                         |        Branham Model API    |
                         |   (FastAPI + SSE streaming) |
                         +--------------+--------------+
                                        |
    +-----------------------------------+-----------------------------------+
    |                                   |                                   |
    v                                   v                                   v
+-------------------+         +-------------------+         +-------------------+
|  vLLM: Embedder   |         |  vLLM: Reranker   |         |     LiteLLM       |
| Qwen3-Embed-0.6B  |         | Qwen3-Rerank-0.6B |         |  (External APIs)  |
| (query embedding) |         |  (conditional)    |         |   (generation)    |
+-------------------+         +-------------------+         +-------------------+
         |                             |                             |
         v                             v                             v
+----------------------------------------------------------------------+
|                         Pipeline (Request)                           |
|----------------------------------------------------------------------|
|  1) Normalize query + regex intent flags                             |
|                                                                      |
|  2) Early BM25 guard                                                 |
|     - if "not good": return fixed refusal (NO generator call)        |
|                                                                      |
|  3) Embed query (vLLM: Qwen3-Embedding-0.6B)                         |
|                                                                      |
|  4) Parallel retrieval                                               |
|     +------------------+      +------------------+                   |
|     |      BM25        |      | Dense (FAISS)   |                   |
|     | (topN chunks)    |      | (topN chunks)   |                   |
|     +---------+--------+      +---------+--------+                   |
|               \__________________________/                           |
|                          |                                           |
|  5) Signals: flatness / overlap / confidence                         |
|                                                                      |
|  6) Conditional reranker (vLLM: Qwen3-Reranker-0.6B)                 |
|     - only if triggered by signals                                   |
|                                                                      |
|  7) Fusion + dedup (chunk_id)                                        |
|                                                                      |
|  8) Collate by date_id (group by sermon, cap sermon count)           |
|                                                                      |
|  9) Select top-K chunks from selected sermons                        |
|                                                                      |
| 10) Bounded neighborhood expansion (±1 or ±2)                        |
|                                                                      |
| 11) Prompt build (system rules + context + user language)            |
|                                                                      |
| 12) Generate response (LiteLLM → External API)                       |
|                                                                      |
| 13) Post-check enforcement                                           |
|     - references present + valid (date_id + paragraph range)         |
|     - format compliance                                              |
|     - mismatch -> fixed refusal                                      |
|                                                                      |
+----------------------------------------------------------------------+
                                        |
                                        v
                         +--------------+--------------+
                         |     Answer + References     |
                         | inline refs + refs block    |
                         | (+ external info if Serper) |
                         +-----------------------------+
```

---

## 19) V1 Model Decisions (Locked)

| Component   | Model                        | Serving       |
|-------------|------------------------------|---------------|
| Embedding   | `Qwen/Qwen3-Embedding-0.6B`  | vLLM          |
| Reranker    | `Qwen/Qwen3-Reranker-0.6B`   | vLLM          |
| Generation  | External API (configurable)  | LiteLLM       |

### Configuration (remains env/config-driven)
- Exact external generator model ID (via LiteLLM)
- FAISS index type (flatip baseline; HNSW optional)

---

## 20) Future Evolution (Out of Scope for V1)

- Self-hosted generation model via vLLM when API costs ≈ hosting costs.
- LoRA / QLoRA fine-tuning (two-stage: domain adaptation → instruction tuning).
- Adding caching layers (query/retrieval/generation) in later versions.
- Optional long-context RoPE scaling for self-hosted generation model if needed.
- Multi-GPU inference.

---

**End of specification.**
