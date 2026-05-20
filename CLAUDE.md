# CLAUDE.md

Guidance for agents working in this repository. The full V1 design lives in `.cursor/rules/design_spec.md`; the SSE contract consumed by the web app lives in `/Users/emeraldhenry/Branham-Web-App/api_contract.md`. This file is the working summary — read those two for ground truth when in doubt.

## Project Overview

Production-grade Python API serving a RAG pipeline + multilingual generator with strict grounding for the William Branham sermon corpus. This is the **only** AI endpoint called by the Branham Web App (Next.js + Supabase, hosted on Cloudflare Workers). English-only queries are supported; non-English queries are politely declined via a translated gate message. All responses stream via Server-Sent Events; there is no non-stream code path.

## Commands

```bash
# Install / update deps
uv sync

# Dev server (hot reload, port 8000)
./scripts/run_dev.sh
# or:
uv run uvicorn branham_model_api.api.main:app --reload --host 0.0.0.0 --port 8000

# Tests
uv run pytest                              # all
uv run pytest tests/test_chat_sse.py -v    # one file
uv run pytest tests/test_chat_sse.py -k "test_name" -v

# Quality gates
uv run black .
uv run ruff check . --fix
uv run mypy .

# Full non-stream debug flow (writes artifacts to data/logs/chat_flow/)
KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/test_chat_full_flow.py \
  --query "What did Brother Branham teach about faith?"

# Index builds (offline)
uv run python scripts/build_bm25_index.py
uv run python scripts/build_faiss_index.py

# Prefetch / warm HF models (used by the Docker build)
uv run python scripts/prefetch_hf_model.py --all --warm --skip-reranker --target-dir ./.hf-cache

# Local Docker
docker build -t branham-api:latest .
docker run -p 8080:8080 -e OPENROUTER_API_KEY_A=<key> branham-api:latest
```

## Architecture

### `POST /api/chat` request lifecycle (streaming SSE)

1. **Auth** (`api/routes/chat.py` ~L201–218) — `Authorization: Bearer <token>`. Token comes from env `CHAT_API_BEARER_KEY`, falling back to `config.api.bearer_key`.
2. **Request validation** — `ChatRequest` schema (`api/schemas/request.py`). Required: `conversation_id`, `query`. Optional: `user_language`, `conversation_summary`, `history_window`.
3. **Language gate** (`_is_english_request` ~L105–128, used ~L839–899). Deterministic — no model inference. If `user_language` is supplied, it's authoritative; otherwise scan for non-ASCII alphabetic characters. Non-English path skips RAG entirely and asks the LLM to translate `_ENGLISH_ONLY_MESSAGE` into the user's language. Emits `start → delta → final → done` (no `rag` event).
4. **RAG pipeline** (`core/pipeline/rag_pipeline.py`):
   - Normalize query (`normalize_query`, NFKC + whitespace).
   - Detect quote intent via regex (`core/pipeline/signals.py: detect_quote_intent`).
   - Parallel BM25 + dense retrieval (`top_n=25` each) via `ThreadPoolExecutor`.
   - Compute signals — `dense_score_std`, `dense_top_score`, `bm25_dense_overlap`, `quote_intent`.
   - Conditional reranker (Qwen3-Reranker-0.6B). Default `reranker.enabled: never`. When enabled, triggered by low std / no overlap / weak top score / quote intent.
   - RRF fusion (`core/pipeline/fusion.py: merge_bm25_dense`, k=60), dedup by `chunk_id`.
   - Collate by `date_id` (sermon), composite-rank by `0.5 * (chunk_count / max) + 0.5 * (best_score / max)`, cap at 8 sermons.
   - Exact-match fallback promotes already-retrieved chunks if the query mentions a specific date_id/title.
   - **Pre-LLM refusal**: refuse only if BOTH `dense_top < 0.20` AND `bm25_top < 0.20` (English path). Refusal short-circuits the LLM call — emits `start → delta(refusal) → final(mode="refusal") → done`. No `rag` event on early refusal.
   - Optional ±N chunk expansion (`expansion.depth` defaults to 0; the DB-Search tool fetches additional context on demand).
5. **Prompt construction** (`core/prompts/templates.py`):
   - System prompt loaded from `core/prompts/system_prompt.txt` with `{{REFUSAL_MESSAGE}}` and `{{INSUFFICIENT_CONTEXT_MESSAGE}}` placeholders substituted.
   - Mode-specific addenda for Bible-only / comparison queries (`_query_mode_prompt_addendum`, ~L161–184).
   - User message = query + last-6-turn history (if provided) + RAG context block.
6. **LLM streaming via LiteLLM** (`generation/litellm_client.py`) — multi-provider with random key selection (OPENROUTER_API_KEY_A..E, DEEPSEEK_API_KEY) and one rate-limit retry with a different key.
7. **Tool loop**, up to `MAX_TOOL_ITERATIONS = 5`. Two execution modes — **see "MODE A vs MODE B" below; this distinction is load-bearing**.
8. **Postcheck** (`core/pipeline/postcheck.py: finalize_answer`) — reference-format validation, "Unverified / External Information" section management when `internet_search` is used. **Never converts an answer to a refusal** — refusal is a pre-LLM decision only.
9. **Conversation summary** generated as a separate LLM call after the answer. First turn returns `{conversation_summary, query_summary}`; follow-ups return only `conversation_summary` (`query_summary: null`). Returned in the `final` event for the FE to persist.

### MODE A vs MODE B (CRITICAL — read before touching `chat.py`)

Two streaming modes coexist in the tool loop in `api/routes/chat.py`:

- **MODE A — tools offered, buffered consumption** (~L1015–1172). Used while the tool budget is not exhausted. The full LLM stream is consumed via `_consume_stream_buffered` so we can detect tool-call deltas without leaking pre-tool "thinking" text to the client. If the LLM returns tool calls → execute, append, loop. If no tool calls → flush the buffered text as `delta` events.
- **MODE B — no tools offered, real-time per-chunk yield** (~L1188–1202). Used when `tools_exhausted` is true (final round). Pattern:
  ```python
  stream = runtime.llm_client.stream_completion(messages=working_messages)
  for chunk in stream:
      text = _extract_delta_text(chunk)
      if text:
          yield _sse_event("delta", {"text": text})
  ```
  This per-token yield is the **contract that makes the FE markdown render arrive incrementally**. Replacing it with a buffered-then-flush pattern (e.g. wrapping in `run_in_executor` + `_consume_stream_buffered`) breaks live markdown rendering on the web app — content appears seconds late, all-at-once. **Do not change MODE B without an explicit reason and end-to-end FE verification.** Memory: see `feedback_reproduce_before_fixing`.

### Tool budgets (`core/tools/registry.py`, `core/tools/factory.py`)

- `db_search`: 3 calls per request
- `biography_search`: 2 calls per request
- `internet_search` (Serper): 2 calls per request
- Hard total: 4 tool-call rounds. Soft guidance in the system prompt: 3 rounds.
- Multiple `db_search` calls in the same assistant turn are auto-batched into one `batch_mixed` call (counts as one round).
- When budget is reached, the tool registry returns a structured `limit_reached_output()` instructing the model to finalize with existing evidence; the next iteration runs in MODE B.

### SSE event contract (matches `Branham-Web-App/api_contract.md`)

Happy path:
```
start  → { conversation_id }
rag    → { retrieval_query, rag_context, retrieval: { signals, hit counts, ... } }
delta* → { text }
final  → { mode: "answer" | "refusal" | "error",
           answer,
           external_info,            // { disclaimer, sources[] } | null
           conversation_summary,     // string | null
           query_summary }           // string | null (first turn only)
done   → { ok: true }
```

Early refusal (off-topic / insufficient context):
```
start → delta(refusal_text) → final(mode="refusal", *_summary=null) → done
```
No `rag` event is emitted on early refusal or on the non-English path.

Error path:
```
start → error{ mode, answer } → done{ ok: false }
```

Field names on the wire are **snake_case**. The web app's parser maps to camelCase (`conversation_id → conversationId`, `external_info → externalInfo`, etc.). Renaming any wire field is a breaking change.

### Key modules

| Path | Responsibility |
|---|---|
| `api/main.py` | FastAPI app + lifespan that preloads `ChatRuntime` |
| `api/routes/chat.py` | The streaming endpoint — language gate, RAG orchestration, tool loop (MODE A/B), SSE event emission. Large file (~52KB); changes here are high-risk |
| `api/routes/health.py` | `GET /api/health` readiness (bearer key, pipeline, tool registry, llm client, key count) |
| `config.py` + `config/default.yaml` | Typed config; env vars override at load time |
| `core/pipeline/rag_pipeline.py` | Retrieval orchestration |
| `core/pipeline/fusion.py` | RRF merge, dedup, sermon collation, exact-match fallback |
| `core/pipeline/signals.py` | Signal computation + `detect_quote_intent` |
| `core/pipeline/expansion.py` | ±N chunk expansion (off by default) |
| `core/pipeline/postcheck.py` | Output normalization, external section handling |
| `core/prompts/templates.py` + `system_prompt.txt` | Prompt construction; addenda per query mode |
| `core/tools/registry.py` | Per-tool / total budgets, `limit_reached_output` |
| `core/tools/{db_search,biography,serper}_tool.py` | Tool implementations |
| `core/tools/loop_runner.py` | Non-streaming tool loop (used by `scripts/test_chat_full_flow.py`) |
| `generation/litellm_client.py` | LiteLLM wrapper, provider/key rotation, one rate-limit retry |
| `retrieval/dense/{embedder,index_faiss}.py` | Query embedding (HF or vLLM) + FAISS search |
| `retrieval/bm25/index.py` | BM25 index load + query |
| `retrieval/store/chunk_store.py` | SQLite chunk text store |
| `scripts/deploy_cloudrun.py` | Cloud Run / Cloud Build deployment driver |

### Data layout

- `data/indices/` — `bm25.index`, `bm25_doc_map.jsonl`, `bm25_meta.json`, `bm25_vocab.json`, `faiss.index`, `faiss_id_map.jsonl`, `faiss_meta.json` (loaded at startup)
- `data/processed/chunks.sqlite` — canonical chunk text store (~207k chunks). Tables: `sermons(date_id, title, year, language)`, `paragraphs(id, date_id, paragraph_no, sub_id, text)`, `chunks(chunk_id, date_id, paragraph_start, paragraph_end, chunk_index, text)`
- `data/reference/biography.txt` — biography corpus used by the biography tool
- `config/default.yaml` — all tunables (retrieval params, LLM provider, tool limits, etc.)

### Environment variables

| Var | Purpose |
|---|---|
| `CHAT_API_BEARER_KEY` | Overrides `api.bearer_key` |
| `LLM_MODEL` | Overrides selected model |
| `OPENROUTER_API_KEY_A` … `_E` | OpenRouter keys (random pick + one-shot retry on 429) |
| `DEEPSEEK_API_KEY` | DeepSeek provider key |
| `HF_HOME` | HF cache (default `.hf-cache`) |
| `HF_OFFLINE_ONLY` | Force offline model loading (used in container) |
| `LITELLM_DEBUG` | Verbose LiteLLM logs (troubleshooting only) |

### Design constraints

- **Streaming-first.** No non-stream response path. All outcomes (answer, refusal, error, language gate) are SSE.
- **Composition over orchestration.** No LlamaIndex / LangChain.
- **Sermon references** are `[TITLE — date_id: ¶X–¶Y]`; never character/token offsets.
- **MODE B real-time yield is non-negotiable** (see above).
- **Long-stable code paths** in `chat.py` deserve high-confidence reproduction before any change. See `feedback_reproduce_before_fixing` memory: do not ship architectural fixes from code reading + circumstantial logs alone — reproduce the user's exact symptom first, or add observability rather than rewriting.
- Apple Silicon MPS (dev), CPU (production), CUDA (training).

## How the Branham Web App consumes this API (cross-repo coupling)

The web app at `/Users/emeraldhenry/Branham-Web-App` is a Next.js 16 (App Router) app deployed to Cloudflare Workers via `@opennextjs/cloudflare`. The chat UI is a tightly-coupled SSE consumer — the API contract below is what makes it work; any change to event names, field names, ordering, or per-chunk delivery cadence requires a coordinated FE change.

### How it talks to this API

- The browser POSTs to **`/api/chat` on the same origin** (`src/app/api/chat/route.ts`), which is a Cloudflare-Workers route handler that proxies to `${MODEL_API_BASE_URL}/api/chat`, injecting `Authorization: Bearer ${CHAT_API_BEARER_KEY}` server-side. **The bearer key never reaches the client bundle.**
- The proxy validates request size (conversation_id ≤128 chars, query ≤4000 chars, history ≤12 messages of ≤4000 chars each) and applies a 10-req/min IP rate-limit for anonymous users.
- The browser parses the SSE stream with `src/lib/sse/parser.ts` (`processSSEStream` + `parseChatEvent`).

### FE state machine (in `src/components/chat/ChatShell.tsx`)

`StreamingStatus` ∈ `idle | connecting | rag_received | streaming | complete | error`.

| API event | FE transition / side effect |
|---|---|
| (POST sent) | `connecting`. Spinner shown. |
| `start` | No-op (no transition); used to capture/confirm `conversation_id`. |
| `rag` | `rag_received`. Sources panel fills from `retrieval`. **`MessageList` shows the "Finalizing response…" indicator only while `status === "rag_received"`.** |
| `delta` (first) | `streaming`. Buffer starts accumulating; `StreamingText` renders markdown live with a blinking caret. |
| `delta` (subsequent) | Append to `streamBuffer`; re-render. |
| `final` | `complete`. **Streamed buffer is replaced wholesale with `final.answer`** (the streamed deltas are display-only; `final.answer` is canonical). Triggers Supabase persistence. |
| `done` | `idle`. Cleanup. |
| `error` | `error`. Banner shown; buffer cleared; `conversation_summary` not overwritten. |

### Supabase persistence triggered by the stream

Logged-in users only. All writes happen client-side using the user's session.

- **User message** — written to `chat_messages` immediately on send (fire-and-forget); a new `conversations` row is created if needed.
- **Assistant message** — written to `chat_messages` only after `final` arrives, with `content = final.answer` (markdown).
- **RAG context** — `upsertRag(conversation_id, rag_context, retrieval_query, retrieval_metadata)` into `conversation_rag` (one-to-one with conversation, latest only). Run after `final`.
- **Conversation summary** — `final.conversation_summary` is written to `conversations.conversation_summary`. **The next request must send this field back as `conversation_summary` for memory continuity.** If the API ever sends `null` here, follow-up retrieval quality degrades.
- **Auto title** — on the first turn of a new conversation, `final.query_summary` is used to rename the conversation row. Optional; nothing else relies on it.

### Wire-format invariants — what breaks the FE if we change them

The FE parser hard-codes both event names and field names. Treat these as the public contract:

1. **Event names**: `start`, `rag`, `delta`, `final`, `done`, `error`. Renaming any of these silently drops the event (parser switch returns `null` → `onEvent` not called).
2. **Field names** (snake_case → camelCase mapping in the parser):
   - `start.conversation_id`
   - `rag.retrieval_query`, `rag.rag_context`, `rag.retrieval`
   - `delta.text`
   - `final.mode`, `final.answer`, `final.external_info`, `final.conversation_summary`, `final.query_summary`
   - `done.ok`
   - `error.mode`, `error.answer`
3. **Ordering**: `start → rag? → delta* → final → done` (or `error → done`). `rag` must arrive before the first `delta` for the "Finalizing response…" placeholder to behave correctly.
4. **Per-chunk delta cadence**: deltas must arrive as the model produces them. Buffering all deltas and flushing at the end keeps the placeholder visible too long and renders the answer all-at-once. (See MODE B above.)
5. **`final.answer` must be present and non-empty on success** — it overwrites the streamed buffer.
6. **Markdown only — no raw HTML.** The FE's `renderMarkdown` strips all raw HTML tags as an XSS guard. Don't rely on emitting `<div>`, `<table>`, `<script>`, etc.

### Safe (additive) changes

- **New SSE event types**: the parser silently ignores unknown events. Adding `tool_call`, `tool_result`, `progress`, etc. is safe — but the FE won't render anything until you wire it.
- **New fields on existing events**: parser uses optional fall-throughs; unknown fields are dropped.
- **New values for `final.mode`**: unknown modes fall through to "answer" treatment.

### When an API change requires a coordinated FE change

If you change any of the wire-format invariants above, you **must** also patch:
- `Branham-Web-App/src/lib/sse/parser.ts` (event/field mapping)
- `Branham-Web-App/src/components/chat/ChatShell.tsx` (state machine + persistence)
- `Branham-Web-App/src/lib/db/queries.ts` (if you change what's persisted)
- `Branham-Web-App/api_contract.md` (the source of truth)
- `Branham-Web-App/CLAUDE.md` (architecture doc)

Plan such changes by first reading `Branham-Web-App/CLAUDE.md` and the affected files, then making both repos' diffs together.

## Deployment (Google Cloud Run via Cloud Build)

Driver: `scripts/deploy_cloudrun.py`. Two build modes; **prefer `cloudbuild`** (no local Docker required).

```bash
# Login once if needed (interactive — run yourself in the terminal):
#   gcloud auth login admin@branhamsermons.ai
#   gcloud config set project elevated-codex-487017-a6

# Standard remote build + deploy:
uv run python scripts/deploy_cloudrun.py --build-mode cloudbuild
```

What the script does:
1. Verifies `gcloud` auth + project (`elevated-codex-487017-a6`).
2. Enables `run.googleapis.com`, `artifactregistry.googleapis.com`, `cloudbuild.googleapis.com`.
3. Ensures Artifact Registry repo `branham-llm-api` exists in `us-central1`.
4. Reads `.env`, injects values via `--set-env-vars` to Cloud Run.
5. Builds & pushes the image to `us-central1-docker.pkg.dev/elevated-codex-487017-a6/branham-llm-api/branham-llm-api`. With `--build-mode cloudbuild`, the build runs remotely via `gcloud builds submit`.
6. Deploys to Cloud Run service `branham-llm-api` with: region `us-central1`, memory `4Gi`, cpu `2`, concurrency `10`, min-instances `1`, max-instances `5`, request timeout `300s`, startup CPU boost on, port `8080`.

The Dockerfile is multi-stage:
- **builder**: installs deps with `uv`, prefetches HF models into `/app/.hf-cache` via `scripts/prefetch_hf_model.py --all --warm`.
- **runtime**: copies the venv, HF cache, source, indices, `chunks.sqlite`, `biography.txt`. Runs with `HF_OFFLINE_ONLY=1` so no model fetches happen at request time.

Smoke test post-deploy:
```bash
SERVICE_URL=$(gcloud run services describe branham-llm-api --region us-central1 --format='value(status.url)')
curl -fsS "$SERVICE_URL/api/health" | jq
curl -N -X POST "$SERVICE_URL/api/chat" \
  -H "Authorization: Bearer $CHAT_API_BEARER_KEY" \
  -H "Content-Type: application/json" \
  -d '{"conversation_id":"smoke-1","query":"What did Brother Branham teach about faith?"}'
```

The web app is deployed independently — see `Branham-Web-App/CLAUDE.md` for the wrangler/OpenNext flow.

## Reference

- Full V1 design: `.cursor/rules/design_spec.md`
- SSE contract (FE-facing): `/Users/emeraldhenry/Branham-Web-App/api_contract.md`
- Web app architecture: `/Users/emeraldhenry/Branham-Web-App/CLAUDE.md`
