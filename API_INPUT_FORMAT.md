# Frontend Integration Contract: `POST /api/chat`

This document is the frontend instruction guide for request payloads and SSE response handling.

## Endpoint

- Method: `POST`
- Path: `/api/chat`
- Response type: `text/event-stream` (SSE)

## Required Auth Header

- Every request must include:
  - `Authorization: Bearer <CHAT_API_BEARER_KEY>`
- Backend validates bearer token before processing body.
- `conversation_id` remains in request JSON body (not in headers).
- Current integration key (as configured):
  - `CHAT_API_BEARER_KEY=b6766b2e-9a26-4342-9bef-5da4ad67e51c`
- Recommended:
  - keep this key in FE environment/config and inject into the header at request time
  - do not hardcode in UI source files

## Request Body

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000",
  "query": "What did Brother Branham teach about faith?",
  "user_language": "en",
  "conversation_summary": "User asked about faith from Branham sermons and references in Hebrews 11 were discussed.",
  "history_window": [
    { "role": "user", "content": "What is faith?" },
    { "role": "assistant", "content": "Faith is the substance..." },
    { "role": "user", "content": "Can you explain from Branham sermons?" }
  ]
}
```

## Request Field Rules

- `conversation_id` (required, string)
  - Stable ID for one conversation.
  - Keep the same value across turns in the same chat.
  - Backward-compatible alias accepted by backend: `session_id`.

- `query` (required, string)
  - Current user turn.
  - Must be non-empty.

- `user_language` (optional, string)
  - ISO/BCP-47 language hint (`en`, `es`, `fr`, etc.).

- `conversation_summary` (optional, string)
  - Compact memory handoff from previous turn.
  - Used to improve retrieval quality.

- `history_window` (optional, array)
  - Each item:
    - `role`: `"user"` or `"assistant"`
    - `content`: string
  - Order: oldest to newest.
  - Latest turn must be last.

## Backend Input Composition

- Retrieval query: `query` + `conversation_summary` (if provided), otherwise `query` alone.
- LLM context includes:
  - system prompt
  - current query
  - optional recent `history_window`
  - RAG context
  - optional tool outputs (if tools are called)

## SSE Response Events

Frontend must parse events in this order pattern:

1. `start`
2. one or more `delta`
3. `final`
4. `done`

Error path may include:
- `error`
- `done`

### `start` event

```json
{
  "conversation_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

### `delta` event

```json
{
  "text": "partial streamed text chunk"
}
```

- Append each `delta.text` to live UI output.
- `delta` is used for both normal answers and refusal text.

### `rag` event (NEW: retrieval-first evidence)

This event is emitted immediately after retrieval completes and **before** any model tokens stream.

```json
{
  "retrieval_query": "What did Brother Branham teach about faith?\n\nConversation summary:\n...",
  "rag_context": "## Context Group 1\n### Sermon: ...\n...",
  "retrieval": {
    "should_refuse": false,
    "refuse_reason": null,
    "bm25_hit_count": 25,
    "dense_hit_count": 25,
    "fused_hit_count": 40,
    "sermon_count": 8,
    "total_chunks": 64,
    "reranker_triggered": false,
    "signals": {
      "dense_score_std": 0.021,
      "dense_top_score": 0.77,
      "bm25_dense_overlap": 2,
      "quote_intent": false
    }
  }
}
```

Frontend handling:
- Render `rag_context` immediately as the “Evidence / Retrieved context” panel.
- Cache `rag_context` locally per-turn (and optionally persist it in DB) so the UI can show what evidence the answer was based on.
- This event is **not** emitted for:
  - the **English-only language gate** path (non-English queries), and
  - **early retrieval refusals** (off-topic / below thresholds), where the server streams only the refusal.

### `final` event

```json
{
  "mode": "answer",
  "answer": "full final text",
  "external_info": {
    "disclaimer": "Unverified external search results.",
    "sources": ["https://..."]
  },
  "conversation_summary": "Compact summary for next-turn memory."
}
```

Fields:
- `mode`: `"answer" | "refusal" | "error"`
- `answer`: final authoritative response text
- `external_info`:
  - `null` unless external web tool was used
  - when present, show disclaimer and sources clearly
- `conversation_summary`:
  - non-stream metadata
  - use as FE memory handoff for next request
  - expected behavior:
    - present for normal answer flow
    - present for LLM-side refusal flow when summary generation succeeds
    - `null` for early retrieval refusal (fail-fast before generation)
    - may be `null` on internal failures

### `error` event

```json
{
  "mode": "error",
  "answer": "Request could not be processed."
}
```

### `done` event

```json
{
  "ok": true
}
```

## Frontend Handling Rules by `mode`

- `answer`
  - Render final answer.
  - Save `conversation_summary` for next turn.
  - If `external_info` exists, show unverified notice and sources.

- `refusal`
  - Render refusal text normally.
  - Still persist returned `conversation_summary` when present.

- `error`
  - Show generic error UI state.
  - Do not overwrite existing conversation summary with null.

## UI / UX Recommendations (Multi-turn)

### Large screens (desktop/tablet)
- Use a split layout:
  - **Evidence panel** (top or left): show latest `rag_context` plus an expandable history of previous turns’ evidence.
  - **Chat panel** (bottom or right): stream `delta` into the assistant message for the current turn.
- Keep the chat input anchored; do not “switch modes” per turn.

### Small screens (phones)
- Keep **chat as the primary full-height view** to avoid layout bouncing per turn.
- Provide an **Evidence drawer** (collapsible bottom-sheet or side drawer):
  - Default collapsed.
  - Opens to show the latest `rag_context`.
  - Include a “View evidence” button per assistant turn to open the drawer to that turn’s evidence.
- This preserves multi-turn flow without toggling between two modes on every turn.

## FE Validation Checklist

- Always send `conversation_id` and `query`.
- Always send `Authorization: Bearer <CHAT_API_BEARER_KEY>`.
- Send `history_window` oldest -> newest.
- Keep roles restricted to `user` / `assistant`.
- Treat `final.answer` as source of truth (not concatenated deltas alone).
- Store `final.conversation_summary` for the next request when provided.
- Handle `external_info` as unverified external data.

