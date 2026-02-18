# Latency Improvement Plan

Purpose: reduce end-to-end chat latency and make streamed UX meaningful before Cloud Run rollout.

---

## Baseline Measurements (Before Optimizations)

Date: 2026-02-11

### 1) End-to-end `/api/chat` (normal query)

Architecture: **double LLM call** — non-streaming tool loop + re-stream final answer.

Observed (3 runs, warm):
- Run 1: TTFT `42449ms`, total `42450ms`, mode `answer`
- Run 2: TTFT `36497ms`, total `36499ms`, mode `refusal`
- Run 3: TTFT `33702ms`, total `33704ms`, mode `refusal`

Problem: TTFT ≈ total. Streaming transport active but first delta arrives only after the entire non-streaming tool loop completes.

### 2) Retrieval-only path latency

- Cold run: `1323ms` (includes `langid` + embedding model initialization)
- Warm runs: `41ms`

### 3) Early retrieval-refusal SSE path

- Cold: ~`3769ms`
- Warm: ~`42ms`

---

## Optimizations Implemented (2026-02-18)

### A) Streaming-first architecture (highest impact)

**Problem:** The old architecture ran a non-streaming tool loop (all iterations + final answer), then re-streamed the final answer — making the LLM produce the answer **twice**.

**Fix:** Rewrote `chat.py` with a streaming-first flow:
- Every LLM call is now streaming.
- For text-answer responses, tokens are forwarded as SSE `delta` events immediately (TTFT ≈ provider TTFT).
- For tool-call responses, the stream is buffered, tools are executed, and the next call streams again.
- The final answer is **never regenerated** — eliminating the double LLM call.

### B) Configurable langid (cold-start elimination)

- Added `retrieval.language_detection.mode` to `config/default.yaml` (default: `never`).
- When `never`: langid is not imported, not downloaded, not warmed at container startup.
- Saves ~800ms cold-start latency.
- Frontend-provided `user_language` is used instead; if absent and mode=never, BM25 is skipped (dense-only retrieval).

### C) Parallel BM25 + Dense retrieval

- BM25 and dense retrieval now run in parallel via `ThreadPoolExecutor(max_workers=2)`.
- Previous serial execution: BM25 (~8ms) + Dense (~30ms) = ~38ms.
- Parallel execution: ~30ms (bounded by the slower Dense path).
- Saves ~8ms per query (modest but consistent).

### D) Latency instrumentation

- Per-request TTFT, total, iteration count, and mode logged to server output.
- `scripts/profile_ttft.py` created for SSE-level TTFT profiling.

---

## Post-Optimization Measurements

Date: 2026-02-18 (warm server, 5 queries)

| Query | TTFT (ms) | Total (ms) | Mode | Deltas |
|-------|-----------|------------|------|--------|
| "What did Brother Branham teach about faith?" | 17,904 | 17,905 | refusal | 218 |
| "What is the third pull?" | 95 | 95 | refusal | 1 |
| "Tell me about the seven church ages" | 48 | 48 | refusal | 1 |
| "What does the Bible say about John 3:16?" | 14,825 | 14,827 | answer | 167 |
| "How does Branham interpret the book of Revelation?" | 10,317 | 10,318 | answer | 25 |

**Summary:**
- **TTFT min:** `48ms`  |  **max:** `17,904ms`  |  **avg:** `8,638ms`
- **Early retrieval refusals:** `48–95ms` (sub-100ms) ✓
- **LLM-powered answers:** `10–18s` (down from 33–42s) — **~60% improvement**

---

## Remaining Bottleneck Analysis

| Factor | Impact | Notes |
|--------|--------|-------|
| Tool-call iterations | ~5–10s per iteration | Each iteration requires a full streaming LLM round-trip. Queries that trigger 1-2 tool calls add 5–15s to TTFT. |
| Provider first-token latency | ~2–5s | DeepSeek on OpenRouter. This is the floor for any LLM-powered TTFT. |
| Retrieval (warm) | ~40–95ms | Negligible after optimization. |
| Python orchestration | <5ms | Prompt build, message assembly, postcheck — all negligible. |

## Further Improvement Opportunities

1. **System prompt tuning** to reduce unnecessary tool calls → fewer buffered iterations → lower TTFT.
2. **Provider/model selection** — faster models or providers with lower TTFT.
3. **Speculative streaming** — start yielding tokens even during first-chunk ambiguity (already partially implemented via first-chunk detection).
4. **Cloud Run min-instances** — keep at least 1 warm instance to avoid container cold starts.
5. **Context window tuning** — smaller RAG context = faster provider processing.
