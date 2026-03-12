"""
Chat endpoint for query processing.

Streaming-first architecture:
- Every LLM call is streamed.
- Text tokens are forwarded as SSE `delta` events on the first pass.
- Tool-call responses are buffered, executed, then the loop repeats.
- The final answer is never regenerated — eliminating the old double-call.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import os
import time
from dataclasses import dataclass
from typing import Any

from fastapi import APIRouter, Header, HTTPException, status
from sse_starlette.sse import EventSourceResponse

from branham_model_api.api.schemas.request import ChatRequest
from branham_model_api.config import get_config, load_yaml_config
from branham_model_api.core.pipeline import (
    RAGPipeline,
    RetrievalConfig,
    create_rag_pipeline,
    finalize_answer,
    is_comparison_query,
    is_bible_query,
)
from branham_model_api.core.prompts import (
    build_chat_messages,
    build_rag_context,
    build_retrieval_query,
    build_system_prompt,
)
from branham_model_api.core.tools import ToolLoopRunner, create_default_tool_registry
from branham_model_api.core.tools.registry import ToolLimitError
from branham_model_api.generation import (
    LiteLLMClient,
    LiteLLMClientConfig,
    LiteLLMMixedClient,
    LiteLLMRouteConfig,
    LiteLLMRateLimitError,
    LiteLLMServiceUnavailableError,
    LLMKeyManager,
    MixedLLMKeyManager,
)
from branham_model_api.retrieval.dense import DenseEmbedder, EmbedderConfig
from branham_model_api.retrieval.reranker import Reranker, RerankerConfig

router = APIRouter()
logger = logging.getLogger(__name__)

_chat_runtime: "ChatRuntime | None" = None
_cached_chat_bearer_key: str | None = None

MAX_TOOL_ITERATIONS = 5


def _get_fixed_refusal_message() -> str:
    try:
        raw = load_yaml_config()
        return raw.get("refusal", {}).get(
            "fixed_message",
            "I can only answer questions based on William Branham's sermons. I don't have enough relevant information to answer your question.",
        )
    except Exception:
        return "I can only answer questions based on William Branham's sermons. I don't have enough relevant information to answer your question."


_ENGLISH_ONLY_MESSAGE = (
    "Sorry — this API currently supports English queries only. "
    "We’re working to add multilingual support in the future. "
    "Please re-ask your question in English."
)


def _is_english_request(*, query: str, user_language: str | None) -> bool:
    """
    Best-effort language gate.

    The sermon corpus is English-only, so we currently accept English queries only.
    This must be deterministic (do not rely on the model to self-enforce).

    Rules:
    - If the client explicitly supplies a user_language hint and it is not English,
      treat as non-English.
    - Otherwise: if the query contains any non-ASCII alphabetic character, treat as non-English.
      (Catches CJK, accented Latin, Cyrillic, etc.)
    """
    ul = (user_language or "").strip().lower()
    if ul:
        # Accept common English tags: en, en-us, en_US, etc.
        if ul.startswith("en"):
            return True
        return False

    for ch in query:
        if ch.isalpha() and not ch.isascii():
            return False
    return True


def _normalize_english_only_reply(text: str) -> str:
    """
    Normalize the english-only gate response to exactly:
      Answer:
      <one short paragraph>

    This runs ONLY on the english-only gate path (not on normal sermon answers).
    """
    raw = (text or "").strip()
    if not raw:
        return "Answer:\n" + _ENGLISH_ONLY_MESSAGE

    lines = [ln.strip() for ln in raw.splitlines() if ln.strip()]
    if not lines:
        return "Answer:\n" + _ENGLISH_ONLY_MESSAGE

    if lines[0] == "Answer:":
        para = lines[1] if len(lines) > 1 else _ENGLISH_ONLY_MESSAGE
        return "Answer:\n" + para

    if lines[0].startswith("Answer:"):
        after = lines[0][len("Answer:") :].strip()
        if after:
            return "Answer:\n" + after
        para = lines[1] if len(lines) > 1 else _ENGLISH_ONLY_MESSAGE
        return "Answer:\n" + para

    return "Answer:\n" + lines[0]


def _query_mode_prompt_addendum(query: str) -> str | None:
    """Return runtime prompt addendum for Bible/comparison intents."""
    if is_bible_query(query):
        return (
            "BIBLE-QUERY MODE:\n"
            "- This query is in scope even when sermon retrieval is weak.\n"
            "- Do not output the fixed Branham-sermon refusal just because sermon chunks are weak.\n"
            "- Answer from biblical context directly.\n"
            "- If there are multiple mainstream interpretations, summarize them briefly and neutrally.\n"
            "- Do NOT write fallback language like 'the provided/current sermon context does not contain...'.\n"
            "- If there are no sermon citations in your answer, do NOT include `References: [N/A]` and do NOT include `Reader Note:` / The Table link.\n"
            "- Never mention internal mechanics (RAG/tools/retrieval pipeline)."
        )
    if is_comparison_query(query):
        return (
            "COMPARISON MODE (Branham vs other authors/preachers/sermons):\n"
            "- This query is in scope.\n"
            "- Do not output the fixed Branham-sermon refusal just because one side is weak.\n"
            "- Provide a balanced comparison.\n"
            "- Branham-side claims must remain sermon-grounded when possible.\n"
            "- If no sermon citations are used in the final answer, do NOT include `Reader Note:` / The Table link.\n"
            "- If non-Branham side lacks direct evidence in context, state that limitation plainly."
        )
    return None


def _get_expected_chat_bearer_key() -> str:
    global _cached_chat_bearer_key
    if _cached_chat_bearer_key is not None:
        return _cached_chat_bearer_key
    env_key = os.getenv("CHAT_API_BEARER_KEY", "").strip()
    if env_key:
        _cached_chat_bearer_key = env_key
        return _cached_chat_bearer_key
    raw = load_yaml_config()
    key = str(raw.get("api", {}).get("bearer_key", "")).strip()
    _cached_chat_bearer_key = key
    return _cached_chat_bearer_key


def _validate_bearer_auth(authorization: str | None) -> None:
    expected = _get_expected_chat_bearer_key()
    if not expected:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Chat bearer key is not configured.",
        )
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header.",
        )
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or token.strip() != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid bearer token.",
        )


SERVICE_UNAVAILABLE_MESSAGE = "The service is temporarily unavailable right now. Please try again later."


def _should_append_unverified_external_section(query: str) -> bool:
    """
    Only force the Unverified/External section when the internet-derived facts are
    about William/Bro Branham himself. For org/admin questions (VGR, tabernacle),
    allow external facts to remain in Answer without the external section.
    """
    q = f" {(query or '').strip().lower()} "
    if not q.strip():
        return False

    # Org/admin queries: do not force the external section.
    org_signals = (
        "voice of god recordings",
        "vgr",
        "branham tabernacle",
        "tabernacle",
        "branham.org",
        "who runs",
        "who is running",
        "leadership",
        "president",
        "director",
        "board",
    )
    if any(s in q for s in org_signals):
        return False

    # William/Bro Branham person-focused queries: force the external section.
    if "bro branham" in q or "brother branham" in q:
        return True
    if "william" in q and "branham" in q:
        return True
    if "w. m. branham" in q or "w.m. branham" in q:
        return True

    return False


class ChatRuntime:
    """Holds warm singletons for the retrieval pipeline, LLM client, and tools."""

    def __init__(self) -> None:
        cfg = get_config()

        # Enforce offline-only HF loads when requested.
        #
        # This prevents any network calls to Hugging Face during local testing/dev
        # (will fail fast if the model is not present in cache).
        offline = os.getenv("HF_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes"}

        embedder = DenseEmbedder(
            EmbedderConfig(
                model_id=cfg.models.embedding_model_id,
                query_instruction_template="Instruct: {task}\nQuery:{query}",
                query_task_description=(
                    "Given a question about William Branham's teachings or sermons, "
                    "retrieve relevant sermon passages that answer the query"
                ),
                local_files_only=offline,
            )
        )

        retrieval_config = RetrievalConfig.from_yaml()
        reranker = None
        if retrieval_config.reranker_mode != "never":
            reranker = Reranker(
                RerankerConfig(
                    model_id=cfg.models.reranker_model_id,
                    local_files_only=offline,
                )
            )

        self.pipeline: RAGPipeline = create_rag_pipeline(
            bm25_index_path=cfg.bm25_path,
            faiss_index_path=cfg.faiss_path,
            faiss_id_map_path="./data/indices/faiss_id_map.jsonl",
            chunk_store_path=cfg.chunk_store_path,
            embedder=embedder,
            reranker=reranker,
            config=retrieval_config,
        )

        llm_cfg = cfg.models.llm
        provider = (llm_cfg.provider or "").strip().lower()
        if provider == "mixed":
            routes = {}
            prefixes = {}
            for rid, rcfg in (llm_cfg.routes or {}).items():
                model = str(rcfg.get("model") or "").strip()
                base_url = str(rcfg.get("base_url") or "").strip() or None
                key_prefix = str(rcfg.get("key_prefix") or "").strip()
                if model and key_prefix:
                    routes[rid] = LiteLLMRouteConfig(model=model, base_url=base_url)
                    prefixes[rid] = key_prefix
            key_mgr = MixedLLMKeyManager(route_key_prefixes=prefixes)
            self.llm_client = LiteLLMMixedClient(
                routes=routes,
                timeout=llm_cfg.timeout,
                temperature=llm_cfg.temperature,
                key_manager=key_mgr,
            )
            logger.info(
                "LLM provider=mixed routes=%s keys=%d",
                ",".join(sorted(routes.keys())) or "(none)",
                key_mgr.key_count,
            )
        else:
            key_mgr = LLMKeyManager(key_prefix=llm_cfg.key_prefix)
            self.llm_client = LiteLLMClient(
                config=LiteLLMClientConfig(
                    model=llm_cfg.effective_model,
                    base_url=llm_cfg.base_url,
                    timeout=llm_cfg.timeout,
                    temperature=llm_cfg.temperature,
                    reasoning=llm_cfg.reasoning,
                ),
                key_manager=key_mgr,
            )
            logger.info(
                "LLM provider=%s model=%s base_url=%s keys=%d",
                llm_cfg.provider,
                llm_cfg.effective_model,
                llm_cfg.base_url or "(litellm default)",
                key_mgr.key_count,
            )
        self.tool_registry = create_default_tool_registry(
            chunk_store=self.pipeline.chunk_store,
            rag_pipeline=self.pipeline,
        )
        # Keep non-streaming runner for test scripts
        self.tool_runner = ToolLoopRunner(
            llm_client=self.llm_client,
            tool_registry=self.tool_registry,
        )

    def retrieve(self, retrieval_query: str, *, user_language: str | None = None):
        return self.pipeline.retrieve(retrieval_query, user_language=user_language)

    def generate(self, request: ChatRequest, rag_context: str):
        """Non-streaming generate (kept for test_chat_full_flow.py)."""
        system_prompt = build_system_prompt(
            refusal_message=_get_fixed_refusal_message(),
            extra_instructions=_query_mode_prompt_addendum(request.query),
        )
        messages = build_chat_messages(
            system_prompt=system_prompt,
            query=request.query,
            rag_context=rag_context,
            history_window=request.history_window,
        )
        loop_result = self.tool_runner.run(messages)

        external_info = None
        if _should_append_unverified_external_section(request.query):
            for output in loop_result.tool_outputs:
                if output.get("name") == "internet_search":
                    payload = output.get("output", {})
                    external_info = {
                        "disclaimer": payload.get(
                            "disclaimer", "Unverified external search results."
                        ),
                        "sources": [
                            s.get("url")
                            for s in payload.get("sources", [])
                            if s.get("url")
                        ],
                    }
                    break

        return _GenerationOutcome(
            answer=loop_result.answer,
            external_info=external_info,
            tool_outputs=loop_result.tool_outputs,
        )

    def summarize_conversation(
        self,
        *,
        query: str,
        answer: str,
        prior_summary: str | None,
        mode: str,
    ) -> dict[str, str | None]:
        """Return {"conversation_summary": ..., "query_summary": ...}.

        On the first query (no prior_summary), both fields are generated in a
        single LLM call.  On follow-up queries only conversation_summary is
        produced; query_summary is None.
        """
        empty = {"conversation_summary": prior_summary or None, "query_summary": None}
        if not query.strip() or not answer.strip():
            return empty

        is_first_query = not (prior_summary or "").strip()

        if is_first_query:
            summary_instruction = (
                "You will receive a user query and an AI answer. Produce a JSON object with exactly two keys:\n"
                "1. \"conversation_summary\": A compact conversation summary for frontend memory handoff.\n"
                "2. \"query_summary\": A 5–7 word title/summary of the user's first query.\n\n"
                "Rules for conversation_summary:\n"
                "- Plain text only (no markdown).\n"
                "- Rich enough to feed into a sermon retrieval system for the next query.\n"
                "- Same language as the AI response.\n"
                "- Concise but relevant.\n"
                f"- Final mode was: {mode}.\n"
                "- If refusal, summarize why briefly.\n\n"
                "Rules for query_summary:\n"
                "- Exactly 5–7 words.\n"
                "- Captures the essence of what the user asked.\n"
                "- Plain text only, no markdown or special characters.\n\n"
                "Output ONLY a valid JSON object. No markdown, no code fences, no extra text.\n"
            )
            max_tokens = 180
        else:
            summary_instruction = (
                "Create a compact conversation summary for frontend memory handoff.\n"
                "Rules:\n"
                "- Output plain text only (no markdown).\n"
                "- Ensure that the summary is rich as it will be fed into a sermon retrieval system\n"
                "- Keep the same language as the ai respons to user's query\n"
                "- keep it concise but relevant to assist the next query retrieve the best context from Rag, we send new query + summary into RAG\n"
                f"- Final mode was: {mode}.\n"
                "- If refusal, summarize why briefly.\n"
            )
            max_tokens = 120

        user_payload = (
            f"Prior summary:\n{prior_summary or '[none]'}\n\n"
            f"User query:\n{query}\n\n"
            f"Assistant final answer:\n{answer}\n"
        )
        try:
            resp = self.llm_client.completion(
                messages=[
                    {"role": "system", "content": summary_instruction},
                    {"role": "user", "content": user_payload},
                ],
                max_tokens=max_tokens,
            )
            content = (resp.choices[0].message.content or "").strip()

            if is_first_query:
                clean = content
                if clean.startswith("```"):
                    clean = "\n".join(
                        ln for ln in clean.splitlines()
                        if not ln.strip().startswith("```")
                    ).strip()
                data = json.loads(clean)
                return {
                    "conversation_summary": str(data.get("conversation_summary") or "").strip() or (prior_summary or None),
                    "query_summary": str(data.get("query_summary") or "").strip() or None,
                }
            return {
                "conversation_summary": content or (prior_summary or None),
                "query_summary": None,
            }
        except Exception as exc:
            logger.warning("Conversation summary generation failed: %s", exc)
            return empty


@dataclass
class _GenerationOutcome:
    answer: str
    external_info: dict[str, Any] | None = None
    tool_outputs: list[dict[str, Any]] | None = None


def get_chat_runtime() -> ChatRuntime:
    global _chat_runtime
    if _chat_runtime is None:
        _chat_runtime = ChatRuntime()
    return _chat_runtime


def _sse_event(event: str, payload: dict[str, Any]) -> dict[str, str]:
    return {"event": event, "data": json.dumps(payload, ensure_ascii=True)}


# ---------------------------------------------------------------------------
# Streaming tool-call helpers
# ---------------------------------------------------------------------------

def _extract_delta_text(chunk: Any) -> str:
    try:
        choices = getattr(chunk, "choices", None)
        if choices:
            delta = getattr(choices[0], "delta", None)
            if delta is not None:
                content = getattr(delta, "content", None)
                if isinstance(content, str):
                    return content
    except Exception:
        pass
    try:
        choices = chunk.get("choices", [])
        if choices:
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if isinstance(content, str):
                return content
    except Exception:
        pass
    return ""


def _extract_delta_tool_calls(chunk: Any) -> list[dict[str, Any]] | None:
    """Extract incremental tool-call deltas from a streaming chunk."""
    try:
        choices = getattr(chunk, "choices", None) or []
        if not choices:
            return None
        delta = getattr(choices[0], "delta", None)
        if delta is None:
            return None
        tc = getattr(delta, "tool_calls", None)
        if tc:
            result = []
            for item in tc:
                fn = getattr(item, "function", None)
                result.append({
                    "index": getattr(item, "index", 0),
                    "id": getattr(item, "id", None),
                    "function": {
                        "name": getattr(fn, "name", None) if fn else None,
                        "arguments": getattr(fn, "arguments", None) if fn else None,
                    },
                })
            return result
    except Exception:
        pass
    try:
        # Dict-shaped chunks (some providers / LiteLLM modes)
        choices = (chunk or {}).get("choices", []) if isinstance(chunk, dict) else []
        if not choices:
            return None
        delta = choices[0].get("delta", {}) or {}
        tc = delta.get("tool_calls")
        if tc:
            result = []
            for item in tc:
                fn = (item or {}).get("function") or {}
                result.append(
                    {
                        "index": item.get("index", 0),
                        "id": item.get("id"),
                        "function": {
                            "name": fn.get("name"),
                            "arguments": fn.get("arguments"),
                        },
                    }
                )
            return result
    except Exception:
        pass
    return None


def _maybe_batch_db_search_tool_calls(
    *,
    tool_registry: Any,
    tool_calls: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], bool]:
    """
    If multiple db_search tool calls are requested in the same streamed assistant turn,
    execute them as one batch_mixed call (counts once), then return per-tool_call_id
    outputs to feed back to the model.

    Returns (tool_call_id -> output, did_batch).
    """
    db_calls: list[tuple[str, dict[str, Any]]] = []
    for c in tool_calls:
        fn = (c or {}).get("function") or {}
        if fn.get("name") != "db_search":
            continue
        tcid = c.get("id") or ""
        raw_args = fn.get("arguments", "{}")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed_args = {}
        # If the model already requested a batch mode, do not wrap it again.
        # Nested batch_mixed would be skipped by the tool and cause result misalignment.
        if isinstance(parsed_args, dict) and parsed_args.get("mode") in {"batch_mixed", "batch_read"}:
            return ({}, False)
        db_calls.append((tcid, parsed_args))

    if len(db_calls) <= 1:
        return ({}, False)

    operations = [args for _, args in db_calls]
    try:
        batch = tool_registry.execute_tool(
            "db_search",
            {"mode": "batch_mixed", "operations": operations},
        )
    except ToolLimitError:
        return ({}, False)

    results = []
    if isinstance(batch, dict) and batch.get("ok") and batch.get("mode") == "batch_mixed":
        results = batch.get("results") or []

    out: dict[str, dict[str, Any]] = {}
    for i, (tcid, args) in enumerate(db_calls):
        if not tcid:
            continue
        result_i = results[i] if i < len(results) else {"ok": False, "error": "Missing batched result."}
        out[tcid] = {
            "ok": True,
            "batched": True,
            "batch_mode": "batch_mixed",
            "original_tool": "db_search",
            "original_mode": (args.get("mode") if isinstance(args, dict) else None),
            "result": result_i,
        }
    return (out, True)


@dataclass
class _StreamedResponse:
    """Accumulated result from consuming one streaming LLM call."""
    text_parts: list[str]
    tool_calls: list[dict[str, Any]]

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)

    @property
    def full_text(self) -> str:
        return "".join(self.text_parts).strip()


def _accumulate_tool_calls_from_deltas(
    deltas: list[list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """
    Merge incremental tool-call deltas into complete tool-call objects.

    OpenAI streaming sends tool calls as incremental deltas across multiple
    chunks: first chunk has the tool id and function name, subsequent chunks
    append to function.arguments.
    """
    calls: dict[int, dict[str, Any]] = {}
    for delta_list in deltas:
        for d in delta_list:
            idx = d.get("index", 0)
            if idx not in calls:
                calls[idx] = {"id": "", "function": {"name": "", "arguments": ""}}
            if d.get("id"):
                calls[idx]["id"] = d["id"]
            fn = d.get("function", {})
            if fn.get("name"):
                calls[idx]["function"]["name"] = fn["name"]
            if fn.get("arguments"):
                calls[idx]["function"]["arguments"] += fn["arguments"]
    return [calls[k] for k in sorted(calls)]


def _consume_stream_buffered(stream: Any) -> _StreamedResponse:
    """
    Consume the entire stream, accumulating text and tool-call deltas.

    Used for tool-call iterations where we need the full response before
    executing tools.
    """
    text_parts: list[str] = []
    tc_deltas: list[list[dict[str, Any]]] = []
    for chunk in stream:
        text = _extract_delta_text(chunk)
        if text:
            text_parts.append(text)
        tc = _extract_delta_tool_calls(chunk)
        if tc:
            tc_deltas.append(tc)

    tool_calls = _accumulate_tool_calls_from_deltas(tc_deltas) if tc_deltas else []
    return _StreamedResponse(text_parts=text_parts, tool_calls=tool_calls)


def _safe_conversation_summary(
    *,
    runtime: Any,
    request: ChatRequest,
    mode: str,
    answer: str,
    fallback: str | None = None,
) -> dict[str, str | None]:
    """Return {"conversation_summary": ..., "query_summary": ...}."""
    empty = {"conversation_summary": fallback, "query_summary": None}
    if not hasattr(runtime, "summarize_conversation"):
        return empty
    try:
        result = runtime.summarize_conversation(
            query=request.query,
            answer=answer,
            prior_summary=request.conversation_summary,
            mode=mode,
        )
        if isinstance(result, dict):
            return {
                "conversation_summary": result.get("conversation_summary"),
                "query_summary": result.get("query_summary"),
            }
        if isinstance(result, str):
            return {"conversation_summary": result or fallback, "query_summary": None}
        return empty
    except Exception as exc:
        logger.warning("conversation summary call failed: %s", exc)
        return empty


def _generate_allowed_query_fallback(
    *,
    runtime: Any,
    query: str,
    is_bible: bool,
) -> str | None:
    """
    Generate a non-refusal fallback for allowed non-sermon intents.

    This is used only when the primary pass produced a refusal-style output
    for intents that should not be hard-refused (Bible-only and comparison).
    """
    addendum = _query_mode_prompt_addendum(query) or (
        "ALLOWED-QUERY FALLBACK MODE:\n"
        "- Do not output the fixed Branham-sermon refusal.\n"
        "- Answer the user directly and concisely."
    )
    system = build_system_prompt(
        refusal_message=_get_fixed_refusal_message(),
        extra_instructions=(
            addendum
            + "\n"
            + "- Previous draft incorrectly refused this allowed query; provide a compliant answer now."
        ),
    )
    try:
        resp = runtime.llm_client.completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            max_tokens=320,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return None
        if text == _get_fixed_refusal_message():
            return None
        return text
    except Exception as exc:
        logger.warning("Allowed-query fallback generation failed: %s", exc)
        return None


def _maybe_override_refusal_for_allowed_query(
    *,
    runtime: Any,
    query: str,
    checked: Any,
) -> Any:
    """Convert refusal to answer for Bible/comparison intents when fallback succeeds."""
    if getattr(checked, "mode", "") != "refusal":
        return checked
    bible = is_bible_query(query)
    comparison = is_comparison_query(query)
    if not (bible or comparison):
        return checked
    fallback = _generate_allowed_query_fallback(
        runtime=runtime,
        query=query,
        is_bible=bible,
    )
    if not fallback:
        return checked
    checked.mode = "answer"
    checked.answer = fallback
    checked.external_info = None
    return checked


@router.post("/chat")
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> EventSourceResponse:
    """
    Process a user query and stream SSE events.

    Architecture: streaming-first with inline tool-call handling.
    - The first LLM call is streamed.  If the response is a text answer,
      tokens are forwarded to the client immediately (TTFT ≈ provider TTFT).
    - If the response contains tool calls, the stream is buffered, tools are
      executed, and the next LLM call is streamed — repeating until a text
      answer is produced.
    - The final answer is NEVER regenerated (no double LLM call).
    """
    _validate_bearer_auth(authorization)

    async def event_stream():
        _loop = asyncio.get_running_loop()
        t_request = time.perf_counter()
        yield _sse_event(
            "start",
            {"conversation_id": request.conversation_id},
        )
        await asyncio.sleep(0)
        runtime = get_chat_runtime()

        # ---- English-only language gate (deterministic) ----
        if not _is_english_request(query=request.query, user_language=request.user_language):
            ul = (request.user_language or "").strip()
            # NOTE: We do NOT stream the translator output directly because models may add
            # extra formatting/lines. Instead, we request strict JSON and render the
            # final two-line reply deterministically server-side.
            system_prompt = (
                "You are a translator.\n"
                "Task: translate the provided English message into the user's language.\n"
                + (f"Target language (hint): {ul}\n" if ul else "")
                + "Output rules (hard):\n"
                "- Output ONLY a JSON object with exactly one key: paragraph\n"
                "- The value must be ONE short paragraph (2–4 sentences) in the user's language.\n"
                "- Do NOT include markdown, code fences, explanations, or extra keys.\n"
                "- Do NOT include citations or sermon claims.\n"
            )
            messages = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": (
                        f"User query language reference (do not answer it):\n{request.query}\n\n"
                        f"English message to translate:\n{_ENGLISH_ONLY_MESSAGE}"
                    ),
                },
            ]

            t_first_delta: float | None = None
            paragraph = ""
            try:
                resp = runtime.llm_client.completion(messages=messages, max_tokens=200)
                raw = (resp.choices[0].message.content or "").strip()
                data = json.loads(raw) if raw else {}
                paragraph = str((data or {}).get("paragraph") or "").strip()
            except Exception:
                paragraph = ""

            if not paragraph:
                # Fallback: stable English-only message (still two-line format).
                paragraph = _ENGLISH_ONLY_MESSAGE

            final_text = "Answer:\n" + paragraph
            t_first_delta = time.perf_counter()
            yield _sse_event("delta", {"text": final_text})
            yield _sse_event(
                "final",
                {
                    "mode": "answer",
                    "answer": final_text,
                    "external_info": None,
                    "conversation_summary": None,
                    "query_summary": None,
                },
            )
            yield _sse_event("done", {"ok": True})
            logger.info(
                "TTFT(english_only)=%.0fms total=%.0fms",
                ((t_first_delta - t_request) * 1000) if t_first_delta else -1,
                (time.perf_counter() - t_request) * 1000,
            )
            return
        retrieval_query = build_retrieval_query(
            request.query,
            request.conversation_summary,
        )
        try:
            # ---- Retrieval ----
            t_retrieval_start = time.perf_counter()
            retrieval_result = await _loop.run_in_executor(
                None,
                lambda: runtime.retrieve(
                    retrieval_query,
                    user_language=request.user_language,
                ),
            )
            t_retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
            logger.info("Retrieval completed in %.1fms", t_retrieval_ms)

            if retrieval_result.should_refuse and not (
                is_bible_query(request.query) or is_comparison_query(request.query)
            ):
                refusal = _get_fixed_refusal_message()
                yield _sse_event("delta", {"text": refusal})
                yield _sse_event(
                    "final",
                    {
                        "mode": "refusal",
                        "answer": refusal,
                        "external_info": None,
                        "conversation_summary": None,
                        "query_summary": None,
                    },
                )
                yield _sse_event("done", {"ok": True})
                logger.info(
                    "TTFT(refusal)=%.0fms total=%.0fms",
                    (time.perf_counter() - t_request) * 1000,
                    (time.perf_counter() - t_request) * 1000,
                )
                return

            # Emit UI RAG context as soon as it's ready to hide downstream latency.
            # Build the LLM RAG context concurrently (we need it before starting the LLM call).
            all_expanded_sermons = getattr(
                retrieval_result,
                "all_expanded_sermons",
                retrieval_result.expanded_sermons,
            )
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                rag_llm_f = pool.submit(
                    build_rag_context,
                    retrieval_result.expanded_sermons,
                    audience="llm",
                )
                rag_context_ui = build_rag_context(
                    all_expanded_sermons,
                    audience="ui",
                )
            yield _sse_event(
                "rag",
                {
                    "retrieval_query": retrieval_query,
                    "rag_context": rag_context_ui,
                    "retrieval": {
                        "should_refuse": retrieval_result.should_refuse,
                        "refuse_reason": retrieval_result.refuse_reason,
                        "bm25_hit_count": retrieval_result.bm25_hit_count,
                        "dense_hit_count": retrieval_result.dense_hit_count,
                        "fused_hit_count": retrieval_result.fused_hit_count,
                        "sermon_count": len(all_expanded_sermons),
                        "total_chunks": retrieval_result.total_chunks,
                        "reranker_triggered": retrieval_result.reranker_triggered,
                        "signals": {
                            "dense_score_std": retrieval_result.signals.dense_score_std,
                            "dense_top_score": retrieval_result.signals.dense_top_score,
                            "bm25_dense_overlap": retrieval_result.signals.bm25_dense_overlap,
                            "quote_intent": retrieval_result.signals.quote_intent,
                        },
                    },
                },
            )
            await asyncio.sleep(0)
            rag_context_llm = rag_llm_f.result()

            # ---- Build messages ----
            system_prompt = build_system_prompt(
                refusal_message=_get_fixed_refusal_message(),
                extra_instructions=_query_mode_prompt_addendum(request.query),
            )
            working_messages = build_chat_messages(
                system_prompt=system_prompt,
                query=request.query,
                rag_context=rag_context_llm,
                history_window=request.history_window,
            )

            tool_defs = runtime.tool_registry.definitions()
            runtime.tool_registry.reset_counts()
            tool_outputs_all: list[dict[str, Any]] = []
            external_used = False
            t_first_delta: float | None = None
            tools_exhausted = False

            # ---- Streaming tool loop ----
            #
            # Two modes per iteration:
            #   A) Tools offered  → buffer entire response (avoids leaking
            #      model "thinking" text that precedes tool calls, which
            #      DeepSeek does consistently).
            #   B) No tools       → stream tokens directly for best TTFT.
            #
            for iteration in range(MAX_TOOL_ITERATIONS):
                t_iter = time.perf_counter()
                offer_tools = tool_defs if (tool_defs and not tools_exhausted) else None

                # ============================================================
                # MODE A — Tools offered: buffer completely, then decide
                # ============================================================
                if offer_tools:
                    buffered = await _loop.run_in_executor(
                        None,
                        lambda: _consume_stream_buffered(
                            runtime.llm_client.stream_completion(
                                messages=working_messages,
                                tools=offer_tools,
                                tool_choice="auto",
                            )
                        ),
                    )

                    if buffered.has_tool_calls:
                        # --- Execute tool calls ---
                        logger.info(
                            "Tool-call iteration %d: %d call(s) in %.0fms",
                            iteration,
                            len(buffered.tool_calls),
                            (time.perf_counter() - t_iter) * 1000,
                        )
                        assistant_msg: dict[str, Any] = {
                            "role": "assistant",
                            # Keep tool-call assistant turns lean: the model can emit
                            # pre-tool reasoning text, but we do not feed that text back.
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": c["id"],
                                    "type": "function",
                                    "function": c["function"],
                                }
                                for c in buffered.tool_calls
                            ],
                        }
                        working_messages.append(assistant_msg)

                        # Begin a single tool-call "round" (counts once for total budget).
                        try:
                            runtime.tool_registry.begin_tool_round()
                        except ToolLimitError as exc:
                            for c in buffered.tool_calls:
                                tool_name = c["function"]["name"]
                                output = runtime.tool_registry.limit_reached_output(
                                    tool_name=tool_name,
                                    error=str(exc),
                                )
                                output["context_hints"] = {
                                    "has_original_query": True,
                                    "has_history_window": bool(request.history_window),
                                    "has_rag_context": bool(rag_context_llm.strip()),
                                    "prior_tool_outputs_count": len(tool_outputs_all),
                                }
                                tool_outputs_all.append({"name": tool_name, "output": output})
                                working_messages.append(
                                    {
                                        "role": "tool",
                                        "tool_call_id": c["id"],
                                        "name": tool_name,
                                        "content": json.dumps(output, ensure_ascii=True),
                                    }
                                )
                            if runtime.tool_registry.total_exhausted:
                                tools_exhausted = True
                            continue  # next iteration

                        batched_db_outputs, did_batch_db = _maybe_batch_db_search_tool_calls(
                            tool_registry=runtime.tool_registry,
                            tool_calls=buffered.tool_calls,
                        )

                        # Reserve counts synchronously, execute tools in parallel.
                        tasks: list[tuple[dict[str, Any], str, tuple[Any, dict[str, Any], list[dict[str, Any]]] | None, dict[str, Any] | None]] = []
                        # Each entry: (tool_call, tool_name, prepared(spec,args,soft), immediate_output)
                        for c in buffered.tool_calls:
                            tool_name = c["function"]["name"]
                            raw_args = c["function"].get("arguments", "{}")
                            try:
                                parsed_args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                parsed_args = {}

                            tcid = c.get("id") or ""
                            if did_batch_db and tool_name == "db_search":
                                if tcid and tcid in batched_db_outputs:
                                    tasks.append((c, tool_name, None, batched_db_outputs[tcid]))
                                continue

                            try:
                                prepared = runtime.tool_registry.prepare_tool_execution(
                                    tool_name, parsed_args
                                )
                                tasks.append((c, tool_name, prepared, None))
                            except ToolLimitError as exc:
                                immediate = runtime.tool_registry.limit_reached_output(
                                    tool_name=tool_name,
                                    error=str(exc),
                                )
                                immediate["context_hints"] = {
                                    "has_original_query": True,
                                    "has_history_window": bool(request.history_window),
                                    "has_rag_context": bool(rag_context_llm.strip()),
                                    "prior_tool_outputs_count": len(tool_outputs_all),
                                }
                                tasks.append((c, tool_name, None, immediate))

                        futures: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
                        with concurrent.futures.ThreadPoolExecutor(
                            max_workers=min(8, max(1, len(tasks)))
                        ) as ex:
                            for c, tool_name, prepared, immediate in tasks:
                                if immediate is not None or prepared is None:
                                    continue
                                tcid = c.get("id") or ""
                                spec, args2, soft_events = prepared

                                def _run(spec=spec, args2=args2, soft_events=soft_events) -> dict[str, Any]:
                                    try:
                                        out = spec.tool.execute(args2)
                                    except Exception as exc:
                                        out = {"ok": False, "error": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                                    if soft_events and isinstance(out, dict):
                                        out["_soft_limits"] = soft_events
                                    return out

                                if tcid:
                                    futures[tcid] = ex.submit(_run)

                        # Collect in original order
                        for c, tool_name, prepared, immediate in tasks:
                            tcid = c.get("id") or ""
                            if immediate is not None:
                                output = immediate
                            elif tcid and tcid in futures:
                                try:
                                    output = futures[tcid].result()
                                except Exception as exc:
                                    output = {"ok": False, "error": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                            else:
                                continue

                            if output.get("external"):
                                external_used = True
                            tool_outputs_all.append({"name": tool_name, "output": output})
                            working_messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tcid,
                                    "name": tool_name,
                                    "content": json.dumps(output, ensure_ascii=True),
                                }
                            )

                        if runtime.tool_registry.total_exhausted:
                            tools_exhausted = True
                        continue  # next iteration

                    else:
                        # --- No tool calls: buffered text IS the answer ---
                        for part in buffered.text_parts:
                            if part:
                                if t_first_delta is None:
                                    t_first_delta = time.perf_counter()
                                yield _sse_event("delta", {"text": part})

                        streamed_answer = buffered.full_text
                        # fall through to finalization below

                # ============================================================
                # MODE B — No tools offered: stream tokens directly
                # ============================================================
                else:
                    streamed_parts: list[str] = []
                    stream = runtime.llm_client.stream_completion(
                        messages=working_messages,
                    )
                    for chunk in stream:
                        text = _extract_delta_text(chunk)
                        if text:
                            if t_first_delta is None:
                                t_first_delta = time.perf_counter()
                            streamed_parts.append(text)
                            yield _sse_event("delta", {"text": text})

                    streamed_answer = "".join(streamed_parts).strip()
                    # fall through to finalization below

                # ============================================================
                # Finalization (shared by both modes when answer is ready)
                # ============================================================
                external_info = None
                if external_used and _should_append_unverified_external_section(request.query):
                    for to in tool_outputs_all:
                        if to.get("name") == "internet_search":
                            payload = to.get("output", {})
                            external_info = {
                                "disclaimer": payload.get(
                                    "disclaimer",
                                    "Unverified external search results.",
                                ),
                                "sources": [
                                    s.get("url")
                                    for s in payload.get("sources", [])
                                    if s.get("url")
                                ],
                            }
                            break

                checked = finalize_answer(
                    query=request.query,
                    answer=streamed_answer,
                    external_info=external_info,
                    refusal_message=_get_fixed_refusal_message(),
                )
                checked = _maybe_override_refusal_for_allowed_query(
                    runtime=runtime,
                    query=request.query,
                    checked=checked,
                )
                summary_result = _safe_conversation_summary(
                    runtime=runtime,
                    request=request,
                    mode=checked.mode,
                    answer=checked.answer,
                    fallback=None,
                )
                t_total = (time.perf_counter() - t_request) * 1000
                t_ttft = (
                    (t_first_delta - t_request) * 1000
                    if t_first_delta
                    else t_total
                )
                logger.info(
                    "TTFT=%.0fms total=%.0fms mode=%s iterations=%d tool_calls=%s",
                    t_ttft,
                    t_total,
                    checked.mode,
                    iteration + 1,
                    runtime.tool_registry.call_counts(),
                )
                yield _sse_event(
                    "final",
                    {
                        "mode": checked.mode,
                        "answer": checked.answer,
                        "external_info": checked.external_info,
                        "conversation_summary": summary_result["conversation_summary"],
                        "query_summary": summary_result["query_summary"],
                    },
                )
                yield _sse_event("done", {"ok": True})
                return

            # Exhausted max iterations — should not normally happen
            logger.warning(
                "Tool loop exhausted %d iterations; tool_calls=%s",
                MAX_TOOL_ITERATIONS,
                runtime.tool_registry.call_counts(),
            )
            yield _sse_event(
                "final",
                {
                    "mode": "error",
                    "answer": "Request could not be processed.",
                    "external_info": None,
                    "conversation_summary": None,
                    "query_summary": None,
                },
            )
            yield _sse_event("done", {"ok": False})
            return

        except (LiteLLMRateLimitError, LiteLLMServiceUnavailableError):
            refusal = SERVICE_UNAVAILABLE_MESSAGE
            summary_result = _safe_conversation_summary(
                runtime=runtime,
                request=request,
                mode="refusal",
                answer=refusal,
                fallback=None,
            )
            yield _sse_event("delta", {"text": refusal})
            yield _sse_event(
                "final",
                {
                    "mode": "refusal",
                    "answer": refusal,
                    "external_info": None,
                    "conversation_summary": summary_result["conversation_summary"],
                    "query_summary": summary_result["query_summary"],
                },
            )
            yield _sse_event("done", {"ok": True})
            return
        except Exception as exc:
            logger.exception("Unhandled /chat error: %s", exc)
            yield _sse_event(
                "error",
                {
                    "mode": "error",
                    "answer": "Request could not be processed.",
                },
            )
            yield _sse_event("done", {"ok": False})

    return EventSourceResponse(event_stream())

