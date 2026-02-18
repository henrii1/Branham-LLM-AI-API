"""
Chat endpoint for query processing.

Streaming-first architecture:
- Every LLM call is streamed.
- Text tokens are forwarded as SSE `delta` events on the first pass.
- Tool-call responses are buffered, executed, then the loop repeats.
- The final answer is never regenerated — eliminating the old double-call.
"""
from __future__ import annotations

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
    LiteLLMRateLimitError,
    LiteLLMServiceUnavailableError,
    LLMKeyManager,
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


class ChatRuntime:
    """Holds warm singletons for the retrieval pipeline, LLM client, and tools."""

    def __init__(self) -> None:
        cfg = get_config()

        embedder = DenseEmbedder(
            EmbedderConfig(
                model_id=cfg.models.embedding_model_id,
                query_instruction_template="Instruct: {task}\nQuery:{query}",
                query_task_description=(
                    "Given a question about William Branham's teachings or sermons, "
                    "retrieve relevant sermon passages that answer the query"
                ),
            )
        )

        retrieval_config = RetrievalConfig.from_yaml()
        reranker = None
        if retrieval_config.reranker_mode != "never":
            reranker = Reranker(
                RerankerConfig(model_id=cfg.models.reranker_model_id)
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
        key_mgr = LLMKeyManager(key_prefix=llm_cfg.key_prefix)
        self.llm_client = LiteLLMClient(
            config=LiteLLMClientConfig(
                model=llm_cfg.effective_model,
                base_url=llm_cfg.base_url,
                timeout=llm_cfg.timeout,
                temperature=llm_cfg.temperature,
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
            chunk_store=self.pipeline.chunk_store
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
            refusal_message=_get_fixed_refusal_message()
        )
        messages = build_chat_messages(
            system_prompt=system_prompt,
            query=request.query,
            rag_context=rag_context,
            history_window=request.history_window,
        )
        loop_result = self.tool_runner.run(messages)

        external_info = None
        for output in loop_result.tool_outputs:
            if output.get("name") == "internet_search":
                payload = output.get("output", {})
                external_info = {
                    "disclaimer": payload.get(
                        "disclaimer", "Unverified external search results."
                    ),
                    "sources": [s.get("url") for s in payload.get("sources", []) if s.get("url")],
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
    ) -> str | None:
        if not query.strip() or not answer.strip():
            return prior_summary or None
        summary_instruction = (
            "Create a compact conversation summary for frontend memory handoff.\n"
            "Rules:\n"
            "- Output plain text only (no markdown).\n"
            "- Max 2 short sentences.\n"
            "- Keep the same language as the user query.\n"
            "- Include: user intent, key grounded outcome, and next context needed.\n"
            f"- Final mode was: {mode}.\n"
            "- If refusal, summarize why briefly.\n"
        )
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
                max_tokens=120,
            )
            content = (resp.choices[0].message.content or "").strip()
            return content or (prior_summary or None)
        except Exception as exc:
            logger.warning("Conversation summary generation failed: %s", exc)
            return prior_summary or None


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
    return None


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
) -> str | None:
    if not hasattr(runtime, "summarize_conversation"):
        return fallback
    try:
        return runtime.summarize_conversation(
            query=request.query,
            answer=answer,
            prior_summary=request.conversation_summary,
            mode=mode,
        )
    except Exception as exc:
        logger.warning("conversation summary call failed: %s", exc)
        return fallback


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
        t_request = time.perf_counter()
        yield _sse_event(
            "start",
            {"conversation_id": request.conversation_id},
        )
        runtime = get_chat_runtime()
        retrieval_query = build_retrieval_query(
            request.query,
            request.conversation_summary,
        )
        try:
            # ---- Retrieval ----
            t_retrieval_start = time.perf_counter()
            retrieval_result = runtime.retrieve(
                retrieval_query,
                user_language=request.user_language,
            )
            t_retrieval_ms = (time.perf_counter() - t_retrieval_start) * 1000
            logger.info("Retrieval completed in %.1fms", t_retrieval_ms)

            if retrieval_result.should_refuse and not is_bible_query(request.query):
                refusal = _get_fixed_refusal_message()
                yield _sse_event("delta", {"text": refusal})
                yield _sse_event(
                    "final",
                    {
                        "mode": "refusal",
                        "answer": refusal,
                        "external_info": None,
                        "conversation_summary": None,
                    },
                )
                yield _sse_event("done", {"ok": True})
                logger.info(
                    "TTFT(refusal)=%.0fms total=%.0fms",
                    (time.perf_counter() - t_request) * 1000,
                    (time.perf_counter() - t_request) * 1000,
                )
                return

            # ---- Build messages ----
            rag_context = build_rag_context(retrieval_result.expanded_sermons)
            system_prompt = build_system_prompt(
                refusal_message=_get_fixed_refusal_message()
            )
            working_messages = build_chat_messages(
                system_prompt=system_prompt,
                query=request.query,
                rag_context=rag_context,
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
                    buffered = _consume_stream_buffered(
                        runtime.llm_client.stream_completion(
                            messages=working_messages,
                            tools=offer_tools,
                            tool_choice="auto",
                        )
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

                        for c in buffered.tool_calls:
                            tool_name = c["function"]["name"]
                            raw_args = c["function"].get("arguments", "{}")
                            try:
                                parsed_args = json.loads(raw_args) if raw_args else {}
                            except json.JSONDecodeError:
                                parsed_args = {}
                            try:
                                output = runtime.tool_registry.execute_tool(
                                    tool_name, parsed_args
                                )
                            except ToolLimitError as exc:
                                output = runtime.tool_registry.limit_reached_output(
                                    tool_name=tool_name,
                                    error=str(exc),
                                )
                                output["context_hints"] = {
                                    "has_original_query": True,
                                    "has_history_window": bool(request.history_window),
                                    "has_rag_context": bool(rag_context.strip()),
                                    "prior_tool_outputs_count": len(tool_outputs_all),
                                }
                            if output.get("external"):
                                external_used = True
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
                if external_used:
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
                summary = _safe_conversation_summary(
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
                        "conversation_summary": summary,
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
                },
            )
            yield _sse_event("done", {"ok": False})
            return

        except (LiteLLMRateLimitError, LiteLLMServiceUnavailableError):
            refusal = SERVICE_UNAVAILABLE_MESSAGE
            summary = _safe_conversation_summary(
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
                    "conversation_summary": summary,
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

