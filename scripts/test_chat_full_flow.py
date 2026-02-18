#!/usr/bin/env python3
"""
Manual full-flow chat debugger (non-stream).

Purpose:
- Run the same retrieval + generation + tool-loop logic used by the API
- Emit deep debug traces and artifacts for manual inspection
- Overwrite artifact files on every run
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from branham_model_api.api.routes.chat import (  # noqa: E402
    SERVICE_UNAVAILABLE_MESSAGE,
    _get_fixed_refusal_message,
    get_chat_runtime,
)
from branham_model_api.core.pipeline import finalize_answer, is_bible_query  # noqa: E402
from branham_model_api.core.prompts import (  # noqa: E402
    build_chat_messages,
    build_rag_context,
    build_retrieval_query,
    build_system_prompt,
)
from branham_model_api.generation import (  # noqa: E402
    LiteLLMRateLimitError,
    LiteLLMServiceUnavailableError,
)

# ---- Editable defaults for quick local manual checks ----
DEFAULT_QUERY = "what does william branham preach about faith, compared with joseph prince recent sermons and branham's recent followers belief, check the internet.?"
DEFAULT_HISTORY_WINDOW = [
    {"role": "user", "content": "I want to ask a question?"},
    {"role": "assistant", "content": "I can help with that. what are the questions?"},
]
DEFAULT_CONVERSATION_SUMMARY = "User is asking for Branham teaching on faith and supporting quotes."
DEFAULT_CONVERSATION_ID = "manual-debug-conversation"

LOG_DIR = Path("data/logs/chat_flow")
LATEST_JSON = LOG_DIR / "latest_run.json"
LATEST_MARKDOWN = LOG_DIR / "latest_answer.md"
LATEST_LLM_MESSAGES = LOG_DIR / "latest_llm_messages.json"
LATEST_LLM_TRACES = LOG_DIR / "latest_llm_traces.json"
LATEST_RAG_CONTEXT = LOG_DIR / "latest_rag_context.txt"


def _setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("chat_full_flow")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logger.addHandler(stream_handler)
    return logger


def _dump_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _dump_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def _extract_external_info(tool_outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    for output in tool_outputs:
        if output.get("name") == "internet_search":
            payload = output.get("output", {})
            return {
                "disclaimer": payload.get("disclaimer", "Unverified external search results."),
                "sources": [
                    s.get("url")
                    for s in payload.get("sources", [])
                    if isinstance(s, dict) and s.get("url")
                ],
            }
    return None


def main() -> None:
    logger = _setup_logging()

    from branham_model_api.config import get_config  # noqa: E402
    cfg = get_config()
    llm_cfg = cfg.models.llm
    logger.info("LLM provider=%s model=%s base_url=%s", llm_cfg.provider, llm_cfg.effective_model, llm_cfg.base_url or "(litellm default)")

    runtime = get_chat_runtime()

    request = SimpleNamespace(
        conversation_id=DEFAULT_CONVERSATION_ID,
        query=DEFAULT_QUERY,
        history_window=DEFAULT_HISTORY_WINDOW,
        conversation_summary=DEFAULT_CONVERSATION_SUMMARY,
        user_language=None,
    )

    logger.debug("Starting full-flow run for conversation_id=%s", request.conversation_id)
    logger.debug("Query: %s", request.query)
    logger.debug("History turns: %d", len(request.history_window or []))
    logger.debug("Summary present: %s", bool(request.conversation_summary))

    retrieval_query = build_retrieval_query(request.query, request.conversation_summary)
    logger.debug("Retrieval query built (len=%d).", len(retrieval_query))

    run_report: dict[str, Any] = {
        "conversation_id": request.conversation_id,
        "query": request.query,
        "retrieval_query": retrieval_query,
        "retrieval_query_includes_summary": (
            bool(request.conversation_summary)
            and "Conversation summary:" in retrieval_query
        ),
        "history_window": request.history_window,
        "conversation_summary_in": request.conversation_summary,
    }

    try:
        retrieval_result = runtime.retrieve(
            retrieval_query,
            user_language=request.user_language,
        )
        run_report["retrieval"] = {
            "should_refuse": retrieval_result.should_refuse,
            "refuse_reason": retrieval_result.refuse_reason,
            "bm25_hit_count": retrieval_result.bm25_hit_count,
            "dense_hit_count": retrieval_result.dense_hit_count,
            "fused_hit_count": retrieval_result.fused_hit_count,
            "sermon_count": len(retrieval_result.expanded_sermons),
            "total_chunks": retrieval_result.total_chunks,
            "signals": asdict(retrieval_result.signals),
            "refusal_thresholds": {
                "min_dense_score": runtime.pipeline.config.min_dense_score,
                "min_bm25_score": runtime.pipeline.config.min_bm25_score,
            },
        }

        if retrieval_result.should_refuse and not is_bible_query(request.query):
            refusal = _get_fixed_refusal_message()
            run_report["final"] = {
                "mode": "refusal",
                "answer": refusal,
                "external_info": None,
                "conversation_summary_out": None,
                "issues": ["retrieval_refusal"],
            }
            _dump_markdown_and_logs(run_report, logger)
            return

        rag_context = build_rag_context(retrieval_result.expanded_sermons)
        _dump_text(LATEST_RAG_CONTEXT, rag_context)
        logger.debug("RAG context written: %s", LATEST_RAG_CONTEXT.resolve())

        system_prompt = build_system_prompt(refusal_message=_get_fixed_refusal_message())
        llm_messages = build_chat_messages(
            system_prompt=system_prompt,
            query=request.query,
            rag_context=rag_context,
            history_window=request.history_window,
        )
        _dump_json(LATEST_LLM_MESSAGES, llm_messages)
        logger.debug("LLM input messages written: %s", LATEST_LLM_MESSAGES.resolve())

        loop_result = runtime.tool_runner.run(llm_messages)
        _dump_json(LATEST_LLM_TRACES, loop_result.llm_traces)
        logger.debug("LLM request/response traces written: %s", LATEST_LLM_TRACES.resolve())

        tool_limit_events = [
            t for t in loop_result.tool_outputs
            if isinstance(t.get("output"), dict) and t["output"].get("tool_limit_reached")
        ]

        external_info = _extract_external_info(loop_result.tool_outputs)
        checked = finalize_answer(
            query=request.query,
            answer=loop_result.answer,
            external_info=external_info,
            refusal_message=_get_fixed_refusal_message(),
        )
        run_report["tool_outputs"] = loop_result.tool_outputs
        run_report["tool_call_counts"] = runtime.tool_registry.call_counts()
        run_report["tool_total_exhausted"] = runtime.tool_registry.total_exhausted
        run_report["tool_limit_event_count"] = len(tool_limit_events)
        run_report["tool_limit_events"] = tool_limit_events
        run_report["postcheck"] = {"mode": checked.mode, "issues": checked.issues}
        run_report["llm_provider"] = llm_cfg.provider
        run_report["llm_model"] = llm_cfg.effective_model
        run_report["final"] = {
            "mode": checked.mode,
            "answer": checked.answer,
            "external_info": checked.external_info,
            "conversation_summary_out": (
                runtime.summarize_conversation(
                    query=request.query,
                    answer=checked.answer,
                    prior_summary=request.conversation_summary,
                    mode=checked.mode,
                )
                if hasattr(runtime, "summarize_conversation")
                else None
            ),
        }
        _dump_markdown_and_logs(run_report, logger)
    except (LiteLLMRateLimitError, LiteLLMServiceUnavailableError) as exc:
        run_report["final"] = {
            "mode": "refusal",
            "answer": SERVICE_UNAVAILABLE_MESSAGE,
            "error": str(exc),
        }
        _dump_markdown_and_logs(run_report, logger)
    except Exception as exc:  # pragma: no cover
        run_report["final"] = {"mode": "error", "answer": "Request could not be processed.", "error": str(exc)}
        _dump_markdown_and_logs(run_report, logger)
        raise


def _dump_markdown_and_logs(run_report: dict[str, Any], logger: logging.Logger) -> None:
    _dump_json(LATEST_JSON, run_report)
    final_answer = run_report.get("final", {}).get("answer", "")
    _dump_text(LATEST_MARKDOWN, final_answer)
    logger.info("Run report written: %s", LATEST_JSON.resolve())
    logger.info("Final answer written: %s", LATEST_MARKDOWN.resolve())
    logger.info("Large payload paths:")
    logger.info("- RAG context: %s", LATEST_RAG_CONTEXT.resolve())
    logger.info("- LLM messages: %s", LATEST_LLM_MESSAGES.resolve())
    logger.info("- LLM traces: %s", LATEST_LLM_TRACES.resolve())


if __name__ == "__main__":
    main()
