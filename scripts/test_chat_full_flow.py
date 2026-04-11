#!/usr/bin/env python3
"""
Manual full-flow chat debugger (non-stream).

Purpose:
- Run the same retrieval + generation + tool-loop logic used by the API
- Emit deep debug traces and artifacts for manual inspection
- Overwrite artifact files on every run
"""

from __future__ import annotations

import random
import os
import json
import logging
import argparse
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

# Work around a common macOS OpenMP duplicate load crash (FAISS + torch, etc.).
# Scoped to this script only.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from branham_model_api.api.routes.chat import (  # noqa: E402
    SERVICE_UNAVAILABLE_MESSAGE,
    _ENGLISH_ONLY_MESSAGE,
    _get_fixed_refusal_message,
    _get_insufficient_context_message,
    _is_english_request,
    _maybe_override_refusal_for_allowed_query,
    _normalize_english_only_reply,
    _query_mode_prompt_addendum,
    _should_append_unverified_external_section,
    get_chat_runtime,
)
from branham_model_api.core.pipeline import (  # noqa: E402
    finalize_answer,
    is_bible_query,
    is_comparison_query,
)
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
DEFAULT_QUERY = """can you tell me  """
DEFAULT_HISTORY_WINDOW = []

"""[
    {"role": "user", "content": "I want to ask a question?"},
    {"role": "assistant", "content": "I can help with that. what are the questions?"},
]"""
DEFAULT_CONVERSATION_SUMMARY = "" #"User is asking for Branham teaching on faith and supporting quotes."
DEFAULT_CONVERSATION_ID = "manual-debug-conversation"

LOG_DIR = Path("data/logs/chat_flow")
LATEST_JSON = LOG_DIR / "latest_run.json"
LATEST_MARKDOWN = LOG_DIR / "latest_answer.md"
LATEST_LLM_MESSAGES = LOG_DIR / "latest_llm_messages.json"
LATEST_LLM_TRACES = LOG_DIR / "latest_llm_traces.json"
LATEST_RAG_CONTEXT = LOG_DIR / "latest_rag_context.txt"


def _utc_ts() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _slug(s: str) -> str:
    safe = "".join(ch.lower() if ch.isalnum() else "-" for ch in (s or "").strip())
    safe = "-".join([p for p in safe.split("-") if p])
    return (safe[:60] or "case").strip("-")


def _stress_cases() -> list[dict[str, Any]]:
    """
    Prompt suite for Grok stress testing across query categories.

    Notes:
    - Keep some intentionally vague/incomplete to test guardrails.
    - Include non-English to exercise the English-only gate.
    - Include off-domain to trigger refusal.
    """
    return [
        # Verbatim / quote seeking
        {
            "name": "verbatim_quote_faith",
            "query": "Give me a verbatim quote from Brother Branham about faith, with the exact paragraph citations.",
        },
        # Comparison
        {
            "name": "comparison_serpent_seed",
            "query": "Compare what Brother Branham taught about the serpent seed with mainstream Christian teaching. Use sermon citations.",
        },
        # Biography (should encourage biography tool use)
        {
            "name": "biography_branham",
            "query": "Give me a concise biography of William Marrion Branham (dates, major events) and cite sermons where he mentions key points.",
        },
        # Detailed analysis
        {
            "name": "detailed_analysis_predestination",
            "query": "Do a detailed analysis of Brother Branham's teaching on predestination and election, and support it with multiple sermon citations.",
        },
        # Incomplete / unclear
        {"name": "incomplete_i_want_to", "query": "I want to"},
        {"name": "unclear_help_me", "query": "help me understand this"},
        {"name": "unclear_what_about_it", "query": "What about it?"},
        # Bible-only prompts (allowed)
        {
            "name": "bible_only_john_316",
            "query": "Explain John 3:16 plainly using the Bible only (no external sources).",
        },
        {
            "name": "bible_only_revelation_10_7",
            "query": "Explain Revelation 10:7 and why it matters. Keep it Bible-focused.",
        },
        # Other language prompts (English-only gate)
        {"name": "spanish_question", "query": "¿Qué enseñó el Hermano Branham sobre la fe?", "user_language": "es"},
        {"name": "french_question", "query": "Parle-moi des sept âges de l'Église.", "user_language": "fr"},
        # External “men of God” / off-domain comparisons
        {
            "name": "external_men_of_god",
            "query": "Compare the teachings of Brother Branham and Billy Graham on salvation. Give 5 points each.",
        },
        # Refusal triggers (off-domain)
        {"name": "refusal_capital_france", "query": "What is the capital of France?"},
        {"name": "refusal_superbowl", "query": "Who won the Super Bowl in 2024?"},
        {"name": "refusal_stock_price", "query": "What is the current stock price of Apple?"},
        # Refusal-ish ambiguous theology outside sermons
        {
            "name": "refusal_non_branham_theology",
            "query": "Give me a full systematic theology of Calvinism vs Arminianism with citations from modern theologians.",
        },
    ]


def _tool_smoke_cases() -> list[dict[str, Any]]:
    """
    Small set of prompts intended to *force* each tool to be used at least once.
    """
    return [
        # Force db_search via verbatim lookup discipline + explicit sermon pointer.
        {
            "name": "tool_db_search_verbatim_serpent_seed",
            "query": (
                "Verbatim lookup: In the sermon \"THE SERPENT’S SEED\" (58-0928E), "
                "give the exact quote around paragraph 162 about \"enmity\" and \"the serpent’s seed\". "
                "Include 2 paragraphs before and 2 after, and keep the quote verbatim."
            ),
        },
        # Force biography_search (verified local biography source required by system prompt).
        {
            "name": "tool_biography_search_early_life",
            "query": (
                "Biography: Summarize William Marrion Branham’s early life and calling "
                "(birth, childhood, early ministry calling). Use the verified biography source."
            ),
        },
        # Force internet_search by requiring non-sermon citations for the non-Branham side.
        {
            "name": "tool_internet_search_billy_graham_comparison",
            "query": (
                "Comparison: Compare Bro Branham’s teaching on salvation with Billy Graham’s preaching on salvation.\n"
                "- Branham-side claims must be sermon-cited.\n"
                "- Billy Graham side must include at least 2 external source links (markdown links) and short supporting snippets.\n"
                "Do not refuse."
            ),
        },
    ]


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


def _extract_external_info(query: str, tool_outputs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not _should_append_unverified_external_section(query):
        return None
    for output in tool_outputs:
        if output.get("name") == "internet_search":
            payload = output.get("output", {})
            return {
                "disclaimer": payload.get("disclaimer", "Unverified external search results."),
                "sources": [
                    {"title": s.get("title") or "", "url": s.get("url") or ""}
                    for s in payload.get("sources", [])
                    if isinstance(s, dict) and s.get("url")
                ],
            }
    return None


def _summarize_llm_tool_calls(llm_traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize whether tools were offered and requested in model traces."""
    offered_counts: list[int] = []
    requested_counts: list[int] = []
    requested_names: list[str] = []
    for trace in llm_traces:
        req = trace.get("request", {})
        resp = trace.get("response", {})
        tools = req.get("tools") or []
        offered_counts.append(len(tools))
        tcalls = resp.get("tool_calls") or []
        requested_counts.append(len(tcalls))
        for tc in tcalls:
            fn = tc.get("function", {})
            name = fn.get("name")
            if isinstance(name, str) and name:
                requested_names.append(name)
    return {
        "iterations": len(llm_traces),
        "tools_offered_per_iteration": offered_counts,
        "tools_requested_per_iteration": requested_counts,
        "total_tools_requested": sum(requested_counts),
        "requested_tool_names": requested_names,
    }


def _run_full_flow(
    *,
    logger: logging.Logger,
    runtime: Any,
    llm_cfg: Any,
    conversation_id: str,
    query: str,
    user_language: str | None,
    history_window: list[dict[str, Any]] | None,
    conversation_summary: str | None,
    dump_artifacts: bool,
) -> dict[str, Any]:
    """
    Run the same retrieval + generation + tool-loop logic used by the API route.

    When dump_artifacts=True, overwrites the latest_* files (matches original script behavior).
    """
    request = SimpleNamespace(
        conversation_id=conversation_id,
        query=query,
        history_window=history_window or [],
        conversation_summary=conversation_summary or "",
        user_language=user_language,
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
        "llm_provider": llm_cfg.provider,
        "llm_model": llm_cfg.effective_model,
    }

    # ---- English-only language gate (match API route behavior) ----
    if not _is_english_request(query=request.query, user_language=request.user_language):
        system_prompt = (
            "You are a translator.\n"
            "Task: translate the provided English message into the user's language.\n"
            "Output rules (hard):\n"
            "- Output markdown only.\n"
            "- Output EXACTLY two lines:\n"
            "  1) Answer:\n"
            "  2) <one short paragraph in the user's language>\n"
            "- Do NOT add any other lines.\n"
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
        if dump_artifacts:
            _dump_json(LATEST_LLM_MESSAGES, messages)
            logger.debug("LLM input messages written: %s", LATEST_LLM_MESSAGES.resolve())

        response = runtime.llm_client.completion(
            messages=messages,
            tools=None,
            tool_choice="none",
        )
        content = _normalize_english_only_reply(
            (getattr(response.choices[0].message, "content", None) or "").strip()
        )
        if dump_artifacts:
            _dump_json(
                LATEST_LLM_TRACES,
                [
                    {
                        "request": {"messages": messages, "tools": None},
                        "response": {"content": content, "tool_calls": []},
                    }
                ],
            )
        run_report["retrieval"] = {"skipped": True, "reason": "english_only_gate"}
        run_report["tool_outputs"] = []
        run_report["tool_call_counts"] = {"db_search": 0, "biography_search": 0, "internet_search": 0}
        run_report["tool_total_exhausted"] = False
        run_report["tool_limit_event_count"] = 0
        run_report["tool_limit_events"] = []
        run_report["llm_tool_call_trace"] = {
            "iterations": 1,
            "tools_offered_per_iteration": [0],
            "tools_requested_per_iteration": [0],
            "total_tools_requested": 0,
            "requested_tool_names": [],
        }
        run_report["postcheck"] = {"mode": "answer", "issues": []}
        run_report["final"] = {
            "mode": "answer",
            "answer": content,
            "external_info": None,
            "conversation_summary_out": None,
        }
        return run_report

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
        "sermon_count": len(retrieval_result.all_expanded_sermons),
        "total_chunks": retrieval_result.total_chunks,
        "signals": asdict(retrieval_result.signals),
        "refusal_thresholds": {
            "min_dense_score": runtime.pipeline.config.min_dense_score,
            "min_bm25_score": runtime.pipeline.config.min_bm25_score,
        },
    }

    if retrieval_result.should_refuse and not (
        is_bible_query(request.query) or is_comparison_query(request.query)
    ):
        refusal = _get_insufficient_context_message()
        run_report["final"] = {
            "mode": "refusal",
            "answer": refusal,
            "external_info": None,
            "conversation_summary_out": None,
            "issues": ["retrieval_refusal"],
        }
        return run_report

    # Build UI + LLM contexts in parallel. UI is for artifact viewing;
    # LLM context preserves the metadata-rich format used in production.
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rag_ui_f = pool.submit(build_rag_context, retrieval_result.all_expanded_sermons, audience="ui")
        rag_llm_f = pool.submit(build_rag_context, retrieval_result.expanded_sermons, audience="llm")
        rag_context_ui = rag_ui_f.result()
        rag_context_llm = rag_llm_f.result()
    if dump_artifacts:
        _dump_text(LATEST_RAG_CONTEXT, rag_context_ui)
        logger.debug("RAG context written: %s", LATEST_RAG_CONTEXT.resolve())

    system_prompt = build_system_prompt(
        refusal_message=_get_fixed_refusal_message(),
        insufficient_context_message=_get_insufficient_context_message(),
        extra_instructions=_query_mode_prompt_addendum(request.query),
    )
    llm_messages = build_chat_messages(
        system_prompt=system_prompt,
        query=request.query,
        rag_context=rag_context_llm,
        history_window=request.history_window,
    )
    if dump_artifacts:
        _dump_json(LATEST_LLM_MESSAGES, llm_messages)
        logger.debug("LLM input messages written: %s", LATEST_LLM_MESSAGES.resolve())

    loop_result = runtime.tool_runner.run(llm_messages)
    if dump_artifacts:
        _dump_json(LATEST_LLM_TRACES, loop_result.llm_traces)
        logger.debug("LLM request/response traces written: %s", LATEST_LLM_TRACES.resolve())

    tool_limit_events = [
        t for t in loop_result.tool_outputs
        if isinstance(t.get("output"), dict) and t["output"].get("tool_limit_reached")
    ]

    external_info = _extract_external_info(request.query, loop_result.tool_outputs)
    checked = finalize_answer(
        query=request.query,
        answer=loop_result.answer,
        external_info=external_info,
        refusal_message=_get_fixed_refusal_message(),
        insufficient_context_message=_get_insufficient_context_message(),
    )
    checked = _maybe_override_refusal_for_allowed_query(
        runtime=runtime,
        query=request.query,
        checked=checked,
    )
    run_report["tool_outputs"] = loop_result.tool_outputs
    run_report["tool_call_counts"] = runtime.tool_registry.call_counts()
    run_report["tool_total_exhausted"] = runtime.tool_registry.total_exhausted
    run_report["tool_limit_event_count"] = len(tool_limit_events)
    run_report["tool_limit_events"] = tool_limit_events
    run_report["llm_tool_call_trace"] = _summarize_llm_tool_calls(loop_result.llm_traces)
    run_report["postcheck"] = {"mode": checked.mode, "issues": checked.issues}
    summary_result = (
        runtime.summarize_conversation(
            query=request.query,
            answer=checked.answer,
            prior_summary=request.conversation_summary,
            mode=checked.mode,
        )
        if hasattr(runtime, "summarize_conversation")
        else {"conversation_summary": None, "query_summary": None}
    )
    run_report["final"] = {
        "mode": checked.mode,
        "answer": checked.answer,
        "external_info": checked.external_info,
        "conversation_summary_out": summary_result["conversation_summary"],
        "query_summary_out": summary_result["query_summary"],
    }
    return run_report


def main() -> None:
    logger = _setup_logging()

    from branham_model_api.config import get_config  # noqa: E402
    cfg = get_config()
    llm_cfg = cfg.models.llm
    logger.info(
        "LLM provider=%s model=%s base_url=%s reasoning=%s",
        llm_cfg.provider,
        llm_cfg.effective_model,
        llm_cfg.base_url or "(litellm default)",
        "on" if llm_cfg.reasoning else "off",
    )

    runtime = get_chat_runtime()

    parser = argparse.ArgumentParser(description="Manual full-flow chat debugger (non-stream).")
    parser.add_argument("--query", type=str, default=None, help="Override query text.")
    parser.add_argument(
        "--query-file",
        type=str,
        default=None,
        help="Path to a UTF-8 text file containing the query.",
    )
    parser.add_argument(
        "--user-language",
        type=str,
        default=None,
        help="Optional ISO language code hint (e.g. es, ja, af).",
    )
    parser.add_argument(
        "--stress",
        action="store_true",
        help="Run the built-in stress suite (many prompts).",
    )
    parser.add_argument(
        "--tool-smoke",
        action="store_true",
        help="Run a small suite intended to force each tool to be called.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Repeat the entire suite N times (default: 1).",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle stress cases before running (helps spot flaky failures).",
    )
    parser.add_argument(
        "--save-artifacts",
        type=str,
        default="failures",
        choices=["none", "failures", "all"],
        help="When stress testing, save per-case artifacts (run_report + answer).",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Optional cap on the number of stress cases to run.",
    )
    args = parser.parse_args()

    if args.repeat < 1:
        raise SystemExit("--repeat must be >= 1")

    # ---------------------------------------------------------------------
    # Suite runner (stress / tool-smoke)
    # ---------------------------------------------------------------------
    if args.stress or args.tool_smoke:
        suite_name = "tool_smoke" if args.tool_smoke else "stress"
        cases = _tool_smoke_cases() if args.tool_smoke else _stress_cases()
        if args.shuffle:
            random.shuffle(cases)
        if args.max_cases is not None:
            cases = cases[: max(0, int(args.max_cases))]

        run_ts = _utc_ts()
        out_dir = LOG_DIR / suite_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{suite_name}_run_{run_ts}.json"

        logger.info(
            "Starting %s run ts=%s cases=%d repeat=%d",
            suite_name,
            run_ts,
            len(cases),
            args.repeat,
        )

        results: list[dict[str, Any]] = []
        t0 = time.perf_counter()

        for rep in range(args.repeat):
            for idx, case in enumerate(cases):
                name = str(case.get("name") or f"case_{idx}")
                q = str(case.get("query") or "")
                ul = case.get("user_language")
                conv_id = f"{suite_name}-{run_ts}-r{rep}-{idx:03d}-{_slug(name)}"

                t_case0 = time.perf_counter()
                ok = True
                error: str | None = None
                run_report: dict[str, Any] | None = None
                try:
                    run_report = _run_full_flow(
                        logger=logger,
                        runtime=runtime,
                        llm_cfg=llm_cfg,
                        conversation_id=conv_id,
                        query=q,
                        user_language=ul,
                        history_window=[],
                        conversation_summary="",
                        dump_artifacts=False,
                    )
                except (LiteLLMRateLimitError, LiteLLMServiceUnavailableError) as exc:
                    ok = False
                    error = str(exc)
                except Exception as exc:  # pragma: no cover
                    ok = False
                    error = str(exc)

                elapsed_ms = (time.perf_counter() - t_case0) * 1000
                final = (run_report or {}).get("final", {}) if run_report else {}
                mode = final.get("mode")
                answer = final.get("answer") or ""
                retrieval = (run_report or {}).get("retrieval", {}) if run_report else {}
                tool_counts = (run_report or {}).get("tool_call_counts", {}) if run_report else {}
                issues = ((run_report or {}).get("postcheck", {}) or {}).get("issues", []) if run_report else []

                row = {
                    "ok": ok and bool(run_report) and mode in {"answer", "refusal"},
                    "error": error,
                    "rep": rep,
                    "idx": idx,
                    "name": name,
                    "conversation_id": conv_id,
                    "user_language": ul,
                    "query": q,
                    "elapsed_ms": elapsed_ms,
                    "mode": mode,
                    "issues": issues,
                    "retrieval": retrieval,
                    "tool_call_counts": tool_counts,
                    "answer_chars": len(str(answer)),
                }
                results.append(row)

                should_save = args.save_artifacts == "all" or (
                    args.save_artifacts == "failures" and not row["ok"]
                )
                if should_save and run_report:
                    case_dir = out_dir / f"{run_ts}_cases" / f"r{rep}_{idx:03d}_{_slug(name)}"
                    case_dir.mkdir(parents=True, exist_ok=True)
                    _dump_json(case_dir / "run_report.json", run_report)
                    _dump_text(case_dir / "answer.txt", str(answer))

                if not row["ok"]:
                    logger.warning(
                        "FAIL rep=%d idx=%d name=%s mode=%s elapsed=%.0fms err=%s",
                        rep,
                        idx,
                        name,
                        mode,
                        elapsed_ms,
                        (error or "")[:200],
                    )
                else:
                    logger.info(
                        "OK rep=%d idx=%d name=%s mode=%s elapsed=%.0fms",
                        rep,
                        idx,
                        name,
                        mode,
                        elapsed_ms,
                    )

        total_ms = (time.perf_counter() - t0) * 1000
        ok_count = sum(1 for r in results if r.get("ok"))
        refusal_count = sum(1 for r in results if r.get("mode") == "refusal")
        english_gate_count = sum(
            1 for r in results
            if isinstance(r.get("retrieval"), dict) and r["retrieval"].get("reason") == "english_only_gate"
        )

        payload = {
            "ts_utc": run_ts,
            "suite": suite_name,
            "llm_provider": llm_cfg.provider,
            "llm_model": llm_cfg.effective_model,
            "reasoning": llm_cfg.reasoning,
            "case_count": len(cases),
            "repeat": args.repeat,
            "total_runs": len(results),
            "ok_runs": ok_count,
            "failed_runs": len(results) - ok_count,
            "refusal_runs": refusal_count,
            "english_gate_runs": english_gate_count,
            "total_elapsed_ms": total_ms,
            "results": results,
        }
        _dump_json(out_path, payload)

        logger.info(
            "%s run complete: ok=%d/%d failed=%d",
            suite_name,
            ok_count,
            len(results),
            len(results) - ok_count,
        )
        logger.info("Refusals=%d english_gate=%d total_elapsed=%.0fms", refusal_count, english_gate_count, total_ms)
        logger.info("Saved %s report: %s", suite_name, out_path.resolve())
        return

    try:
        query = DEFAULT_QUERY
        if args.query_file:
            query = Path(args.query_file).read_text(encoding="utf-8").strip()
        elif args.query:
            query = args.query.strip()

        run_report = _run_full_flow(
            logger=logger,
            runtime=runtime,
            llm_cfg=llm_cfg,
            conversation_id=DEFAULT_CONVERSATION_ID,
            query=query,
            user_language=args.user_language,
            history_window=DEFAULT_HISTORY_WINDOW,
            conversation_summary=DEFAULT_CONVERSATION_SUMMARY,
            dump_artifacts=True,
        )
        _dump_markdown_and_logs(run_report, logger)
    except (LiteLLMRateLimitError, LiteLLMServiceUnavailableError) as exc:
        query = (args.query or DEFAULT_QUERY).strip()
        run_report: dict[str, Any] = {
            "conversation_id": DEFAULT_CONVERSATION_ID,
            "query": query,
            "llm_provider": llm_cfg.provider,
            "llm_model": llm_cfg.effective_model,
        }
        run_report["final"] = {
            "mode": "refusal",
            "answer": SERVICE_UNAVAILABLE_MESSAGE,
            "error": str(exc),
        }
        _dump_markdown_and_logs(run_report, logger)
    except Exception as exc:  # pragma: no cover
        query = (args.query or DEFAULT_QUERY).strip()
        run_report = {
            "conversation_id": DEFAULT_CONVERSATION_ID,
            "query": query,
            "llm_provider": llm_cfg.provider,
            "llm_model": llm_cfg.effective_model,
        }
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
