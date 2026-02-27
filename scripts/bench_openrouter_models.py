#!/usr/bin/env python3
"""
Benchmark a few OpenRouter models with the same RAG+prompt context.

Focus:
- Speed: retrieval time, LLM TTFT (first token), total generation time
- Output: basic format compliance (Answer/Evidence/References), citation presence
- Reliability: request failures / empty outputs

Usage:
  OPENROUTER_API_KEY_A=... uv run python scripts/bench_openrouter_models.py

Optional:
  HF_OFFLINE_ONLY=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1  # enforce offline HF loads for embedder
"""

from __future__ import annotations

import os
import sys
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Work around a common macOS OpenMP duplicate load crash (FAISS + torch, etc.).
# This is scoped to this benchmark script only.
if sys.platform == "darwin":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import litellm  # noqa: E402

from branham_model_api.config import get_config  # noqa: E402
from branham_model_api.core.pipeline import RetrievalConfig, create_rag_pipeline  # noqa: E402
from branham_model_api.core.prompts import (  # noqa: E402
    build_chat_messages,
    build_rag_context,
    build_retrieval_query,
    build_system_prompt,
)
from branham_model_api.retrieval.dense import DenseEmbedder, EmbedderConfig  # noqa: E402


MODELS = [
    # Baseline used previously via OpenRouter
    "openrouter/deepseek/deepseek-chat",
    # Candidates
    "openrouter/x-ai/grok-4.1-fast",
    "openrouter/deepseek/deepseek-v3.2",
]

# 3 representative in-scope queries (no explicit tool requirement).
QUERIES = [
    "What did Brother Branham teach about faith?",
    "What is the third pull?",
    "Tell me about the seven church ages.",
]


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


def _compliance_metrics(text: str) -> dict[str, bool]:
    t = (text or "").strip()
    return {
        "has_answer_header": t.startswith("Answer:"),
        "has_evidence_header": "\nEvidence Summary:" in t or t.startswith("Evidence Summary:"),
        "has_references_header": "\nReferences:" in t or t.startswith("References:"),
        "has_any_bracket_citation": ("[" in t and "¶" in t and "]" in t),
    }


@dataclass
class RunResult:
    model: str
    query: str
    retrieval_ms: float
    rag_chars: int
    ttft_ms: float | None
    total_ms: float
    ok: bool
    error: str | None
    output_chars: int
    metrics: dict[str, bool]
    preview: str
    output: str


def main() -> None:
    api_key = (
        os.getenv("OPENROUTER_API_KEY_A", "").strip()
        or os.getenv("OPENROUTER_API_KEY", "").strip()
        or next(
            (
                v.strip()
                for k, v in os.environ.items()
                if k.startswith("OPENROUTER_API_KEY_") and v.strip()
            ),
            "",
        )
    )
    if not api_key:
        raise SystemExit(
            "Missing OpenRouter API key. Set OPENROUTER_API_KEY (or OPENROUTER_API_KEY_A)."
        )

    cfg = get_config()

    # Build retrieval pipeline (same as API uses)
    data_dir = Path(__file__).parent.parent / "data"
    bm25_path = data_dir / "indices" / "bm25.index"
    faiss_path = data_dir / "indices" / "faiss.index"
    faiss_id_map_path = data_dir / "indices" / "faiss_id_map.jsonl"
    chunk_store_path = data_dir / "processed" / "chunks.sqlite"

    retrieval_cfg = RetrievalConfig.from_yaml()
    embedder = DenseEmbedder(
        EmbedderConfig(
            model_id=cfg.models.embedding_model_id,
            pooling="last_token",
            padding_side="left",
            max_length=512,
            batch_size=1,
            normalize=True,
            trust_remote_code=True,
            local_files_only=os.getenv("HF_OFFLINE_ONLY", "").strip().lower() in {"1", "true", "yes"},
            query_instruction_template="Instruct: {task}\nQuery:{query}",
            query_task_description=(
                "Given a question about William Branham's teachings or sermons, "
                "retrieve relevant sermon passages that answer the query"
            ),
        )
    )
    pipeline = create_rag_pipeline(
        bm25_index_path=bm25_path,
        faiss_index_path=faiss_path,
        faiss_id_map_path=faiss_id_map_path,
        chunk_store_path=chunk_store_path,
        embedder=embedder,
        reranker=None,
        config=retrieval_cfg,
    )

    # Reuse the same helper used by the API for exact refusal contract injection.
    from branham_model_api.api.routes.chat import _get_fixed_refusal_message  # noqa: E402

    fixed_refusal = _get_fixed_refusal_message()

    # Reasoning control for speed: disable reasoning and exclude it if any.
    reasoning = {"effort": "none", "exclude": True}

    results: list[RunResult] = []

    for model in MODELS:
        for query in QUERIES:
            # Retrieval
            retrieval_query = build_retrieval_query(query, None)
            t0 = time.perf_counter()
            r = pipeline.retrieve(retrieval_query, user_language="en")
            retrieval_ms = (time.perf_counter() - t0) * 1000
            rag_context = build_rag_context(r.expanded_sermons)

            # Prompt build
            system_prompt = build_system_prompt(refusal_message=fixed_refusal)
            messages = build_chat_messages(
                system_prompt=system_prompt,
                query=query,
                rag_context=rag_context,
                history_window=None,
            )

            # LLM stream
            ttft_ms: float | None = None
            out_parts: list[str] = []
            ok = True
            err: str | None = None
            t_llm0 = time.perf_counter()
            try:
                stream = litellm.completion(
                    model=model,
                    messages=messages,
                    api_key=api_key,
                    base_url="https://openrouter.ai/api/v1",
                    stream=True,
                    temperature=0.2,
                    timeout=60.0,
                    reasoning=reasoning,
                )
                for chunk in stream:
                    text = _extract_delta_text(chunk)
                    if text and ttft_ms is None:
                        ttft_ms = (time.perf_counter() - t_llm0) * 1000
                    if text:
                        out_parts.append(text)
            except Exception as e:
                ok = False
                err = str(e)

            total_ms = (time.perf_counter() - t_llm0) * 1000
            output = "".join(out_parts).strip()
            metrics = _compliance_metrics(output)

            results.append(
                RunResult(
                    model=model,
                    query=query,
                    retrieval_ms=retrieval_ms,
                    rag_chars=len(rag_context),
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    ok=ok,
                    error=err,
                    output_chars=len(output),
                    metrics=metrics,
                    preview=(output[:220].replace("\n", " ") if output else ""),
                    output=output,
                )
            )

    # Persist full results for manual review.
    out_dir = Path(__file__).parent.parent / "data" / "benchmarks"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"openrouter_model_bench_{ts}.json"
    out_payload = {
        "generated_at_utc": ts,
        "models": MODELS,
        "queries": QUERIES,
        "reasoning": {"effort": "none", "exclude": True},
        "results": [
            {
                "model": r.model,
                "query": r.query,
                "ok": r.ok,
                "error": r.error,
                "retrieval_ms": r.retrieval_ms,
                "rag_chars": r.rag_chars,
                "ttft_ms": r.ttft_ms,
                "total_ms": r.total_ms,
                "output_chars": r.output_chars,
                "metrics": r.metrics,
                "output": r.output,
            }
            for r in results
        ],
    }
    out_path.write_text(json.dumps(out_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Print summary
    print("=" * 96)
    print("OpenRouter Model Benchmark (reasoning.effort=none, exclude=true)")
    print("=" * 96)
    print(f"Saved full outputs to: {out_path}")
    for model in MODELS:
        rows = [x for x in results if x.model == model]
        oks = [x for x in rows if x.ok]
        fails = [x for x in rows if not x.ok]
        ttfts = [x.ttft_ms for x in oks if x.ttft_ms is not None]
        totals = [x.total_ms for x in oks]
        print(f"\nModel: {model}")
        print(f"  ok={len(oks)}/{len(rows)} fail={len(fails)}")
        if ttfts:
            print(f"  ttft_ms: min={min(ttfts):.0f} max={max(ttfts):.0f} avg={sum(ttfts)/len(ttfts):.0f}")
        if totals:
            print(f"  total_ms: min={min(totals):.0f} max={max(totals):.0f} avg={sum(totals)/len(totals):.0f}")
        # Compliance rates
        if oks:
            def rate(key: str) -> str:
                return f"{sum(1 for x in oks if x.metrics.get(key)):d}/{len(oks)}"
            print(
                "  compliance: "
                f"Answer={rate('has_answer_header')} "
                f"Evidence={rate('has_evidence_header')} "
                f"Refs={rate('has_references_header')} "
                f"Citations={rate('has_any_bracket_citation')}"
            )

        for x in rows:
            status = "OK" if x.ok else "FAIL"
            ttft_str = f"{x.ttft_ms:.0f}ms" if x.ttft_ms is not None else "None"
            print(
                f"  - {status} q='{x.query[:42]}' "
                f"retrieval={x.retrieval_ms:.0f}ms ttft={ttft_str} "
                f"total={x.total_ms:.0f}ms out_chars={x.output_chars}"
            )
            if x.error:
                print(f"    error: {x.error[:200]}")
            else:
                print(f"    preview: {x.preview}")


if __name__ == "__main__":
    main()

