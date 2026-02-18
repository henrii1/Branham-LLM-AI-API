#!/usr/bin/env python3
"""
Provider latency & tool-call behaviour comparison.

Sends 5 queries against the live /api/chat SSE endpoint and reports:
  - TTFT (time to first delta)
  - Total time
  - Tool calls observed (via server logs emitted in SSE events)
  - Answer mode (refusal, answer, etc.)
  - Active LLM provider/model (from config)
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from branham_model_api.config import get_config  # noqa: E402

BASE_URL = os.getenv("TEST_BASE_URL", "http://127.0.0.1:8000")
BEARER_KEY = os.getenv(
    "CHAT_API_BEARER_KEY",
    "b6766b2e-9a26-4342-9bef-5da4ad67e51c",
)

QUERIES = [
    "What did Brother Branham teach about the seven church ages?",
    "Give me the exact quote from paragraph 45 of 65-1128M about the rapture.",
    "Who was William Branham? Tell me about his early life.",
    "What does Branham say about the serpent seed doctrine?",
    "What does branham preach about faith, compared with joseph prince recent sermons? check the internet.",
]


def _parse_sse_events(raw: str) -> list[tuple[str, dict]]:
    """Parse raw SSE text into (event_type, data_dict) pairs."""
    events = []
    # Normalize line endings
    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    # Split on double newline (event boundary)
    blocks = re.split(r"\n\n+", raw)
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        event_type = None
        event_data = None
        for line in block.split("\n"):
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                event_data = line[len("data:"):].strip()
        if event_type and event_data:
            try:
                events.append((event_type, json.loads(event_data)))
            except json.JSONDecodeError:
                pass
    return events


def _stream_query(query: str, idx: int) -> dict:
    """Send one query, consume SSE, return timing + tool info."""
    payload = {
        "query": query,
        "conversation_id": f"latency-test-{idx}",
        "history_window": [],
        "conversation_summary": None,
        "user_language": "en",
    }
    headers = {
        "Authorization": f"Bearer {BEARER_KEY}",
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    t0 = time.perf_counter()
    t_first_delta = None
    final_event: dict = {}
    delta_count = 0
    answer_parts: list[str] = []

    with httpx.Client(timeout=120.0) as client:
        with client.stream("POST", f"{BASE_URL}/api/chat", json=payload, headers=headers) as resp:
            resp.raise_for_status()
            raw_buf = ""
            for chunk in resp.iter_text():
                raw_buf += chunk
                # Try parsing completed events
                while "\n\n" in raw_buf:
                    block, raw_buf = raw_buf.split("\n\n", 1)
                    for evt_type, evt_data in _parse_sse_events(block + "\n\n"):
                        if evt_type == "delta":
                            delta_count += 1
                            if t_first_delta is None:
                                t_first_delta = time.perf_counter()
                            text = evt_data.get("text", "")
                            answer_parts.append(text)
                        elif evt_type == "final":
                            final_event = evt_data

            # Parse any remaining data in buffer
            if raw_buf.strip():
                for evt_type, evt_data in _parse_sse_events(raw_buf):
                    if evt_type == "delta":
                        delta_count += 1
                        if t_first_delta is None:
                            t_first_delta = time.perf_counter()
                        answer_parts.append(evt_data.get("text", ""))
                    elif evt_type == "final":
                        final_event = evt_data

    t_total = time.perf_counter() - t0
    ttft = (t_first_delta - t0) if t_first_delta else t_total

    full_answer = "".join(answer_parts).strip()
    mode = final_event.get("mode", "unknown")
    return {
        "query": query,
        "ttft_ms": round(ttft * 1000),
        "total_ms": round(t_total * 1000),
        "delta_count": delta_count,
        "mode": mode,
        "answer_preview": full_answer[:150],
        "answer_length": len(full_answer),
        "has_external": final_event.get("external_info") is not None,
    }


def main() -> None:
    cfg = get_config()
    llm = cfg.models.llm
    print("=" * 72)
    print(f"Provider Latency & Tool-Call Test")
    print(f"  provider : {llm.provider}")
    print(f"  model    : {llm.effective_model}")
    print(f"  base_url : {llm.base_url or '(litellm default)'}")
    print(f"  key_pfx  : {llm.key_prefix}")
    print("=" * 72)

    # Warm-up query
    print("\n[warm-up] Sending warm-up query...")
    try:
        warmup = _stream_query("Hello, who is Brother Branham?", 0)
        print(f"  warm-up: TTFT={warmup['ttft_ms']}ms total={warmup['total_ms']}ms mode={warmup['mode']} deltas={warmup['delta_count']}")
    except Exception as e:
        print(f"  warm-up failed: {e}")

    results = []
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {q[:65]}...")
        try:
            r = _stream_query(q, i)
            results.append(r)
            print(f"  TTFT={r['ttft_ms']}ms  total={r['total_ms']}ms  mode={r['mode']}")
            print(f"  deltas={r['delta_count']}  answer_len={r['answer_length']}  external={r['has_external']}")
            print(f"  preview: {r['answer_preview'][:100]}...")
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append({"query": q, "error": str(e)})

    # Summary
    valid = [r for r in results if "ttft_ms" in r]
    if valid:
        ttfts = [r["ttft_ms"] for r in valid]
        totals = [r["total_ms"] for r in valid]
        modes = [r.get("mode", "?") for r in valid]
        print("\n" + "=" * 72)
        print("SUMMARY")
        print(f"  Provider: {llm.provider} ({llm.effective_model})")
        print(f"  Queries : {len(valid)}/{len(QUERIES)}")
        print(f"  TTFT    : min={min(ttfts)}ms  max={max(ttfts)}ms  avg={sum(ttfts)//len(ttfts)}ms")
        print(f"  Total   : min={min(totals)}ms  max={max(totals)}ms  avg={sum(totals)//len(totals)}ms")
        print(f"  Modes   : {', '.join(modes)}")
        # Separate refusals from actual answers
        answers = [r for r in valid if r.get("mode") not in ("refusal",)]
        refusals = [r for r in valid if r.get("mode") == "refusal"]
        if answers:
            a_ttft = [r["ttft_ms"] for r in answers]
            a_total = [r["total_ms"] for r in answers]
            print(f"  Answers : {len(answers)} queries")
            print(f"    TTFT  : min={min(a_ttft)}ms  max={max(a_ttft)}ms  avg={sum(a_ttft)//len(a_ttft)}ms")
            print(f"    Total : min={min(a_total)}ms  max={max(a_total)}ms  avg={sum(a_total)//len(a_total)}ms")
        if refusals:
            print(f"  Refusals: {len(refusals)} queries")
        print("=" * 72)

    # Dump detailed JSON
    out_path = Path("data/logs/provider_latency_results.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "provider": llm.provider,
        "model": llm.effective_model,
        "results": results,
    }, indent=2, ensure_ascii=True))
    print(f"\nDetailed results: {out_path.resolve()}")


if __name__ == "__main__":
    main()
