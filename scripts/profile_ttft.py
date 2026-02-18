#!/usr/bin/env python3
"""
Profile TTFT (time to first token) and total latency for the /api/chat SSE endpoint.

Starts the server in-process, sends real queries, measures:
- TTFT: time from request to first `delta` SSE event
- Total: time from request to `done` SSE event
- Mode: answer / refusal / error

Usage:
    KMP_DUPLICATE_LIB_OK=TRUE uv run python scripts/profile_ttft.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import httpx

QUERIES = [
    "What did Brother Branham teach about faith?",
    "What is the third pull?",
    "Tell me about the seven church ages",
    "What does the Bible say about John 3:16?",
    "How does Branham interpret the book of Revelation?",
]

BASE_URL = "http://127.0.0.1:8000"


def _get_bearer_key() -> str:
    key = os.getenv("CHAT_API_BEARER_KEY", "").strip()
    if key:
        return key
    config_path = Path(__file__).parent.parent / "config" / "default.yaml"
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            raw = yaml.safe_load(f) or {}
        return str(raw.get("api", {}).get("bearer_key", ""))
    return ""


def profile_query(query: str, bearer_key: str) -> dict:
    headers = {
        "Authorization": f"Bearer {bearer_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "conversation_id": "profile-session",
        "query": query,
    }

    t_start = time.perf_counter()
    t_first_delta = None
    mode = "unknown"
    answer_preview = ""
    delta_count = 0

    with httpx.Client(timeout=120.0) as client:
        with client.stream(
            "POST",
            f"{BASE_URL}/api/chat",
            json=payload,
            headers=headers,
        ) as response:
            current_event = None
            for line in response.iter_lines():
                if line.startswith("event: "):
                    current_event = line[len("event: "):]
                elif line.startswith("data: "):
                    data = json.loads(line[len("data: "):])
                    if current_event == "delta":
                        delta_count += 1
                        if t_first_delta is None:
                            t_first_delta = time.perf_counter()
                        text = data.get("text", "")
                        if len(answer_preview) < 80:
                            answer_preview += text
                    elif current_event == "final":
                        mode = data.get("mode", "unknown")
                    elif current_event == "done":
                        break

    t_end = time.perf_counter()
    ttft_ms = (t_first_delta - t_start) * 1000 if t_first_delta else None
    total_ms = (t_end - t_start) * 1000

    return {
        "query": query[:60],
        "ttft_ms": round(ttft_ms, 1) if ttft_ms else None,
        "total_ms": round(total_ms, 1),
        "mode": mode,
        "delta_count": delta_count,
        "answer_preview": answer_preview[:80].replace("\n", " "),
    }


def main():
    bearer_key = _get_bearer_key()
    if not bearer_key:
        print("ERROR: No bearer key found. Set CHAT_API_BEARER_KEY or check config/default.yaml")
        sys.exit(1)

    print("=" * 80)
    print("TTFT Profiler — /api/chat SSE endpoint")
    print("=" * 80)
    print(f"Server: {BASE_URL}")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Warm-up: first query (cold start)
    print("--- Warm-up query (absorbs cold-start costs) ---")
    warmup = profile_query("warm up query", bearer_key)
    print(f"  TTFT={warmup['ttft_ms']}ms  total={warmup['total_ms']}ms  mode={warmup['mode']}")
    print()

    results = []
    print("--- Profiling queries ---")
    for i, q in enumerate(QUERIES, 1):
        print(f"\n[{i}/{len(QUERIES)}] {q[:60]}...")
        r = profile_query(q, bearer_key)
        results.append(r)
        print(f"  TTFT={r['ttft_ms']}ms  total={r['total_ms']}ms  mode={r['mode']}  deltas={r['delta_count']}")
        print(f"  Preview: {r['answer_preview']}")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    ttft_values = [r["ttft_ms"] for r in results if r["ttft_ms"] is not None]
    total_values = [r["total_ms"] for r in results]

    if ttft_values:
        print(f"TTFT  — min: {min(ttft_values):.0f}ms  max: {max(ttft_values):.0f}ms  avg: {sum(ttft_values)/len(ttft_values):.0f}ms")
    print(f"Total — min: {min(total_values):.0f}ms  max: {max(total_values):.0f}ms  avg: {sum(total_values)/len(total_values):.0f}ms")
    print(f"Modes: {', '.join(r['mode'] for r in results)}")
    print()

    for r in results:
        print(f"  {r['query'][:50]:50s}  TTFT={str(r['ttft_ms']):>8s}ms  total={r['total_ms']:>8.0f}ms  {r['mode']}")

    print("=" * 80)


if __name__ == "__main__":
    main()
