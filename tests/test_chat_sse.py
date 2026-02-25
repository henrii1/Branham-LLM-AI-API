from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

from fastapi.testclient import TestClient

from branham_model_api.api.main import app
from branham_model_api.api.routes import chat as chat_route
from branham_model_api.generation import LiteLLMServiceUnavailableError


def _auth_headers() -> dict[str, str]:
    return {"Authorization": "Bearer test-chat-key"}


def _parse_sse(body: str) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current_event = None
    current_data = None
    for line in body.splitlines():
        if line.startswith("event: "):
            current_event = line[len("event: ") :]
        elif line.startswith("data: "):
            raw = line[len("data: ") :]
            current_data = json.loads(raw)
        elif not line.strip():
            if current_event is not None and current_data is not None:
                events.append((current_event, current_data))
            current_event = None
            current_data = None
    if current_event is not None and current_data is not None:
        events.append((current_event, current_data))
    return events


def _chunk(text: str):
    """Simulate a streaming LLM chunk with content text."""
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
    )


class _FakeLLMClient:
    """LLM client that returns canned streaming responses."""

    def __init__(self, chunks: list | None = None, raise_on_call: Exception | None = None):
        self._chunks = chunks or []
        self._raise = raise_on_call

    def stream_completion(self, *, messages, tools=None, tool_choice=None, **kw):
        if self._raise:
            raise self._raise
        return iter(self._chunks)

    def completion(self, *, messages, max_tokens=None, **kw):
        content = "".join(
            getattr(getattr(c.choices[0], "delta", None), "content", "") or ""
            for c in self._chunks
        ).strip()
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _FakeToolRegistry:
    def definitions(self):
        return []

    def reset_counts(self):
        pass

    @property
    def total_exhausted(self):
        return False

    def call_counts(self):
        return {}

    def execute_tool(self, name, args):
        return {}


# ---------------------------------------------------------------------------
# Runtime mocks
# ---------------------------------------------------------------------------


class _RefusalRuntime:
    llm_client = _FakeLLMClient()
    tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=True,
            refuse_reason="no hits",
            expanded_sermons=[],
            bm25_hit_count=0,
            dense_hit_count=0,
            fused_hit_count=0,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.0,
                dense_top_score=0.0,
                bm25_dense_overlap=0,
                quote_intent=False,
            ),
        )


class _AnswerRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[_chunk("streamed answer with citation [47-0412M: ¶2–¶3]")]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )

    def summarize_conversation(self, *, query, answer, prior_summary, mode):
        return "short summary"


class _UnavailableRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            raise_on_call=LiteLLMServiceUnavailableError("down")
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )


class _ErrorRuntime:
    llm_client = _FakeLLMClient()
    tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        raise RuntimeError("boom")


class _BibleExceptionRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[_chunk("For God so loved the world (John 3:16).")]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=True,
            refuse_reason="weak",
            expanded_sermons=[],
            bm25_hit_count=0,
            dense_hit_count=0,
            fused_hit_count=0,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.0,
                dense_top_score=0.0,
                bm25_dense_overlap=0,
                quote_intent=False,
            ),
        )


class _ComparisonExceptionRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[_chunk("Comparison answer between Branham and other authors.")]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=True,
            refuse_reason="weak",
            expanded_sermons=[],
            bm25_hit_count=0,
            dense_hit_count=0,
            fused_hit_count=0,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.0,
                dense_top_score=0.0,
                bm25_dense_overlap=0,
                quote_intent=False,
            ),
        )


class _ExternalRuntime:
    """Simulates a flow where the serper tool was called (external_used)."""

    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[_chunk("Grounded answer [47-0412M: ¶2–¶3].")]
        )
        self.tool_registry = _FakeExternalToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )


class _FakeExternalToolRegistry(_FakeToolRegistry):
    """Registry that simulates having already run internet_search."""

    def definitions(self):
        return []


class _StreamAnswerRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[
                _chunk("This "),
                _chunk("is "),
                _chunk("streamed [47-0412M: ¶2–¶3]."),
            ]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )


class _StreamRefusalRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[
                _chunk("I can only answer questions "),
                _chunk("based on William Branham's sermons."),
            ]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )


class _SummaryRuntime:
    def __init__(self):
        self.llm_client = _FakeLLMClient(
            chunks=[_chunk("Grounded answer [47-0412M: ¶2–¶3].")]
        )
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )

    def summarize_conversation(self, *, query, answer, prior_summary, mode):
        assert mode == "answer"
        return "generated summary metadata"


class _BibleFallbackLLMClient:
    def stream_completion(self, *, messages, tools=None, tool_choice=None, **kw):
        return iter(
            [
                _chunk("I can only answer questions based on William Branham's sermons. "),
                _chunk("I don't have enough relevant information to answer your question."),
            ]
        )

    def completion(self, *, messages, max_tokens=None, **kw):
        system = (messages[0].get("content") or "").lower()
        if "bible-query mode" in system:
            content = "In 1 Corinthians 12, spiritual gifts are distributed by the Holy Spirit without a gender-only gift list."
        else:
            content = "short summary"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class _BibleFallbackRuntime:
    def __init__(self):
        self.llm_client = _BibleFallbackLLMClient()
        self.tool_registry = _FakeToolRegistry()

    def retrieve(self, retrieval_query: str, **kw):
        return SimpleNamespace(
            should_refuse=False,
            refuse_reason=None,
            expanded_sermons=[],
            bm25_hit_count=25,
            dense_hit_count=25,
            fused_hit_count=40,
            total_chunks=0,
            reranker_triggered=False,
            signals=SimpleNamespace(
                dense_score_std=0.02,
                dense_top_score=0.7,
                bm25_dense_overlap=2,
                quote_intent=False,
            ),
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_chat_sse_streams_early_refusal(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _RefusalRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c1", "query": "off topic"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names[0] == "start"
    # For early retrieval refusals we do not emit a rag payload.
    assert "delta" in names
    assert "final" in names
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "refusal"
    assert isinstance(final["answer"], str) and final["answer"]
    assert final["conversation_summary"] is None


def test_chat_sse_streams_answer_flow(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _AnswerRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c2", "query": "what is faith?"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    names = [n for n, _ in events]
    assert names[0] == "start"
    assert "rag" in names
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "[47-0412M: ¶2–¶3]" in final["answer"]
    assert final["conversation_summary"] == "short summary"


def test_chat_sse_converts_llm_unavailable_to_refusal(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _UnavailableRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c3", "query": "question"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "refusal"
    assert "temporarily unavailable" in final["answer"].lower()


def test_chat_sse_unhandled_error_emits_error_event(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _ErrorRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c4", "query": "question"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    error = [p for n, p in events if n == "error"][-1]
    assert error["mode"] == "error"


def test_chat_sse_bible_exception_can_bypass_early_refusal(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _BibleExceptionRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c5", "query": "What does the Bible say in John 3:16?"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "John 3:16" in final["answer"]


def test_chat_sse_comparison_exception_can_bypass_early_refusal(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _ComparisonExceptionRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "conversation_id": "c6",
            "query": "Compare Branham and other authors on 1 Corinthians 12 gifts",
        },
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "Comparison answer" in final["answer"]


def test_chat_sse_streams_token_deltas_for_normal_answer(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _StreamAnswerRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c7", "query": "question"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    deltas = [p["text"] for n, p in events if n == "delta"]
    assert len(deltas) >= 2
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "streamed [47-0412M: ¶2–¶3]" in final["answer"]


def test_chat_sse_streams_token_deltas_for_llm_refusal(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _StreamRefusalRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c8", "query": "question"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    deltas = [p["text"] for n, p in events if n == "delta"]
    assert len(deltas) >= 2
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "I can only answer questions" in final["answer"]


def test_chat_sse_bible_query_uses_fallback_when_model_refuses(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _BibleFallbackRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c10", "query": "According to 1 Corinthians 12, are women limited in gifts?"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final = [p for n, p in events if n == "final"][-1]
    assert final["mode"] == "answer"
    assert "without a gender-only gift list" in final["answer"]


def test_chat_sse_final_contains_generated_summary_metadata(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _SummaryRuntime())
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={"conversation_id": "c9", "query": "question"},
        headers=_auth_headers(),
    )
    assert resp.status_code == 200
    events = _parse_sse(resp.text)
    final = [p for n, p in events if n == "final"][-1]
    assert final["conversation_summary"] == "generated summary metadata"


def test_chat_sse_rejects_missing_bearer_token(monkeypatch):
    monkeypatch.setattr(chat_route, "_get_expected_chat_bearer_key", lambda: "test-chat-key")
    monkeypatch.setattr(chat_route, "get_chat_runtime", lambda: _AnswerRuntime())
    client = TestClient(app)
    resp = client.post("/api/chat", json={"conversation_id": "cx", "query": "question"})
    assert resp.status_code == 401
