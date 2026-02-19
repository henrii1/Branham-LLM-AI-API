from __future__ import annotations

from branham_model_api.core.pipeline.expansion import ExpandedChunk, ExpandedSermon
from branham_model_api.core.prompts.templates import (
    build_chat_messages,
    build_rag_context,
    build_retrieval_query,
    build_system_prompt,
)


def _sample_sermon() -> ExpandedSermon:
    return ExpandedSermon(
        date_id="47-0412M",
        title="Faith Is The Substance",
        source=None,
        language="en",
        min_paragraph_no=1,
        max_paragraph_no=277,
        best_score=0.8123,
        chunks=[
            ExpandedChunk(
                chunk_id="47-0412M_chunk_001",
                date_id="47-0412M",
                chunk_index=1,
                text="Now faith is the substance of things hoped for.",
                paragraph_start=2,
                paragraph_end=3,
                word_count=10,
                char_count=48,
                is_tail_chunk=False,
                score=0.8123,
                is_retrieved=True,
            ),
            ExpandedChunk(
                chunk_id="47-0412M_chunk_002",
                date_id="47-0412M",
                chunk_index=2,
                text="This is supporting expanded context.",
                paragraph_start=4,
                paragraph_end=5,
                word_count=6,
                char_count=36,
                is_tail_chunk=True,
                score=None,
                is_retrieved=False,
            ),
        ],
    )


def test_build_retrieval_query_query_only() -> None:
    out = build_retrieval_query("  What is faith?  ")
    assert out == "What is faith?"


def test_build_retrieval_query_with_summary() -> None:
    out = build_retrieval_query("What is faith?", "User asked about Hebrews and Branham.")
    assert "What is faith?" in out
    assert "Conversation summary" in out
    assert "Hebrews" in out


def test_build_rag_context_contains_sermon_metadata_and_chunks() -> None:
    context = build_rag_context([_sample_sermon()])
    assert "Sermon: Faith Is The Substance" in context
    assert "date_id: 47-0412M" in context
    assert "chunk_id: 47-0412M_chunk_001" in context
    assert "paragraphs: ¶2–¶3" in context
    assert "valid_paragraph_bounds_for_db_search: ¶1–¶277" in context
    assert "db_search_guardrail: never request paragraph numbers outside these bounds" in context
    assert "is_tail_chunk: false" in context


def test_build_system_prompt_includes_enforced_reference_policies() -> None:
    prompt = build_system_prompt(
        refusal_message="I can only answer questions based on William Branham's sermons. I don't have enough relevant information to answer your question."
    )
    assert "no 12a/12b/etc" in prompt
    assert "Unverified / External Information" in prompt
    assert "The Table" in prompt


def test_build_system_prompt_appends_runtime_addendum() -> None:
    prompt = build_system_prompt(
        refusal_message="I can only answer questions based on William Branham's sermons. I don't have enough relevant information to answer your question.",
        extra_instructions="BIBLE-QUERY MODE:\n- This query is in scope.",
    )
    assert "MODE-SPECIFIC ADDENDUM (RUNTIME)" in prompt
    assert "BIBLE-QUERY MODE" in prompt


def test_build_chat_messages_includes_query_history_and_context() -> None:
    rag_context = build_rag_context([_sample_sermon()])
    messages = build_chat_messages(
        system_prompt="system",
        query="What did he say about faith?",
        rag_context=rag_context,
        history_window=[{"role": "assistant", "content": "Prior response"}],
    )
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    user_content = messages[1]["content"]
    assert "What did he say about faith?" in user_content
    assert "Recent chat history:" in user_content
    assert "RAG context:" in user_content
