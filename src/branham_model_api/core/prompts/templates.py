"""
Prompt/context builders for retrieval + generation flow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from branham_model_api.core.pipeline.expansion import ExpandedSermon

DEFAULT_SYSTEM_PROMPT_PATH = (
    Path(__file__).resolve().parent / "system_prompt.txt"
)


def build_retrieval_query(
    query: str,
    conversation_summary: str | None = None,
) -> str:
    """
    Build retrieval input text.

    Strategy:
    - If summary exists: query + summary
    - Else: query only
    """
    query_norm = " ".join(query.strip().split())
    if conversation_summary and conversation_summary.strip():
        summary_norm = " ".join(conversation_summary.strip().split())
        return f"{query_norm}\n\nConversation summary:\n{summary_norm}"
    return query_norm


def format_history_window(
    history_window: list[dict[str, str]] | None,
    *,
    max_turns: int = 6,
) -> str:
    """Render short chat history for LLM continuity."""
    if not history_window:
        return ""

    trimmed = history_window[-max_turns:]
    lines: list[str] = []
    for i, turn in enumerate(trimmed, start=1):
        role = (turn.get("role") or "user").strip().lower()
        content = (turn.get("content") or "").strip()
        if not content:
            continue
        lines.append(f"{i}. {role}: {content}")
    return "\n".join(lines)


def _format_sermon_header(sermon: ExpandedSermon) -> str:
    title = sermon.title or "Unknown Title"
    valid_bounds = (
        f"¶{sermon.min_paragraph_no}–¶{sermon.max_paragraph_no}"
        if sermon.min_paragraph_no is not None and sermon.max_paragraph_no is not None
        else "unknown"
    )
    return (
        f"### Sermon: {title}\n"
        f"- date_id: {sermon.date_id}\n"
        f"- valid_paragraph_bounds_for_db_search: {valid_bounds}\n"
        f"- best_score: {sermon.best_score:.4f}\n"
        f"- chunk_count: {len(sermon.chunks)}"
    )


def format_sermon_context_block(sermon: ExpandedSermon) -> str:
    """
    Format one sermon's chunk context for prompt injection.

    Includes stable metadata + chunk markers for reference traceability.
    """
    lines = [_format_sermon_header(sermon), "", "Chunks:"]
    for chunk in sermon.chunks:
        retrieved = "yes" if chunk.is_retrieved else "expansion"
        valid_bounds = (
            f"¶{sermon.min_paragraph_no}–¶{sermon.max_paragraph_no}"
            if sermon.min_paragraph_no is not None and sermon.max_paragraph_no is not None
            else "unknown"
        )
        lines.append(
            f"- chunk_id: {chunk.chunk_id} | paragraphs: ¶{chunk.paragraph_start}–¶{chunk.paragraph_end} "
            f"| source: {retrieved} | is_tail_chunk: {str(chunk.is_tail_chunk).lower()} "
            f"| valid_paragraph_bounds_for_db_search: {valid_bounds} "
            f"| db_search_guardrail: never request paragraph numbers outside these bounds"
        )
        lines.append(chunk.text)
        lines.append("")
    return "\n".join(lines).strip()


def build_rag_context(sermons: Sequence[ExpandedSermon]) -> str:
    """Build the full RAG context from grouped sermons."""
    if not sermons:
        return "No RAG context available."
    blocks = []
    for idx, sermon in enumerate(sermons, start=1):
        blocks.append(f"## Context Group {idx}")
        blocks.append(format_sermon_context_block(sermon))
        blocks.append("")
    return "\n".join(blocks).strip()


def load_system_prompt(path: Path | None = None) -> str:
    """Load editable base system prompt text."""
    p = path or DEFAULT_SYSTEM_PROMPT_PATH
    return p.read_text(encoding="utf-8").strip()


def build_system_prompt(
    *,
    refusal_message: str,
    base_prompt_path: Path | None = None,
) -> str:
    """
    Build runtime system prompt with dynamic refusal message injection.

    The base prompt is the single source of policy truth.
    We only replace the refusal placeholder to avoid duplicated rules.
    """
    base = load_system_prompt(base_prompt_path)
    placeholder = "{{REFUSAL_MESSAGE}}"
    if placeholder in base:
        return base.replace(placeholder, refusal_message).strip()
    return (base + f'\n\nRefusal message: "{refusal_message}"').strip()


def build_chat_messages(
    *,
    system_prompt: str,
    query: str,
    rag_context: str,
    history_window: list[dict[str, str]] | None = None,
) -> list[dict[str, Any]]:
    """
    Build OpenAI/LiteLLM-compatible messages.

    User content structure:
      query + optional history + rag context
    """
    history_text = format_history_window(history_window)
    user_parts = [query.strip()]
    if history_text:
        user_parts.append("Recent chat history:\n" + history_text)
    user_parts.append("RAG context:\n" + rag_context)
    user_content = "\n\n".join(p for p in user_parts if p.strip())

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
