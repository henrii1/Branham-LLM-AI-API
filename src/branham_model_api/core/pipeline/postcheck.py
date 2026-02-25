"""
Post-generation output finalization.

Note:
- Refusal gating is intentionally disabled here.
- This module only handles lightweight normalization of the external-info section.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Canonical sermon inline reference (preferred):
#   [Contending For The Faith — 59-0823: ¶89–¶92]
# Backward-compatible legacy format:
#   [47-0412M: ¶2–¶3]
SERMON_REF_PATTERN = re.compile(
    r"\[(?:[^\[\]\n]+?\s+[—-]\s+)?[0-9]{2}-[0-9]{4}[A-Z]?:\s*¶\d+(?:–¶?\d+)?\]"
)

# Bible-style references:
#   John 3:16, 1 Corinthians 13:4-7, Genesis 1:1
BIBLE_REF_PATTERN = re.compile(
    r"\b(?:[1-3]\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+\d+:\d+(?:-\d+)?\b"
)

_BIBLE_KEYWORDS = (
    "bible",
    "biblical",
    "scriptural",
    "scripturally",
    "scripture",
    "verse",
    "verses",
    "old testament",
    "new testament",
    "genesis",
    "exodus",
    "psalm",
    "habakkuk",
    "isaiah",
    "matthew",
    "mark",
    "luke",
    "john",
    "romans",
    "corinthians",
    "galatians",
    "ephesians",
    "revelation",
    "tithe",
    "tithing",
    "cain",
    "abel",
)

_BIBLE_QUERY_REF_PATTERN = re.compile(
    r"\b(?:[1-3]\s*)?[A-Za-z]+\s+\d+:\d+(?:-\d+)?\b",
    re.IGNORECASE,
)

_COMPARISON_KEYWORDS = (
    "compare",
    "comparison",
    "contrast",
    "versus",
    " vs ",
    "against",
    "difference between",
    "similarities between",
)

UNVERIFIED_SECTION_HEADER = "## Unverified / External Information"
_NON_COMPLIANT_REFUSAL_PATTERNS = (
    "provided rag context",
    "provided sermon context",
    "current sermon context i have",
    "cannot answer your question based on",
    "if you'd like, i can search",
    "let me know if you'd like me to proceed",
)


def is_bible_query(query: str) -> bool:
    q = (query or "").lower()
    return any(k in q for k in _BIBLE_KEYWORDS) or bool(
        _BIBLE_QUERY_REF_PATTERN.search(query or "")
    )


def is_comparison_query(query: str) -> bool:
    q = f" {(query or '').lower()} "
    has_compare_signal = any(k in q for k in _COMPARISON_KEYWORDS)
    has_branham = "branham" in q or "bro branham" in q
    has_other_source = any(
        marker in q
        for marker in (
            "other author",
            "other authors",
            "author",
            "authors",
            "other preacher",
            "other preachers",
            "other sermon",
            "other sermons",
        )
    )
    return has_compare_signal or (has_branham and has_other_source)


def has_sermon_reference(answer: str) -> bool:
    return bool(SERMON_REF_PATTERN.search(answer or ""))


def has_bible_reference(answer: str) -> bool:
    return bool(BIBLE_REF_PATTERN.search(answer or ""))


def _append_external_section(answer: str, external_info: dict[str, Any]) -> str:
    if UNVERIFIED_SECTION_HEADER in answer:
        return answer
    lines = [answer.rstrip(), "", UNVERIFIED_SECTION_HEADER]
    disclaimer = str(external_info.get("disclaimer") or "Unverified external search results.")
    lines.append(disclaimer)
    sources = external_info.get("sources") or []
    if sources:
        lines.append("")
        lines.append("Sources:")
        for idx, src in enumerate(sources, start=1):
            url = str(src).strip()
            if url:
                lines.append(f"- [Source {idx}]({url})")
    return "\n".join(lines).strip()


def _remove_external_section(answer: str) -> str:
    if UNVERIFIED_SECTION_HEADER not in answer:
        return answer
    before, _, _ = answer.partition(UNVERIFIED_SECTION_HEADER)
    return before.rstrip()


def _strip_non_sermon_sections(answer: str) -> str:
    """
    Remove sermon-only sections when answer has no sermon references.

    For Bible/comparison answers with no sermon citations, we do not want:
    - Reader Note pointing to The Table
    - placeholder References blocks like [N/A]
    """
    text = answer or ""
    lines = text.splitlines()

    # Remove everything from Reader Note onward (it should be terminal when present).
    reader_idx = next(
        (i for i, ln in enumerate(lines) if ln.strip().startswith("Reader Note:")),
        None,
    )
    if reader_idx is not None:
        lines = lines[:reader_idx]

    cleaned = "\n".join(lines).strip()

    # Remove placeholder References section(s).
    cleaned = re.sub(
        r"\n*References:\s*\n\s*-\s*\[N/A\]\s*(?:\n|$)",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\n*References:\s*\n\s*\[N/A\]\s*(?:\n|$)",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    return cleaned.strip()


def _is_non_compliant_refusal_style(answer: str) -> bool:
    text = (answer or "").strip().lower()
    if not text:
        return False
    return any(pat in text for pat in _NON_COMPLIANT_REFUSAL_PATTERNS)


def _strip_internal_mechanics_language(answer: str) -> tuple[str, bool]:
    """
    Remove user-visible internal pipeline wording (RAG/retrieval context talk).

    We keep this lightweight and conservative: only remove explicit mechanics
    phrases that should never appear in final user-facing text.
    """
    text = answer or ""
    original = text
    patterns = (
        r"(?im)^[^\n]*\bprovided\s+rag\s+context\b[^\n]*\n?",
        r"(?im)^[^\n]*\bprovided\s+sermon\s+context\b[^\n]*\n?",
        r"(?im)^[^\n]*\bcurrent\s+sermon\s+context\b[^\n]*\n?",
        r"(?im)^[^\n]*\bretrieval\s+pipeline\b[^\n]*\n?",
        r"(?im)^[^\n]*\btool\s+loop\b[^\n]*\n?",
    )
    for pat in patterns:
        text = re.sub(pat, "", text)
    # If phrase appears inline inside a sentence, scrub only the phrase.
    inline_patterns = (
        r"(?i)\bprovided\s+rag\s+context\b",
        r"(?i)\bprovided\s+sermon\s+context\b",
        r"(?i)\bcurrent\s+sermon\s+context\b",
    )
    for pat in inline_patterns:
        text = re.sub(pat, "available evidence", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, (text != original)


@dataclass
class PostcheckResult:
    mode: str
    answer: str
    external_info: dict[str, Any] | None
    issues: list[str]


def finalize_answer(
    *,
    query: str,
    answer: str,
    external_info: dict[str, Any] | None,
    refusal_message: str,
) -> PostcheckResult:
    """
    Normalize model output before returning to client.

    Postcheck refusal gating is intentionally disabled. Retrieval-time refusal
    remains the primary refusal mechanism.
    """
    _ = query  # kept for signature stability
    text = (answer or "").strip()
    text, removed_internal_language = _strip_internal_mechanics_language(text)

    # Normalize model-produced refusal variants to the fixed refusal contract.
    if not external_info:
        normalized_for_refusal_check = text
        if normalized_for_refusal_check.lower().startswith("answer:"):
            normalized_for_refusal_check = normalized_for_refusal_check[7:].strip()
        if normalized_for_refusal_check == refusal_message:
            return PostcheckResult(
                mode="refusal",
                answer=refusal_message,
                external_info=None,
                issues=[],
            )
        if _is_non_compliant_refusal_style(text):
            return PostcheckResult(
                mode="refusal",
                answer=refusal_message,
                external_info=None,
                issues=["normalized_non_compliant_refusal_style"],
            )

    if external_info:
        text = _append_external_section(text, external_info)
    else:
        text = _remove_external_section(text)

    # Bible/comparison responses without sermon citations should not include
    # sermon-only footers/placeholders.
    if not has_sermon_reference(text):
        text = _strip_non_sermon_sections(text)

    return PostcheckResult(
        mode="answer",
        answer=text,
        external_info=external_info,
        issues=(["removed_internal_mechanics_language"] if removed_internal_language else []),
    )

