"""
Post-generation policy checks and output finalization.
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
INVALID_PARAGRAPH_SUFFIX_PATTERN = re.compile(
    r"\[(?:[^\[\]\n]+?\s+[—-]\s+)?[0-9]{2}-[0-9]{4}[A-Z]?:\s*¶\d+[a-z](?:–¶?\d+[a-z]?)?\]"
)

# Bible-style references:
#   John 3:16, 1 Corinthians 13:4-7, Genesis 1:1
BIBLE_REF_PATTERN = re.compile(
    r"\b(?:[1-3]\s+)?[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?\s+\d+:\d+(?:-\d+)?\b"
)

_BIBLE_KEYWORDS = (
    "bible",
    "scripture",
    "verse",
    "verses",
    "old testament",
    "new testament",
    "genesis",
    "exodus",
    "psalm",
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
    "old testament",
    "new testament",
    "genesis",
    "exodus",
    "leviticus",
    "numbers",
    "deuteronomy",
    "joshua",
    "judges",
    "ruth",
    "1 samuel",
    "2 samuel",
    "1 kings",
    "2 kings",
    "1 chronicles",
    "2 chronicles",
    "ezra",
    "nehemiah",
    "esther",
    "job",
    "psalms",
    "proverbs",
    "ecclesiastes",
    "song of solomon",
    "isaiah",
    "jeremiah",
    "lamentations",
    "ezekiel",
    "daniel",
    "hosea",
    "joel",
    "amos",
    "obadiah",
    "jonah",
    "micah",
    "nahum",
    "habakkuk",
    "zephaniah",
    "haggai",
    "zechariah",
    "malachi",
    "matthew",
    "mark",
    "luke",
    "john",
    "acts",
    "romans",
    "1 corinthians",
    "2 corinthians",
    "galatians",
    "ephesians",
    "philippians",
    "colossians",
    "1 thessalonians",
    "2 thessalonians",
    "1 timothy",
    "2 timothy",
    "titus",
    "philemon",
    "hebrews",
    "james",
    "1 peter",
    "2 peter",
    "1 john",
    "2 john",
    "3 john",
    "jude",
    "revelation",
)

UNVERIFIED_SECTION_HEADER = "## Unverified / External Information"


def is_bible_query(query: str) -> bool:
    q = query.lower()
    return any(k in q for k in _BIBLE_KEYWORDS)


def has_sermon_reference(answer: str) -> bool:
    return bool(SERMON_REF_PATTERN.search(answer))


def has_bible_reference(answer: str) -> bool:
    return bool(BIBLE_REF_PATTERN.search(answer))


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

    Postcheck refusal gating is intentionally disabled. The caller keeps
    retrieval-time refusal behavior, while LLM outputs (including biography
    and external-info answers) are returned as-is.
    """
    issues: list[str] = []
    text = (answer or "").strip()

    if external_info:
        text = _append_external_section(text, external_info)
    else:
        text = _remove_external_section(text)

    return PostcheckResult(
        mode="answer",
        answer=text,
        external_info=external_info,
        issues=issues,
    )
