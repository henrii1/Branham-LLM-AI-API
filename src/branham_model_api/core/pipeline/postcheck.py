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

_REF_BULLET_PATTERN = re.compile(
    r"^\s*[-*]\s*\[(.+?)\s*[—–-]+\s*(\d{2}-\d{4}[A-Z]?):\s*(.+?)\]\s*$"
)

UNVERIFIED_SECTION_HEADER = "## Unverified / External Information"

_BIOGRAPHY_LINK_PATTERN = re.compile(r"lifeboatchurch\.org")

_BRANHAM_MENTION_PATTERN = re.compile(
    r"(?i)\b(?:branham|bro\.?\s*branham|brother\s+branham|william\s+branham)\b"
)


def _is_pure_internet_answer(text: str) -> bool:
    """
    True when the Answer body contains Branham references but has no sermon
    citations and no biography link — meaning it is purely internet-derived
    content that the model placed in Answer instead of Unverified.
    """
    if not _BRANHAM_MENTION_PATTERN.search(text):
        return False
    if SERMON_REF_PATTERN.search(text):
        return False
    if _BIOGRAPHY_LINK_PATTERN.search(text):
        return False
    return True


def _relocate_internet_answer_to_unverified(text: str) -> tuple[str, bool]:
    """
    When the model writes only internet-derived Branham content in Answer
    (no sermon refs, no biography link), move that content into the
    Unverified section so Answer stays clean.

    Returns (new_text, was_relocated).
    """
    # Split off any existing Unverified section the model may have written
    if UNVERIFIED_SECTION_HEADER in text:
        before, _, after = text.partition(UNVERIFIED_SECTION_HEADER)
    else:
        before = text
        after = ""

    answer_body = before.strip()

    # Extract the "Answer:" header line and the body
    body_without_header = answer_body
    if body_without_header.lower().startswith("answer:"):
        body_without_header = body_without_header[len("answer:"):].strip()

    if not body_without_header:
        return text, False

    if not _is_pure_internet_answer(body_without_header):
        return text, False

    # Relocate: keep just a brief Answer header, move body to Unverified
    moved = body_without_header
    existing_unverified = after.strip()
    unverified_content = (moved + "\n\n" + existing_unverified).strip() if existing_unverified else moved

    new_text = (
        "Answer:\n\n"
        + UNVERIFIED_SECTION_HEADER + "\n"
        + unverified_content
    )
    return new_text.strip(), True


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
        for src in sources:
            if isinstance(src, dict):
                url = str(src.get("url") or "").strip()
                title = str(src.get("title") or "").strip()
                if url:
                    label = title if title else _label_from_url(url)
                    lines.append(f"- [{label}]({url})")
            elif isinstance(src, str) and src.strip():
                # Backward compat: bare URL strings
                url = src.strip()
                lines.append(f"- [{_label_from_url(url)}]({url})")
    return "\n".join(lines).strip()


def _label_from_url(url: str) -> str:
    """Derive a short human-readable label from a URL when no title is available."""
    try:
        from urllib.parse import urlparse
        host = urlparse(url).hostname or ""
        # Strip www. prefix
        if host.startswith("www."):
            host = host[4:]
        return host.split(".")[0].capitalize() if host else "External Source"
    except Exception:
        return "External Source"


def _remove_external_section(answer: str) -> str:
    if UNVERIFIED_SECTION_HEADER not in answer:
        return answer
    before, _, _ = answer.partition(UNVERIFIED_SECTION_HEADER)
    return before.rstrip()


def _strip_empty_unverified_section(answer: str) -> str:
    """
    Remove the Unverified section if it has no substantive narrative content
    (only the boilerplate disclaimer and/or source links, no actual text).
    """
    if UNVERIFIED_SECTION_HEADER not in answer:
        return answer
    before, _, after = answer.partition(UNVERIFIED_SECTION_HEADER)
    # Check if the section has real narrative beyond disclaimer + source links
    lines = after.strip().splitlines()
    has_narrative = False
    for line in lines:
        stripped = line.strip()
        if not stripped:
            continue
        # Skip boilerplate disclaimer
        if stripped.lower() in (
            "unverified external search results.",
            "unverified external search results",
        ):
            continue
        # Skip "Sources:" header
        if stripped.lower() == "sources:":
            continue
        # Skip source link bullets: - [label](url)
        if re.match(r"^-\s*\[.+\]\(https?://.+\)\s*$", stripped):
            continue
        # Anything else is real narrative
        has_narrative = True
        break

    if has_narrative:
        return answer
    # No narrative — strip the entire section
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
        (
            i
            for i, ln in enumerate(lines)
            if "reader note:" in (ln or "").strip().lower()
        ),
        None,
    )
    if reader_idx is not None:
        lines = lines[:reader_idx]

    cleaned = "\n".join(lines).strip()

    # Remove stale "note: these are unverified external claims" lines
    # but preserve the official ## Unverified / External Information header.
    cleaned = re.sub(
        r"(?im)^\s*note:\s*these\s+are\s+unverified\s+external\s+claims.*$",
        "",
        cleaned,
    )

    # Remove placeholder References section(s).
    # We only call this helper when there are NO sermon citations at all,
    # so "References" is typically empty/placeholder and should be removed.
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
    cleaned = re.sub(
        r"\n*#{0,6}\s*References:\s*\n(?:\s*-\s*(?:\[\]|\-|\[\s*\]|None|\[N/?A\])\s*(?:\n|$))+",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\n*#{0,6}\s*References:\s*\n\s*\(none\s+from\s+sermons\.\)\s*(?:\n|$)",
        "\n",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\n*#{0,6}\s*References:\s*\n\s*-\s*$",
        "\n",
        cleaned,
        flags=re.IGNORECASE | re.MULTILINE,
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
        (r"(?i)\bprovided\s+rag\s+context\b", "available evidence"),
        (r"(?i)\bprovided\s+sermon\s+context\b", "available evidence"),
        (r"(?i)\bcurrent\s+sermon\s+context\b", "available evidence"),
        (r"(?i)\bthe\s+biography\s+tool\b", ""),
        (r"(?i)\bbiography\s+tool\b", ""),
        (r"(?i)\bbiography\s+search\b", ""),
        (r"(?i)\ba\s+curated\s+biography\s+states:?\s*", ""),
        (r"(?i)\bthe\s+biography\s+states:?\s*", ""),
        (r"(?i)\baccording\s+to\s+the\s+biography,?\s*", ""),
        (r"(?i)\bbiographical\s+sources?\s+(?:state|indicate|confirm|note)s?:?\s*", ""),
        (r"(?i)\bfrom\s+(?:the\s+)?(?:verified\s+)?biography,?\s*", ""),
        (r"(?i)\b(?:the\s+)?searched\s+sermon\s+archives?\b", "the sermons"),
        (r"(?i)\b(?:the\s+)?sermon\s+archives?\b", "the sermons"),
    )
    for pat, replacement in inline_patterns:
        text = re.sub(pat, replacement, text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, (text != original)


def _normalize_leading_answer_header(answer: str) -> tuple[str, bool]:
    """
    Normalize model-styled first-line Answer headers to the exact contract form.

    Example:
      **Answer:** -> Answer:
    """
    text = answer or ""
    normalized = re.sub(
        r"(?im)\A\s*\*{1,2}\s*Answer:\s*\*{1,2}\s*(?:\r?\n)?",
        "Answer:\n\n",
        text,
        count=1,
    )
    return normalized, (normalized != text)


def _merge_sermon_references(answer: str) -> str:
    """Merge references to the same sermon into one line with comma-separated paragraph ranges.

    Before:
      - [THE SERPENT'S SEED — 58-0928E: ¶157–¶163]
      - [THE SERPENT'S SEED — 58-0928E: ¶164–¶171]
    After:
      - [THE SERPENT'S SEED — 58-0928E: ¶157–¶163, ¶164–¶171]
    """
    lines = answer.split("\n")

    ref_header_idx = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if re.match(r"^#{0,6}\s*References:\s*$", stripped, re.IGNORECASE):
            ref_header_idx = i
            break

    if ref_header_idx is None:
        return answer

    ref_end_idx = len(lines)
    for i in range(ref_header_idx + 1, len(lines)):
        stripped = lines[i].strip()
        if not stripped:
            continue
        if stripped.startswith("#") or re.match(r"(?i)^#{0,6}\s*reader\s+note", stripped):
            ref_end_idx = i
            break

    sermon_groups: dict[tuple[str, str], list[str]] = {}
    sermon_order: list[tuple[str, str]] = []
    parsed_count = 0

    for i in range(ref_header_idx + 1, ref_end_idx):
        m = _REF_BULLET_PATTERN.match(lines[i])
        if m:
            title = m.group(1).strip()
            date_id = m.group(2).strip()
            para_range = m.group(3).strip()
            key = (title, date_id)
            if key not in sermon_groups:
                sermon_groups[key] = []
                sermon_order.append(key)
            for pr in para_range.split(","):
                pr = pr.strip()
                if pr:
                    sermon_groups[key].append(pr)
            parsed_count += 1

    if not sermon_groups:
        return answer

    total_ranges = sum(len(v) for v in sermon_groups.values())
    if total_ranges == len(sermon_groups) and parsed_count == len(sermon_groups):
        return answer

    merged_bullets = []
    for key in sermon_order:
        title, date_id = key
        merged_bullets.append(f"- [{title} — {date_id}: {', '.join(sermon_groups[key])}]")

    result_lines = lines[: ref_header_idx + 1] + merged_bullets + [""] + lines[ref_end_idx:]
    return "\n".join(result_lines)


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
    insufficient_context_message: str | None = None,
) -> PostcheckResult:
    """
    Normalize model output before returning to client.

    Postcheck refusal gating is intentionally disabled. Retrieval-time refusal
    remains the primary refusal mechanism.
    """
    _ = query  # kept for signature stability
    text = (answer or "").strip()
    text, removed_internal_language = _strip_internal_mechanics_language(text)
    text, normalized_answer_header = _normalize_leading_answer_header(text)

    # Normalize model-produced refusal variants to the fixed refusal contract.
    # Recognize both off-topic and insufficient-context refusal messages.
    refusal_messages = [refusal_message]
    if insufficient_context_message:
        refusal_messages.append(insufficient_context_message)

    if not external_info:
        normalized_for_refusal_check = text
        if normalized_for_refusal_check.lower().startswith("answer:"):
            normalized_for_refusal_check = normalized_for_refusal_check[7:].strip()
        for rmsg in refusal_messages:
            if normalized_for_refusal_check == rmsg:
                return PostcheckResult(
                    mode="refusal",
                    answer=rmsg,
                    external_info=None,
                    issues=[],
                )
        if _is_non_compliant_refusal_style(text):
            return PostcheckResult(
                mode="refusal",
                answer=insufficient_context_message or refusal_message,
                external_info=None,
                issues=["normalized_non_compliant_refusal_style"],
            )

    # Bible/comparison responses without sermon citations should not include
    # sermon-only footers/placeholders. Run BEFORE appending external section
    # so the server-appended Unverified section is never stripped.
    if not has_sermon_reference(text):
        text = _strip_non_sermon_sections(text)

    # If the model put only internet-derived Branham content in Answer
    # (no sermon refs, no biography link), relocate it to Unverified.
    relocated = False
    if external_info:
        text, relocated = _relocate_internet_answer_to_unverified(text)

    if external_info:
        text = _append_external_section(text, external_info)
    else:
        text = _remove_external_section(text)

    text = _merge_sermon_references(text)

    # Strip Unverified section if it has no real narrative (just disclaimer + links)
    text = _strip_empty_unverified_section(text)

    return PostcheckResult(
        mode="answer",
        answer=text,
        external_info=external_info,
        issues=[
            *(
                ["removed_internal_mechanics_language"]
                if removed_internal_language
                else []
            ),
            *(["normalized_answer_header"] if normalized_answer_header else []),
            *(["relocated_internet_content_to_unverified"] if relocated else []),
        ],
    )

