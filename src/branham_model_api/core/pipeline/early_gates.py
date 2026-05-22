"""
Pre-LLM deterministic refusal gates.

These match user-query patterns that the corpus reliably cannot answer, OR
short/vague queries that have no clear referent on a first turn. When a gate
fires, the server emits an INSUFFICIENT_CONTEXT refusal before retrieval —
the LLM is never called.

Scope is intentionally narrow. We only match patterns whose failure modes
were observed in benchmark cases and where the Kimi K2.5 (reasoning off)
prompt-level rule was not reliably triggered. Broader analysis queries that
happen to mention "model" or "address" must NOT be caught — see the negative
guards in each regex.

Returns a (matched, reason) tuple so callers can log why they refused.
"""
from __future__ import annotations

import re


# Specific-fact query: asks for a single literal datum (a model number, year,
# trim, exact address, named hymn, etc.) about Branham that the sermon corpus
# does not state literally. The model otherwise fabricates "Branham drove
# Cadillacs" instead of refusing.
#
# Each pattern is anchored to a Branham subject so non-Branham queries pass
# through (e.g. "what model of car did Henry Ford build?" must NOT trigger).
_SUBJECT = r"(?:bro\.?\s+branham|brother\s+branham|branham|william\s+(?:marrion\s+)?branham|he|his|him)"
_FACT_NOUN = (
    r"(?:model|year|trim|brand|make|color|colour|license\s*plate|"
    r"phone|height|weight|"
    # Address variants: "address", "home address", "street address",
    # "home street address", "exact home street address", etc.
    r"(?:exact\s+)?(?:home\s+)?(?:street\s+)?address|"
    r"(?:favorite|favourite)\s+(?:hymn|song|verse|chapter|book|movie|food|meal|drink))"
)
_SPECIFIC_FACT_PATTERNS = [
    # "What model of Cadillac did Brother Branham drive?" / similar
    re.compile(
        rf"\bwhat\s+{_FACT_NOUN}\s+(?:of\s+\w+\s+)?(?:did|was|is|does)\s+{_SUBJECT}\b",
        re.IGNORECASE,
    ),
    # "What was Bro Branham's exact home street address?" / "Bro Branham's
    # Cadillac model" — allow one optional intervening object/qualifier word
    # (e.g. "Cadillac model", "favorite hymn") between possessive and fact noun.
    re.compile(
        rf"\bwhat\s+(?:was|is)\s+{_SUBJECT}'?s?\s+(?:\w+\s+){{0,2}}{_FACT_NOUN}\b",
        re.IGNORECASE,
    ),
    # "What was Bro Branham's personal favorite hymn?" (personal qualifier)
    re.compile(
        rf"\bwhat\s+(?:was|is)\s+{_SUBJECT}'?s?\s+personal\s+(?:favorite|favourite)\s+\w+\b",
        re.IGNORECASE,
    ),
    # "Which Cadillac (model) did Bro Branham …?" — "Which" variant of pattern 1.
    re.compile(
        rf"\bwhich\s+\w+(?:\s+{_FACT_NOUN})?\s+(?:did|was|is|does)\s+{_SUBJECT}\b",
        re.IGNORECASE,
    ),
]

# Negative guard: queries that include analysis verbs are NOT specific-fact.
# A "thorough analysis of Branham's teaching on faith" should never trigger.
_ANALYSIS_GUARD = re.compile(
    r"\b(?:analysis|analyze|analyse|compare|contrast|explain|teach|teaching|"
    r"thorough|detailed|comprehensive|overview|summary|study)\b",
    re.IGNORECASE,
)


def is_specific_fact_query(query: str) -> tuple[bool, str | None]:
    """Detect a SPECIFIC-datum query whose answer the sermon corpus does not state.

    Returns (True, reason) when matched. Conservative — only triggers when:
      - the query syntactically asks for a single literal datum, AND
      - the subject is Branham (not a generic entity), AND
      - the query is NOT phrased as analysis/comparison/teaching.

    Falls back to (False, None) on any ambiguity so the LLM still gets to try.
    """
    if not query or not query.strip():
        return False, None
    text = query.strip()

    if _ANALYSIS_GUARD.search(text):
        return False, None

    for pat in _SPECIFIC_FACT_PATTERNS:
        m = pat.search(text)
        if m:
            return True, f"specific-fact: matched '{m.group(0)[:60]}'"
    return False, None


# Unclear-query stems: short fragments that have no clear referent on a first
# turn (no chat history, no conversation summary). Kept narrow.
_UNCLEAR_STEMS = [
    # "I want to" / "I'd like to" / "I would like to" — same vague stem family.
    re.compile(r"^\s*i\s+want\s+to\s*\.?\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*i'?\s*d\s+like\s+to\s*\.?\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*i\s+would\s+like\s+to\s*\.?\s*\??\s*$", re.IGNORECASE),
    # "help me" + optional verb (understand/with) + optional pronoun (it/this/that). NO trailing topic noun.
    re.compile(
        r"^\s*help\s+me(?:\s+(?:understand|with))?(?:\s+(?:it|this|that))?\s*\??\s*$",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*what\s+about\s+(?:it|that|this|him|them)\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*tell\s+me\s+more\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*continue\s*\.?\s*$", re.IGNORECASE),
    re.compile(r"^\s*more\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*and\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*explain\s+(?:it|that|this)\s*\??\s*$", re.IGNORECASE),
    re.compile(r"^\s*(?:go on|keep going)\s*\??\s*$", re.IGNORECASE),
]


def is_unclear_query(
    query: str,
    *,
    has_history: bool,
    has_summary: bool,
) -> tuple[bool, str | None]:
    """Detect a short/vague query with no clear referent and no prior context.

    When chat history OR conversation summary is present, this gate does NOT
    fire — the LLM can resolve "it"/"that" against the prior turn.

    Returns (True, reason) only when:
      - both history AND summary are absent (first turn), AND
      - the query matches one of the vague stems.

    Word-count alone is NOT sufficient: "John 3:16?" is short but legitimate.
    """
    if has_history or has_summary:
        return False, None
    if not query or not query.strip():
        return False, None
    text = query.strip()
    for pat in _UNCLEAR_STEMS:
        m = pat.search(text)
        if m:
            return True, f"unclear-stem: matched '{m.group(0)[:60]}'"
    return False, None
