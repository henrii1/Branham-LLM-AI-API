"""
Tests for post-check enforcement.
"""

from branham_model_api.core.pipeline.postcheck import (
    finalize_answer,
    has_bible_reference,
    has_sermon_reference,
    is_comparison_query,
    is_bible_query,
)

REFUSAL = "I can only answer questions based on William Branham's sermons. I don't have enough relevant information to answer your question."


def test_has_sermon_reference_detection() -> None:
    assert has_sermon_reference("He taught this [47-0412M: ¶2–¶3].") is True
    assert has_sermon_reference(
        "He taught this [Contending For The Faith — 59-0823: ¶89–¶92]."
    ) is True
    assert has_sermon_reference("No citation here.") is False


def test_has_bible_reference_detection() -> None:
    assert has_bible_reference("John 3:16 is often cited.") is True
    assert has_bible_reference("No verse format") is False


def test_non_bible_answer_without_sermon_reference_passes() -> None:
    out = finalize_answer(
        query="What did Brother Branham teach about faith?",
        answer="He taught many things.",
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert out.answer == "He taught many things."
    assert out.issues == []


def test_fixed_refusal_message_maps_to_refusal_mode() -> None:
    out = finalize_answer(
        query="What did Brother Branham teach about faith?",
        answer=REFUSAL,
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "refusal"
    assert out.answer == REFUSAL
    assert out.issues == []


def test_non_compliant_refusal_style_is_normalized() -> None:
    out = finalize_answer(
        query="According to 1 Corinthians 12, are women limited in gifts?",
        answer=(
            "The provided RAG context does not contain any direct references to 1 Corinthians 12. "
            "Therefore, I cannot answer your question based on the current sermon context I have.\n\n"
            "If you'd like, I can search Bro Branham's sermons for related teachings. "
            "Let me know if you'd like me to proceed with that."
        ),
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "refusal"
    assert out.answer == REFUSAL
    assert "normalized_non_compliant_refusal_style" in out.issues


def test_invalid_paragraph_suffix_reference_passes_without_refusal() -> None:
    out = finalize_answer(
        query="What did Brother Branham teach about faith?",
        answer="He taught this [47-0412M: ¶12a–¶13b].",
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert "[47-0412M: ¶12a–¶13b]" in out.answer
    assert out.issues == []


def test_invalid_paragraph_suffix_reference_titled_passes_without_refusal() -> None:
    out = finalize_answer(
        query="What did Brother Branham teach about faith?",
        answer="He taught this [Contending For The Faith — 59-0823: ¶12a–¶13b].",
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert "59-0823: ¶12a–¶13b" in out.answer
    assert out.issues == []


def test_bible_query_can_pass_with_bible_reference() -> None:
    out = finalize_answer(
        query="What does the Bible say about faith?",
        answer="Hebrews says faith is substance (Hebrews 11:1).",
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert "Hebrews 11:1" in out.answer


def test_external_section_added_only_when_external_info_present() -> None:
    out = finalize_answer(
        query="Tell me current events about Branham ministries",
        answer="Summary with sermon ref [47-0412M: ¶2–¶3].",
        external_info={"disclaimer": "Unverified external search results.", "sources": ["https://example.com"]},
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert "## Unverified / External Information" in out.answer
    assert "[Source 1](https://example.com)" in out.answer


def test_external_section_removed_when_external_info_absent() -> None:
    out = finalize_answer(
        query="What did Brother Branham teach about healing?",
        answer="Grounded answer [47-0412M: ¶2–¶3].\n\n## Unverified / External Information\nShould not remain.",
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert out.mode == "answer"
    assert "## Unverified / External Information" not in out.answer


def test_no_sermon_reference_strips_reader_note_and_na_references() -> None:
    out = finalize_answer(
        query="From biblical points,by which woman did cain produce the giants",
        answer=(
            "Answer:\n"
            "The provided sermon context does not contain this.\n\n"
            "References:\n"
            "[N/A]\n\n"
            "Reader Note: For full context and to verify wording, read the full sermon and cited paragraphs "
            "in The Table: [The Table](https://table.branham.org/) (plain link: https://table.branham.org/)"
        ),
        external_info=None,
        refusal_message=REFUSAL,
    )
    assert "Reader Note:" not in out.answer
    assert "table.branham.org" not in out.answer.lower()
    assert "[N/A]" not in out.answer


def test_is_bible_query_keyword_check() -> None:
    assert is_bible_query("Explain John 3:16") is True
    assert is_bible_query("What does Habakkuk say about faith?") is True
    assert is_bible_query("From biblical points, by which woman did Cain produce the giants?") is True
    assert is_bible_query("What is the spiritual implication/significance in tithe-paying?") is True
    assert is_bible_query("What did Branham preach in 1963?") is False


def test_is_comparison_query_detection() -> None:
    assert is_comparison_query("Compare Branham and other authors on water baptism") is True
    assert is_comparison_query("What is the difference between Branham and other preachers?") is True
    assert is_comparison_query("What did Branham preach in 1963?") is False

