from __future__ import annotations

from branham_model_api.core.pipeline.rag_pipeline import detect_query_language


def test_detect_query_language_english() -> None:
    assert detect_query_language("What did Brother Branham teach about faith?") == "en"


def test_detect_query_language_non_english() -> None:
    assert detect_query_language("¿Qué enseñó el Hermano Branham sobre la fe?") == "non-en"
