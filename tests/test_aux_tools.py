from __future__ import annotations

from pathlib import Path

import pytest

from branham_model_api.core.tools.biography_tool import BiographyTool
from branham_model_api.core.tools.serper_tool import SerperTool


def test_biography_tool_reads_local_file(tmp_path: Path) -> None:
    bio = tmp_path / "bio.txt"
    bio.write_text("William Branham biography sample.", encoding="utf-8")
    tool = BiographyTool(file_path=bio)
    out = tool.execute({"query": "early ministry"})
    assert out["ok"] is True
    assert "biography sample" in out["content"]
    assert out["query"] == "early ministry"


def test_serper_tool_handles_missing_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    tool = SerperTool()
    out = tool.execute({"query": "test"})
    assert out["ok"] is False
    assert "SERPER_API_KEY" in out["error"]


def test_serper_tool_success_with_mocked_request(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Resp:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "organic": [
                    {
                        "title": "Result 1",
                        "link": "https://example.com/1",
                        "snippet": "Snippet 1",
                    }
                ]
            }

    def _fake_post(*args, **kwargs):
        return _Resp()

    monkeypatch.setenv("SERPER_API_KEY", "x")
    monkeypatch.setattr("branham_model_api.core.tools.serper_tool.requests.post", _fake_post)

    tool = SerperTool()
    out = tool.execute({"query": "faith", "num_results": 1})
    assert out["ok"] is True
    assert out["external"] is True
    assert out["sources"][0]["url"] == "https://example.com/1"
