from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from branham_model_api.core.tools import (
    DbSearchTool,
    ToolLimitError,
    ToolLoopRunner,
    ToolRegistry,
    ToolSpec,
)
from branham_model_api.retrieval.store.chunk_store import ChunkStore


def _build_test_db(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE sermons (
            date_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            language TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE chunks (
            chunk_id TEXT PRIMARY KEY,
            date_id TEXT,
            paragraph_start INTEGER,
            paragraph_end INTEGER,
            chunk_index INTEGER,
            text TEXT,
            word_count INTEGER,
            char_count INTEGER
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE paragraphs (
            date_id TEXT,
            paragraph_no INTEGER,
            sub_id TEXT DEFAULT '',
            text TEXT,
            PRIMARY KEY (date_id, paragraph_no, sub_id)
        )
        """
    )
    conn.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES ('47-0412M','Faith Is The Substance','src','en')"
    )
    conn.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES ('63-0318E','The Spoken Word Is The Original Seed','src','en')"
    )
    # Add a same-day suffix variant to test prefix expansion behavior
    conn.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES ('63-0318M','The Spoken Word Is The Original Seed','src','en')"
    )
    # Add title duplicates to test fuzzy resolution + year disambiguation
    conn.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES ('62-0128A','A PARADOX','src','en')"
    )
    conn.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES ('63-0801','A PARADOX','src','en')"
    )
    rows = [
        ("47-0412M_c1", "47-0412M", 1, 2, 0, "Faith text first.", 3, 20),
        ("47-0412M_c2", "47-0412M", 3, 4, 1, "Faith text second.", 3, 21),
        ("63-0318E_c1", "63-0318E", 10, 12, 0, "Seed sermon content.", 3, 20),
        ("63-0318M_c1", "63-0318M", 10, 12, 0, "Seed sermon (morning) content.", 3, 28),
    ]
    conn.executemany(
        """
        INSERT INTO chunks(
            chunk_id, date_id, paragraph_start, paragraph_end, chunk_index, text, word_count, char_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    paragraph_rows = [
        ("47-0412M", 1, "", "Faith text first."),
        ("47-0412M", 1, "a", "Faith text first continuation."),
        ("47-0412M", 2, "", "Faith text second."),
        ("47-0412M", 3, "", "Faith text third."),
        ("47-0412M", 4, "", "Faith text fourth."),
        ("63-0318E", 10, "", "Seed sermon content."),
        ("63-0318E", 11, "", "Seed sermon continuation."),
        ("63-0318E", 12, "", "Seed sermon ending."),
        ("63-0318M", 10, "", "Seed sermon morning content."),
        ("63-0318M", 11, "", "Seed sermon morning M_ONLY continuation."),
        ("63-0318M", 12, "", "Seed sermon morning ending."),
        ("62-0128A", 30, "", "Paradox older paragraph 30."),
        ("62-0128A", 31, "", "Paradox older paragraph 31."),
        ("62-0128A", 32, "", "Paradox older paragraph 32."),
        ("63-0801", 29, "", "Paradox 1963 paragraph 29."),
        ("63-0801", 30, "", "Paradox 1963 paragraph 30."),
        ("63-0801", 31, "", "Paradox 1963 paragraph 31 story."),
        ("63-0801", 32, "", "Paradox 1963 paragraph 32."),
        ("63-0801", 33, "", "Paradox 1963 paragraph 33."),
    ]
    conn.executemany(
        """
        INSERT INTO paragraphs(date_id, paragraph_no, sub_id, text)
        VALUES (?, ?, ?, ?)
        """,
        paragraph_rows,
    )
    conn.commit()
    conn.close()


def test_db_search_tool_read_single_and_batch(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    single = tool.execute(
        {"mode": "read_paragraphs", "date_id": "47-0412M", "paragraph_start": "1a", "paragraph_end": "1b"}
    )
    assert single["ok"] is True
    assert single["sermons"][0]["date_id"] == "47-0412M"
    assert single["sermons"][0]["ranges"][0]["paragraph_start"] == 1
    assert single["sermons"][0]["ranges"][0]["paragraphs"][0]["paragraph_no"] == 1

    batch = tool.execute(
        {
            "mode": "batch_read",
            "requests": [
                {"date_id": "47-0412M", "paragraph_start": 1, "paragraph_end": 4},
                {"date_id": "63-0318E", "paragraph_start": 10, "paragraph_end": 12},
            ],
        }
    )
    assert batch["ok"] is True
    by_id = {s["date_id"]: s for s in batch["sermons"]}
    assert "47-0412M" in by_id
    assert "63-0318E" in by_id
    first_range = by_id["47-0412M"]["ranges"][0]
    assert "paragraphs" in first_range


def test_db_search_tool_read_sermon_default_head_and_override(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store, max_paragraphs_per_query=80, default_sermon_head=2)

    default_head = tool.execute({"mode": "read_sermon", "date_id": "47-0412M"})
    paragraphs = default_head["sermons"][0]["ranges"][0]["paragraphs"]
    assert len(paragraphs) == 2

    overridden = tool.execute({"mode": "read_sermon", "date_id": "47-0412M", "head": 3})
    paragraphs_override = overridden["sermons"][0]["ranges"][0]["paragraphs"]
    assert len(paragraphs_override) == 3


def test_db_search_tool_can_resolve_title_only_for_read_sermon(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store, max_paragraphs_per_query=80, default_sermon_head=2)

    out = tool.execute({"mode": "read_sermon", "title_query": "A Paradox"})
    assert out["ok"] is True
    assert out["sermons"][0]["date_id"] in ("62-0128A", "63-0801")
    assert "resolution" in out


def test_db_search_tool_resolves_title_plus_year_and_reads_context_paragraph(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute(
        {"mode": "read_paragraphs", "title_query": "A Paradox", "year": 63, "paragraph_start": 31}
    )
    assert out["ok"] is True
    assert out["sermons"][0]["date_id"] == "63-0801"
    paras = out["sermons"][0]["ranges"][0]["paragraphs"]
    # Default ±2 context => 29..33
    nos = [p["paragraph_no"] for p in paras]
    assert min(nos) == 29
    assert max(nos) == 33


def test_db_search_tool_resolves_date_id_prefix(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    # Test DB includes "63-0318E" and "63-0318M"; prefix should expand to both.
    out = tool.execute(
        {"mode": "read_paragraphs", "date_id": "63-0318", "paragraph_start": 10, "paragraph_end": 11}
    )
    assert out["ok"] is True
    assert {s["date_id"] for s in out["sermons"]} == {"63-0318E", "63-0318M"}


def test_db_search_tool_read_sermon_splits_head_across_prefix_suffixes(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute({"mode": "read_sermon", "date_id": "63-0318", "head": 6})
    assert out["ok"] is True
    assert {s["date_id"] for s in out["sermons"]} == {"63-0318E", "63-0318M"}
    # head=6, two suffix sermons => 3 paragraphs each
    counts = {s["date_id"]: len(s["ranges"][0]["paragraphs"]) for s in out["sermons"]}
    assert counts["63-0318E"] == 3
    assert counts["63-0318M"] == 3


def test_db_search_tool_quote_local_searches_across_prefix_suffixes(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute(
        {"mode": "search_quote_local", "date_id": "63-0318", "query": "M_ONLY", "limit": 5}
    )
    assert out["ok"] is True
    assert any(s["date_id"] == "63-0318M" for s in out["sermons"])


def test_db_search_tool_falls_back_to_title_when_date_id_unknown(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute(
        {
            "mode": "read_paragraphs",
            "date_id": "63-1201",  # unknown in test DB
            "title_query": "A Paradox",
            "year": 63,
            "paragraph_start": 31,
        }
    )
    assert out["ok"] is True
    assert out["sermons"][0]["date_id"] == "63-0801"


def test_db_search_tool_invalid_range_reports_bounds(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    partially_invalid = tool.execute(
        {"mode": "read_paragraphs", "date_id": "47-0412M", "paragraph_start": 3, "paragraph_end": 7}
    )
    info = partially_invalid["sermons"][0]["ranges"][0]["range_info"]
    assert info["final_paragraph_no"] == 4
    assert info["has_invalid_request"] is True
    assert info["invalid_ranges"] == [{"from": 5, "to": 7}]

    fully_invalid = tool.execute(
        {"mode": "read_paragraphs", "date_id": "47-0412M", "paragraph_start": 9, "paragraph_end": 10}
    )
    info2 = fully_invalid["sermons"][0]["ranges"][0]["range_info"]
    assert info2["final_paragraph_no"] == 4
    assert info2["valid_start"] is None
    assert info2["valid_end"] is None
    assert info2["invalid_ranges"] == [{"from": 9, "to": 10}]


def test_db_search_tool_quote_search(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute({"mode": "search_quote", "query": "Faith text", "limit": 5})
    assert out["ok"] is True
    assert any(s["date_id"] == "47-0412M" for s in out["sermons"])
    faith = [s for s in out["sermons"] if s["date_id"] == "47-0412M"][0]
    assert all("chunk_id" not in p for p in faith["ranges"][0]["paragraphs"])

    local = tool.execute(
        {"mode": "search_quote_local", "date_id": "47-0412M", "query": "Faith text", "limit": 5}
    )
    assert local["ok"] is True
    assert len(local["sermons"]) == 1
    assert local["sermons"][0]["date_id"] == "47-0412M"


def test_db_search_tool_batch_mixed(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    tool = DbSearchTool(chunk_store=store)

    out = tool.execute(
        {
            "mode": "batch_mixed",
            "operations": [
                {"mode": "read_paragraphs", "date_id": "47-0412M", "paragraph_start": 1, "paragraph_end": 2},
                {"mode": "search_quote_local", "date_id": "47-0412M", "query": "Faith text", "limit": 3},
            ],
        }
    )
    assert out["ok"] is True
    assert out["mode"] == "batch_mixed"
    assert len(out["results"]) == 2
    assert out["results"][0]["ok"] is True
    assert out["results"][1]["ok"] is True


class _DummyTool:
    name = "dummy_tool"

    def definition(self):
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "dummy",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        }

    def execute(self, args):
        return {"ok": True, "echo": "tool-ok"}


class _FakeLLM:
    def __init__(self) -> None:
        self._calls = 0

    def completion(self, **kwargs):
        self._calls += 1
        if self._calls == 1:
            tc = SimpleNamespace(
                id="tc_1",
                type="function",
                function=SimpleNamespace(name="dummy_tool", arguments="{}"),
            )
            msg = SimpleNamespace(content="", tool_calls=[tc])
        else:
            msg = SimpleNamespace(content="final answer", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_tool_loop_runner_executes_tool_then_returns_answer() -> None:
    registry = ToolRegistry([ToolSpec(tool=_DummyTool(), max_calls=2)])
    runner = ToolLoopRunner(llm_client=_FakeLLM(), tool_registry=registry, max_iterations=4)
    result = runner.run(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert result.answer == "final answer"
    assert len(result.tool_outputs) == 1
    assert result.tool_outputs[0]["name"] == "dummy_tool"


class _AlwaysToolCallLLM:
    def completion(self, **kwargs):
        tc = SimpleNamespace(
            id="tc_x",
            type="function",
            function=SimpleNamespace(name="dummy_tool", arguments="{}"),
        )
        msg = SimpleNamespace(content="", tool_calls=[tc])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_tool_loop_runner_returns_tool_limit_output_without_raising() -> None:
    registry = ToolRegistry([ToolSpec(tool=_DummyTool(), max_calls=1)])
    runner = ToolLoopRunner(llm_client=_AlwaysToolCallLLM(), tool_registry=registry, max_iterations=4)
    out = runner.run(messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}])
    assert any(
        t["output"].get("tool_limit_reached") is True for t in out.tool_outputs
    )


def test_tool_registry_total_limit_enforced() -> None:
    registry = ToolRegistry([ToolSpec(tool=_DummyTool(), max_calls=10)], max_total_calls=2)
    registry.begin_tool_round()
    assert registry.execute_tool("dummy_tool", {})["ok"] is True
    registry.begin_tool_round()
    assert registry.execute_tool("dummy_tool", {})["ok"] is True
    with pytest.raises(ToolLimitError):
        registry.begin_tool_round()


class _DbSearchSuffixLLM:
    def __init__(self) -> None:
        self._calls = 0

    def completion(self, **kwargs):
        self._calls += 1
        if self._calls == 1:
            tc = SimpleNamespace(
                id="tc_db_1",
                type="function",
                function=SimpleNamespace(
                    name="db_search",
                    arguments='{"mode":"read_paragraphs","date_id":"47-0412M","paragraph_start":"1a","paragraph_end":"1b"}',
                ),
            )
            msg = SimpleNamespace(content="", tool_calls=[tc])
        else:
            msg = SimpleNamespace(content="final answer", tool_calls=[])
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def test_tool_loop_runner_db_search_handles_suffix_args(tmp_path: Path) -> None:
    db_path = tmp_path / "chunks.sqlite"
    _build_test_db(db_path)
    store = ChunkStore(db_path)
    registry = ToolRegistry([ToolSpec(tool=DbSearchTool(chunk_store=store), max_calls=2)])
    runner = ToolLoopRunner(llm_client=_DbSearchSuffixLLM(), tool_registry=registry, max_iterations=4)
    result = runner.run(
        messages=[{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    )
    assert result.answer == "final answer"
    assert result.tool_outputs[0]["name"] == "db_search"
    first_sermon = result.tool_outputs[0]["output"]["sermons"][0]
    assert first_sermon["ranges"][0]["paragraph_start"] == 1
