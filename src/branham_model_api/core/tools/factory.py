"""
Factory helpers for default tool wiring.
"""

from __future__ import annotations

from pathlib import Path

from branham_model_api.core.tools.biography_tool import BiographyTool
from branham_model_api.core.tools.db_search_tool import DbSearchTool
from branham_model_api.core.tools.registry import ToolRegistry, ToolSpec
from branham_model_api.core.tools.serper_tool import SerperTool
from branham_model_api.retrieval.store.chunk_store import ChunkStore


def create_default_tool_registry(
    *,
    chunk_store: ChunkStore,
    biography_file_path: str | Path = "data/reference/biography.txt",
) -> ToolRegistry:
    """
    Build tool registry with project call limits.

    Total budget is enforced per tool-loop "round" (one assistant tool-call turn),
    not per individual tool call. This matches the latency/loop structure.

    Per-tool hard limits:
      - db_search: 3  (system prompt pushes batching → usually 1 call)
      - biography_search: 2  (system prompt targets 1)
      - internet_search: 2  (system prompt targets 1)
      - total rounds: 4 code / 3 system prompt target (soft)

    When limits are reached, ToolLimitError tells the LLM to answer
    with what it already has.
    """
    return ToolRegistry(
        [
            ToolSpec(tool=DbSearchTool(chunk_store=chunk_store), max_calls=3),
            ToolSpec(tool=BiographyTool(file_path=Path(biography_file_path)), max_calls=2),
            ToolSpec(tool=SerperTool(), max_calls=2),
        ],
        max_total_calls=4,
        # Soft limits: align with the system prompt's "prefer" guidance.
        # These do NOT block; they annotate tool outputs with `_soft_limits` when exceeded.
        soft_max_total_calls=3,
        soft_max_calls={
            "db_search": 2,
            "biography_search": 1,
            "internet_search": 1,
        },
    )
