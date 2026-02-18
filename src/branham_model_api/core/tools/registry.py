"""
Tool registry with call-limit enforcement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ToolProtocol(Protocol):
    name: str

    def definition(self) -> dict[str, Any]:
        ...

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        ...


class ToolLimitError(Exception):
    """Raised when tool call limit is exceeded."""


@dataclass
class ToolSpec:
    tool: ToolProtocol
    max_calls: int


class ToolRegistry:
    """Registers tools and enforces per-request call limits."""

    def __init__(self, specs: list[ToolSpec], *, max_total_calls: int = 3) -> None:
        self._specs = {spec.tool.name: spec for spec in specs}
        self._counts: dict[str, int] = {name: 0 for name in self._specs}
        self._max_total_calls = max_total_calls
        self._total_calls = 0

    def reset_counts(self) -> None:
        for name in self._counts:
            self._counts[name] = 0
        self._total_calls = 0

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.tool.definition() for spec in self._specs.values()]

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        spec = self._specs.get(tool_name)
        if spec is None:
            return {"ok": False, "error": f"Unknown tool: {tool_name}"}

        if self._total_calls >= self._max_total_calls:
            raise ToolLimitError(
                f"Total tool call limit reached ({self._total_calls}/{self._max_total_calls}). "
                "You must now answer using the evidence you already have. Do not request more tools."
            )
        used = self._counts[tool_name]
        if used >= spec.max_calls:
            raise ToolLimitError(
                f"Call limit reached for {tool_name} ({used}/{spec.max_calls}). "
                "Answer using the evidence you already have. Do not call this tool again."
            )
        self._counts[tool_name] = used + 1
        self._total_calls += 1
        return spec.tool.execute(args)

    def limit_reached_output(
        self,
        *,
        tool_name: str,
        error: str,
    ) -> dict[str, Any]:
        """
        Structured tool output for limit-reached situations.

        This is returned to the model as a normal tool payload so the loop
        continues without hard-failing the turn.
        """
        return {
            "ok": False,
            "tool_limit_reached": True,
            "error": error,
            "tool_name": tool_name,
            "call_counts": self.call_counts(),
            "total_calls_used": self._total_calls,
            "max_total_calls": self._max_total_calls,
            "total_limit_reached": self.total_exhausted,
            "guidance": (
                "Tool-call limit has been reached. Continue the answer using the "
                "existing RAG context and all prior tool outputs. Do not request "
                "additional tool calls unless absolutely necessary."
            ),
        }

    @property
    def total_exhausted(self) -> bool:
        """True when total call limit is reached and no more tools can run."""
        return self._total_calls >= self._max_total_calls

    @property
    def max_total_calls(self) -> int:
        return self._max_total_calls

    @property
    def total_calls(self) -> int:
        return self._total_calls

    def call_counts(self) -> dict[str, int]:
        return dict(self._counts)
