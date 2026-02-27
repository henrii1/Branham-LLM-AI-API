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

    def __init__(
        self,
        specs: list[ToolSpec],
        *,
        max_total_calls: int = 3,
        soft_max_total_calls: int | None = None,
        soft_max_calls: dict[str, int] | None = None,
    ) -> None:
        self._specs = {spec.tool.name: spec for spec in specs}
        self._counts: dict[str, int] = {name: 0 for name in self._specs}
        self._max_total_calls = max_total_calls
        self._soft_max_total_calls = soft_max_total_calls
        self._soft_max_calls = dict(soft_max_calls or {})
        # Total call budget is enforced per tool-loop "round" (per assistant tool-call turn),
        # not per individual tool call, to reflect latency/loop structure.
        self._total_calls = 0
        self._active_round_soft_events: list[dict[str, Any]] = []

    def reset_counts(self) -> None:
        for name in self._counts:
            self._counts[name] = 0
        self._total_calls = 0
        self._active_round_soft_events = []

    def definitions(self) -> list[dict[str, Any]]:
        return [spec.tool.definition() for spec in self._specs.values()]

    def begin_tool_round(self) -> None:
        """
        Begin a single tool-call round (one assistant turn containing one-or-many tool calls).

        This increments the total budget exactly once per round.
        """
        if self._total_calls >= self._max_total_calls:
            raise ToolLimitError(
                f"Total tool-call rounds limit reached ({self._total_calls}/{self._max_total_calls}). "
                "You must now answer using the evidence you already have. Do not request more tools."
            )
        self._total_calls += 1
        self._active_round_soft_events = []
        if self._soft_max_total_calls is not None and self._total_calls > self._soft_max_total_calls:
            self._active_round_soft_events.append(
                {
                    "kind": "soft_total_limit_exceeded",
                    "soft_limit": self._soft_max_total_calls,
                    "total_rounds_used": self._total_calls,
                    "note": "Preferred total tool-call rounds budget exceeded (soft). Continue only if necessary.",
                }
            )

    def _reserve_tool_call(self, tool_name: str) -> tuple[ToolSpec, list[dict[str, Any]]]:
        """
        Reserve a single tool call within the current tool-round.

        This increments per-tool call counts and returns the ToolSpec + any soft-limit
        events that should be annotated onto the eventual tool output.
        """
        spec = self._specs.get(tool_name)
        if spec is None:
            raise ToolLimitError(f"Unknown tool: {tool_name}")
        # Soft limits (do not block): annotate outputs when exceeded.
        # Total soft limit is tracked per round (via begin_tool_round()).
        soft_events: list[dict[str, Any]] = list(self._active_round_soft_events)
        would_tool = self._counts.get(tool_name, 0) + 1
        soft_tool_limit = self._soft_max_calls.get(tool_name)
        if soft_tool_limit is not None and would_tool > soft_tool_limit:
            soft_events.append(
                {
                    "kind": "soft_tool_limit_exceeded",
                    "tool_name": tool_name,
                    "soft_limit": soft_tool_limit,
                    "would_tool_calls": would_tool,
                    "note": "Preferred per-tool budget exceeded (soft). Batch/reduce tool calls when possible.",
                }
            )

        used = self._counts[tool_name]
        if used >= spec.max_calls:
            raise ToolLimitError(
                f"Call limit reached for {tool_name} ({used}/{spec.max_calls}). "
                "Answer using the evidence you already have. Do not call this tool again."
            )
        self._counts[tool_name] = used + 1
        return spec, soft_events

    def prepare_tool_execution(self, tool_name: str, args: dict[str, Any]) -> tuple[ToolSpec, dict[str, Any], list[dict[str, Any]]]:
        """
        Prepare a tool execution for parallel running.

        - Reserves budget/counters synchronously (safe).
        - Returns the ToolSpec + args + soft events to attach to output.
        """
        spec, soft_events = self._reserve_tool_call(tool_name)
        return spec, args, soft_events

    def execute_tool(self, tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
        """
        Execute tool synchronously (reserves counts + runs tool).

        For parallel execution, use prepare_tool_execution() and run spec.tool.execute(args)
        in a threadpool.
        """
        spec, args2, soft_events = self.prepare_tool_execution(tool_name, args)
        output = spec.tool.execute(args2)
        if soft_events and isinstance(output, dict):
            output["_soft_limits"] = soft_events
        return output

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
