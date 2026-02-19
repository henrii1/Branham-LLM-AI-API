"""
Internal tool-calling loop runner.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from branham_model_api.core.tools.registry import ToolLimitError, ToolRegistry


class ToolLoopError(Exception):
    """Raised when tool loop cannot complete successfully."""


@dataclass
class ToolLoopResult:
    """Final result from internal tool loop."""

    answer: str
    messages: list[dict[str, Any]]
    tool_outputs: list[dict[str, Any]]
    external_used: bool
    llm_traces: list[dict[str, Any]] = field(default_factory=list)
    prefinal_messages: list[dict[str, Any]] = field(default_factory=list)


def _tool_call_to_dict(tool_call: Any) -> dict[str, Any]:
    fn = getattr(tool_call, "function", None)
    return {
        "id": getattr(tool_call, "id", ""),
        "type": getattr(tool_call, "type", "function"),
        "function": {
            "name": getattr(fn, "name", ""),
            "arguments": getattr(fn, "arguments", "{}"),
        },
    }


class ToolLoopRunner:
    """
    Handles server-internal LLM tool loop until final assistant content.
    """

    def __init__(
        self,
        *,
        llm_client: Any,
        tool_registry: ToolRegistry,
        max_iterations: int = 5,
    ) -> None:
        self.llm_client = llm_client
        self.tool_registry = tool_registry
        self.max_iterations = max_iterations

    def run(
        self,
        messages: list[dict[str, Any]],
    ) -> ToolLoopResult:
        working_messages = list(messages)
        tool_outputs: list[dict[str, Any]] = []
        external_used = False
        llm_traces: list[dict[str, Any]] = []
        last_content = ""
        self.tool_registry.reset_counts()

        for iteration in range(self.max_iterations):
            offer_tools = (
                self.tool_registry.definitions()
                if not self.tool_registry.total_exhausted
                else None
            )
            request_snapshot = {
                "messages": working_messages,
                "tools": offer_tools,
            }
            response = self.llm_client.completion(
                messages=working_messages,
                tools=offer_tools,
                tool_choice="auto" if offer_tools else None,
            )
            choice = response.choices[0]
            message = choice.message
            tool_calls = getattr(message, "tool_calls", None) or []
            content = (getattr(message, "content", None) or "").strip()
            if content:
                last_content = content
            llm_traces.append(
                {
                    "request": request_snapshot,
                    "response": {
                        "content": content,
                        "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
                    },
                }
            )

            if not tool_calls:
                return ToolLoopResult(
                    answer=content,
                    messages=working_messages,
                    tool_outputs=tool_outputs,
                    external_used=external_used,
                    llm_traces=llm_traces,
                    prefinal_messages=list(working_messages),
                )

            assistant_message = {
                "role": "assistant",
                # Keep tool-call assistant turns compact; only tool_calls are needed.
                "content": None,
                "tool_calls": [_tool_call_to_dict(tc) for tc in tool_calls],
            }
            working_messages.append(assistant_message)

            for tc in tool_calls:
                tool_name = getattr(getattr(tc, "function", None), "name", "")
                raw_args = getattr(getattr(tc, "function", None), "arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {}
                try:
                    output = self.tool_registry.execute_tool(tool_name, parsed_args)
                except ToolLimitError as exc:
                    output = self.tool_registry.limit_reached_output(
                        tool_name=tool_name,
                        error=str(exc),
                    )

                if output.get("external"):
                    external_used = True
                tool_outputs.append({"name": tool_name, "output": output})
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": getattr(tc, "id", ""),
                        "name": tool_name,
                        "content": json.dumps(output, ensure_ascii=True),
                    }
                )

        return ToolLoopResult(
            answer=last_content,
            messages=working_messages,
            tool_outputs=tool_outputs,
            external_used=external_used,
            llm_traces=llm_traces,
            prefinal_messages=list(working_messages),
        )
