"""
Internal tool-calling loop runner.
"""

from __future__ import annotations

import concurrent.futures
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


def _batched_db_search_outputs(
    *,
    tool_registry: ToolRegistry,
    tool_calls: list[Any],
) -> dict[str, dict[str, Any]]:
    """
    If the model requests multiple db_search tool calls in the same assistant turn,
    execute them as ONE batch_mixed call (counts once), and return per-tool_call_id
    outputs that wrap each operation's result.

    Returns: mapping tool_call_id -> output payload
    """
    db_calls: list[tuple[str, dict[str, Any]]] = []
    for tc in tool_calls:
        fn = getattr(tc, "function", None)
        if getattr(fn, "name", "") != "db_search":
            continue
        raw_args = getattr(fn, "arguments", "{}")
        try:
            parsed_args = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError:
            parsed_args = {}
        # If the model already requested a batch mode, do not wrap it again.
        # Nested batch_mixed would be skipped by the tool and cause result misalignment.
        if isinstance(parsed_args, dict) and parsed_args.get("mode") in {"batch_mixed", "batch_read"}:
            return {}
        db_calls.append((getattr(tc, "id", ""), parsed_args))

    if len(db_calls) <= 1:
        return {}

    operations = [args for _, args in db_calls]
    try:
        batch = tool_registry.execute_tool("db_search", {"mode": "batch_mixed", "operations": operations})
    except ToolLimitError as exc:
        # Hard limit reached: let caller handle normally by executing individually.
        return {}

    results = []
    if isinstance(batch, dict) and batch.get("ok") and batch.get("mode") == "batch_mixed":
        results = batch.get("results") or []
    out: dict[str, dict[str, Any]] = {}
    for i, (tcid, args) in enumerate(db_calls):
        result_i = results[i] if i < len(results) else {"ok": False, "error": "Missing batched result."}
        out[tcid] = {
            "ok": True,
            "batched": True,
            "batch_mode": "batch_mixed",
            "original_tool": "db_search",
            "original_mode": (args.get("mode") if isinstance(args, dict) else None),
            "result": result_i,
        }
    return out


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

            # Begin a single tool-call "round" (counts once for total budget).
            try:
                self.tool_registry.begin_tool_round()
            except ToolLimitError as exc:
                # Hard total limit reached: feed limit payloads back to the model for each requested call.
                for tc in tool_calls:
                    tool_name = getattr(getattr(tc, "function", None), "name", "")
                    output = self.tool_registry.limit_reached_output(
                        tool_name=tool_name,
                        error=str(exc),
                    )
                    tool_outputs.append({"name": tool_name, "output": output})
                    working_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": getattr(tc, "id", ""),
                            "name": tool_name,
                            "content": json.dumps(output, ensure_ascii=True),
                        }
                    )
                continue

            batched_db = _batched_db_search_outputs(
                tool_registry=self.tool_registry,
                tool_calls=tool_calls,
            )

            # Reserve counts synchronously, execute tools in parallel.
            tasks: list[tuple[Any, str, dict[str, Any] | None, tuple[Any, dict[str, Any], list[dict[str, Any]]] | None, dict[str, Any] | None]] = []
            # Each entry: (tc, tool_name, parsed_args, prepared(spec,args,soft), immediate_output)
            for tc in tool_calls:
                tool_name = getattr(getattr(tc, "function", None), "name", "")
                raw_args = getattr(getattr(tc, "function", None), "arguments", "{}")
                try:
                    parsed_args = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    parsed_args = {}

                tcid = getattr(tc, "id", "")
                if tcid and tcid in batched_db:
                    tasks.append((tc, tool_name, None, None, batched_db[tcid]))
                    continue
                if tool_name == "db_search" and batched_db:
                    # Already served via the batch call.
                    continue

                try:
                    prepared = self.tool_registry.prepare_tool_execution(tool_name, parsed_args)
                    tasks.append((tc, tool_name, None, prepared, None))
                except ToolLimitError as exc:
                    immediate = self.tool_registry.limit_reached_output(
                        tool_name=tool_name,
                        error=str(exc),
                    )
                    tasks.append((tc, tool_name, None, None, immediate))

            futures: dict[str, concurrent.futures.Future[dict[str, Any]]] = {}
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(tasks)))) as ex:
                for tc, tool_name, _, prepared, immediate in tasks:
                    tcid = getattr(tc, "id", "")
                    if immediate is not None or prepared is None:
                        continue
                    spec, args2, soft_events = prepared

                    def _run(spec=spec, args2=args2, soft_events=soft_events) -> dict[str, Any]:
                        try:
                            out = spec.tool.execute(args2)
                        except Exception as exc:
                            out = {"ok": False, "error": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                        if soft_events and isinstance(out, dict):
                            out["_soft_limits"] = soft_events
                        return out

                    if tcid:
                        futures[tcid] = ex.submit(_run)

            # Collect in original order
            for tc, tool_name, _, prepared, immediate in tasks:
                tcid = getattr(tc, "id", "")
                if immediate is not None:
                    output = immediate
                elif tcid and tcid in futures:
                    try:
                        output = futures[tcid].result()
                    except Exception as exc:
                        output = {"ok": False, "error": f"Tool execution failed: {type(exc).__name__}: {exc}"}
                else:
                    continue

                if output.get("external"):
                    external_used = True
                tool_outputs.append({"name": tool_name, "output": output})
                working_messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tcid,
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
