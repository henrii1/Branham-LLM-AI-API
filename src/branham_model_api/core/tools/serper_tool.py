"""
Serper web search tool.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import requests


@dataclass
class SerperTool:
    """Minimal Serper wrapper for controlled external search."""

    timeout_seconds: float = 10.0
    endpoint: str = "https://google.serper.dev/search"
    name: str = "internet_search"

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Search the public web for external/current information. "
                    "Use only when needed and treat output as unverified."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        api_key = os.getenv("SERPER_API_KEY", "").strip()
        if not api_key:
            return {"ok": False, "error": "SERPER_API_KEY not configured"}

        query = str(args.get("query", "")).strip()
        if not query:
            return {"ok": False, "error": "query is required"}
        num_results = int(args.get("num_results", 5))
        num_results = max(1, min(num_results, 10))

        try:
            response = requests.post(
                self.endpoint,
                headers={
                    "X-API-KEY": api_key,
                    "Content-Type": "application/json",
                },
                json={"q": query, "num": num_results},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return {"ok": False, "error": f"Serper request failed: {exc}"}

        organic = payload.get("organic", [])[:num_results]
        sources = []
        snippets = []
        for item in organic:
            link = item.get("link")
            title = item.get("title")
            snippet = item.get("snippet")
            if link:
                sources.append({"title": title, "url": link})
            if snippet:
                snippets.append(snippet)

        return {
            "ok": True,
            "query": query,
            "external": True,
            "disclaimer": "Unverified external search results.",
            "sources": sources,
            "snippets": snippets,
        }
