"""
Biography tool backed by local curated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class BiographyTool:
    """Simple local biography lookup tool."""

    file_path: Path
    max_chars: int = 4000
    name: str = "biography_search"

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": "Read curated biography information about William Branham.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Optional focus hint for biography lookup.",
                        }
                    },
                    "required": [],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        if not self.file_path.exists():
            return {"ok": False, "error": f"Biography file not found: {self.file_path}"}
        text = self.file_path.read_text(encoding="utf-8").strip()
        if len(text) > self.max_chars:
            text = text[: self.max_chars].rstrip() + "\n..."
        return {
            "ok": True,
            "query": query or None,
            "content": text,
            "source": str(self.file_path),
        }
