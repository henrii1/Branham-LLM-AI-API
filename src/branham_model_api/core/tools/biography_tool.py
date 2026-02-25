"""
Biography tool backed by local curated text.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


_DEFAULT_BIOGRAPHY_SOURCE_TITLE = "William Marion Branham - Life Boat Gospel Ministry"
_DEFAULT_BIOGRAPHY_SOURCE_URL = "https://lifeboatchurch.org/william-marion-branham/"

_SECTION_MARKER_PREFIX = "=== SECTION:"


@dataclass
class BiographyTool:
    """Simple local biography lookup tool."""

    file_path: Path
    # Keep tool output bounded to protect prompt size.
    # The backing file can be longer; we return a clipped excerpt.
    max_chars: int = 12000
    name: str = "biography_search"
    source_title: str = _DEFAULT_BIOGRAPHY_SOURCE_TITLE
    source_url: str = _DEFAULT_BIOGRAPHY_SOURCE_URL

    def _load_sections(self, text: str) -> tuple[list[str], dict[str, str]]:
        """
        Parse biography sections from the backing file.

        Format:
          === SECTION: <section_id> ===
          <content...>

        If no markers are present, treat the entire file as one section.
        """
        raw = (text or "").strip()
        if not raw:
            return ([], {})

        lines = raw.splitlines()
        order: list[str] = []
        sections: dict[str, list[str]] = {}

        current_id: str | None = None
        for ln in lines:
            stripped = ln.strip()
            if stripped.startswith(_SECTION_MARKER_PREFIX) and stripped.endswith("==="):
                # Example: "=== SECTION: early_life_and_calling ==="
                inside = stripped[len(_SECTION_MARKER_PREFIX) :].strip()
                inside = inside[:-3].strip() if inside.endswith("===") else inside
                section_id = inside.strip()
                current_id = section_id or None
                if current_id and current_id not in sections:
                    sections[current_id] = []
                    order.append(current_id)
                continue

            if current_id is None:
                # No section marker seen yet → accumulate into implicit first section.
                current_id = "section_1"
                if current_id not in sections:
                    sections[current_id] = []
                    order.append(current_id)

            sections[current_id].append(ln)

        rendered = {k: "\n".join(v).strip() for k, v in sections.items()}
        rendered = {k: v for k, v in rendered.items() if v}
        order = [k for k in order if k in rendered]
        return (order, rendered)

    def definition(self) -> dict[str, Any]:
        # Keep this list stable (prompt + tool schema teach the model how to choose).
        section_enum = [
            "early_life_and_calling",
            "healing_ministry_and_campaigns",
            "later_visions_and_death",
            "full",
        ]
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
                        ,
                        "section": {
                            "type": "string",
                            "enum": section_enum,
                            "description": (
                                "Optional section selector. Use when the question targets a timeframe/theme. "
                                "If omitted, the tool returns the first section by default. "
                                "Sections: early_life_and_calling (birth/call/early years), "
                                "healing_ministry_and_campaigns (commissioning/healing campaigns/miracle accounts), "
                                "later_visions_and_death (1962-1965 / seven seals / death), "
                                "full (entire biography)."
                            ),
                        },
                    },
                    "required": [],
                },
            },
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query", "")).strip()
        section = str(args.get("section", "")).strip() or None
        if not self.file_path.exists():
            return {"ok": False, "error": f"Biography file not found: {self.file_path}"}
        text = self.file_path.read_text(encoding="utf-8").strip()
        order, sections = self._load_sections(text)
        if not sections:
            return {"ok": False, "error": "Biography file is empty."}

        default_section = order[0] if order else next(iter(sections))
        selected_key = default_section
        if section and section != "full" and section in sections:
            selected_key = section

        if section == "full":
            content = text
            selected_key = "full"
        else:
            content = sections.get(selected_key, sections[default_section])

        if len(content) > self.max_chars:
            content = content[: self.max_chars].rstrip() + "\n..."
        return {
            "ok": True,
            "query": query or None,
            "section": selected_key,
            "available_sections": order,
            "content": content,
            "source_file": str(self.file_path),
            "sources": [
                {
                    "title": self.source_title,
                    "url": self.source_url,
                }
            ],
        }
