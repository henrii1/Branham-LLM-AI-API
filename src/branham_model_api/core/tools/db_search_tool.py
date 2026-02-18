"""
DB Search tool for sermon paragraph lookup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from branham_model_api.retrieval.store.chunk_store import ChunkStore, ParagraphRecord


def _paragraph_payload(row: ParagraphRecord) -> dict[str, Any]:
    return {
        "paragraph_no": row.paragraph_no,
        "sub_id": row.sub_id,
        "text": row.text,
    }


def _invalid_range_payload(start: int, end: int) -> dict[str, int]:
    return {"from": min(start, end), "to": max(start, end)}


@dataclass
class DbSearchTool:
    """Tool for sermon paragraph and quote lookup."""

    chunk_store: ChunkStore
    max_paragraphs_per_query: int = 80
    default_sermon_head: int = 80

    name: str = "db_search"

    def definition(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": (
                    "Sermon paragraph retrieval tool with latency-aware batching. "
                    "Prefer one call using batch modes when possible. "
                    "Supports: read_sermon, read_paragraphs, search_quote_global, "
                    "search_quote_local, batch_read, and batch_mixed."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "mode": {
                            "type": "string",
                            "enum": [
                                "read_sermon",
                                "read_paragraphs",
                                "search_quote_global",
                                "search_quote_local",
                                "search_quote",
                                "batch_read",
                                "batch_mixed",
                            ],
                        },
                        "date_id": {"type": "string"},
                        "paragraph_start": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}],
                            "description": "Paragraph number; suffixes like 12a are accepted and normalized to 12.",
                        },
                        "paragraph_end": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}],
                            "description": "Paragraph number; suffixes like 12b are accepted and normalized to 12.",
                        },
                        "query": {"type": "string"},
                        "limit": {"type": "integer"},
                        "head": {
                            "type": "integer",
                            "description": "For read_sermon: number of paragraph rows to return (default 80).",
                        },
                        "requests": {
                            "type": "array",
                            "description": "For batch_read: list of paragraph ranges to fetch in one call.",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "date_id": {"type": "string"},
                                    "paragraph_start": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                    "paragraph_end": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                },
                                "required": ["date_id", "paragraph_start", "paragraph_end"],
                            },
                        },
                        "operations": {
                            "type": "array",
                            "description": (
                                "For batch_mixed: execute multiple db_search operations in one call "
                                "(e.g. read_paragraphs + search_quote_local)."
                            ),
                            "items": {
                                "type": "object",
                                "properties": {
                                    "mode": {
                                        "type": "string",
                                        "enum": [
                                            "read_sermon",
                                            "read_paragraphs",
                                            "search_quote_global",
                                            "search_quote_local",
                                            "search_quote",
                                            "batch_read",
                                        ],
                                    },
                                    "date_id": {"type": "string"},
                                    "paragraph_start": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                    "paragraph_end": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                    "query": {"type": "string"},
                                    "limit": {"type": "integer"},
                                    "head": {"type": "integer"},
                                    "requests": {"type": "array"},
                                },
                                "required": ["mode"],
                            },
                        },
                    },
                    "required": ["mode"],
                },
            },
        }

    def _parse_paragraph_no(self, raw: Any) -> int | None:
        """Parse paragraph number; tolerate suffix styles like '12a' or '¶12b'."""
        if isinstance(raw, int):
            return raw if raw > 0 else None
        text = str(raw or "").strip()
        if not text:
            return None
        match = re.search(r"(\d+)", text)
        if not match:
            return None
        value = int(match.group(1))
        return value if value > 0 else None

    def _group_result(
        self,
        *,
        date_id: str,
        paragraph_start: int,
        paragraph_end: int,
        paragraphs: list[ParagraphRecord],
        range_info: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        sermon = self.chunk_store.get_sermon(date_id)
        return {
            "date_id": date_id,
            "title": sermon.title if sermon else None,
            "ranges": [
                {
                    "paragraph_start": paragraph_start,
                    "paragraph_end": paragraph_end,
                    "paragraphs": [_paragraph_payload(p) for p in paragraphs],
                    "range_info": range_info,
                }
            ],
        }

    def _build_range_info(
        self,
        *,
        date_id: str,
        requested_start: int,
        requested_end: int,
    ) -> dict[str, Any]:
        bounds = self.chunk_store.get_paragraph_bounds(date_id)
        req_start = min(requested_start, requested_end)
        req_end = max(requested_start, requested_end)
        if bounds is None:
            return {
                "requested_start": req_start,
                "requested_end": req_end,
                "final_paragraph_no": None,
                "valid_start": None,
                "valid_end": None,
                "invalid_ranges": [_invalid_range_payload(req_start, req_end)],
                "has_invalid_request": True,
            }

        min_no, max_no = bounds
        valid_start = max(req_start, min_no)
        valid_end = min(req_end, max_no)
        invalid_ranges: list[dict[str, int]] = []
        if req_start < min_no:
            invalid_ranges.append(
                _invalid_range_payload(req_start, min(req_end, min_no - 1))
            )
        if req_end > max_no:
            invalid_ranges.append(
                _invalid_range_payload(max(req_start, max_no + 1), req_end)
            )
        if valid_start > valid_end:
            valid_start = None
            valid_end = None
        return {
            "requested_start": req_start,
            "requested_end": req_end,
            "final_paragraph_no": max_no,
            "valid_start": valid_start,
            "valid_end": valid_end,
            "invalid_ranges": invalid_ranges,
            "has_invalid_request": len(invalid_ranges) > 0 or valid_start is None or valid_end is None,
        }

    def execute(self, args: dict[str, Any]) -> dict[str, Any]:
        mode = str(args.get("mode", "")).strip()
        if mode == "read_sermon":
            date_id = str(args.get("date_id", "")).strip()
            if not date_id:
                return {"ok": False, "error": "date_id is required for read_sermon"}
            paragraphs = self.chunk_store.get_paragraphs_by_sermon(date_id)
            if not paragraphs:
                return {"ok": True, "mode": mode, "sermons": []}
            try:
                head = int(args.get("head", self.default_sermon_head))
            except (TypeError, ValueError):
                head = self.default_sermon_head
            head = max(1, min(head, self.max_paragraphs_per_query))
            paragraphs = paragraphs[:head]
            paragraph_nos = [p.paragraph_no for p in paragraphs]
            group = self._group_result(
                date_id=date_id,
                paragraph_start=min(paragraph_nos),
                paragraph_end=max(paragraph_nos),
                paragraphs=paragraphs,
                range_info=self._build_range_info(
                    date_id=date_id,
                    requested_start=min(paragraph_nos),
                    requested_end=max(paragraph_nos),
                ),
            )
            return {"ok": True, "mode": mode, "sermons": [group]}

        if mode == "read_paragraphs":
            date_id = str(args.get("date_id", "")).strip()
            start = self._parse_paragraph_no(args.get("paragraph_start"))
            end = self._parse_paragraph_no(args.get("paragraph_end", args.get("paragraph_start")))
            if not date_id or start is None or end is None:
                return {"ok": False, "error": "date_id, paragraph_start, paragraph_end are required"}
            paragraphs = self.chunk_store.get_paragraphs_by_range(date_id, start, end)[
                : self.max_paragraphs_per_query
            ]
            group = self._group_result(
                date_id=date_id,
                paragraph_start=min(start, end),
                paragraph_end=max(start, end),
                paragraphs=paragraphs,
                range_info=self._build_range_info(
                    date_id=date_id,
                    requested_start=start,
                    requested_end=end,
                ),
            )
            return {"ok": True, "mode": mode, "sermons": [group]}

        if mode in {"search_quote", "search_quote_global", "search_quote_local"}:
            query = str(args.get("query", "")).strip()
            date_id_raw = str(args.get("date_id", "")).strip()
            if mode == "search_quote_local":
                if not date_id_raw:
                    return {"ok": False, "error": "date_id is required for search_quote_local"}
                date_id = date_id_raw
            else:
                date_id = date_id_raw or None
            try:
                limit = int(args.get("limit", 10))
            except (TypeError, ValueError):
                limit = 10
            if not query:
                return {"ok": False, "error": "query is required for quote search"}
            row_limit = max(1, min(limit, self.max_paragraphs_per_query))
            rows = self.chunk_store.search_paragraphs_by_text(query, date_id=date_id, limit=row_limit * 3)
            grouped: dict[str, list[ParagraphRecord]] = {}
            for row in rows:
                grouped.setdefault(row.date_id, []).append(row)
            sermons: list[dict[str, Any]] = []
            for sid, sid_rows in grouped.items():
                selected_rows = sid_rows[:row_limit]
                if not selected_rows:
                    continue
                sermons.append(
                    self._group_result(
                        date_id=sid,
                        paragraph_start=min(r.paragraph_no for r in selected_rows),
                        paragraph_end=max(r.paragraph_no for r in selected_rows),
                        paragraphs=selected_rows,
                        range_info=self._build_range_info(
                            date_id=sid,
                            requested_start=min(r.paragraph_no for r in selected_rows),
                            requested_end=max(r.paragraph_no for r in selected_rows),
                        ),
                    )
                )
            return {"ok": True, "mode": mode, "sermons": sermons}

        if mode == "batch_read":
            requests = args.get("requests", []) or []
            if not isinstance(requests, list) or not requests:
                return {"ok": False, "error": "requests (non-empty list) is required for batch_read"}
            sermon_map: dict[str, dict[str, Any]] = {}
            total = 0
            for req in requests:
                date_id = str(req.get("date_id", "")).strip()
                start = self._parse_paragraph_no(req.get("paragraph_start"))
                end = self._parse_paragraph_no(req.get("paragraph_end", req.get("paragraph_start")))
                if not date_id or start is None or end is None:
                    continue
                rows = self.chunk_store.get_paragraphs_by_range(date_id, start, end)
                if total >= self.max_paragraphs_per_query:
                    break
                remaining = self.max_paragraphs_per_query - total
                rows = rows[:remaining]
                total += len(rows)

                sermon = sermon_map.get(date_id)
                if sermon is None:
                    meta = self.chunk_store.get_sermon(date_id)
                    sermon = {
                        "date_id": date_id,
                        "title": meta.title if meta else None,
                        "ranges": [],
                    }
                    sermon_map[date_id] = sermon
                sermon["ranges"].append(
                    {
                        "paragraph_start": min(start, end),
                        "paragraph_end": max(start, end),
                        "paragraphs": [_paragraph_payload(p) for p in rows],
                        "range_info": self._build_range_info(
                            date_id=date_id,
                            requested_start=start,
                            requested_end=end,
                        ),
                    }
                )
            return {
                "ok": True,
                "mode": mode,
                "sermons": list(sermon_map.values()),
            }

        if mode == "batch_mixed":
            operations = args.get("operations", []) or []
            if not isinstance(operations, list) or not operations:
                return {
                    "ok": False,
                    "error": "operations (non-empty list) is required for batch_mixed",
                }
            results: list[dict[str, Any]] = []
            for op in operations:
                if not isinstance(op, dict):
                    continue
                op_mode = str(op.get("mode", "")).strip()
                if not op_mode or op_mode == "batch_mixed":
                    continue
                results.append(self.execute(op))
            return {"ok": True, "mode": mode, "results": results}

        return {"ok": False, "error": f"Unsupported mode: {mode}"}
