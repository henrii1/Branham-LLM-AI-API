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
                        "title_query": {
                            "type": "string",
                            "description": (
                                "Optional fuzzy sermon title query (SQLite LIKE NOCASE). "
                                "Use when you have the sermon name but not the date_id, e.g. 'A Paradox'."
                            ),
                        },
                        "year": {
                            "oneOf": [{"type": "integer"}, {"type": "string"}],
                            "description": (
                                "Optional year hint to disambiguate title matches. "
                                "Accepts 63 or 1963 (filters date_id by YY- prefix)."
                            ),
                        },
                        "date_id_hint": {
                            "type": "string",
                            "description": (
                                "Optional date_id prefix hint to disambiguate title matches "
                                "(e.g. '63-0801')."
                            ),
                        },
                        "context_delta": {
                            "type": "integer",
                            "description": (
                                "For single-paragraph reads: fetch ±delta paragraphs around paragraph_start "
                                "(default 2 when paragraph_end is omitted)."
                            ),
                        },
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
                                    "title_query": {"type": "string"},
                                    "year": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                                    "date_id_hint": {"type": "string"},
                                    "paragraph_start": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                    "paragraph_end": {
                                        "oneOf": [{"type": "integer"}, {"type": "string"}],
                                    },
                                },
                                "required": ["paragraph_start", "paragraph_end"],
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
                                    "title_query": {"type": "string"},
                                    "year": {"oneOf": [{"type": "integer"}, {"type": "string"}]},
                                    "date_id_hint": {"type": "string"},
                                    "context_delta": {"type": "integer"},
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

    def _resolve_date_ids(
        self, args: dict[str, Any]
    ) -> tuple[list[str], dict[str, Any] | None]:
        """
        Resolve one-or-many date_ids.

        Key behavior:
        - If args.date_id is an exact sermon id: return [date_id].
        - If args.date_id is NOT exact: treat it as a prefix and return ALL sermons
          whose date_id starts with that prefix (e.g. "63-1201" -> ["63-1201M","63-1201E"]).
        - If prefix yields no candidates, fall back to title_query resolution when present.
        """
        date_id_raw = str(args.get("date_id", "")).strip()
        if date_id_raw:
            sermon = self.chunk_store.get_sermon(date_id_raw)
            if sermon is not None:
                return [date_id_raw], {
                    "resolved_by": "date_id",
                    "date_ids": [date_id_raw],
                    "candidates": [],
                }

            candidates = self.chunk_store.search_sermons_by_date_id_prefix(
                date_id_raw, limit=10
            )
            if candidates:
                date_ids = [c.date_id for c in candidates]
                return date_ids, {
                    "resolved_by": "date_id_prefix_all",
                    "date_id_prefix": date_id_raw,
                    "date_ids": date_ids,
                    "candidates": [{"date_id": c.date_id, "title": c.title} for c in candidates],
                }

            # Prefix resolution failed; fall back to title query (human-friendly)
            title_present = bool(
                str(
                    args.get("title_query")
                    or args.get("sermon_title")
                    or args.get("sermon_name")
                    or args.get("title")
                    or ""
                ).strip()
            )
            if title_present:
                args2 = dict(args)
                args2.pop("date_id", None)
                date_id_one, title_resolution = self._resolve_date_id(args2)
                if date_id_one:
                    debug = title_resolution or {}
                    debug["unknown_date_id"] = date_id_raw
                    debug["resolved_by"] = "title_query_fallback"
                    debug["date_ids"] = [date_id_one]
                    return [date_id_one], debug
                debug = title_resolution or {}
                debug["unknown_date_id"] = date_id_raw
                debug["resolved_by"] = "title_query_fallback"
                debug["date_ids"] = []
                debug["resolution_failed"] = True
                return [], debug

            return [], {
                "resolved_by": "date_id_prefix_all",
                "date_id_prefix": date_id_raw,
                "date_ids": [],
                "candidates": [],
                "resolution_failed": True,
            }

        # No date_id provided; use existing title_query resolver (single date_id).
        date_id_one, resolution = self._resolve_date_id(args)
        if not date_id_one:
            return [], resolution
        debug = resolution or {}
        debug["date_ids"] = [date_id_one]
        return [date_id_one], debug

    def _resolve_date_id(
        self, args: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any] | None]:
        """
        Resolve date_id from either explicit date_id or fuzzy title_query (+ optional year/date_id_hint).

        Returns:
          (date_id or None, resolution_debug or None)
        """
        date_id = str(args.get("date_id", "")).strip()
        if date_id:
            # Validate exact date_id first (prevents silent empties on partial/typo IDs).
            sermon = self.chunk_store.get_sermon(date_id)
            if sermon is not None:
                return date_id, {"resolved_by": "date_id", "candidates": []}

            # If user also provided title metadata, prefer resolving by title instead of
            # trusting an unknown/partial date_id.
            title_present = bool(
                str(
                    args.get("title_query")
                    or args.get("sermon_title")
                    or args.get("sermon_name")
                    or args.get("title")
                    or ""
                ).strip()
            )
            if title_present:
                # Continue to title resolution below, but record what happened.
                args = dict(args)
                args["_date_id_was_unknown"] = date_id
            else:
                # Try prefix resolution (e.g. "63-1201" -> "63-1201M"/"63-1201E").
                candidates = self.chunk_store.search_sermons_by_date_id_prefix(date_id, limit=10)
                debug: dict[str, Any] = {
                    "resolved_by": "date_id_prefix",
                    "date_id_prefix": date_id,
                    "candidates": [{"date_id": c.date_id, "title": c.title} for c in candidates],
                }
                if not candidates:
                    debug["resolution_failed"] = True
                    return None, debug
                if len(candidates) == 1:
                    debug["picked_reason"] = "single_prefix_match"
                    debug["picked_date_id"] = candidates[0].date_id
                    return candidates[0].date_id, debug
                picked = sorted(candidates, key=lambda c: c.date_id)[-1]
                debug["picked_reason"] = "latest_date_id"
                debug["picked_date_id"] = picked.date_id
                return picked.date_id, debug

        title_query = (
            str(
                args.get("title_query")
                or args.get("sermon_title")
                or args.get("sermon_name")
                or args.get("title")
                or ""
            ).strip()
        )
        if not title_query:
            return None, None

        year = args.get("year")
        date_id_hint = str(args.get("date_id_hint", "")).strip() or None
        candidates = self.chunk_store.search_sermons_by_title(
            title_query, year=year, date_id_hint=date_id_hint, limit=10
        )
        debug: dict[str, Any] = {
            "resolved_by": "title_query",
            "unknown_date_id": args.get("_date_id_was_unknown"),
            "title_query": title_query,
            "year": year,
            "date_id_hint": date_id_hint,
            "candidates": [{"date_id": c.date_id, "title": c.title} for c in candidates],
        }
        if not candidates:
            return None, debug
        if len(candidates) == 1:
            return candidates[0].date_id, debug

        tq_norm = title_query.casefold().strip()
        exact = [c for c in candidates if (c.title or "").casefold().strip() == tq_norm]
        if len(exact) == 1:
            debug["picked_reason"] = "exact_title_match"
            debug["picked_date_id"] = exact[0].date_id
            return exact[0].date_id, debug

        picked = sorted(candidates, key=lambda c: c.date_id)[-1]
        debug["picked_reason"] = "latest_date_id"
        debug["picked_date_id"] = picked.date_id
        return picked.date_id, debug

    def _context_range_for_single_paragraph(
        self,
        *,
        date_id: str,
        paragraph_no: int,
        delta: int,
    ) -> tuple[int, int]:
        bounds = self.chunk_store.get_paragraph_bounds(date_id)
        start = max(1, paragraph_no - delta)
        end = paragraph_no + delta
        if bounds is not None:
            min_no, max_no = bounds
            start = max(start, min_no)
            end = min(end, max_no)
        return start, end

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
            date_ids, resolution = self._resolve_date_ids(args)
            if not date_ids:
                return {
                    "ok": False,
                    "error": "date_id or title_query is required for read_sermon",
                    "resolution": resolution,
                }
            try:
                head = int(args.get("head", self.default_sermon_head))
            except (TypeError, ValueError):
                head = self.default_sermon_head
            head = max(1, min(head, self.max_paragraphs_per_query))

            sermons: list[dict[str, Any]] = []
            if len(date_ids) == 1:
                date_id = date_ids[0]
                paragraphs = self.chunk_store.get_paragraphs_by_sermon(date_id)
                if not paragraphs:
                    return {
                        "ok": True,
                        "mode": mode,
                        "sermons": [],
                        "resolution": resolution,
                    }
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
                sermons = [group]
                return {"ok": True, "mode": mode, "sermons": sermons, "resolution": resolution}

            # Multiple suffix sermons for a day prefix: split head across all.
            n = len(date_ids)
            per_sermon = max(1, head // n)
            for date_id in date_ids:
                paragraphs = self.chunk_store.get_paragraphs_by_sermon(date_id)[:per_sermon]
                if not paragraphs:
                    continue
                paragraph_nos = [p.paragraph_no for p in paragraphs]
                sermons.append(
                    self._group_result(
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
                )
            return {
                "ok": True,
                "mode": mode,
                "sermons": sermons,
                "resolution": resolution,
                "head_split": {
                    "requested_head": head,
                    "sermon_count": n,
                    "head_per_sermon": per_sermon,
                },
            }

        if mode == "read_paragraphs":
            date_ids, resolution = self._resolve_date_ids(args)
            start = self._parse_paragraph_no(args.get("paragraph_start"))
            end = self._parse_paragraph_no(args.get("paragraph_end"))
            if not date_ids or start is None:
                return {
                    "ok": False,
                    "error": "date_id/title_query and paragraph_start are required",
                    "resolution": resolution,
                }

            try:
                delta = int(args.get("context_delta", 2))
            except (TypeError, ValueError):
                delta = 2
            delta = max(0, min(delta, 10))

            n = len(date_ids)
            per_sermon_limit = (
                self.max_paragraphs_per_query
                if n <= 1
                else max(1, self.max_paragraphs_per_query // n)
            )

            sermons: list[dict[str, Any]] = []
            for date_id in date_ids:
                s = start
                e = end
                if e is None:
                    # Single-paragraph reads: compute bounds per sermon.
                    if delta > 0:
                        s, e = self._context_range_for_single_paragraph(
                            date_id=date_id,
                            paragraph_no=s,
                            delta=delta,
                        )
                    else:
                        e = s
                assert e is not None
                paragraphs = self.chunk_store.get_paragraphs_by_range(date_id, s, e)[
                    :per_sermon_limit
                ]
                sermons.append(
                    self._group_result(
                        date_id=date_id,
                        paragraph_start=min(s, e),
                        paragraph_end=max(s, e),
                        paragraphs=paragraphs,
                        range_info=self._build_range_info(
                            date_id=date_id,
                            requested_start=s,
                            requested_end=e,
                        ),
                    )
                )
            out: dict[str, Any] = {"ok": True, "mode": mode, "sermons": sermons}
            if resolution is not None:
                out["resolution"] = resolution
            if n > 1:
                out["multi_sermon"] = {
                    "sermon_count": n,
                    "paragraph_limit_per_sermon": per_sermon_limit,
                }
            return out

        if mode in {"search_quote", "search_quote_global", "search_quote_local"}:
            query = str(args.get("query", "")).strip()
            date_id_raw = str(args.get("date_id", "")).strip()
            resolution = None
            date_ids: list[str] = []
            if mode == "search_quote_local":
                date_ids, resolution = self._resolve_date_ids(args)
                if not date_ids:
                    return {
                        "ok": False,
                        "error": "date_id or title_query is required for search_quote_local",
                        "resolution": resolution,
                    }
            else:
                # Optional narrowing: if title_query resolves, treat as local search.
                if date_id_raw:
                    date_ids, resolution = self._resolve_date_ids(args)
                    if not date_ids:
                        return {
                            "ok": False,
                            "error": "Unknown date_id (no prefix matches)",
                            "resolution": resolution,
                        }
                else:
                    date_ids, resolution = self._resolve_date_ids(args)
            try:
                limit = int(args.get("limit", 10))
            except (TypeError, ValueError):
                limit = 10
            if not query:
                return {"ok": False, "error": "query is required for quote search"}
            row_limit = max(1, min(limit, self.max_paragraphs_per_query))
            if date_ids:
                if len(date_ids) == 1:
                    rows = self.chunk_store.search_paragraphs_by_text(
                        query, date_id=date_ids[0], limit=row_limit * 3
                    )
                else:
                    rows = self.chunk_store.search_paragraphs_by_text_many(
                        query, date_ids=date_ids, limit=row_limit * 3
                    )
            else:
                rows = self.chunk_store.search_paragraphs_by_text(query, date_id=None, limit=row_limit * 3)
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
            out: dict[str, Any] = {"ok": True, "mode": mode, "sermons": sermons}
            if resolution:
                out["resolution"] = resolution
            return out

        if mode == "batch_read":
            requests = args.get("requests", []) or []
            if not isinstance(requests, list) or not requests:
                return {"ok": False, "error": "requests (non-empty list) is required for batch_read"}
            sermon_map: dict[str, dict[str, Any]] = {}
            total = 0
            skipped: list[dict[str, Any]] = []
            for req in requests:
                if not isinstance(req, dict):
                    continue
                date_ids, resolution = self._resolve_date_ids(req)
                start = self._parse_paragraph_no(req.get("paragraph_start"))
                end = self._parse_paragraph_no(req.get("paragraph_end", req.get("paragraph_start")))
                if not date_ids or start is None or end is None:
                    skipped.append(
                        {
                            "request": req,
                            "error": "could_not_resolve_date_id_or_range",
                            "resolution": resolution,
                        }
                    )
                    continue
                for date_id in date_ids:
                    if total >= self.max_paragraphs_per_query:
                        break
                    rows = self.chunk_store.get_paragraphs_by_range(date_id, start, end)
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
                            "resolution": resolution,
                        }
                    )
            return {
                "ok": True,
                "mode": mode,
                "sermons": list(sermon_map.values()),
                "skipped": skipped,
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
