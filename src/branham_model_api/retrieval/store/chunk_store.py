"""
SQLite chunk store for text lookup.

Provides efficient lookup of chunks by:
- chunk_id (single chunk)
- date_id (all chunks from a sermon)
- chunk_id list (batch lookup)
- Adjacent chunks for expansion (±N)

Tables used (from DATA_FORMAT.md):
- chunks: chunk_id, date_id, paragraph_start, paragraph_end, chunk_index, text, word_count, char_count, is_tail_chunk
- sermons: date_id, title, source, language
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence
import re


@dataclass(frozen=True)
class ChunkRecord:
    """A chunk record from the database."""

    chunk_id: str
    date_id: str
    paragraph_start: int
    paragraph_end: int
    chunk_index: int
    text: str
    word_count: int
    char_count: int
    is_tail_chunk: bool


@dataclass(frozen=True)
class SermonRecord:
    """A sermon metadata record."""

    date_id: str
    title: str | None
    source: str | None
    language: str


@dataclass(frozen=True)
class ParagraphRecord:
    """A raw paragraph record from the canonical paragraphs table."""

    date_id: str
    paragraph_no: int
    sub_id: str
    text: str


class ChunkStore:
    """
    SQLite-backed chunk store for text retrieval.

    Thread-safe via check_same_thread=False and read-only queries.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Chunk database not found: {self.db_path}")

        # Open connection with WAL mode for concurrent reads
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=30.0,
            check_same_thread=False,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode = WAL;")
        self._conn.execute("PRAGMA synchronous = NORMAL;")
        self._conn.execute("PRAGMA cache_size = -64000;")  # 64MB cache
        self._has_is_tail_chunk = self._detect_column("chunks", "is_tail_chunk")

    def _detect_column(self, table: str, column: str) -> bool:
        cur = self._conn.execute(f"PRAGMA table_info({table})")
        return any(row["name"] == column for row in cur.fetchall())

    def _chunk_tail_sql(self, *, table_alias: str = "") -> str:
        prefix = f"{table_alias}." if table_alias else ""
        if self._has_is_tail_chunk:
            return f"COALESCE({prefix}is_tail_chunk, 0) AS is_tail_chunk"
        # Fallback inference when migration has not been applied yet.
        return (
            "CASE WHEN "
            f"{prefix}chunk_index = (SELECT MAX(c2.chunk_index) FROM chunks c2 WHERE c2.date_id = {prefix}date_id) "
            "THEN 1 ELSE 0 END AS is_tail_chunk"
        )

    def _chunk_record_from_row(self, row: sqlite3.Row) -> ChunkRecord:
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            date_id=row["date_id"],
            paragraph_start=row["paragraph_start"],
            paragraph_end=row["paragraph_end"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            word_count=row["word_count"],
            char_count=row["char_count"],
            is_tail_chunk=bool(row["is_tail_chunk"]),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self) -> "ChunkStore":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def get_chunk(self, chunk_id: str) -> ChunkRecord | None:
        """Get a single chunk by ID."""
        cur = self._conn.execute(
            f"""
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count,
                   {self._chunk_tail_sql()}
            FROM chunks
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._chunk_record_from_row(row)

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, ChunkRecord]:
        """
        Batch lookup of chunks by IDs.

        Returns:
            Dict mapping chunk_id → ChunkRecord (missing IDs are omitted)
        """
        if not chunk_ids:
            return {}

        placeholders = ",".join("?" for _ in chunk_ids)
        cur = self._conn.execute(
            f"""
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count,
                   {self._chunk_tail_sql()}
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            tuple(chunk_ids),
        )

        result: dict[str, ChunkRecord] = {}
        for row in cur.fetchall():
            result[row["chunk_id"]] = self._chunk_record_from_row(row)
        return result

    def get_chunks_by_sermon(self, date_id: str) -> list[ChunkRecord]:
        """Get all chunks from a sermon, ordered by chunk_index."""
        cur = self._conn.execute(
            f"""
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count,
                   {self._chunk_tail_sql()}
            FROM chunks
            WHERE date_id = ?
            ORDER BY chunk_index ASC
            """,
            (date_id,),
        )
        return [self._chunk_record_from_row(row) for row in cur.fetchall()]

    def get_chunks_by_paragraph_range(
        self,
        date_id: str,
        paragraph_start: int,
        paragraph_end: int,
    ) -> list[ChunkRecord]:
        """
        Get chunks overlapping a paragraph range within one sermon.

        Overlap condition:
          chunk.paragraph_end >= paragraph_start
          chunk.paragraph_start <= paragraph_end
        """
        start = min(paragraph_start, paragraph_end)
        end = max(paragraph_start, paragraph_end)
        cur = self._conn.execute(
            f"""
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count,
                   {self._chunk_tail_sql()}
            FROM chunks
            WHERE date_id = ?
              AND paragraph_end >= ?
              AND paragraph_start <= ?
            ORDER BY chunk_index ASC
            """,
            (date_id, start, end),
        )
        return [self._chunk_record_from_row(row) for row in cur.fetchall()]

    def get_paragraphs_by_sermon(self, date_id: str) -> list[ParagraphRecord]:
        """Get all canonical paragraphs for a sermon ordered by paragraph_no + sub_id."""
        cur = self._conn.execute(
            """
            SELECT date_id, paragraph_no, sub_id, text
            FROM paragraphs
            WHERE date_id = ?
            ORDER BY
              paragraph_no ASC,
              CASE WHEN sub_id = '' THEN 0 ELSE 1 END ASC,
              sub_id ASC
            """,
            (date_id,),
        )
        return [
            ParagraphRecord(
                date_id=row["date_id"],
                paragraph_no=int(row["paragraph_no"]),
                sub_id=row["sub_id"] or "",
                text=row["text"] or "",
            )
            for row in cur.fetchall()
        ]

    def get_paragraphs_by_range(
        self,
        date_id: str,
        paragraph_start: int,
        paragraph_end: int,
    ) -> list[ParagraphRecord]:
        """Get canonical paragraphs overlapping a paragraph number range."""
        start = min(paragraph_start, paragraph_end)
        end = max(paragraph_start, paragraph_end)
        cur = self._conn.execute(
            """
            SELECT date_id, paragraph_no, sub_id, text
            FROM paragraphs
            WHERE date_id = ?
              AND paragraph_no >= ?
              AND paragraph_no <= ?
            ORDER BY
              paragraph_no ASC,
              CASE WHEN sub_id = '' THEN 0 ELSE 1 END ASC,
              sub_id ASC
            """,
            (date_id, start, end),
        )
        return [
            ParagraphRecord(
                date_id=row["date_id"],
                paragraph_no=int(row["paragraph_no"]),
                sub_id=row["sub_id"] or "",
                text=row["text"] or "",
            )
            for row in cur.fetchall()
        ]

    def search_paragraphs_by_text(
        self,
        query: str,
        *,
        date_id: str | None = None,
        limit: int = 40,
    ) -> list[ParagraphRecord]:
        """
        Search canonical paragraphs by substring.

        Returns raw paragraph rows (including sub_id parts) so callers can merge
        split paragraph parts (e.g., 12a/12b) back to canonical paragraph 12.
        """
        q = f"%{query}%"
        if date_id:
            cur = self._conn.execute(
                """
                SELECT date_id, paragraph_no, sub_id, text
                FROM paragraphs
                WHERE date_id = ? AND text LIKE ?
                ORDER BY
                  paragraph_no ASC,
                  CASE WHEN sub_id = '' THEN 0 ELSE 1 END ASC,
                  sub_id ASC
                LIMIT ?
                """,
                (date_id, q, limit),
            )
        else:
            cur = self._conn.execute(
                """
                SELECT date_id, paragraph_no, sub_id, text
                FROM paragraphs
                WHERE text LIKE ?
                ORDER BY
                  date_id ASC,
                  paragraph_no ASC,
                  CASE WHEN sub_id = '' THEN 0 ELSE 1 END ASC,
                  sub_id ASC
                LIMIT ?
                """,
                (q, limit),
            )
        return [
            ParagraphRecord(
                date_id=row["date_id"],
                paragraph_no=int(row["paragraph_no"]),
                sub_id=row["sub_id"] or "",
                text=row["text"] or "",
            )
            for row in cur.fetchall()
        ]

    def search_paragraphs_by_text_many(
        self,
        query: str,
        *,
        date_ids: Sequence[str],
        limit: int = 40,
    ) -> list[ParagraphRecord]:
        """
        Search canonical paragraphs by substring, restricted to a set of date_ids.

        This supports "date_id prefix" expansion in tools (e.g. "63-1201" -> multiple suffix sermons).
        """
        if not date_ids:
            return []
        q = f"%{query}%"
        placeholders = ",".join("?" for _ in date_ids)
        cur = self._conn.execute(
            f"""
            SELECT date_id, paragraph_no, sub_id, text
            FROM paragraphs
            WHERE date_id IN ({placeholders}) AND text LIKE ?
            ORDER BY
              date_id ASC,
              paragraph_no ASC,
              CASE WHEN sub_id = '' THEN 0 ELSE 1 END ASC,
              sub_id ASC
            LIMIT ?
            """,
            (*tuple(date_ids), q, limit),
        )
        return [
            ParagraphRecord(
                date_id=row["date_id"],
                paragraph_no=int(row["paragraph_no"]),
                sub_id=row["sub_id"] or "",
                text=row["text"] or "",
            )
            for row in cur.fetchall()
        ]

    def get_paragraph_bounds(self, date_id: str) -> tuple[int, int] | None:
        """Return (min_paragraph_no, max_paragraph_no) for a sermon."""
        cur = self._conn.execute(
            """
            SELECT MIN(paragraph_no) AS min_no, MAX(paragraph_no) AS max_no
            FROM paragraphs
            WHERE date_id = ?
            """,
            (date_id,),
        )
        row = cur.fetchone()
        if row is None or row["min_no"] is None or row["max_no"] is None:
            return None
        return int(row["min_no"]), int(row["max_no"])

    def get_paragraph_bounds_many(
        self,
        date_ids: Sequence[str],
    ) -> dict[str, tuple[int, int]]:
        """Batch paragraph bounds lookup keyed by date_id."""
        if not date_ids:
            return {}
        placeholders = ",".join("?" for _ in date_ids)
        cur = self._conn.execute(
            f"""
            SELECT date_id, MIN(paragraph_no) AS min_no, MAX(paragraph_no) AS max_no
            FROM paragraphs
            WHERE date_id IN ({placeholders})
            GROUP BY date_id
            """,
            tuple(date_ids),
        )
        out: dict[str, tuple[int, int]] = {}
        for row in cur.fetchall():
            if row["min_no"] is None or row["max_no"] is None:
                continue
            out[row["date_id"]] = (int(row["min_no"]), int(row["max_no"]))
        return out

    def search_chunks_by_text(
        self,
        query: str,
        *,
        date_id: str | None = None,
        limit: int = 20,
    ) -> list[ChunkRecord]:
        """
        Search chunks by a text substring.

        This is a lightweight LIKE-based helper for tool lookup.
        """
        q = f"%{query}%"
        if date_id:
            cur = self._conn.execute(
                f"""
                SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                       chunk_index, text, word_count, char_count,
                       {self._chunk_tail_sql()}
                FROM chunks
                WHERE date_id = ? AND text LIKE ?
                ORDER BY chunk_index ASC
                LIMIT ?
                """,
                (date_id, q, limit),
            )
        else:
            cur = self._conn.execute(
                f"""
                SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                       chunk_index, text, word_count, char_count,
                       {self._chunk_tail_sql()}
                FROM chunks
                WHERE text LIKE ?
                ORDER BY date_id ASC, chunk_index ASC
                LIMIT ?
                """,
                (q, limit),
            )
        return [self._chunk_record_from_row(row) for row in cur.fetchall()]

    def get_adjacent_chunks(
        self,
        date_id: str,
        chunk_index: int,
        *,
        delta: int = 1,
    ) -> list[ChunkRecord]:
        """
        Get adjacent chunks within the same sermon.

        Args:
            date_id: Sermon identifier
            chunk_index: Current chunk index
            delta: Number of chunks before and after (default ±1)

        Returns:
            List of adjacent chunks (may include the original chunk),
            ordered by chunk_index ASC.
        """
        min_idx = max(0, chunk_index - delta)
        max_idx = chunk_index + delta

        cur = self._conn.execute(
            f"""
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count,
                   {self._chunk_tail_sql()}
            FROM chunks
            WHERE date_id = ? AND chunk_index >= ? AND chunk_index <= ?
            ORDER BY chunk_index ASC
            """,
            (date_id, min_idx, max_idx),
        )
        return [self._chunk_record_from_row(row) for row in cur.fetchall()]

    def get_sermon(self, date_id: str) -> SermonRecord | None:
        """Get sermon metadata by date_id."""
        cur = self._conn.execute(
            """
            SELECT date_id, title, source, language
            FROM sermons
            WHERE date_id = ?
            """,
            (date_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return SermonRecord(
            date_id=row["date_id"],
            title=row["title"],
            source=row["source"],
            language=row["language"] or "en",
        )

    def search_sermons_by_title(
        self,
        title_query: str,
        *,
        year: int | str | None = None,
        date_id_hint: str | None = None,
        limit: int = 10,
    ) -> list[SermonRecord]:
        """
        Fuzzy search sermons by title.

        - Uses LIKE with NOCASE (SQLite-friendly ILIKE equivalent).
        - If year is provided (e.g. 63 or "1963"), filters date_id by YY- prefix.
        - If date_id_hint is provided (e.g. "63-0801"), filters date_id by prefix.
        """
        q = (title_query or "").strip()
        if not q:
            return []

        # Build a forgiving LIKE pattern: "%token1%token2%..."
        tokens = [t for t in re.split(r"[^A-Za-z0-9]+", q) if t]
        if not tokens:
            return []
        like_pat = "%" + "%".join(tokens) + "%"

        where = ["title LIKE ? COLLATE NOCASE"]
        params: list[object] = [like_pat]

        if date_id_hint:
            where.append("date_id LIKE ?")
            params.append(f"{date_id_hint.strip()}%")
        elif year is not None:
            y = str(year).strip()
            # Accept 1963 -> 63, or "63" -> 63
            yy_match = re.search(r"(\d{2})(?:\D|$)", y[-2:]) if len(y) >= 2 else None
            yy = y[-2:] if yy_match is None else yy_match.group(1)
            if yy.isdigit():
                where.append("date_id LIKE ?")
                params.append(f"{yy}-%")

        try:
            lim = int(limit)
        except (TypeError, ValueError):
            lim = 10
        lim = max(1, min(lim, 50))

        cur = self._conn.execute(
            f"""
            SELECT date_id, title, source, language
            FROM sermons
            WHERE {" AND ".join(where)}
            ORDER BY date_id ASC
            LIMIT ?
            """,
            (*params, lim),
        )
        out: list[SermonRecord] = []
        for row in cur.fetchall():
            out.append(
                SermonRecord(
                    date_id=row["date_id"],
                    title=row["title"],
                    source=row["source"],
                    language=row["language"] or "en",
                )
            )
        return out

    def get_sermons(self, date_ids: Sequence[str]) -> dict[str, SermonRecord]:
        """Batch lookup of sermon metadata."""
        if not date_ids:
            return {}

        placeholders = ",".join("?" for _ in date_ids)
        cur = self._conn.execute(
            f"""
            SELECT date_id, title, source, language
            FROM sermons
            WHERE date_id IN ({placeholders})
            """,
            tuple(date_ids),
        )

        result: dict[str, SermonRecord] = {}
        for row in cur.fetchall():
            result[row["date_id"]] = SermonRecord(
                date_id=row["date_id"],
                title=row["title"],
                source=row["source"],
                language=row["language"] or "en",
            )
        return result

    def search_sermons_by_date_id_prefix(
        self,
        date_id_prefix: str,
        *,
        limit: int = 10,
    ) -> list[SermonRecord]:
        """
        Search sermons by date_id prefix, e.g. "63-1201" -> ["63-1201M", "63-1201E"].
        """
        pfx = (date_id_prefix or "").strip()
        if not pfx:
            return []
        try:
            lim = int(limit)
        except (TypeError, ValueError):
            lim = 10
        lim = max(1, min(lim, 50))
        cur = self._conn.execute(
            """
            SELECT date_id, title, source, language
            FROM sermons
            WHERE date_id LIKE ?
            ORDER BY date_id ASC
            LIMIT ?
            """,
            (f"{pfx}%", lim),
        )
        out: list[SermonRecord] = []
        for row in cur.fetchall():
            out.append(
                SermonRecord(
                    date_id=row["date_id"],
                    title=row["title"],
                    source=row["source"],
                    language=row["language"] or "en",
                )
            )
        return out

    def get_chunk_count(self) -> int:
        """Get total number of chunks in the database."""
        cur = self._conn.execute("SELECT COUNT(*) FROM chunks")
        return cur.fetchone()[0]

    def get_sermon_count(self) -> int:
        """Get total number of sermons in the database."""
        cur = self._conn.execute("SELECT COUNT(*) FROM sermons")
        return cur.fetchone()[0]

    def get_max_chunk_index(self, date_id: str) -> int | None:
        """Get the maximum chunk index for a sermon."""
        cur = self._conn.execute(
            "SELECT MAX(chunk_index) FROM chunks WHERE date_id = ?",
            (date_id,),
        )
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None
