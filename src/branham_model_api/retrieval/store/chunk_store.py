"""
SQLite chunk store for text lookup.

Provides efficient lookup of chunks by:
- chunk_id (single chunk)
- date_id (all chunks from a sermon)
- chunk_id list (batch lookup)
- Adjacent chunks for expansion (±N)

Tables used (from DATA_FORMAT.md):
- chunks: chunk_id, date_id, paragraph_start, paragraph_end, chunk_index, text, word_count, char_count
- sermons: date_id, title, source, language
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


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


@dataclass(frozen=True)
class SermonRecord:
    """A sermon metadata record."""

    date_id: str
    title: str | None
    source: str | None
    language: str


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
            """
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count
            FROM chunks
            WHERE chunk_id = ?
            """,
            (chunk_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return ChunkRecord(
            chunk_id=row["chunk_id"],
            date_id=row["date_id"],
            paragraph_start=row["paragraph_start"],
            paragraph_end=row["paragraph_end"],
            chunk_index=row["chunk_index"],
            text=row["text"],
            word_count=row["word_count"],
            char_count=row["char_count"],
        )

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
                   chunk_index, text, word_count, char_count
            FROM chunks
            WHERE chunk_id IN ({placeholders})
            """,
            tuple(chunk_ids),
        )

        result: dict[str, ChunkRecord] = {}
        for row in cur.fetchall():
            result[row["chunk_id"]] = ChunkRecord(
                chunk_id=row["chunk_id"],
                date_id=row["date_id"],
                paragraph_start=row["paragraph_start"],
                paragraph_end=row["paragraph_end"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                word_count=row["word_count"],
                char_count=row["char_count"],
            )
        return result

    def get_chunks_by_sermon(self, date_id: str) -> list[ChunkRecord]:
        """Get all chunks from a sermon, ordered by chunk_index."""
        cur = self._conn.execute(
            """
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count
            FROM chunks
            WHERE date_id = ?
            ORDER BY chunk_index ASC
            """,
            (date_id,),
        )
        return [
            ChunkRecord(
                chunk_id=row["chunk_id"],
                date_id=row["date_id"],
                paragraph_start=row["paragraph_start"],
                paragraph_end=row["paragraph_end"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                word_count=row["word_count"],
                char_count=row["char_count"],
            )
            for row in cur.fetchall()
        ]

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
            """
            SELECT chunk_id, date_id, paragraph_start, paragraph_end,
                   chunk_index, text, word_count, char_count
            FROM chunks
            WHERE date_id = ? AND chunk_index >= ? AND chunk_index <= ?
            ORDER BY chunk_index ASC
            """,
            (date_id, min_idx, max_idx),
        )
        return [
            ChunkRecord(
                chunk_id=row["chunk_id"],
                date_id=row["date_id"],
                paragraph_start=row["paragraph_start"],
                paragraph_end=row["paragraph_end"],
                chunk_index=row["chunk_index"],
                text=row["text"],
                word_count=row["word_count"],
                char_count=row["char_count"],
            )
            for row in cur.fetchall()
        ]

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
