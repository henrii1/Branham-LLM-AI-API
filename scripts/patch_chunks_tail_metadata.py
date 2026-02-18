#!/usr/bin/env python3
"""
Add and backfill tail-chunk metadata on chunks table.

Adds:
  - chunks.is_tail_chunk INTEGER DEFAULT 0

Backfill rule:
  - is_tail_chunk = 1 for the last chunk_index per sermon (date_id)
  - is_tail_chunk = 0 otherwise
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from branham_model_api.config import get_config


def has_column(conn: sqlite3.Connection, table: str, column: str) -> bool:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return any(r[1] == column for r in rows)


def main() -> None:
    db_path = Path(get_config().chunk_store_path)
    if not db_path.exists():
        raise FileNotFoundError(f"chunks sqlite not found: {db_path}")

    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

        if not has_column(conn, "chunks", "is_tail_chunk"):
            conn.execute("ALTER TABLE chunks ADD COLUMN is_tail_chunk INTEGER DEFAULT 0")

        conn.execute("UPDATE chunks SET is_tail_chunk = 0")
        conn.execute(
            """
            UPDATE chunks
            SET is_tail_chunk = 1
            WHERE (date_id, chunk_index) IN (
                SELECT date_id, MAX(chunk_index)
                FROM chunks
                GROUP BY date_id
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chunks_date_tail ON chunks(date_id, is_tail_chunk)"
        )
        conn.commit()

        stats = conn.execute(
            """
            SELECT
              COUNT(*) AS total_chunks,
              SUM(CASE WHEN is_tail_chunk = 1 THEN 1 ELSE 0 END) AS tail_chunks,
              COUNT(DISTINCT date_id) AS sermons
            FROM chunks
            """
        ).fetchone()
        print(
            {
                "db_path": str(db_path),
                "total_chunks": int(stats[0]),
                "tail_chunks": int(stats[1] or 0),
                "sermons": int(stats[2]),
            }
        )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
