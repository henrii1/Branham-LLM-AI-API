#!/usr/bin/env python3
"""
Audit the rebuilt SQLite corpus for:
  - Paragraph numbering sequentiality (per sermon)
  - Longest paragraph (approx word count)
  - Chunk suffix usage (a/b/.../aa/ab) and maximum suffix observed
  - Basic table integrity between sermons/paragraphs/chunks

This is a diagnostics tool to help confirm Stage 1 (paragraphing) and Stage 2 (chunking)
behave as intended.

Usage:
  uv run python datasets/ingest/audit_db.py --db-path data/processed/chunks.sqlite
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


PARA_REF_RE = re.compile(r"^(\d+)([a-z]*)$")  # allow multi-letter suffixes


def approx_word_count(text: str) -> int:
    # Fast approximate: spaces + 1
    if not text:
        return 0
    return len([w for w in text.split() if w.strip()])


def parse_ref(ref: str) -> Tuple[int, str]:
    m = PARA_REF_RE.match((ref or "").strip())
    if not m:
        raise ValueError(f"Invalid paragraph ref: {ref!r}")
    return (int(m.group(1)), m.group(2) or "")


def suffix_to_idx(suffix: str) -> int:
    # '' -> 0, 'a' -> 1, ..., 'z' -> 26, 'aa' -> 27
    s = suffix.strip().lower()
    if not s:
        return 0
    n = 0
    for ch in s:
        if not ("a" <= ch <= "z"):
            raise ValueError(f"invalid suffix char: {ch!r}")
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n


def table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    """
    Return True if `table_name` exists in the SQLite database.

    This audit script is used across Stage 1 and Stage 2.
    Stage 1-only DBs legitimately do not have a `chunks` table yet.
    """
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=? LIMIT 1",
        (table_name,),
    )
    return cur.fetchone() is not None


@dataclass(frozen=True)
class GapSermon:
    date_id: str
    min_no: int
    max_no: int
    count_distinct: int


def find_non_sequential_sermons(conn: sqlite3.Connection) -> List[GapSermon]:
    """
    A sermon is sequential if its distinct paragraph_no values cover [min..max] without gaps.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT date_id,
               MIN(paragraph_no) AS min_no,
               MAX(paragraph_no) AS max_no,
               COUNT(DISTINCT paragraph_no) AS cnt
        FROM paragraphs
        GROUP BY date_id
        HAVING cnt != (max_no - min_no + 1)
        ORDER BY (max_no - min_no + 1) - cnt DESC
        """
    )
    return [GapSermon(r[0], int(r[1]), int(r[2]), int(r[3])) for r in cur.fetchall()]


def missing_numbers_for_sermon(conn: sqlite3.Connection, date_id: str) -> List[int]:
    cur = conn.cursor()
    cur.execute(
        "SELECT DISTINCT paragraph_no FROM paragraphs WHERE date_id=? ORDER BY paragraph_no ASC",
        (date_id,),
    )
    nos = [int(r[0]) for r in cur.fetchall()]
    if not nos:
        return []
    s = set(nos)
    return [n for n in range(nos[0], nos[-1] + 1) if n not in s]


def max_paragraph_word_count(conn: sqlite3.Connection) -> Tuple[str, int, int]:
    """
    Returns (date_id, paragraph_no, approx_words) for the max paragraph.
    """
    cur = conn.cursor()
    cur.execute("SELECT date_id, paragraph_no, text FROM paragraphs")
    best = ("", 0, 0)
    for date_id, pno, text in cur.fetchall():
        wc = approx_word_count(text)
        if wc > best[2]:
            best = (date_id, int(pno), wc)
    return best


def chunk_suffix_stats(conn: sqlite3.Connection) -> Dict[str, object]:
    cur = conn.cursor()
    if not table_exists(conn, "chunks"):
        return {
            "max_suffix_item": None,
            "total_suffix_refs": 0,
            "multi_letter_suffix_refs": 0,
        }

    cur.execute("SELECT chunk_id, date_id, paragraph_start, paragraph_end FROM chunks")

    max_item = None  # (idx, suffix, side, chunk_id, date_id, ref)
    multi_letter_count = 0
    total_suffix_count = 0

    for chunk_id, date_id, p_start, p_end in cur.fetchall():
        for side, ref in (("start", p_start), ("end", p_end)):
            try:
                _, suffix = parse_ref(str(ref))
            except Exception:
                # Keep going; other audits will catch invalid refs
                continue
            if suffix:
                total_suffix_count += 1
                if len(suffix) > 1:
                    multi_letter_count += 1
                idx = suffix_to_idx(suffix)
                item = (idx, suffix, side, str(chunk_id), str(date_id), str(ref))
                if max_item is None or idx > max_item[0]:
                    max_item = item

    return {
        "max_suffix_item": max_item,
        "total_suffix_refs": total_suffix_count,
        "multi_letter_suffix_refs": multi_letter_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit chunks.sqlite for paragraph + chunk integrity")
    parser.add_argument("--db-path", type=str, default="data/processed/chunks.sqlite")
    parser.add_argument("--max-gap-sermons", type=int, default=20)
    parser.add_argument("--max-missing-show", type=int, default=40)
    args = parser.parse_args()

    db_path = Path(args.db_path)
    conn = sqlite3.connect(db_path)

    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM sermons")
    sermons_count = int(cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM paragraphs")
    paragraphs_rows = int(cur.fetchone()[0])
    if table_exists(conn, "chunks"):
        cur.execute("SELECT COUNT(*) FROM chunks")
        chunks_rows = int(cur.fetchone()[0])
    else:
        chunks_rows = 0

    cur.execute("SELECT COUNT(DISTINCT date_id) FROM paragraphs")
    paragraph_sermons = int(cur.fetchone()[0])
    if table_exists(conn, "chunks"):
        cur.execute("SELECT COUNT(DISTINCT date_id) FROM chunks")
        chunk_sermons = int(cur.fetchone()[0])
    else:
        chunk_sermons = 0

    # Integrity: missing joins
    cur.execute("SELECT COUNT(*) FROM sermons s LEFT JOIN paragraphs p ON p.date_id=s.date_id WHERE p.date_id IS NULL")
    sermons_without_paragraphs = int(cur.fetchone()[0])
    if table_exists(conn, "chunks"):
        cur.execute("SELECT COUNT(*) FROM sermons s LEFT JOIN chunks c ON c.date_id=s.date_id WHERE c.date_id IS NULL")
        sermons_without_chunks = int(cur.fetchone()[0])
    else:
        sermons_without_chunks = 0
    cur.execute("SELECT COUNT(*) FROM paragraphs p LEFT JOIN sermons s ON s.date_id=p.date_id WHERE s.date_id IS NULL")
    paragraphs_without_sermon = int(cur.fetchone()[0])
    if table_exists(conn, "chunks"):
        cur.execute("SELECT COUNT(*) FROM chunks c LEFT JOIN sermons s ON s.date_id=c.date_id WHERE s.date_id IS NULL")
        chunks_without_sermon = int(cur.fetchone()[0])
    else:
        chunks_without_sermon = 0

    # Basic word stats for chunks
    if table_exists(conn, "chunks"):
        cur.execute("SELECT MIN(word_count), MAX(word_count), AVG(word_count) FROM chunks")
        min_wc, max_wc, avg_wc = cur.fetchone()
    else:
        min_wc, max_wc, avg_wc = (0, 0, 0.0)

    # Under/over constraints
    if table_exists(conn, "chunks"):
        cur.execute("SELECT SUM(CASE WHEN word_count>380 THEN 1 ELSE 0 END) FROM chunks")
        over_380 = int(cur.fetchone()[0])
        cur.execute(
            """
            WITH max_idx AS (
              SELECT date_id, MAX(chunk_index) AS max_chunk_index FROM chunks GROUP BY date_id
            )
            SELECT
              SUM(CASE WHEN c.word_count < 260 AND c.chunk_index < m.max_chunk_index THEN 1 ELSE 0 END) AS nonfinal_under260,
              SUM(CASE WHEN c.word_count < 260 AND c.chunk_index = m.max_chunk_index THEN 1 ELSE 0 END) AS final_under260
            FROM chunks c
            JOIN max_idx m USING(date_id)
            """
        )
        nonfinal_under260, final_under260 = [int(x) for x in cur.fetchone()]
    else:
        over_380 = 0
        nonfinal_under260 = 0
        final_under260 = 0

    # Paragraph sequentiality
    gaps = find_non_sequential_sermons(conn)

    # Longest paragraph
    max_para_date_id, max_para_no, max_para_wc = max_paragraph_word_count(conn)

    # Chunk suffix stats
    suffix_stats = chunk_suffix_stats(conn)

    print("## DB AUDIT")
    print(f"- db: {db_path}")
    print("")
    print("### Table counts")
    print(f"- sermons: {sermons_count}")
    print(f"- paragraphs (rows): {paragraphs_rows}")
    print(f"- chunks (rows): {chunks_rows}")
    print(f"- distinct date_id in paragraphs: {paragraph_sermons}")
    print(f"- distinct date_id in chunks: {chunk_sermons}")
    print("")
    print("### Join integrity")
    print(f"- sermons without paragraphs: {sermons_without_paragraphs}")
    print(f"- sermons without chunks: {sermons_without_chunks}")
    print(f"- paragraphs without sermon row: {paragraphs_without_sermon}")
    print(f"- chunks without sermon row: {chunks_without_sermon}")
    print("")
    if table_exists(conn, "chunks"):
        print("### Chunk word-count stats")
        print(f"- word_count min/avg/max: {int(min_wc)}/{float(avg_wc):.1f}/{int(max_wc)}")
        print(f"- chunks >380: {over_380}")
        print(f"- non-final chunks <260: {nonfinal_under260}")
        print(f"- final chunks <260: {final_under260}")
        print("")
    else:
        print("### Chunk word-count stats")
        print("- <chunks table not present; run Stage 2 to audit chunk constraints>")
        print("")
    print("### Paragraph stats")
    print(f"- longest paragraph (approx words): {max_para_wc} at {max_para_date_id} ¶{max_para_no}")
    print("")
    print("### Chunk suffix stats (split refs like 12a / 12aa)")
    max_item = suffix_stats["max_suffix_item"]
    print(f"- suffix refs present (start/end): {suffix_stats['total_suffix_refs']}")
    print(f"- multi-letter suffix refs (aa, ab, ...): {suffix_stats['multi_letter_suffix_refs']}")
    if max_item:
        idx, suffix, side, chunk_id, date_id, ref = max_item
        print(f"- max suffix: '{suffix}' (idx={idx}) at {date_id} {chunk_id} ({side}={ref})")
    else:
        print("- max suffix: <none>")
    print("")
    print("### Paragraph sequentiality (per sermon)")
    print(f"- sermons with gaps: {len(gaps)}")
    if gaps:
        print(f"- showing first {min(len(gaps), args.max_gap_sermons)} gap sermons (largest gap first):")
        for g in gaps[: args.max_gap_sermons]:
            expected = (g.max_no - g.min_no + 1)
            missing_cnt = expected - g.count_distinct
            miss = missing_numbers_for_sermon(conn, g.date_id)[: args.max_missing_show]
            miss_preview = ", ".join(map(str, miss)) if miss else "<none>"
            print(
                f"  - {g.date_id}: range {g.min_no}-{g.max_no}, distinct={g.count_distinct}, missing={missing_cnt}; "
                f"missing_sample=[{miss_preview}]"
            )

    conn.close()


if __name__ == "__main__":
    main()

