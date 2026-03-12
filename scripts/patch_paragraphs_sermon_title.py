#!/usr/bin/env python3
"""
Patch the canonical paragraphs table in-place.

What this script does:
1. Adds `paragraphs.sermon_title` if missing
2. Backfills `sermon_title` from `sermons.title`
3. Removes repeated sermon-title header leakage from paragraph text
4. Removes the leading sermon-start `` marker from first paragraphs
5. Removes trailing footer metadata that leaked after the closing ``

This is a DB patch only. It does NOT rerun PDF parsing.

Usage:
  uv run python scripts/patch_paragraphs_sermon_title.py --dry-run
  uv run python scripts/patch_paragraphs_sermon_title.py --apply
  uv run python scripts/patch_paragraphs_sermon_title.py --date-id 61-0212M --apply
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path


LEAK_MARKERS = (
    "VOICE OF GOD RECORDINGS",
    "ALL RIGHTS RESERVED",
    "Copyright Notice",
    "www.branham.org",
    "P.O. BOX",
    "U.S.A.",
    "ENGLISH",
)


@dataclass
class PatchResult:
    rowid: int
    date_id: str
    paragraph_no: int
    sub_id: str
    old_text: str
    new_text: str
    old_title: str
    new_title: str


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def remove_leading_sermon_start_marker(text: str) -> str:
    """Remove the leading sermon-start marker while preserving paragraph text."""
    if not text:
        return text
    return re.sub(r"^\s*\s*", "", text, count=1)


def remove_repeated_title_leaks(text: str, sermon_title: str) -> str:
    """
    Remove repeated page-header sermon titles embedded into paragraph text.

    Examples:
      "And so, JEHOVAH-JIREH"
      "Would a JEHOVAH-JIREH 25 real father ..."
    """
    if not text or not sermon_title:
        return text

    cleaned = text
    escaped_title = re.escape(sermon_title.strip())

    # Embedded title plus page number.
    cleaned = re.sub(
        rf"(?:(?<=^)|(?<=[\s,;:.(\"'“”])){escaped_title}\s+\d{{1,3}}(?=(?:\s|$))",
        "",
        cleaned,
    )

    # Standalone embedded title without retained page number.
    cleaned = re.sub(
        rf"(?:(?<=^)|(?<=[\s,;:.(\"'“”])){escaped_title}(?=(?:\s|$))",
        "",
        cleaned,
    )

    # Conservative cleanup if THE SPOKEN WORD survived inline with a page number.
    cleaned = re.sub(r"(?:(?<=^)|(?<=\s))\d{1,3}\s+THE SPOKEN WORD(?=(?:\s|$))", "", cleaned, flags=re.IGNORECASE)

    # Tighten punctuation/space after removals.
    cleaned = re.sub(r"\s+([,.;:!?])", r"\1", cleaned)
    cleaned = re.sub(r"([,.;:!?])([A-Za-z])", r"\1 \2", cleaned)
    return normalize_whitespace(cleaned)


def remove_trailing_footer_metadata(text: str, date_id: str, sermon_title: str) -> str:
    """
    Remove footer metadata that leaked into the last paragraph after the closing ``.
    """
    if not text:
        return text

    marker_idx = text.rfind("")
    if marker_idx == -1:
        return text

    trailing = text[marker_idx + 1 :]
    if not trailing.strip():
        return text

    if (
        date_id in trailing
        or sermon_title in trailing
        or any(marker in trailing for marker in LEAK_MARKERS)
    ):
        return text[: marker_idx + 1].rstrip()

    return text


def clean_paragraph_text(text: str, date_id: str, sermon_title: str) -> str:
    cleaned = text or ""
    cleaned = remove_leading_sermon_start_marker(cleaned)
    cleaned = remove_repeated_title_leaks(cleaned, sermon_title)
    cleaned = remove_trailing_footer_metadata(cleaned, date_id, sermon_title)
    cleaned = normalize_whitespace(cleaned)
    return cleaned


def ensure_paragraph_title_column(conn: sqlite3.Connection) -> None:
    cols = conn.execute("PRAGMA table_info(paragraphs)").fetchall()
    col_names = {row[1] for row in cols}
    if "sermon_title" not in col_names:
        conn.execute("ALTER TABLE paragraphs ADD COLUMN sermon_title TEXT")
        conn.commit()


def iter_patch_results(conn: sqlite3.Connection, date_id: str | None = None) -> list[PatchResult]:
    where = ""
    params: tuple[object, ...] = ()
    if date_id:
        where = "WHERE p.date_id = ?"
        params = (date_id,)

    cur = conn.execute(
        f"""
        SELECT
          p.rowid AS rowid,
          p.date_id AS date_id,
          p.paragraph_no AS paragraph_no,
          COALESCE(p.sub_id, '') AS sub_id,
          COALESCE(p.text, '') AS text,
          COALESCE(p.sermon_title, '') AS paragraph_sermon_title,
          COALESCE(s.title, '') AS sermon_title
        FROM paragraphs p
        LEFT JOIN sermons s
          ON s.date_id = p.date_id
        {where}
        ORDER BY p.date_id ASC, p.paragraph_no ASC, p.sub_id ASC
        """,
        params,
    )

    out: list[PatchResult] = []
    for row in cur.fetchall():
        rowid = int(row[0])
        did = str(row[1] or "")
        para_no = int(row[2])
        sub_id = str(row[3] or "")
        old_text = str(row[4] or "")
        old_title = str(row[5] or "")
        new_title = str(row[6] or "")
        cleaned_text = clean_paragraph_text(old_text, did, new_title)

        if cleaned_text != old_text or old_title != new_title:
            out.append(
                PatchResult(
                    rowid=rowid,
                    date_id=did,
                    paragraph_no=para_no,
                    sub_id=sub_id,
                    old_text=old_text,
                    new_text=cleaned_text,
                    old_title=old_title,
                    new_title=new_title,
                )
            )
    return out


def apply_patch_results(conn: sqlite3.Connection, patch_results: list[PatchResult]) -> None:
    with conn:
        for item in patch_results:
            conn.execute(
                """
                UPDATE paragraphs
                SET text = ?, sermon_title = ?
                WHERE rowid = ?
                """,
                (item.new_text, item.new_title, item.rowid),
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch paragraphs.sermon_title and clean leaked title/start/footer text."
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "processed" / "chunks.sqlite",
        help="Path to chunks.sqlite (default: data/processed/chunks.sqlite)",
    )
    parser.add_argument("--date-id", type=str, default=None, help="Patch only one sermon")
    parser.add_argument("--apply", action="store_true", help="Apply the patch (default is dry-run)")
    parser.add_argument("--sample-limit", type=int, default=10, help="Number of changed rows to print in dry-run")
    args = parser.parse_args()

    if not args.db_path.exists():
        raise FileNotFoundError(f"DB not found: {args.db_path}")

    conn = sqlite3.connect(str(args.db_path))
    try:
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA busy_timeout = 60000;")

        ensure_paragraph_title_column(conn)
        patch_results = iter_patch_results(conn, date_id=args.date_id)

        changed_text = sum(1 for r in patch_results if r.old_text != r.new_text)
        changed_title = sum(1 for r in patch_results if r.old_title != r.new_title)

        print(
            {
                "db_path": str(args.db_path),
                "date_id": args.date_id,
                "mode": "apply" if args.apply else "dry_run",
                "rows_to_update": len(patch_results),
                "rows_with_text_changes": changed_text,
                "rows_with_sermon_title_changes": changed_title,
            }
        )

        text_changed = [r for r in patch_results if r.old_text != r.new_text]
        title_only = [r for r in patch_results if r.old_text == r.new_text and r.old_title != r.new_title]
        sample = (text_changed + title_only)[: max(0, args.sample_limit)]
        for item in sample:
            old_preview = normalize_whitespace(item.old_text)[:180]
            new_preview = normalize_whitespace(item.new_text)[:180]
            print(
                {
                    "date_id": item.date_id,
                    "paragraph_no": item.paragraph_no,
                    "sub_id": item.sub_id,
                    "old_preview": old_preview,
                    "new_preview": new_preview,
                }
            )

        if args.apply:
            apply_patch_results(conn, patch_results)
            print({"status": "applied", "rows_updated": len(patch_results)})

    finally:
        conn.close()


if __name__ == "__main__":
    main()
