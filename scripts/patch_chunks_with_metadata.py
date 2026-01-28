#!/usr/bin/env python3
"""
Patch chunks table to add sermon metadata and composite text for indexing.

This script:
1. Adds sermon_title column to chunks (from sermons table)
2. Adds text_with_metadata column for BM25/FAISS indexing

The composite text format:
    [Sermon: {title} | ID: {date_id} | ¶{para_start}-{para_end}]
    {text}

Usage:
    uv run python scripts/patch_chunks_with_metadata.py
"""

import sqlite3
import sys
from pathlib import Path


def patch_chunks_table(db_path: Path) -> None:
    """Add sermon_title and text_with_metadata columns to chunks table."""
    
    conn = sqlite3.connect(db_path, timeout=60.0)
    conn.execute("PRAGMA busy_timeout = 60000;")
    conn.execute("PRAGMA journal_mode = WAL;")
    
    try:
        cur = conn.cursor()
        
        # Check if columns already exist
        cur.execute("PRAGMA table_info(chunks)")
        existing_cols = {row[1] for row in cur.fetchall()}
        
        # 1. Add sermon_title column if not exists
        if "sermon_title" not in existing_cols:
            print("Adding sermon_title column...")
            cur.execute("ALTER TABLE chunks ADD COLUMN sermon_title TEXT")
            conn.commit()
        else:
            print("sermon_title column already exists")
        
        # 2. Add text_with_metadata column if not exists
        if "text_with_metadata" not in existing_cols:
            print("Adding text_with_metadata column...")
            cur.execute("ALTER TABLE chunks ADD COLUMN text_with_metadata TEXT")
            conn.commit()
        else:
            print("text_with_metadata column already exists")
        
        # 3. Populate sermon_title from sermons table
        print("Populating sermon_title from sermons table...")
        cur.execute("""
            UPDATE chunks
            SET sermon_title = (
                SELECT s.title
                FROM sermons s
                WHERE s.date_id = chunks.date_id
            )
            WHERE sermon_title IS NULL OR sermon_title = ''
        """)
        updated_titles = cur.rowcount
        conn.commit()
        print(f"  Updated {updated_titles} rows with sermon_title")
        
        # 4. Generate text_with_metadata
        print("Generating text_with_metadata...")
        cur.execute("""
            UPDATE chunks
            SET text_with_metadata = 
                '[Sermon: ' || COALESCE(sermon_title, 'Unknown') || 
                ' | ID: ' || date_id || 
                ' | ¶' || paragraph_start || '-' || paragraph_end || ']' ||
                char(10) || text
            WHERE text_with_metadata IS NULL OR text_with_metadata = ''
        """)
        updated_text = cur.rowcount
        conn.commit()
        print(f"  Updated {updated_text} rows with text_with_metadata")
        
        # 5. Verify results
        print("\nVerification:")
        cur.execute("SELECT COUNT(*) FROM chunks WHERE sermon_title IS NULL OR sermon_title = ''")
        missing_titles = cur.fetchone()[0]
        print(f"  Chunks missing sermon_title: {missing_titles}")
        
        cur.execute("SELECT COUNT(*) FROM chunks WHERE text_with_metadata IS NULL OR text_with_metadata = ''")
        missing_text = cur.fetchone()[0]
        print(f"  Chunks missing text_with_metadata: {missing_text}")
        
        # 6. Show sample
        print("\nSample text_with_metadata:")
        cur.execute("""
            SELECT chunk_id, text_with_metadata 
            FROM chunks 
            WHERE text_with_metadata IS NOT NULL
            LIMIT 2
        """)
        for chunk_id, text_meta in cur.fetchall():
            preview = text_meta[:300] + "..." if len(text_meta) > 300 else text_meta
            print(f"\n  {chunk_id}:")
            print(f"  {preview}")
        
        print("\n✓ Patch complete!")
        
    finally:
        conn.close()


def main():
    repo_root = Path(__file__).parent.parent
    db_path = repo_root / "data" / "processed" / "chunks.sqlite"
    
    if not db_path.exists():
        print(f"Error: chunks.sqlite not found: {db_path}", file=sys.stderr)
        sys.exit(1)
    
    print(f"Patching: {db_path}")
    print("=" * 70)
    
    patch_chunks_table(db_path)


if __name__ == "__main__":
    main()
