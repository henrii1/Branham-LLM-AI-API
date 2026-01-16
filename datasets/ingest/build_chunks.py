#!/usr/bin/env python3
"""
Stage 2: Build Chunks from Paragraphs

Reads paragraphs from chunks.sqlite and creates retrieval-ready chunks:
- Packs paragraphs using **word budgeting** (260-320 words preferred, 380 hard max)
- Assigns chunk_index sequentially (0 to N-1) per sermon
- Generates chunk_id = {date_id}_chunk_{chunk_index}
- Writes to chunks table

Usage:
    python datasets/ingest/build_chunks.py [--test] [--date-id 47-1207]
    python datasets/ingest/build_chunks.py [--limit N] [--overwrite-chunks-table]

Options:
    --test                   Test mode: writes a **single-sermon** DB to test_chunk.sqlite
    --date-id DATE_ID        Only process a single sermon (recommended for --test)
    --limit N                Only process first N sermons (for testing)
    --overwrite-chunks-table Drop + recreate chunks table if it exists (destructive)
    --db-path PATH           SQLite DB to read paragraphs from / write chunks into (default: data/processed/chunks.sqlite)
"""

import sqlite3
import sys
from pathlib import Path
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Iterable, List, Optional, Tuple


# ============================================================================
# Configuration (from DATA_FORMAT.md Section 5.2)
# ============================================================================

TARGET_WORDS_MIN = 260
TARGET_WORDS_MAX = 320
HARD_MAX_WORDS = 380


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class Paragraph:
    """Represents a paragraph from the database"""
    date_id: str
    paragraph_no: int
    sub_id: str  # '' (canonical paragraph) or 'a'/'b'/... (rare; primarily for synthetic units)
    text: str
    word_count: int

    @property
    def ref(self) -> str:
        """
        Canonical paragraph reference used in chunk metadata.
        Examples:
          - paragraph_no=15, sub_id=''  -> '15'
          - paragraph_no=15, sub_id='b' -> '15b'
        """
        return f"{self.paragraph_no}{self.sub_id}" if self.sub_id else str(self.paragraph_no)


@dataclass
class Chunk:
    """Represents a chunk to be stored"""
    chunk_id: str
    date_id: str
    paragraph_start: str  # TEXT (e.g., '12', '15a', '15b')
    paragraph_end: str    # TEXT (e.g., '15a', '20')
    chunk_index: int
    text: str
    word_count: int
    char_count: int


@dataclass
class Unit:
    """
    Atomic unit the chunker operates on.

    Normally this corresponds 1:1 with a row in the `paragraphs` table.
    When we must split a paragraph across chunk boundaries, we create synthetic
    Units that share the same paragraph_no but have new sub-identifiers (a/b/c...).
    """
    date_id: str
    paragraph_no: int
    sub_id: str  # '' | 'a' | 'b' | ...
    text: str
    word_count: int

    @property
    def ref(self) -> str:
        return f"{self.paragraph_no}{self.sub_id}" if self.sub_id else str(self.paragraph_no)


# ============================================================================
# Helper Functions
# ============================================================================

def count_words(text: str) -> int:
    """
    Count words in text using whitespace splitting.
    Simple and fast, model-agnostic.
    """
    # NOTE: This is intentionally simple; it matches the spec: "Use word count (not tokenizer)".
    return len([w for w in text.split() if w.strip()])


# NOTE:
# - The `paragraphs` table may already contain `sub_id` rows (e.g., 23a/23b) from
#   earlier pipeline stages.
# - This Stage 2 script MAY ALSO create **additional** split refs (e.g., 15a/15b)
#   when a paragraph must be split across a chunk boundary to satisfy the 380 hard max.


import re

# Sentence boundary regex for splitting (kept intentionally lightweight / fast).
# We split on a punctuation run followed by whitespace or end-of-string.
SENTENCE_END_RE = re.compile(r"([.!?]+)(\s+|$)")


def _suffix_to_idx(suffix: str) -> int:
    """
    Convert a base-26 suffix to an integer index:
      a -> 0
      b -> 1
      ...
      z -> 25
      aa -> 26
      ab -> 27
    """
    s = suffix.strip().lower()
    if not s:
        raise ValueError("suffix must be non-empty")
    n = 0
    for ch in s:
        if not ("a" <= ch <= "z"):
            raise ValueError(f"invalid suffix char: {ch!r}")
        n = n * 26 + (ord(ch) - ord("a") + 1)
    return n - 1


def _idx_to_suffix(idx: int) -> str:
    """
    Convert integer index to base-26 suffix:
      0 -> a
      1 -> b
      ...
      25 -> z
      26 -> aa
    """
    if idx < 0:
        raise ValueError("idx must be >= 0")
    # 1-based conversion
    n = idx + 1
    chars = []
    while n > 0:
        n, rem = divmod(n - 1, 26)
        chars.append(chr(ord("a") + rem))
    return "".join(reversed(chars))


def next_split_letter_after(max_idx_by_para: Dict[int, int], para_no: int) -> str:
    """
    Allocate the next synthetic split letter for a paragraph number.

    We track only the *maximum* used letter per paragraph number, so that:
      - First split creates 'a'/'b' (max becomes 'b')
      - Next split creates 'c' (max becomes 'c'), etc.
    """
    next_idx = max_idx_by_para.get(para_no, -1) + 1
    max_idx_by_para[para_no] = next_idx
    return _idx_to_suffix(next_idx)


def split_text_to_budget_strict(text: str, target_words: int, min_words_needed: int) -> Tuple[str, str]:
    """
    Split text to fit within a word budget, preferring sentence boundaries.

    Contract:
      - Return (left, right) such that wc(left) <= target_words
      - Prefer splitting on sentence boundaries
      - If the best sentence-boundary split would produce wc(left) < min_words_needed,
        we fall back to a **hard word split** to satisfy the minimum (only when needed).
    
    Returns:
        (left_part, remainder)
    """
    if target_words <= 0:
        return ("", text.strip())

    sentences: List[str] = []
    pos = 0
    
    for match in SENTENCE_END_RE.finditer(text):
        end = match.end()
        sentence = text[pos:end].strip()
        if sentence:
            sentences.append(sentence)
        pos = end
    
    # Add remaining text
    if pos < len(text):
        remainder = text[pos:].strip()
        if remainder:
            sentences.append(remainder)
    
    if not sentences:
        return (text, '')
    
    # Best-effort sentence-boundary split: take as many full sentences as fit (maximal <= target).
    accumulated: List[str] = []
    total_words = 0
    for sentence in sentences:
        sent_words = count_words(sentence)
        if total_words + sent_words <= target_words:
            accumulated.append(sentence)
            total_words += sent_words
        else:
            break

    left = " ".join(accumulated).strip()
    right = " ".join(sentences[len(accumulated):]).strip()

    # If sentence-boundary split can't meet the minimum needed (and we're not at end-of-sermon),
    # force a hard split by words up to target_words (or at least min_words_needed).
    if count_words(left) < min_words_needed:
        words_all = text.split()
        cut = min(target_words, len(words_all))
        cut = max(cut, min_words_needed)
        cut = min(cut, len(words_all))
        left = " ".join(words_all[:cut]).strip()
        right = " ".join(words_all[cut:]).strip()
        return (left, right)

    return (left, right)


def _pack_paragraphs_into_chunks_core(
    paragraphs: List[Paragraph],
    date_id: str,
    *,
    allow_final_under_min: bool,
) -> List[Chunk]:
    """
    Pack paragraph Units into chunks using strict word budgeting:
      - Preferred: 260–320 words
      - Hard max: 380 words (never exceed)

    Critical rule:
      - If the current chunk is <260 and the next paragraph would push us >380,
        we split that paragraph at the nearest sentence boundary that fits.
        The remainder becomes the start of the next chunk.

    Final-chunk exception:
      - If allow_final_under_min=True, only the FINAL chunk may be <260 words.
      - If allow_final_under_min=False, ALL chunks must be >=260 words.
    
    Args:
        paragraphs: List of paragraphs for a single sermon (ordered)
        date_id: The sermon's date_id
    
    Returns:
        List of Chunk objects
    """
    if not paragraphs:
        return []
    
    # Initialize Units from the canonical paragraphs list.
    units: Deque[Unit] = deque(
        Unit(
            date_id=p.date_id,
            paragraph_no=p.paragraph_no,
            sub_id=p.sub_id or "",
            text=p.text.strip(),
            word_count=p.word_count,
        )
        for p in paragraphs
    )

    # Track max used split letter per paragraph_no (0='a', 1='b', ...).
    # For unsplit paragraphs, there is no entry until the first split happens.
    max_letter_idx_by_para: Dict[int, int] = {}

    # We build chunks as lists of Units first so we can do small end-of-sermon
    # rebalancing when `allow_final_under_min=False`.
    built_unit_chunks: List[List[Unit]] = []
    chunk_index = 0
    
    while units:
        chunk_units: List[Unit] = []
        chunk_wc = 0

        # Build a single chunk.
        while units:
            u = units[0]

            # 1) Whole-unit add: if it fits under hard max, consider adding it.
            if chunk_wc + u.word_count <= HARD_MAX_WORDS:
                chunk_units.append(units.popleft())
                chunk_wc += u.word_count

                # Stop if we're in preferred range.
                if TARGET_WORDS_MIN <= chunk_wc <= TARGET_WORDS_MAX:
                    break

                # Stop if we're above preferred but still legal (don't chase perfection).
                if TARGET_WORDS_MAX < chunk_wc <= HARD_MAX_WORDS:
                    break

                # Otherwise: keep adding until we reach the band.
                continue

            # 2) Would exceed hard max if we add the next unit whole.
            # If we already hit minimum, finalize chunk and let next unit start the next chunk.
            if chunk_wc >= TARGET_WORDS_MIN:
                break
        
            # 3) Critical case: chunk is under-min AND next unit can't fit.
            # We must split the next unit at a sentence boundary so we:
            #   - never exceed HARD_MAX_WORDS
            #   - avoid producing a non-final chunk under 260 words
            budget = HARD_MAX_WORDS - chunk_wc
            u = units.popleft()

            # We must ensure this chunk reaches the minimum (unless it's the final chunk).
            min_needed = max(0, TARGET_WORDS_MIN - chunk_wc)
            left_text, right_text = split_text_to_budget_strict(u.text, budget, min_needed)
            left_text = left_text.strip()
            right_text = right_text.strip()

            if not left_text:
                # This should be extremely rare (e.g., whitespace-only paragraph).
                # If we can't take any text, we have to finalize to avoid infinite loops.
                break

            # Allocate synthetic split ids (a/b/c...) for the paragraph number.
            #
            # Important nuance:
            # - If we're splitting an *unsplit* paragraph (sub_id == ''), we want 'a' + 'b'.
            # - If we're splitting an already-labeled segment (e.g., 'b'), we keep the
            #   left label ('b') and allocate a new next label for the remainder ('c').
            if u.sub_id:
                left_sub = u.sub_id
                right_sub = next_split_letter_after(max_letter_idx_by_para, u.paragraph_no)
            else:
                left_sub = "a"
                right_sub = "b"
                max_letter_idx_by_para[u.paragraph_no] = _suffix_to_idx("b")

            left_unit = Unit(
                date_id=u.date_id,
                paragraph_no=u.paragraph_no,
                sub_id=left_sub,
                text=left_text,
                word_count=count_words(left_text),
            )
            right_unit = Unit(
                date_id=u.date_id,
                paragraph_no=u.paragraph_no,
                sub_id=right_sub,
                text=right_text,
                word_count=count_words(right_text),
            )

            chunk_units.append(left_unit)
            chunk_wc += left_unit.word_count

            # Push remainder back to the front for the next chunk.
            if right_unit.word_count > 0:
                units.appendleft(right_unit)

            break  # finalize this chunk after a split

        if not chunk_units:
            # Defensive: avoid infinite loops if we somehow couldn't add anything.
            # In practice, this shouldn't happen with real sermon text.
            raise RuntimeError(f"Failed to build chunk for {date_id} at chunk_index={chunk_index}")

        # Enforce minimum word count constraint for NON-FINAL chunks.
        #
        # If `units` still has remaining text, this chunk is non-final and must be >= min.
        # If `units` is empty, this is the final chunk for this (sub)sermon; if
        # allow_final_under_min=False we'll handle tail rebalancing after building.
        if chunk_wc < TARGET_WORDS_MIN and units:
            raise RuntimeError(
                f"Invariant violated: non-final chunk under {TARGET_WORDS_MIN} words "
                f"({date_id}, chunk_index={chunk_index}, words={chunk_wc})"
            )

        built_unit_chunks.append(chunk_units)
        chunk_index += 1
    
    def chunk_wc(units_: List[Unit]) -> int:
        return sum(u.word_count for u in units_)

    def split_text_near_left_words(text: str, left_words_target: int) -> Tuple[str, str]:
        """
        Split `text` into (left, right) where wc(left) is as close as possible to
        left_words_target, preferring sentence boundaries. Falls back to hard split.
        """
        text = text.strip()
        if not text:
            return ("", "")
        words_all = text.split()
        total = len(words_all)
        if left_words_target <= 0:
            return ("", text)
        if left_words_target >= total:
            return (text, "")

        # Try sentence-boundary split near the target.
        sentences: List[str] = []
        pos = 0
        for match in SENTENCE_END_RE.finditer(text):
            end = match.end()
            sentence = text[pos:end].strip()
            if sentence:
                sentences.append(sentence)
            pos = end
        if pos < len(text):
            tail = text[pos:].strip()
            if tail:
                sentences.append(tail)

        if sentences:
            cum = []
            running = 0
            for s in sentences:
                running += count_words(s)
                cum.append(running)

            # Choose boundary (k) that minimizes |cum[k] - left_words_target|
            best_k = None
            best_dist = None
            for k, c in enumerate(cum[:-1]):  # must leave something for right
                dist = abs(c - left_words_target)
                if best_dist is None or dist < best_dist:
                    best_dist = dist
                    best_k = k
            if best_k is not None:
                left = " ".join(sentences[: best_k + 1]).strip()
                right = " ".join(sentences[best_k + 1 :]).strip()
                if left and right:
                    return (left, right)

        # Hard split by words
        left = " ".join(words_all[:left_words_target]).strip()
        right = " ".join(words_all[left_words_target:]).strip()
        return (left, right)

    # If this chunking is being used as a NON-final prefix (e.g., we've reserved the last
    # short paragraph as its own final chunk), we must ensure **every** chunk meets minimum.
    # We do this with a backward "debt propagation" pass: if chunk i is under-min, borrow
    # from the end of chunk i-1; this may underfill chunk i-1, so we continue backwards.
    if not allow_final_under_min and built_unit_chunks:
        for i in range(len(built_unit_chunks) - 1, -1, -1):
            while chunk_wc(built_unit_chunks[i]) < TARGET_WORDS_MIN:
                if i == 0:
                    raise RuntimeError(
                        f"Prefix too short to satisfy minimum chunk size without the final paragraph: {date_id}"
                    )

                donor_chunk = built_unit_chunks[i - 1]
                recv_chunk = built_unit_chunks[i]

                donor_wc = chunk_wc(donor_chunk)
                recv_wc = chunk_wc(recv_chunk)
                needed = TARGET_WORDS_MIN - recv_wc

                spare_from_donor = donor_wc - TARGET_WORDS_MIN
                room_in_recv = HARD_MAX_WORDS - recv_wc
                if spare_from_donor <= 0 or room_in_recv <= 0:
                    raise RuntimeError(
                        f"Cannot rebalance to satisfy minimum: {date_id} "
                        f"(donor_idx={i-1}, recv_idx={i}, donor_wc={donor_wc}, recv_wc={recv_wc}, "
                        f"spare_from_donor={spare_from_donor}, room_in_recv={room_in_recv})"
                    )

                max_move = min(spare_from_donor, room_in_recv)
                if not donor_chunk:
                    raise RuntimeError(f"Cannot rebalance: donor chunk has no units ({date_id})")

                donor = donor_chunk[-1]

                # Prefer moving whole donor unit if possible and helpful.
                if donor.word_count <= max_move and donor.word_count <= needed and (donor_wc - donor.word_count) >= TARGET_WORDS_MIN:
                    donor_chunk.pop()
                    recv_chunk.insert(0, donor)
                    continue

                move_words = min(max_move, max(needed, 1))
                if donor.word_count <= 1:
                    raise RuntimeError(f"Cannot split 1-word donor during rebalance: {date_id}")
                move_words = min(move_words, donor.word_count - 1)
                left_words_target = donor.word_count - move_words

                left_text, right_text = split_text_near_left_words(donor.text, left_words_target)
                if not left_text or not right_text:
                    words_all = donor.text.split()
                    left_text = " ".join(words_all[:left_words_target]).strip()
                    right_text = " ".join(words_all[left_words_target:]).strip()

                if donor.sub_id:
                    left_sub = donor.sub_id
                    right_sub = next_split_letter_after(max_letter_idx_by_para, donor.paragraph_no)
                else:
                    left_sub = "a"
                    right_sub = "b"
                    max_letter_idx_by_para[donor.paragraph_no] = _suffix_to_idx("b")

                left_unit = Unit(
                    date_id=donor.date_id,
                    paragraph_no=donor.paragraph_no,
                    sub_id=left_sub,
                    text=left_text.strip(),
                    word_count=count_words(left_text),
                )
                right_unit = Unit(
                    date_id=donor.date_id,
                    paragraph_no=donor.paragraph_no,
                    sub_id=right_sub,
                    text=right_text.strip(),
                    word_count=count_words(right_text),
                )

                donor_chunk[-1] = left_unit
                recv_chunk.insert(0, right_unit)

    # Convert unit chunks → stored Chunk rows.
    chunks: List[Chunk] = []
    for idx, unit_chunk in enumerate(built_unit_chunks):
        chunk_text = "\n".join(u.text for u in unit_chunk).strip()
        chunks.append(
            Chunk(
                chunk_id=f"{date_id}_chunk_{idx}",
                date_id=date_id,
                paragraph_start=unit_chunk[0].ref,
                paragraph_end=unit_chunk[-1].ref,
                chunk_index=idx,
                text=chunk_text,
                word_count=count_words(chunk_text),
                char_count=len(chunk_text),
            )
        )

    return chunks


def pack_paragraphs_into_chunks(
    paragraphs: List[Paragraph],
    date_id: str
) -> List[Chunk]:
    """
    Public packer for Stage 2 chunking.

    Contract (Option C):
      - All NON-FINAL chunks must be >= 260 words.
      - The FINAL chunk may be < 260 words.
      - No chunk may exceed 380 words.
      - Paragraph order must be preserved; splitting is allowed only when needed.

    Practical edge case:
      - Some sermons are extremely short (total < 260 words). In that case, we emit
        a single (final) chunk even though it is < 260.
    """
    if not paragraphs:
        return []

    total_words = sum(p.word_count for p in paragraphs)
    if total_words < TARGET_WORDS_MIN:
        # Tiny sermon: emit one final chunk (allowed to be <260).
        chunk_text = "\n".join(p.text for p in paragraphs).strip()
        return [
            Chunk(
                chunk_id=f"{date_id}_chunk_0",
                date_id=date_id,
                paragraph_start=paragraphs[0].ref,
                paragraph_end=paragraphs[-1].ref,
                chunk_index=0,
                text=chunk_text,
                word_count=count_words(chunk_text),
                char_count=len(chunk_text),
            )
        ]

    return _pack_paragraphs_into_chunks_core(paragraphs, date_id, allow_final_under_min=True)


# ============================================================================
# Database Operations
# ============================================================================

def create_chunks_table(conn: sqlite3.Connection):
    """
    Create chunks table if it doesn't exist.

    NOTE: This Stage 2 script writes TEXT paragraph refs (e.g., '15a'),
    so paragraph_start / paragraph_end are TEXT.
    """
    cursor = conn.cursor()
    
    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chunks (
            chunk_id TEXT PRIMARY KEY,
            date_id TEXT,
            paragraph_start TEXT,
            paragraph_end TEXT,
            chunk_index INTEGER,
            text TEXT,
            word_count INTEGER,
            char_count INTEGER
        )
    ''')
    
    # Create indexes for fast retrieval
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunks_date 
        ON chunks(date_id)
    ''')
    
    cursor.execute('''
        CREATE INDEX IF NOT EXISTS idx_chunks_range 
        ON chunks(date_id, paragraph_start, paragraph_end)
    ''')
    
    conn.commit()


def load_paragraphs_for_sermon(
    conn: sqlite3.Connection,
    date_id: str
) -> List[Paragraph]:
    """
    Load all paragraphs for a sermon, ordered by paragraph_no and sub_id.
    """
    cursor = conn.cursor()
    # IMPORTANT:
    # The `paragraphs` table may contain "continuation rows" like (97, '') and (97, 'b').
    # For chunking + verification, we want *canonical* paragraphs aligned with the
    # paragraph numbering used in `*_DB_PARAGRAPHS.md` (one paragraph per paragraph_no).
    #
    # So we reconstruct each paragraph_no by concatenating all its sub_id parts in order.
    cursor.execute(
        """
        SELECT date_id, paragraph_no, sub_id, text
        FROM paragraphs
        WHERE date_id = ?
        ORDER BY paragraph_no ASC, sub_id ASC
        """,
        (date_id,),
    )

    paragraphs: List[Paragraph] = []
    current_no: Optional[int] = None
    current_text_parts: List[str] = []

    for row_date_id, para_no, sub_id, text in cursor.fetchall():
        if current_no is None:
            current_no = para_no

        if para_no != current_no:
            combined = " ".join(t.strip() for t in current_text_parts if t and t.strip()).strip()
            if combined:
                paragraphs.append(
                    Paragraph(
                        date_id=date_id,
                        paragraph_no=current_no,
                        sub_id="",  # canonical paragraph
                        text=combined,
                        word_count=count_words(combined),
                    )
                )
            current_no = para_no
            current_text_parts = []

        current_text_parts.append(text or "")

    # Flush last
    if current_no is not None:
        combined = " ".join(t.strip() for t in current_text_parts if t and t.strip()).strip()
        if combined:
            paragraphs.append(
                Paragraph(
                    date_id=date_id,
                    paragraph_no=current_no,
                    sub_id="",
                    text=combined,
                    word_count=count_words(combined),
                )
            )

    return paragraphs


def insert_chunk(conn: sqlite3.Connection, chunk: Chunk):
    """Insert a single chunk into the database."""
    cursor = conn.cursor()
    cursor.execute('''
        INSERT OR REPLACE INTO chunks (
            chunk_id, date_id, paragraph_start, paragraph_end,
            chunk_index, text, word_count, char_count
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        chunk.chunk_id,
        chunk.date_id,
        chunk.paragraph_start,
        chunk.paragraph_end,
        chunk.chunk_index,
        chunk.text,
        chunk.word_count,
        chunk.char_count
    ))


def create_single_sermon_test_db(
    source_db_path: Path,
    target_db_path: Path,
    date_id: str
):
    """
    Create a minimal test DB containing:
      - sermons row for date_id
      - all paragraphs rows for date_id
      - a fresh chunks table (Stage 2 output)

    This avoids mutating the canonical chunks.sqlite during iteration.
    """
    if target_db_path.exists():
        target_db_path.unlink()

    src = sqlite3.connect(source_db_path)
    dst = sqlite3.connect(target_db_path)
    src.row_factory = sqlite3.Row

    # Recreate sermons + paragraphs schema (matches current chunks.sqlite).
    dst.execute('''
        CREATE TABLE sermons (
            date_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            language TEXT DEFAULT 'en'
        );
    ''')
    dst.execute('''
        CREATE TABLE paragraphs (
            date_id TEXT,
            paragraph_no INTEGER,
            sub_id TEXT DEFAULT '',
            text TEXT,
            PRIMARY KEY (date_id, paragraph_no, sub_id)
        );
    ''')

    # Copy sermon
    sermon = src.execute("SELECT date_id, title, source, language FROM sermons WHERE date_id = ?", (date_id,)).fetchone()
    if not sermon:
        raise RuntimeError(f"date_id not found in sermons table: {date_id}")
    dst.execute(
        "INSERT INTO sermons(date_id, title, source, language) VALUES (?, ?, ?, ?)",
        (sermon["date_id"], sermon["title"], sermon["source"], sermon["language"]),
    )

    # Copy paragraphs
    rows = src.execute(
        "SELECT date_id, paragraph_no, sub_id, text FROM paragraphs WHERE date_id = ? ORDER BY paragraph_no, sub_id",
        (date_id,),
    ).fetchall()
    if not rows:
        raise RuntimeError(f"No paragraphs found for date_id={date_id}")

    dst.executemany(
        "INSERT INTO paragraphs(date_id, paragraph_no, sub_id, text) VALUES (?, ?, ?, ?)",
        [(r["date_id"], r["paragraph_no"], r["sub_id"] or "", r["text"]) for r in rows],
    )

    # Create fresh chunks table (new Stage 2 schema)
    create_chunks_table(dst)
    dst.commit()
    src.close()
    dst.close()


def create_test_db_subset(
    source_db_path: Path,
    target_db_path: Path,
    date_ids: List[str],
):
    """
    Create a minimal test DB containing:
      - sermons rows for date_ids
      - all paragraphs rows for date_ids
      - a fresh chunks table (Stage 2 output)

    This is the multi-sermon version of `create_single_sermon_test_db()`.
    """
    if target_db_path.exists():
        target_db_path.unlink()

    src = sqlite3.connect(source_db_path)
    dst = sqlite3.connect(target_db_path)
    src.row_factory = sqlite3.Row

    # Recreate sermons + paragraphs schema (matches current chunks.sqlite).
    dst.execute('''
        CREATE TABLE sermons (
            date_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            language TEXT DEFAULT 'en'
        );
    ''')
    dst.execute('''
        CREATE TABLE paragraphs (
            date_id TEXT,
            paragraph_no INTEGER,
            sub_id TEXT DEFAULT '',
            text TEXT,
            PRIMARY KEY (date_id, paragraph_no, sub_id)
        );
    ''')

    # Copy sermons
    for did in date_ids:
        sermon = src.execute(
            "SELECT date_id, title, source, language FROM sermons WHERE date_id = ?",
            (did,),
        ).fetchone()
        if not sermon:
            raise RuntimeError(f"date_id not found in sermons table: {did}")
        dst.execute(
            "INSERT INTO sermons(date_id, title, source, language) VALUES (?, ?, ?, ?)",
            (sermon["date_id"], sermon["title"], sermon["source"], sermon["language"]),
        )

    # Copy paragraphs
    for did in date_ids:
        rows = src.execute(
            "SELECT date_id, paragraph_no, sub_id, text FROM paragraphs WHERE date_id = ? ORDER BY paragraph_no, sub_id",
            (did,),
        ).fetchall()
        if not rows:
            raise RuntimeError(f"No paragraphs found for date_id={did}")
        dst.executemany(
            "INSERT INTO paragraphs(date_id, paragraph_no, sub_id, text) VALUES (?, ?, ?, ?)",
            [(r["date_id"], r["paragraph_no"], r["sub_id"] or "", r["text"]) for r in rows],
        )

    create_chunks_table(dst)
    dst.commit()
    src.close()
    dst.close()


# ============================================================================
# Main Processing
# ============================================================================

def process_all_sermons(
    source_db_path: Path,
    target_db_path: Path,
    limit: int = None,
    date_id: Optional[str] = None,
    overwrite_chunks_table: bool = False,
):
    """
    Process all sermons and build chunks.
    
    Args:
        source_db_path: Path to chunks.sqlite with paragraphs
        target_db_path: Path to output database (may be same as source)
        limit: Optional limit on number of sermons to process
    """
    same_db = str(source_db_path.resolve()) == str(target_db_path.resolve())

    # IMPORTANT (SQLite):
    # If source_db == target_db and we use two connections, a writer can lock out the reader.
    # So we use ONE shared connection in that case.
    if same_db:
        source_conn = sqlite3.connect(source_db_path, timeout=60.0)
        target_conn = source_conn
    else:
        source_conn = sqlite3.connect(source_db_path, timeout=60.0)
        target_conn = sqlite3.connect(target_db_path, timeout=60.0)

    # Be a good SQLite citizen: wait a bit for locks and prefer WAL.
    # This helps if the user has another process briefly touching the DB.
    for conn in {source_conn, target_conn}:
        conn.execute("PRAGMA busy_timeout = 60000;")
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")

    if overwrite_chunks_table:
        target_conn.execute("DROP TABLE IF EXISTS chunks;")
        target_conn.execute("DROP INDEX IF EXISTS idx_chunks_date;")
        target_conn.execute("DROP INDEX IF EXISTS idx_chunks_range;")
        target_conn.commit()

    create_chunks_table(target_conn)
    
    cursor = source_conn.cursor()
    if date_id:
        all_date_ids = [date_id]
    else:
        cursor.execute('SELECT date_id FROM sermons ORDER BY date_id')
        all_date_ids = [row[0] for row in cursor.fetchall()]
        if limit:
            all_date_ids = all_date_ids[:limit]
    
    print(f'Processing {len(all_date_ids)} sermons...')
    print(f'Word budget: preferred {TARGET_WORDS_MIN}-{TARGET_WORDS_MAX}, hard max {HARD_MAX_WORDS}')
    print('Sentence splitting is applied only when required to avoid a non-final chunk under minimum.')
    print('='*70)
    
    total_chunks = 0
    total_paragraphs = 0
    # Note: We no longer treat pre-split paragraphs as special; sub_id is part of ref.
    
    for idx, date_id in enumerate(all_date_ids, 1):
        # Load paragraphs for this sermon (with auto-splitting)
        paragraphs = load_paragraphs_for_sermon(source_conn, date_id)
        
        if not paragraphs:
            print(f'[{idx}/{len(all_date_ids)}] {date_id}: No paragraphs, skipping')
            continue
        
        # Pack into chunks
        chunks = pack_paragraphs_into_chunks(paragraphs, date_id)
        
        # Insert chunks
        for chunk in chunks:
            insert_chunk(target_conn, chunk)
        
        total_chunks += len(chunks)
        total_paragraphs += len(paragraphs)
        
        # Progress indicator
        avg_words = sum(c.word_count for c in chunks) / len(chunks) if chunks else 0
        print(f'[{idx}/{len(all_date_ids)}] {date_id}: '
              f'{len(paragraphs)} paragraphs → {len(chunks)} chunks '
              f'(avg {avg_words:.0f} words/chunk)')
    
    # Commit all changes
    target_conn.commit()
    
    print('='*70)
    print(f'✓ Total sermons processed: {len(all_date_ids)}')
    print(f'✓ Total paragraphs: {total_paragraphs:,}')
    print(f'✓ Total chunks created: {total_chunks:,}')
    print(f'✓ Average chunks per sermon: {total_chunks/len(all_date_ids):.1f}')
    
    # Cleanup
    source_conn.close()
    if target_conn is not source_conn:
        target_conn.close()


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    """Main entry point for chunking pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build chunks from paragraphs (Stage 2)'
    )
    parser.add_argument(
        '--db-path',
        type=str,
        default=None,
        help='SQLite DB path to read paragraphs from / write chunks into (default: data/processed/chunks.sqlite)',
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Test mode: write a single-sermon DB to test_chunk.sqlite'
    )
    parser.add_argument(
        '--date-id',
        type=str,
        default=None,
        help='Process a single sermon date_id (required for --test)'
    )
    parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limit number of sermons to process (for testing)'
    )
    parser.add_argument(
        '--overwrite-chunks-table',
        action='store_true',
        help='Drop + recreate chunks table if it exists (destructive)'
    )
    
    args = parser.parse_args()
    
    # Paths
    base_dir = Path(__file__).parent.parent.parent
    source_db = Path(args.db_path) if args.db_path else (base_dir / 'data' / 'processed' / 'chunks.sqlite')
    
    if args.test:
        if not args.date_id and not args.limit:
            print('Error: --test requires either --date-id (single sermon) or --limit N (first N sermons)')
            sys.exit(1)
        target_db = base_dir / 'data' / 'processed' / 'test_chunk.sqlite'
        if args.date_id:
            print(f'TEST MODE: Building single-sermon DB for {args.date_id} → test_chunk.sqlite')
        else:
            print(f'TEST MODE: Building subset DB for first {args.limit} sermons → test_chunk.sqlite')
        print('='*70)
    else:
        target_db = source_db
        print(f'PRODUCTION MODE: Writing to {target_db}')
        print('='*70)
    
    # Verify source exists
    if not source_db.exists():
        print(f'Error: Source database not found: {source_db}')
        sys.exit(1)
    
    if args.test:
        if args.date_id:
            create_single_sermon_test_db(source_db, target_db, args.date_id)
            process_all_sermons(
                source_db_path=target_db,
                target_db_path=target_db,
                date_id=args.date_id,
                overwrite_chunks_table=True,  # always regenerate chunks in the test DB
            )
        else:
            # Choose first N sermons by date_id order (deterministic)
            src_conn = sqlite3.connect(source_db)
            cur = src_conn.cursor()
            cur.execute("SELECT date_id FROM sermons ORDER BY date_id LIMIT ?", (args.limit,))
            date_ids = [r[0] for r in cur.fetchall()]
            src_conn.close()

            create_test_db_subset(source_db, target_db, date_ids)
            process_all_sermons(
                source_db_path=target_db,
                target_db_path=target_db,
                overwrite_chunks_table=True,  # always regenerate chunks in the test DB
            )
    else:
        process_all_sermons(
            source_db_path=source_db,
            target_db_path=target_db,
            limit=args.limit,
            date_id=args.date_id,
            overwrite_chunks_table=args.overwrite_chunks_table,
        )
    
    print(f'\n✅ Chunks written to: {target_db}')


if __name__ == '__main__':
    main()
