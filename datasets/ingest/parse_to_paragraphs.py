"""
Stage 1: Parse PDFs to canonical paragraphs

Extracts:
- date_id (from filename)
- title (from PDF top)
- paragraphs with VGR numbering (if present)
- generates deterministic numbering if absent

Writes to SQLite:
- sermons table
- paragraphs table
"""

from __future__ import annotations

import argparse
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF


# ----------------------------
# Config
# ----------------------------

DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SQLITE_DB = PROCESSED_DIR / "chunks.sqlite"

# Paragraph size policy (project requirement):
# - Ideally, VGR paragraphs should not exceed ~800 words.
# - If we observe very large paragraphs in VGR-mode, that's a strong signal we missed
#   paragraph boundaries (parsing bug) and should improve Stage 1 detection.
MAX_PARAGRAPH_WORDS = 800

# VGR detection policy (user requirement):
# - "VGR must always win" means we only fall back to deterministic paragraphing when
#   VGR numbering is truly absent (not just partially detected).
# - We consider VGR numbering present only if we observe a sufficiently long consecutive run.
MIN_VGR_CONSECUTIVE_RUN = 12

# Sermons that require manual paragraph extraction.
#
# These are identified by the full-run warning:
#   "[WARN] VGR marker coverage looks truncated in <date_id> ..."
#
# For these `date_id`s we skip automated Stage 1 ingest entirely (no sermon row, no
# paragraphs). We'll manually extract + insert them later so we don't accidentally
# ship a low-quality deterministic fallback.
MANUAL_INGEST_DATE_IDS: set[str] = {
    "52-0817A",
    "53-0217",
    "53-0405M",
    "53-0729",
    "53-1018",
    "53-1130",
    "53-1206A",
    "54-0513",
    "54-0515",
    "55-0410E",
    "56-0428",
    "57-0127E",
    "57-0810",
    "57-0828",
    "57-0908E",
    "57-0915M",
    "57-0922E",
    "57-1002",
    "57-1006",
    "58-0130",
    "58-0309E",
    "58-0309M",
    "58-0618",
    "59-1129",
    "60-0611E",
    "60-1205",
    "60-1209",
    "61-0402",
    "61-0730M",
    "61-0806",
    "62-0318",
    "63-0112",
    "63-1229M",
    "64-0112",
    "64-0830E",
    "65-1127B",
}

# Paragraph number patterns.
#
# VGR paragraph numbers typically appear as:
#   - Inline: "2 Text starts..." or sometimes "2. Text..." / "2) Text..."
#   - Standalone: "2" on its own line, or sometimes "2." / "2)"
#
# We allow optional punctuation after the number to avoid silently missing paragraphs.
# Some PDFs extract paragraph markers as "21— Text..." or "21- Text..." (dash variants).
PARA_NUM_PATTERN_INLINE = re.compile(r"^(\d+)[\.\)\-–—]?\s+")  # "2 Text..." / "2. Text..." / "2) Text..." / "2— Text..."
PARA_NUM_PATTERN_STANDALONE = re.compile(r"^(\d+)[\.\)\-–—]?$")  # "2" / "2." / "2)" / "2—"

# Some VGR transcripts include sub-paragraph suffix markers like "1a", "1b", "50a", "50b".
# These should become separate paragraph rows using the existing `sub_id` column.
PARA_NUM_SUFFIX_INLINE = re.compile(r"^(\d+)([A-Za-z])[\.\)\-–—]?\s+")  # "50a Text..." / "1b Text..."
PARA_NUM_SUFFIX_STANDALONE = re.compile(r"^(\d+)([A-Za-z])[\.\)\-–—]?$")  # "50a" / "1b." / "50a—"

# Copyright section patterns
COPYRIGHT_PATTERNS = [
    r"Copyright\s+Notice",
    r"All\s+rights\s+reserved",
    r"VOICE\s+OF\s+GOD\s+RECORDINGS",
    r"www\.branham\.org",
    r"This\s+book\s+may\s+be\s+printed",
    r"©\d{4}\s*VGR",  # ©2002 VGR, ALL RIGHTS RESERVED
    r"ALL\s+RIGHTS\s+RESERVED",
    r"ENGLISH\s+©",  # Language tag followed by copyright
]


# ----------------------------
# Data structures
# ----------------------------


@dataclass
class Sermon:
    """Sermon metadata"""
    date_id: str
    title: str
    source: str  # PDF filename
    language: str = "en"


@dataclass
class Paragraph:
    """Paragraph with text"""
    date_id: str
    paragraph_no: int
    text: str
    sub_id: str = ""


# ----------------------------
# Paragraph number plausibility checks (Stage 1 hardening)
# ----------------------------
#
# Real-world problem:
# PDF extraction sometimes produces standalone numbers or "number + text" lines that are NOT VGR
# paragraph numbers (e.g., street addresses like "2002 Gano Avenue...").
#
# Without safeguards, these get misinterpreted as paragraph numbers, creating huge jumps such as:
#   ... 44, 45, 46, 47, 2002
#
# That breaks downstream expectations (chunk references, validation, manual review).

ADDRESS_WORDS = {
    # Full words
    "avenue", "street", "road", "drive", "lane", "boulevard", "highway", "route",
    # Common address components
    "box", "pobox",
    # States commonly seen in addresses (keep as full words only)
    "missouri", "arizona", "illinois", "california", "indiana", "kentucky", "ohio", "texas",
}

# Abbreviations must match as whole tokens (e.g., "dr" should NOT match inside "draw").
ADDRESS_ABBREV = {
    "ave", "st", "rd", "dr", "ln", "blvd", "hwy", "po", "p.o",
}


def looks_like_address_or_metadata(text_after_number: str) -> bool:
    """
    Heuristic: detect address-like / metadata-like continuations after a leading number.
    Example: "2002 Gano Avenue, St. Louis, Missouri..."
    """
    t = text_after_number.strip().lower()
    if not t:
        return False

    # Tokenize into words/abbreviations. This avoids substring false positives:
    #   - "dr" should not match inside "draw"
    #   - "rd" should not match inside "lord"
    tokens = re.findall(r"[a-z]+(?:\.[a-z]+)?", t)
    words = {tok.replace(".", "") for tok in tokens}

    if words & ADDRESS_WORDS:
        return True
    if words & ADDRESS_ABBREV:
        return True

    # Additional strong signal: "PO BOX" often appears as two tokens.
    if "po" in words and "box" in words:
        return True

    return False


def is_plausible_paragraph_number(
    candidate: int,
    current_para_no: Optional[int],
    *,
    is_first_number: bool,
) -> bool:
    """
    Decide whether a detected numeric token is plausibly a VGR paragraph number.

    Rules are conservative: we prefer rejecting suspicious numbers over creating bogus
    paragraph indices (which are extremely harmful downstream).
    """
    if candidate <= 0:
        return False

    # Hard reject obvious non-paragraphs (years).
    if 1900 <= candidate <= 2100:
        return False

    if is_first_number:
        # First numbered paragraph should be small (often 2; sometimes 1).
        return candidate <= 50

    if current_para_no is None:
        return candidate <= 50

    # Must be non-decreasing and reasonably close to expected next.
    if candidate <= current_para_no:
        return False

    # Once we've entered a numbering run, VGR paragraph numbers should be strictly sequential.
    # Our Stage 1 quality metric is "no gaps"; allowing gaps here masks extraction errors and
    # lets prayer-card range calls sneak in as bogus paragraph markers.
    if candidate != current_para_no + 1:
        return False

    return True

# ----------------------------
# Non-paragraph numeric line heuristics
# ----------------------------

def looks_like_non_paragraph_number_prefix(text_after_number: str) -> bool:
    """
    Detect common cases where a leading number is NOT a VGR paragraph number,
    even though it appears at the start of a line.

    Example (prayer card announcements):
      "85 back to 100 maybe."
      "From 1 to 100 in A."

    These are extremely common in sermon transcripts and can create artificial
    paragraph jumps (e.g., 35 -> 85).
    """
    t = (text_after_number or "").strip().lower()
    if not t:
        return False

    # Tokenization for range-based heuristics
    toks = re.findall(r"[a-z']+|\d+", t)
    if not toks:
        return False

    early = toks[:12]

    # Card code pattern (e.g., "U-75", "A-12", "U75").
    card_code = bool(re.search(r"\b[A-Z]{1,2}[-–—]?\d{1,3}\b", text_after_number))

    # We intentionally avoid broad rules like "contains 'to' + a digit" because sermons
    # frequently contain scripture references and enumerations ("to the 1st chapter", etc.).
    #
    # Instead, we target explicit range/announcement patterns.
    has_digit = any(tok.isdigit() for tok in early)
    digit_count = sum(1 for tok in early if tok.isdigit())

    # Explicit prayer-card / card-number phrases are only treated as non-paragraph
    # when they include numeric signals or card codes.
    if re.search(r"\bprayer\s+card(s)?\b", t) and (has_digit or card_code):
        return True
    if re.search(r"\bcard\s+number(s)?\b", t) and (has_digit or card_code):
        return True
    if re.search(r"\bwhat'?s\s+the\s+prayer\s+card\s+number\b", t) and (has_digit or card_code):
        return True
    if re.search(r"\bstart\s+about\b", t) and re.search(r"\bcard\b", t) and (has_digit or card_code):
        return True

    # "who has / raise your hand" lines are only treated as non-paragraphs when they
    # look like card announcements (codes, numbers, or explicit card language).
    if re.search(r"\bwho\s+has\b", t):
        if card_code or has_digit or re.search(r"\bcard(s)?\b", t):
            return True
    if re.search(r"\braise\s+your\s+hand\b", t):
        if card_code or has_digit or re.search(r"\bcard(s)?\b", t):
            return True

    # "from 1 to 100 ..." (range selection)
    if "from" in early and "to" in early and has_digit:
        return True

    # "back to 100 ..." (range selection)
    if "back" in early and "to" in early and has_digit:
        return True

    # Range without explicit "from" (e.g., "50 to 65 ...") is very common in
    # prayer-card announcements. Require at least two digits to avoid false positives
    # like "to the 1st chapter".
    if "to" in early and digit_count >= 2:
        return True

    # "up to 100" is another common prayer-card range call.
    # We only trigger this when a digit is present to avoid false positives.
    if re.search(r"\bup\s+to\b", t) and has_digit:
        return True

    # "60 or 65 ..." is another prayer-card style call. If a numbered line starts
    # with "or <number>", it's almost certainly not a paragraph boundary.
    if re.match(r"^or\s+\d{1,4}\b", t):
        return True

    # NOTE: Avoid broad "then ..." rules here. Legitimate prose frequently begins with
    # "Then ..." (e.g., "Then somebody ..."), and false positives create paragraph gaps.

    # Detect numeric ranges anywhere in the line, not just early tokens.
    # Example: "65. ... 50 to 65. ... line up ..." should be treated as a non-paragraph
    # marker because it's a prayer-card range call, not a paragraph boundary.
    if re.search(r"\b\d{1,4}\s+(to|\-|–|—)\s+\d{1,4}\b", t):
        return True

    # Card-code announcements without explicit phrases (e.g., "U-75") are not prose.
    if card_code and digit_count >= 1:
        return True

    return False
# ----------------------------
# PDF extraction
# ----------------------------


def remove_copyright_text(text: str) -> str:
    """
    Remove copyright notice section from text
    VGR PDFs typically have a copyright notice at the end
    Also removes metadata pages (sermon title, location, date repeated at end)
    """
    # Find where copyright section starts
    copyright_start = -1
    
    for pattern in COPYRIGHT_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            start_pos = match.start()
            # Update if this is earlier than previous matches
            if copyright_start == -1 or start_pos < copyright_start:
                copyright_start = start_pos
    
    # If copyright section found, truncate there
    if copyright_start > 0:
        # Keep some buffer before copyright (might catch partial paragraph)
        # Look for paragraph break before copyright
        before_copyright = text[:copyright_start]
        # Find last double newline before copyright
        last_break = before_copyright.rfind('\n\n')
        if last_break > 0:
            return text[:last_break]
        else:
            return text[:copyright_start]
    
    # Additional check: Look for metadata pattern at end
    # Format: "date_id + TitleNoSpaces + Location U.S.A." (without spacing)
    # Example: "65-0117AParadox WestwardHoHotel Phoenix,ArizonaU.S.A."
    # This appears concatenated at the end of the last paragraph
    metadata_pattern = r'\d{2}-\d{4}[SMABEX]?[A-Z][A-Za-z,\s]*U\.S\.A\.'
    match = re.search(metadata_pattern, text)
    if match:
        # Found metadata section - verify it's near the end (last 500 chars)
        start_pos = match.start()
        if start_pos > len(text) - 500:  # Only if near the end
            before_metadata = text[:start_pos]
            # Find last sentence/meaningful break before metadata
            # Look for end of sentence markers
            for marker in ['. ', '.\n', '! ', '!\n', '? ', '?\n']:
                last_break = before_metadata.rfind(marker)
                if last_break > 0 and last_break > start_pos - 200:
                    # Found a good breaking point
                    return text[:last_break + 1]  # Include the period
            
            # Fallback: just truncate at metadata start if we have content before
            if start_pos > 100:
                return text[:start_pos].rstrip()
    
    return text


def is_page_number(lines: list[str], line_idx: int) -> bool:
    """
    Check if a standalone number is a page number (not a paragraph number)
    
    Page numbers appear with "THE SPOKEN WORD" header:
    - Pattern 1: Number + "THE SPOKEN WORD" on next line
    - Pattern 2: "THE SPOKEN WORD" on previous line + Number
    - Pattern 3: Title + Number + "THE SPOKEN WORD" (less common)
    
    Returns True if this is a page number, False if it's a paragraph number
    """
    def _line_to_int(s: str) -> Optional[int]:
        """
        Safely convert a standalone numeric token to int.
        Accepts optional trailing punctuation (e.g., "35.").
        """
        s = (s or "").strip()
        m = re.fullmatch(r"(\d{1,4})[\\.)-]?", s)
        if not m:
            return None
        return int(m.group(1))

    # Pattern 1: Check if next line is "THE SPOKEN WORD"
    if line_idx + 1 < len(lines):
        next_line = lines[line_idx + 1].strip()
        if next_line == "THE SPOKEN WORD":
            return True
    
    # Pattern 2: Check if previous line is "THE SPOKEN WORD"
    # BUT: if there's already a page number 2 lines back, this is a VGR paragraph!
    if line_idx > 0:
        prev_line = lines[line_idx - 1].strip()
        if prev_line == "THE SPOKEN WORD":
            # Check if there's already a page number before the header
            if line_idx > 1:
                two_back = lines[line_idx - 2].strip()
                # If two_back is a number, it's the page number, so current is VGR paragraph
                two_back_num = _line_to_int(two_back)
                if two_back_num is not None and two_back_num < 300:  # Reasonable page number range
                    return False  # This is a VGR paragraph number, not a page number
            return True  # This is a page number
    
    # Pattern 3: Check if within 2 lines of "THE SPOKEN WORD"
    # (title, number, THE SPOKEN WORD) or (THE SPOKEN WORD, number, body)
    if line_idx > 1:
        two_back = lines[line_idx - 2].strip()
        if two_back == "THE SPOKEN WORD":
            return True
    
    if line_idx + 2 < len(lines):
        two_ahead = lines[line_idx + 2].strip()
        if two_ahead == "THE SPOKEN WORD":
            return True
    
    # Pattern 4: Check for title header pattern (all caps title on previous line)
    # This is more conservative - only if it's a known title pattern
    # AND the number is small (1-10, typical page numbers)
    if line_idx > 0:
        prev_line = lines[line_idx - 1].strip()
        current_num = _line_to_int(lines[line_idx])
        if current_num is None:
            return False
        
        # Check if previous line looks like a sermon title (all caps, mostly letters)
        # AND the number is in page number range (1-20)
        if (prev_line and current_num <= 20 and 
            len(prev_line) < 60 and prev_line.isupper()):
            # Calculate alpha ratio
            if len(prev_line) > 0:
                alpha_ratio = sum(c.isalpha() or c.isspace() for c in prev_line) / len(prev_line)
                if alpha_ratio > 0.85:
                    return True
    
    return False


def remove_page_headers_footers(text: str) -> str:
    """
    Remove page headers and footers that appear on each page
    
    Common patterns:
    - Page number + "THE SPOKEN WORD" (e.g., "2 THE SPOKEN WORD")
    - Title + page number (e.g., "FAITH IS THE SUBSTANCE 3")
    - Standalone page numbers adjacent to "THE SPOKEN WORD"
    
    NOTE: We must carefully distinguish page numbers from VGR paragraph numbers!
    """
    lines = text.split('\n')
    filtered_lines = []
    
    # Track indices to skip (page numbers adjacent to "THE SPOKEN WORD")
    skip_indices = set()
    
    # First pass: identify headers and mark adjacent numbers as page numbers
    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Pattern A: "THE SPOKEN WORD" with adjacent page numbers
        if stripped == "THE SPOKEN WORD":
            # Mark this line for removal
            skip_indices.add(i)
            
            # Check if previous line is a standalone number (page number)
            if i > 0:
                prev_stripped = lines[i-1].strip()
                if re.match(r'^\d{1,3}$', prev_stripped):
                    skip_indices.add(i-1)
            
            # Check if next line is a standalone number
            # But verify it's actually a page number, not a VGR paragraph number!
            if i + 1 < len(lines):
                next_stripped = lines[i+1].strip()
                if re.match(r'^\d{1,3}$', next_stripped):
                    # Only skip if it's actually a page number
                    if is_page_number(lines, i+1):
                        skip_indices.add(i+1)
        
        # Pattern B: Sermon title (all caps) followed by page number
        #
        # IMPORTANT:
        # Earlier versions were too aggressive and could remove legitimate paragraph numbers,
        # e.g. when a line like "SAITH THE LORD." (all caps with punctuation) was followed by "21".
        #
        # So we only apply this removal when it's very likely to be a page header:
        #   - Title line is ALL CAPS, mostly letters/spaces, and does NOT end with sentence punctuation
        #   - Next line is a small number
        #   - "THE SPOKEN WORD" appears near this header block (within a few lines)
        if (
            stripped
            and len(stripped) < 60
            and stripped.isupper()
            and len(stripped) > 5
            and stripped != "THE SPOKEN WORD"
            and not re.search(r"[.!?]$", stripped)  # don't treat sentence-ending caps as title
        ):
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / len(stripped)
            if alpha_ratio > 0.80 and i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if re.match(r"^\d{1,3}$", next_stripped):
                    # Require "THE SPOKEN WORD" near the header to avoid deleting VGR paragraph numbers.
                    window = [lines[k].strip() for k in range(max(0, i - 2), min(len(lines), i + 4))]
                    if any(w == "THE SPOKEN WORD" for w in window):
                        skip_indices.add(i)      # Remove title
                        skip_indices.add(i + 1)  # Remove page number
    
    # Second pass: filter lines
    for i, line in enumerate(lines):
        # Skip if marked
        if i in skip_indices:
            continue
        
        stripped = line.strip()
        
        # Skip empty lines (preserve them)
        if not stripped:
            filtered_lines.append(line)
            continue
        
        # Pattern 1: Page number + "THE SPOKEN WORD" inline
        # Example: "2 THE SPOKEN WORD"
        if re.match(r'^\d{1,3}\s+THE\s+SPOKEN\s+WORD$', stripped, re.IGNORECASE):
            continue
        
        # Pattern 2: Title (all caps, short) + page number at end
        # Example: "FAITH IS THE SUBSTANCE 3"
        if re.match(r'^[A-Z\s]{10,60}\s+\d{1,3}$', stripped):
            # Check if it's mostly uppercase and ends with number
            words = stripped.split()
            if words and words[-1].isdigit():
                # This looks like a header
                continue
        
        # Keep this line
        filtered_lines.append(line)
    
    return '\n'.join(filtered_lines)


# ----------------------------
# NEW: Geometry-based header/footer removal (preferred)
# ----------------------------
#
# Why:
# - Pattern-based stripping can accidentally delete real VGR paragraph markers,
#   especially standalone numbers that happen to follow ALL-CAPS lines.
# - Using PyMuPDF block bounding boxes is much more stable: headers/footers live
#   in consistent page regions across the corpus.
#
# Policy:
# - Drop blocks entirely inside a conservative top/bottom margin window.
# - Keep everything else verbatim (we want paragraph markers preserved).

def extract_text_from_pdf_blocks(
    pdf_path: Path,
    *,
    header_ratio: float = 0.08,
    footer_ratio: float = 0.06,
) -> str:
    """
    Extract text using PyMuPDF block geometry and remove headers/footers by position.

    This is intentionally conservative: we only delete blocks that sit fully within
    the header/footer regions. This reduces the chance of deleting real paragraph
    markers near the start of body text.
    """
    texts: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            page_h = float(page.rect.height) if page.rect else 0.0
            if page_h <= 0:
                # Fallback: if page geometry is missing, use legacy extraction.
                t = page.get_text() or ""
                if t:
                    texts.append(t)
                continue

            header_cut = page_h * header_ratio
            footer_start = page_h - (page_h * footer_ratio)

            def _is_header_footer_text(s: str) -> bool:
                """
                Decide whether a block looks like a repeated header/footer artifact.

                We intentionally keep this narrow to avoid deleting real paragraph markers
                that happen to sit high on a page.
                """
                t = (s or "").strip()
                if not t:
                    return True
                # The canonical repeated header line in VGR PDFs.
                if re.search(r"\bTHE\s+SPOKEN\s+WORD\b", t, flags=re.IGNORECASE):
                    return True
                # Inline page-number header variant: "2 THE SPOKEN WORD"
                if re.fullmatch(r"\d{1,3}\s+THE\s+SPOKEN\s+WORD", t, flags=re.IGNORECASE):
                    return True
                return False

            # Blocks are tuples: (x0, y0, x1, y1, text, block_no, block_type)
            blocks = page.get_text("blocks") or []
            kept: list[tuple[float, float, str]] = []

            # Detect whether this page has the canonical header text somewhere.
            # We'll use this to more carefully drop tiny page-number blocks without
            # deleting real paragraph markers that happen to be near the top of body text.
            has_spoken_word_header = any(
                (len(b) >= 5)
                and isinstance(b[4], str)
                and re.search(r"\bTHE\s+SPOKEN\s+WORD\b", (b[4] or ""), flags=re.IGNORECASE)
                for b in blocks
            )

            for b in blocks:
                if len(b) < 5:
                    continue
                x0, y0, x1, y1, text = float(b[0]), float(b[1]), float(b[2]), float(b[3]), str(b[4] or "")
                if not text.strip():
                    continue

                # If a block is in the top/bottom margin region, only drop it when it
                # looks like a known header/footer artifact.
                #
                # IMPORTANT: Do NOT broadly drop standalone numbers in these regions. Some
                # paragraph markers can sit close to the top of the body and would be lost.
                #
                # We only drop *tiny* standalone numbers when they are extremely close to the
                # top/bottom edge and the page has the canonical VGR header present.
                if has_spoken_word_header and re.fullmatch(r"\d{1,3}", text.strip()):
                    if y0 < (header_cut * 0.5):
                        continue
                    if y1 > (footer_start + ((page_h * footer_ratio) * 0.5)):
                        continue
                if y0 < header_cut and _is_header_footer_text(text):
                    continue
                if y1 > footer_start and _is_header_footer_text(text):
                    continue

                kept.append((y0, x0, text))

            # Sort for stable reading order.
            #
            # IMPORTANT: Some PDFs are effectively 2-column layouts after extraction.
            # A naïve (y, x) sort can interleave columns and scramble paragraph markers
            # (e.g., 3, 5, 2, 4 ...), which then looks like missing paragraphs.
            #
            # Heuristic:
            # - If we have substantial content on both left and right halves of the page,
            #   treat as 2 columns: read left column top-to-bottom, then right column.
            # - Otherwise, fall back to standard top-to-bottom, left-to-right.
            page_w = float(page.rect.width) if page.rect else 0.0
            two_col = False
            if page_w > 0 and len(kept) >= 12:
                left = sum(1 for (_y, x, _t) in kept if x < page_w * 0.45)
                right = sum(1 for (_y, x, _t) in kept if x > page_w * 0.55)
                if left >= 5 and right >= 5:
                    two_col = True

            if two_col and page_w > 0:
                kept2: list[tuple[int, float, float, str]] = []
                for y0, x0, text in kept:
                    col = 0 if x0 < (page_w * 0.5) else 1
                    kept2.append((col, y0, x0, text))
                kept2.sort(key=lambda t: (t[0], t[1], t[2]))
                page_text = "\n".join(t[3].rstrip() for t in kept2).strip()
            else:
                kept.sort(key=lambda t: (t[0], t[1]))
                page_text = "\n".join(t[2].rstrip() for t in kept).strip()
            if page_text:
                texts.append(page_text)
    finally:
        doc.close()

    return "\n".join(texts)


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from PDF using PyMuPDF (fitz)
    PyMuPDF provides excellent word spacing and text extraction quality
    """
    # Preferred: geometry-based extraction to avoid deleting real paragraph markers.
    full_text = extract_text_from_pdf_blocks(pdf_path)

    # Legacy fallback: if block extraction yields nothing (rare), use plain text + legacy stripper.
    if not full_text.strip():
        texts: list[str] = []
        doc = fitz.open(pdf_path)
        for page in doc:
            text = page.get_text()
            if text:
                texts.append(text)
        doc.close()
        full_text = "\n".join(texts)
        full_text = remove_page_headers_footers(full_text)

    # Remove copyright section + trailing metadata pages.
    return remove_copyright_text(full_text)


def clean_title(title_lines: list[str]) -> str:
    """
    Clean title from first few lines
    PyMuPDF extracts titles cleanly, so we just need basic cleanup
    """
    # Join lines and collapse multiple spaces
    combined = " ".join(title_lines)
    cleaned = re.sub(r"\s+", " ", combined).strip()
    
    return cleaned


def extract_date_id_from_filename(filename: str) -> str:
    """
    Extract date_id from filename
    Examples:
    - 47-0412M.pdf -> 47-0412M
    - 65-0117.pdf -> 65-0117
    """
    return Path(filename).stem


def extract_title_from_text(text: str, max_lines: int = 10) -> tuple[str, int]:
    """
    Extract title from first few lines of PDF text
    Title is typically at the very top, before paragraph 1
    Returns (title, line_index_where_title_ends)
    """
    lines = text.split('\n')
    
    # Find where title ends
    # Title characteristics:
    # - Usually all caps or mixed with spaces between letters
    # - Short lines (< 60 chars typically)
    # - At the very top
    # - Ends before substantial paragraph text
    
    title_lines = []
    title_end_idx = 0
    
    for i, line in enumerate(lines[:max_lines]):
        stripped = line.strip()
        
        # Skip empty lines
        if not stripped:
            continue
        
        # Stop if we hit a paragraph number (inline/standalone), including suffix markers like "1a"
        if (
            PARA_NUM_PATTERN_INLINE.match(stripped)
            or PARA_NUM_PATTERN_STANDALONE.match(stripped)
            or PARA_NUM_SUFFIX_INLINE.match(stripped)
            or PARA_NUM_SUFFIX_STANDALONE.match(stripped)
        ):
            title_end_idx = i
            break
        
        # Check if this looks like title (short, mostly caps)
        # or spaced letters pattern
        is_likely_title = (
            len(stripped) < 60 and  # Short line
            (
                stripped.isupper() or  # All caps
                re.match(r'^[A-Z\s]+$', stripped) or  # Letters and spaces
                re.search(r'[A-Z]\s[A-Z]', stripped)  # Spaced letters pattern
            )
        )
        
        if is_likely_title:
            title_lines.append(stripped)
            title_end_idx = i + 1
        else:
            # If we have some title lines already and hit non-title text, stop
            if title_lines:
                title_end_idx = i
                break
            # Otherwise this might be the start of paragraph 1 without a clear title
            title_end_idx = i
            break
        
        # Stop after collecting a few title lines
        if len(title_lines) >= 4:
            title_end_idx = i + 1
            break
    
    title = clean_title(title_lines) if title_lines else "Untitled"
    return title, title_end_idx


def parse_paragraphs_with_numbering(text: str, date_id: str, skip_lines: int = 0) -> list[Paragraph]:
    """
    Parse paragraphs when VGR numbering is present
    Handles two formats:
    1. Inline: "2 Text starts here..."
    2. Standalone: "2\nText starts here..."
    
    IMPORTANT: Must distinguish between page numbers and paragraph numbers!
    Page numbers appear with "THE SPOKEN WORD" or title headers.
    
    Paragraph 1 is usually unnumbered intro text
    
    skip_lines: number of lines to skip at start (title lines)
    """
    lines = text.split('\n')
    paragraphs = []
    current_para_no = None
    current_sub_id = ""
    current_para_text = []
    
    # Track if we've seen any numbered paragraphs (and the last accepted number).
    seen_numbered = False
    
    # Everything before first number is paragraph 1
    para_1_lines = []
    
    i = 0
    while i < len(lines):
        # Skip title lines
        if i < skip_lines:
            i += 1
            continue
        
        stripped = lines[i].strip()
        
        # Check for paragraph markers:
        # - inline: "2 Text..."
        # - standalone: "2" on its own line
        # - suffix inline: "2a Text..."
        # - suffix standalone: "2a"
        match_inline = PARA_NUM_PATTERN_INLINE.match(stripped)
        match_standalone = PARA_NUM_PATTERN_STANDALONE.match(stripped)
        match_suffix_inline = PARA_NUM_SUFFIX_INLINE.match(stripped)
        match_suffix_standalone = PARA_NUM_SUFFIX_STANDALONE.match(stripped)

        def _next_nonempty_line(start_idx: int, max_lookahead: int = 4) -> str:
            """
            Look ahead a few lines for the next non-empty line.
            This helps detect address-like constructs where a number is on its own line.
            """
            for j in range(start_idx, min(len(lines), start_idx + max_lookahead)):
                candidate_line = lines[j].strip()
                if candidate_line:
                    return candidate_line
            return ""
        
        if match_suffix_inline:
            para_no = int(match_suffix_inline.group(1))
            sub_id = (match_suffix_inline.group(2) or "").lower()
            text_after_number = stripped[match_suffix_inline.end():].strip()

            expected_next = (current_para_no + 1) if current_para_no is not None else None
            should_apply_nonpara_guards = (
                current_para_no is None
                or (para_no != current_para_no and (expected_next is None or para_no != expected_next))
            )

            # Avoid treating expected-next or same-number suffix markers as non-paragraphs; VGR numbering is authoritative.
            if should_apply_nonpara_guards and looks_like_non_paragraph_number_prefix(text_after_number):
                if not seen_numbered:
                    para_1_lines.append(stripped)
                elif current_para_no is not None:
                    current_para_text.append(stripped)
                i += 1
                continue

            # Plausibility:
            # - first number must be small
            # - otherwise allow same-number (suffix split) or +1
            if not seen_numbered:
                if para_no > 50:
                    para_1_lines.append(stripped)
                    i += 1
                    continue
            else:
                if current_para_no is None:
                    para_1_lines.append(stripped)
                    i += 1
                    continue
                if para_no != current_para_no and para_no != current_para_no + 1:
                    current_para_text.append(stripped)
                    i += 1
                    continue

            seen_numbered = True

            if current_para_no is not None and current_para_text:
                para_text = " ".join(current_para_text)
                if para_text:
                    paragraphs.append(Paragraph(date_id=date_id, paragraph_no=current_para_no, sub_id=current_sub_id, text=para_text))

            current_para_no = para_no
            current_sub_id = sub_id
            current_para_text = [text_after_number] if text_after_number else []
            i += 1

        elif match_inline:
            # Found an inline numbered paragraph
            para_no = int(match_inline.group(1))
            text_after_number = stripped[match_inline.end():].strip()
            expected_next = (current_para_no + 1) if current_para_no is not None else None

            # Special-case: prayer-card range calls where the **lower bound** is
            # the inline number itself (e.g., "51 to 61 ...").
            #
            # Here the range text immediately follows the number, so we can detect
            # a "to <number>" prefix and treat it as a non-paragraph marker.
            range_suffix_match = re.match(r"^to\s+(\d{1,4})\b", text_after_number.lower())
            if range_suffix_match and (expected_next is None or para_no != expected_next):
                try:
                    upper = int(range_suffix_match.group(1))
                except ValueError:
                    upper = None
                if upper is not None and upper >= para_no and (upper - para_no) <= 150:
                    if not seen_numbered:
                        para_1_lines.append(stripped)
                    elif current_para_no is not None:
                        current_para_text.append(stripped)
                    i += 1
                    continue

            # Special-case: prayer-card range calls where the **upper bound** is the
            # inline number itself (e.g., "65. ... 50 to 65 ...").
            #
            # In these cases, the range text may appear *after* the number, so we
            # can't rely on earlier tokens. We detect a pattern like "50 to" in the
            # trailing text and treat the number as a non-paragraph marker if the
            # implied range width is reasonable.
            range_prefix_match = re.search(r"\b(\d{1,4})\s+to\b", text_after_number.lower())
            if range_prefix_match and (expected_next is None or para_no != expected_next):
                try:
                    lower = int(range_prefix_match.group(1))
                except ValueError:
                    lower = None
                if lower is not None and 0 < lower <= para_no and (para_no - lower) <= 150:
                    # Treat as non-paragraph; append to current paragraph text.
                    if not seen_numbered:
                        para_1_lines.append(stripped)
                    elif current_para_no is not None:
                        current_para_text.append(stripped)
                    i += 1
                    continue

            # Harden: only accept numbers that are plausible *in sequence*.
            #
            # IMPORTANT: Do NOT use address heuristics as a primary gate for small paragraph numbers.
            # Words like "lane" are common in sermons and can create false positives.
            #
            # Additional guard: if the post-number text looks like prayer-card instructions
            # or other numeric ranges, it's not a paragraph boundary *when it would create
            # a jump*. If this is the expected next paragraph number, accept it even if
            # it sounds like an announcement — VGR numbering is authoritative for our
            # sequentiality metric.
            if (expected_next is None or para_no != expected_next) and looks_like_non_paragraph_number_prefix(text_after_number):
                if not seen_numbered:
                    para_1_lines.append(stripped)
                elif current_para_no is not None:
                    current_para_text.append(stripped)
                i += 1
                continue

            plausible = is_plausible_paragraph_number(
                para_no,
                current_para_no,
                is_first_number=not seen_numbered,
            )
            if not plausible:
                # Treat as normal text line
                if not seen_numbered:
                    para_1_lines.append(stripped)
                elif current_para_no is not None:
                    current_para_text.append(stripped)
                i += 1
                continue

            seen_numbered = True
            
            # Save previous paragraph if exists
            if current_para_no is not None and current_para_text:
                para_text = ' '.join(current_para_text)
                if para_text:  # Non-empty
                    paragraphs.append(Paragraph(
                        date_id=date_id,
                        paragraph_no=current_para_no,
                        sub_id=current_sub_id,
                        text=para_text
                    ))
            
            # Start new paragraph
            current_para_no = para_no
            current_sub_id = ""
            current_para_text = [text_after_number] if text_after_number else []
            i += 1
        
        elif match_suffix_standalone:
            # Suffix marker on its own line: "50a"
            para_no = int(match_suffix_standalone.group(1))
            sub_id = (match_suffix_standalone.group(2) or "").lower()

            # Do not treat these as page numbers.
            if not seen_numbered:
                if para_no > 50:
                    if stripped:
                        para_1_lines.append(stripped)
                    i += 1
                    continue
            else:
                if current_para_no is None:
                    if stripped:
                        para_1_lines.append(stripped)
                    i += 1
                    continue
                if para_no != current_para_no and para_no != current_para_no + 1:
                    current_para_text.append(stripped)
                    i += 1
                    continue

            seen_numbered = True

            if current_para_no is not None and current_para_text:
                para_text = " ".join(current_para_text)
                if para_text:
                    paragraphs.append(Paragraph(date_id=date_id, paragraph_no=current_para_no, sub_id=current_sub_id, text=para_text))

            current_para_no = para_no
            current_sub_id = sub_id
            current_para_text = []
            i += 1

        elif match_standalone:
            # Check if this is a page number (not a paragraph number)
            if is_page_number(lines, i):
                # Skip page numbers
                i += 1
                continue
            
            # Found a standalone paragraph number
            para_no = int(match_standalone.group(1))

            # Harden: reject implausible paragraph numbers.
            plausible = is_plausible_paragraph_number(
                para_no,
                current_para_no,
                is_first_number=not seen_numbered,
            )
            if not plausible:
                # Treat as normal text
                if not seen_numbered:
                    para_1_lines.append(stripped)
                elif current_para_no is not None:
                    current_para_text.append(stripped)
                i += 1
                continue

            seen_numbered = True
            
            # Save previous paragraph if exists
            if current_para_no is not None and current_para_text:
                para_text = ' '.join(current_para_text)
                if para_text:  # Non-empty
                    paragraphs.append(Paragraph(
                        date_id=date_id,
                        paragraph_no=current_para_no,
                        sub_id=current_sub_id,
                        text=para_text
                    ))
            
            # Start new paragraph
            current_para_no = para_no
            current_sub_id = ""
            current_para_text = []
            i += 1  # Skip the number line, text will be on next lines
        
        elif not seen_numbered:
            # Before any numbered paragraphs - this is paragraph 1
            if stripped:
                para_1_lines.append(stripped)
            i += 1
        
        elif current_para_no is not None:
            # Continuation of current numbered paragraph
            if stripped:
                current_para_text.append(stripped)
            i += 1
        
        else:
            i += 1
    
    # Save last paragraph
    if current_para_no is not None and current_para_text:
        para_text = ' '.join(current_para_text)
        if para_text:
            paragraphs.append(Paragraph(
                date_id=date_id,
                paragraph_no=current_para_no,
                sub_id=current_sub_id,
                text=para_text
            ))
    
    # Add paragraph 1 (unnumbered intro) if we found any numbered paragraphs
    if seen_numbered and para_1_lines:
        para_1_text = ' '.join(para_1_lines).strip()
        if para_1_text:
            # If the first numbered paragraph is already ¶1, don't create a duplicate row.
            if paragraphs and paragraphs[0].paragraph_no == 1:
                paragraphs[0] = Paragraph(
                    date_id=date_id,
                    paragraph_no=1,
                    sub_id=paragraphs[0].sub_id,
                    text=(para_1_text + " " + paragraphs[0].text).strip(),
                )
            else:
                paragraphs.insert(0, Paragraph(
                    date_id=date_id,
                    paragraph_no=1,
                    sub_id="",
                    text=para_1_text
                ))
    
    return sorted(paragraphs, key=lambda p: (p.paragraph_no, p.sub_id))


def has_vgr_numbering(paragraphs: list[Paragraph]) -> bool:
    """
    Determine whether VGR numbering is truly present.
    We require a consecutive run of at least MIN_VGR_CONSECUTIVE_RUN.
    """
    if not paragraphs:
        return False
    nos = sorted({p.paragraph_no for p in paragraphs})

    # If numbering starts unusually high, it's often not true VGR paragraph numbering
    # (common in "Questions and Answers" documents where small standalone numbers can be
    # page numbers or list indices, and the real body doesn't carry VGR markers).
    #
    # For VGR-numbered sermons, the first detected numbered paragraph is almost always 1 or 2.
    nos_wo_intro = [n for n in nos if n != 1]
    if not nos_wo_intro:
        return False
    if min(nos_wo_intro) > 5:
        return False

    longest = 1
    run = 1
    for a, b in zip(nos, nos[1:]):
        if b == a + 1:
            run += 1
            longest = max(longest, run)
        else:
            run = 1
    return longest >= MIN_VGR_CONSECUTIVE_RUN


def extract_raw_marker_ints(text: str, *, skip_lines: int = 0) -> set[int]:
    """
    Extract numeric markers that look like paragraph markers from the raw extracted text.
    We only consider line-start markers that match our paragraph-marker regexes.
    """
    lines = text.split("\n")
    if skip_lines > 0:
        lines = lines[skip_lines:]

    out: set[int] = set()
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        m = PARA_NUM_PATTERN_INLINE.match(s) or PARA_NUM_PATTERN_STANDALONE.match(s) or PARA_NUM_SUFFIX_INLINE.match(s) or PARA_NUM_SUFFIX_STANDALONE.match(s)
        if not m:
            continue
        try:
            out.add(int(m.group(1)))
        except Exception:
            continue
    return out


def find_last_pdf_paragraph_marker(text: str, *, skip_lines: int = 0) -> tuple[int, str] | None:
    """
    Find the last paragraph marker *as it appears in PDF order* (scan from end).

    IMPORTANT:
    - We do NOT use max(marker) because PDFs can contain unrelated numbers mid-document.
    - We scan from the end and return the first valid paragraph marker we encounter.
    - We exclude page numbers via `is_page_number()` for standalone numeric lines.

    Returns:
        (paragraph_no, sub_id) where sub_id is '' or 'a'/'b'/...
        or None if no marker was found.
    """
    lines = text.split("\n")
    if skip_lines > 0:
        lines = lines[skip_lines:]

    for idx in range(len(lines) - 1, -1, -1):
        s = lines[idx].strip()
        if not s:
            continue

        # Suffix markers first (e.g., "50a" or "50a Text...")
        m = PARA_NUM_SUFFIX_INLINE.match(s)
        if m:
            return (int(m.group(1)), (m.group(2) or "").lower())
        m = PARA_NUM_SUFFIX_STANDALONE.match(s)
        if m:
            return (int(m.group(1)), (m.group(2) or "").lower())

        # Normal markers (e.g., "50" or "50 Text...")
        m = PARA_NUM_PATTERN_INLINE.match(s)
        if m:
            return (int(m.group(1)), "")
        m = PARA_NUM_PATTERN_STANDALONE.match(s)
        if m:
            # Exclude page numbers
            if is_page_number(lines, idx):
                continue
            return (int(m.group(1)), "")

    return None


def generate_paragraphs_deterministic(text: str, date_id: str, skip_lines: int = 0) -> list[Paragraph]:
    """
    Generate paragraphs deterministically when no VGR numbering exists
    
    Strategy:
    - Split on double newlines (paragraph breaks)
    - Number sequentially starting from 1
    - Ignore very short lines (likely formatting artifacts)
    - For very short documents (like ceremonies), more lenient
    """
    lines = text.split('\n')
    
    # Skip title lines
    if skip_lines > 0:
        lines = lines[skip_lines:]
        text = '\n'.join(lines)
    
    # Split on multiple newlines (paragraph breaks)
    raw_paragraphs = re.split(r'\n\s*\n', text)
    
    paragraphs: list[Paragraph] = []
    para_no = 1
    
    # Determine minimum paragraph length based on document size
    # Short documents (< 2000 chars) are more lenient
    min_para_length = 10 if len(text) < 2000 else 20
    
    for raw_para in raw_paragraphs:
        # Clean up the paragraph
        lines = [line.strip() for line in raw_para.split('\n') if line.strip()]
        para_text = ' '.join(lines)
        
        # Skip very short paragraphs (likely artifacts)
        if len(para_text) < min_para_length:
            continue
        
        # In deterministic mode, we still enforce a hard cap by splitting on sentence boundaries
        # so we don't store giant blocks.
        words = para_text.split()
        if len(words) > MAX_PARAGRAPH_WORDS:
            # Sentence-aware split
            sentences = re.split(r"(?<=[.!?])\s+", para_text)
            acc: list[str] = []
            acc_words = 0

            def flush():
                nonlocal para_no, acc, acc_words
                if acc:
                    paragraphs.append(Paragraph(date_id=date_id, paragraph_no=para_no, text=" ".join(acc).strip()))
                    para_no += 1
                    acc = []
                    acc_words = 0

            for s in sentences:
                s = s.strip()
                if not s:
                    continue
                s_wc = len(s.split())

                # If a single sentence exceeds max, hard-split by words.
                if s_wc > MAX_PARAGRAPH_WORDS:
                    flush()
                    w = s.split()
                    for i in range(0, len(w), MAX_PARAGRAPH_WORDS):
                        chunk = " ".join(w[i:i + MAX_PARAGRAPH_WORDS]).strip()
                        if chunk:
                            paragraphs.append(Paragraph(date_id=date_id, paragraph_no=para_no, text=chunk))
                            para_no += 1
                    continue

                if acc_words + s_wc <= MAX_PARAGRAPH_WORDS:
                    acc.append(s)
                    acc_words += s_wc
                else:
                    flush()
                    acc.append(s)
                    acc_words = s_wc

            flush()
        else:
            paragraphs.append(Paragraph(
                date_id=date_id,
                paragraph_no=para_no,
                text=para_text
            ))
            para_no += 1
    
    return paragraphs


def clean_paragraph_metadata(para_text: str, date_id: str) -> str:
    """
    Remove metadata that might be appended to paragraph text
    Format: date_id + TitleNoSpaces + Location
    Example: "65-0117AParadox WestwardHoHotel Phoenix,ArizonaU.S.A."
    """
    # Pattern: date_id followed by concatenated text ending with location
    metadata_pattern = r'\s*\d{2}-\d{4}[SMABEX]?[A-Z][A-Za-z,\s]*U\.S\.A\.\s*$'
    cleaned = re.sub(metadata_pattern, '', para_text)
    
    # Also remove any trailing "ENGLISH" tags
    cleaned = re.sub(r'\s*ENGLISH\s*$', '', cleaned, flags=re.IGNORECASE)
    
    return cleaned.strip()


def parse_pdf_to_sermon(pdf_path: Path) -> tuple[Sermon, list[Paragraph]]:
    """
    Parse a PDF file into Sermon and Paragraphs
    Returns (Sermon, list of Paragraphs)
    """
    # Extract date_id from filename
    date_id = extract_date_id_from_filename(pdf_path.name)
    
    # Extract full text
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        raise ValueError(f"No text extracted from {pdf_path}")
    
    # Extract title and find where it ends
    title, title_end_line = extract_title_from_text(text)
    
    # Create sermon metadata
    sermon = Sermon(
        date_id=date_id,
        title=title,
        source=pdf_path.name,
        language="en"
    )
    
    # Parse paragraphs (skip title lines)
    # First try with VGR numbering
    paragraphs = parse_paragraphs_with_numbering(text, date_id, skip_lines=title_end_line)

    # VGR must always win: only fall back if VGR is truly absent (no real consecutive run)
    # OR if our parsed numbering clearly doesn't cover the raw marker range (truncation).
    if paragraphs:
        raw_ints = extract_raw_marker_ints(text, skip_lines=title_end_line)
        parsed_ints = {p.paragraph_no for p in paragraphs}
        if raw_ints and parsed_ints:
            raw_max = max(raw_ints)
            parsed_max = max(parsed_ints)
            coverage = len(raw_ints & parsed_ints) / max(1, len(raw_ints))
            # If strict sequential parsing misses one marker, it can reject all following markers
            # and silently truncate the numbering run. Detect that and fall back.
            if (raw_max - parsed_max) > 50 or coverage < 0.80:
                print(
                    f"  [WARN] VGR marker coverage looks truncated in {date_id} "
                    f"(raw_max={raw_max}, parsed_max={parsed_max}, coverage={coverage:.2f}). "
                    f"Falling back to deterministic paragraphing."
                )
                paragraphs = []

    if paragraphs and not has_vgr_numbering(paragraphs):
        print(
            f"  [WARN] No reliable VGR numbering run detected in {date_id} "
            f"(parsed {len({p.paragraph_no for p in paragraphs})} paras). Falling back to deterministic paragraphing."
        )
        paragraphs = []
    
    # If no paragraphs found, generate deterministically
    if not paragraphs:
        print(f"  [WARN] No VGR numbering found, generating paragraphs deterministically")
        paragraphs = generate_paragraphs_deterministic(text, date_id, skip_lines=title_end_line)
    
    # Post-process: Clean metadata from last paragraph
    if paragraphs:
        last_para = paragraphs[-1]
        cleaned_text = clean_paragraph_metadata(last_para.text, date_id)
        if cleaned_text != last_para.text:
            # Update last paragraph with cleaned text
            paragraphs[-1] = Paragraph(
                date_id=last_para.date_id,
                paragraph_no=last_para.paragraph_no,
                text=cleaned_text,
                sub_id=last_para.sub_id,
            )
    
    return sermon, paragraphs


# ----------------------------
# SQLite operations
# ----------------------------


def init_database(db_path: Path) -> sqlite3.Connection:
    """Initialize SQLite database with schema"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create sermons table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sermons (
            date_id TEXT PRIMARY KEY,
            title TEXT,
            source TEXT,
            language TEXT DEFAULT 'en'
        )
    """)
    
    # Create paragraphs table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS paragraphs (
            date_id TEXT,
            paragraph_no INTEGER,
            sub_id TEXT DEFAULT '',
            text TEXT,
            PRIMARY KEY (date_id, paragraph_no, sub_id)
        )
    """)
    
    conn.commit()
    return conn


def insert_sermon(conn: sqlite3.Connection, sermon: Sermon) -> None:
    """Insert sermon into database (or replace if exists)"""
    cursor = conn.cursor()
    cursor.execute("""
        INSERT OR REPLACE INTO sermons (date_id, title, source, language)
        VALUES (?, ?, ?, ?)
    """, (sermon.date_id, sermon.title, sermon.source, sermon.language))
    conn.commit()


def insert_paragraphs(conn: sqlite3.Connection, paragraphs: list[Paragraph]) -> None:
    """Insert paragraphs into database (or replace if exist)"""
    cursor = conn.cursor()
    for para in paragraphs:
        cursor.execute("""
            INSERT OR REPLACE INTO paragraphs (date_id, paragraph_no, sub_id, text)
            VALUES (?, ?, ?, ?)
        """, (para.date_id, para.paragraph_no, para.sub_id or "", para.text))
    conn.commit()


def get_sermon_count(conn: sqlite3.Connection) -> int:
    """Get total sermon count"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sermons")
    return cursor.fetchone()[0]


def get_paragraph_count(conn: sqlite3.Connection) -> int:
    """Get total paragraph count"""
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM paragraphs")
    return cursor.fetchone()[0]


# ----------------------------
# Main processing
# ----------------------------


def process_pdf(pdf_path: Path, conn: sqlite3.Connection) -> tuple[bool, str]:
    """
    Process a single PDF file
    Returns (success, message)
    """
    try:
        date_id = extract_date_id_from_filename(pdf_path.name)
        if date_id in MANUAL_INGEST_DATE_IDS:
            # Still extract enough to report the last PDF marker for debugging/manual work.
            text = extract_text_from_pdf(pdf_path)
            title, title_end_line = extract_title_from_text(text)
            last_marker = find_last_pdf_paragraph_marker(text, skip_lines=title_end_line)
            last_marker_ref = (
                f"{last_marker[0]}{last_marker[1]}" if last_marker and last_marker[1] else (str(last_marker[0]) if last_marker else "<none>")
            )
            return True, f"SKIPPED (manual ingest required; last_pdf_marker={last_marker_ref}; title={title})"

        sermon, paragraphs = parse_pdf_to_sermon(pdf_path)
        
        if not paragraphs:
            return False, "No paragraphs extracted"
        
        # Log helpers: last marker from PDF order and last parsed paragraph ref.
        text = extract_text_from_pdf(pdf_path)
        _title, title_end_line = extract_title_from_text(text)
        last_marker = find_last_pdf_paragraph_marker(text, skip_lines=title_end_line)
        last_marker_ref = (
            f"{last_marker[0]}{last_marker[1]}" if last_marker and last_marker[1] else (str(last_marker[0]) if last_marker else "<none>")
        )
        last_parsed = paragraphs[-1]
        last_parsed_ref = f"{last_parsed.paragraph_no}{last_parsed.sub_id}" if last_parsed.sub_id else str(last_parsed.paragraph_no)

        # Insert into database
        insert_sermon(conn, sermon)
        insert_paragraphs(conn, paragraphs)
        
        return True, f"{len(paragraphs)} paragraphs (last_pdf_marker={last_marker_ref}, last_parsed={last_parsed_ref})"
    
    except Exception as e:
        return False, str(e)


def manual_ingest_date_id(db_path: Path, pdf_dir: Path, date_id: str) -> int:
    """
    Manual ingest mode (one sermon at a time):
    - Reads the PDF for date_id
    - Uses the *raw* extraction (no header/footer geometry filtering) to avoid hiding markers
    - Inserts sermon row + all paragraphs into the given DB

    Returns:
      number of paragraph rows inserted
    """
    matches = list(pdf_dir.rglob(f"{date_id}.pdf"))
    if not matches:
        raise FileNotFoundError(f"Could not find PDF for date_id={date_id} under {pdf_dir}")
    pdf_path = matches[0]

    # Raw extraction: keep all text; strip only trailing copyright/metadata pages.
    # Use blocks (not filtered) to avoid dropping standalone numeric markers.
    texts: list[str] = []
    doc = fitz.open(pdf_path)
    try:
        for page in doc:
            blocks = page.get_text("blocks") or []
            kept: list[tuple[float, float, str]] = []
            for b in blocks:
                if len(b) < 5:
                    continue
                x0, y0, textb = float(b[0]), float(b[1]), str(b[4] or "")
                if not textb.strip():
                    continue
                kept.append((y0, x0, textb))
            kept.sort(key=lambda t: (t[0], t[1]))
            page_text = "\n".join(t[2].rstrip() for t in kept).strip()
            if page_text:
                texts.append(page_text)
    finally:
        doc.close()
    text = remove_copyright_text("\n".join(texts))

    title, title_end_line = extract_title_from_text(text)

    def _detect_best_vgr_run(markers_in_order: list[int]) -> tuple[int, int, int] | None:
        """
        Detect the best (longest) consecutive integer run in marker stream.

        Returns (run_start, run_end, run_len) or None.
        """
        if not markers_in_order:
            return None
        # Deduplicate while preserving order
        seen: set[int] = set()
        uniq: list[int] = []
        for n in markers_in_order:
            if n in seen:
                continue
            seen.add(n)
            uniq.append(n)

        uniq.sort()

        candidates: list[tuple[int, int, int]] = []

        run_s = uniq[0]
        run_len = 1
        for a, b in zip(uniq, uniq[1:]):
            if b == a + 1:
                run_len += 1
            else:
                candidates.append((run_s, a, run_len))
                run_s = b
                run_len = 1
        candidates.append((run_s, uniq[-1], run_len))

        # Manual ingest: choose the longest run overall (this preserves continuation
        # numbering like 304–552 or 554–932 for Hebrews/Q&A series).
        good = [c for c in candidates if c[2] >= MIN_VGR_CONSECUTIVE_RUN]
        if not good:
            return None
        # Tie-breaker: longer first, then smaller start
        good.sort(key=lambda t: (-t[2], t[0]))
        return good[0]

    def _parse_marker_driven(text_: str, date_id_: str, title_end_line_: int) -> list[Paragraph]:
        """
        Marker-driven paragraph parser intended for manual-ingest sermons.

        Goals:
        - Be maximally complete: split whenever a line-start marker appears.
        - Avoid bogus markers (page numbers / list indices / stray large numbers).
        - Prefer the most likely VGR numbering run: the longest consecutive run of markers.
        """
        # IMPORTANT:
        # We do NOT blindly skip title lines here because some PDFs place the first
        # paragraph marker very early. Instead:
        # - We still *ignore* non-marker title lines from becoming intro text.
        # - But we allow marker lines even if they appear before title_end_line_.
        lines = text_.split("\n")

        out: list[Paragraph] = []
        intro: list[str] = []
        cur_no: int | None = None
        cur_sub: str = ""
        buf: list[str] = []

        def flush():
            nonlocal buf
            if cur_no is None:
                return
            txt = " ".join(x.strip() for x in buf if x and x.strip()).strip()
            buf = []
            if txt:
                out.append(Paragraph(date_id=date_id_, paragraph_no=cur_no, sub_id=cur_sub, text=txt))

        def accept_marker(n: int) -> bool:
            if n <= 0:
                return False
            if 1900 <= n <= 2100:
                return False
            return True

        # First pass: collect candidate marker ints in appearance order (for run detection).
        marker_stream: list[int] = []
        for idx, ln in enumerate(lines):
            s = (ln or "").strip()
            if not s:
                continue
            m = PARA_NUM_SUFFIX_INLINE.match(s) or PARA_NUM_SUFFIX_STANDALONE.match(s) or PARA_NUM_PATTERN_INLINE.match(s) or PARA_NUM_PATTERN_STANDALONE.match(s)
            if not m:
                continue
            try:
                n = int(m.group(1))
            except Exception:
                continue
            if not accept_marker(n):
                continue
            marker_stream.append(n)

        best_run = _detect_best_vgr_run(marker_stream)
        run_start = run_end = None
        if best_run and best_run[2] >= MIN_VGR_CONSECUTIVE_RUN:
            run_start, run_end, _ = best_run

        # Manual ingest MUST preserve paragraph numbering exactly as in the PDF, since
        # these numbers are the canonical reference users will verify against.
        normalize_base = None

        for idx, ln in enumerate(lines):
            s = (ln or "").strip()
            if not s:
                continue
            if title_end_line_ and idx < title_end_line_:
                # Ignore title/header lines as content, but still allow a marker line through.
                maybe_marker = (
                    PARA_NUM_SUFFIX_INLINE.match(s)
                    or PARA_NUM_SUFFIX_STANDALONE.match(s)
                    or PARA_NUM_PATTERN_INLINE.match(s)
                    or PARA_NUM_PATTERN_STANDALONE.match(s)
                )
                if not maybe_marker:
                    continue

            msi = PARA_NUM_SUFFIX_INLINE.match(s)
            mss = PARA_NUM_SUFFIX_STANDALONE.match(s)
            mi = PARA_NUM_PATTERN_INLINE.match(s)
            ms = PARA_NUM_PATTERN_STANDALONE.match(s)

            if msi:
                n0 = int(msi.group(1))
                sub = (msi.group(2) or "").lower()
                after = s[msi.end():].strip()
                if accept_marker(n0) and (run_start is None or (run_start <= n0 <= run_end)):
                    n = (n0 - normalize_base + 1) if normalize_base is not None else n0
                    if cur_no is None:
                        intro_txt = " ".join(x.strip() for x in intro if x.strip()).strip()
                        if intro_txt:
                            after = (intro_txt + " " + after).strip() if after else intro_txt
                    flush()
                    cur_no, cur_sub = n, sub
                    buf = [after] if after else []
                    continue

            if mi:
                n0 = int(mi.group(1))
                after = s[mi.end():].strip()
                if accept_marker(n0) and (run_start is None or (run_start <= n0 <= run_end)):
                    n = (n0 - normalize_base + 1) if normalize_base is not None else n0
                    if cur_no is None:
                        intro_txt = " ".join(x.strip() for x in intro if x.strip()).strip()
                        if intro_txt:
                            after = (intro_txt + " " + after).strip() if after else intro_txt
                    flush()
                    cur_no, cur_sub = n, ""
                    buf = [after] if after else []
                    continue

            if mss:
                n0 = int(mss.group(1))
                sub = (mss.group(2) or "").lower()
                if accept_marker(n0) and (run_start is None or (run_start <= n0 <= run_end)):
                    n = (n0 - normalize_base + 1) if normalize_base is not None else n0
                    if cur_no is None:
                        intro_txt = " ".join(x.strip() for x in intro if x.strip()).strip()
                        if intro_txt:
                            buf = [intro_txt]
                    flush()
                    cur_no, cur_sub = n, sub
                    buf = [] if not buf else buf
                    continue

            if ms:
                n0 = int(ms.group(1))
                if accept_marker(n0) and (run_start is None or (run_start <= n0 <= run_end)):
                    n = (n0 - normalize_base + 1) if normalize_base is not None else n0
                    if cur_no is None:
                        intro_txt = " ".join(x.strip() for x in intro if x.strip()).strip()
                        if intro_txt:
                            buf = [intro_txt]
                    flush()
                    cur_no, cur_sub = n, ""
                    buf = [] if not buf else buf
                    continue

            # normal text
            if cur_no is None:
                intro.append(s)
            else:
                buf.append(s)

        flush()
        if not out:
            return []
        # de-dup (keep first) and sort
        seen = set()
        deduped: list[Paragraph] = []
        for p in sorted(out, key=lambda p: (p.paragraph_no, p.sub_id)):
            k = (p.paragraph_no, p.sub_id)
            if k in seen:
                continue
            seen.add(k)
            deduped.append(p)
        return deduped

    paragraphs = _parse_marker_driven(text, date_id, title_end_line)
    if not paragraphs:
        paragraphs = generate_paragraphs_deterministic(text, date_id, skip_lines=title_end_line)

    sermon = Sermon(date_id=date_id, title=title, source=pdf_path.name, language="en")

    conn = init_database(db_path)
    # Replace semantics: wipe existing rows for this date_id first so we don't leave stale
    # paragraph numbers behind when re-running manual ingest.
    cur = conn.cursor()
    cur.execute("DELETE FROM paragraphs WHERE date_id = ?", (date_id,))
    cur.execute("DELETE FROM sermons WHERE date_id = ?", (date_id,))
    conn.commit()
    insert_sermon(conn, sermon)
    insert_paragraphs(conn, paragraphs)
    conn.close()
    return len(paragraphs)


def sort_by_date_id(pdf_path: Path) -> tuple:
    """
    Sort key for chronological ordering by date_id
    Returns tuple (year, month, day, time_code)
    Example: 47-0412M -> (1947, 4, 12, 'M')
    """
    date_id = extract_date_id_from_filename(pdf_path.name)
    
    # Parse date_id: dd-mm-yy<T> or dd-mm-yy
    match = re.match(r"(\d{2})-(\d{4})([SMABEX])?$", date_id)
    if not match:
        # Fallback: sort by string
        return (9999, 99, 99, 'Z')
    
    yy, mmdd, time_code = match.groups()
    mm = mmdd[:2]
    dd = mmdd[2:]
    
    # Convert 2-digit year to 4-digit
    year = 1900 + int(yy)
    month = int(mm)
    day = int(dd)
    
    # Time code for ordering (empty string sorts before letters)
    time_code = time_code or ''
    
    return (year, month, day, time_code)


def process_directory(
    pdf_dir: Path,
    conn: sqlite3.Connection,
    limit: Optional[int] = None
) -> dict:
    """
    Process all PDFs in a directory (recursively)
    Processes in chronological order (earliest to latest)
    Returns statistics
    """
    # Find all PDFs
    pdf_files = list(pdf_dir.rglob("*.pdf"))
    
    # Sort chronologically by date_id
    pdf_files = sorted(pdf_files, key=sort_by_date_id)
    
    if limit:
        pdf_files = pdf_files[:limit]
    
    stats = {"total": len(pdf_files), "success": 0, "failed": 0}
    
    print(f"\nProcessing {stats['total']} PDF files (chronological order: earliest to latest)...")
    print(f"{'='*70}")
    
    for i, pdf_path in enumerate(pdf_files, 1):
        date_id = extract_date_id_from_filename(pdf_path.name)
        print(f"\n[{i}/{stats['total']}] {date_id}: ", end="")
        
        success, message = process_pdf(pdf_path, conn)
        
        if success:
            stats["success"] += 1
            print(f"✓ {message}")
        else:
            stats["failed"] += 1
            print(f"✗ {message}")
    
    print(f"\n{'='*70}")
    print(f"Completed: {stats['success']} success, {stats['failed']} failed")
    
    return stats


# ----------------------------
# CLI
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stage 1: Parse PDFs to canonical paragraphs"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: process only first 3 PDFs"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of PDFs to process"
    )
    parser.add_argument(
        "--pdf-dir",
        type=Path,
        default=DATA_DIR / "raw" / "pdfs",
        help="Directory containing PDFs (default: data/raw/pdfs)"
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=SQLITE_DB,
        help="SQLite DB path to write sermons/paragraphs into (default: data/processed/chunks.sqlite)",
    )
    parser.add_argument(
        "--date-id",
        type=str,
        default=None,
        help="Process only a single sermon by date_id (e.g., 64-0823M). Searches under --pdf-dir.",
    )
    parser.add_argument(
        "--rebuild-db",
        action="store_true",
        help="DANGER: wipe sermons/paragraphs tables before rebuilding from PDFs",
    )
    parser.add_argument(
        "--manual-date-id",
        type=str,
        default=None,
        help="Manual ingest a single denylisted sermon into --db-path (reads raw PDF text, inserts sermon+paragraphs).",
    )
    
    args = parser.parse_args()
    
    # Setup
    print(f"SQLite database: {args.db_path}")
    print(f"PDF directory: {args.pdf_dir}")
    
    # Initialize database
    conn = init_database(args.db_path)

    if args.rebuild_db:
        # Fully wipe tables so old bad rows (e.g., bogus paragraph_no=2002) don't linger.
        cur = conn.cursor()
        cur.execute("DELETE FROM paragraphs;")
        cur.execute("DELETE FROM sermons;")
        conn.commit()
        print("\n[REBUILD] Wiped sermons + paragraphs tables before ingesting PDFs.")
    
    # Determine limit
    limit = 3 if args.test else args.limit
    
    if args.manual_date_id:
        conn.close()
        n = manual_ingest_date_id(args.db_path, args.pdf_dir, args.manual_date_id)
        print(f"\n[MANUAL] Inserted {args.manual_date_id}: {n} paragraphs into {args.db_path}")
        return

    if args.test:
        print("\n[TEST MODE] Processing first 3 PDFs")

    # Process PDFs
    if args.date_id:
        # Find a matching PDF under the pdf root (supports nested year directories).
        matches = list(args.pdf_dir.rglob(f"{args.date_id}.pdf"))
        if not matches:
            print(f"\n[ERROR] Could not find PDF for date_id={args.date_id} under {args.pdf_dir}")
            conn.close()
            raise SystemExit(1)
        if len(matches) > 1:
            print(f"\n[WARN] Multiple PDFs matched {args.date_id}. Using first match: {matches[0]}")
        pdf_path = matches[0]
        print(f"\nProcessing single PDF: {pdf_path}")
        ok, msg = process_pdf(pdf_path, conn)
        stats = {"total": 1, "success": 1 if ok else 0, "failed": 0 if ok else 1}
        print(f"\nResult: {'✓' if ok else '✗'} {msg}")
    else:
        stats = process_directory(args.pdf_dir, conn, limit=limit)
    
    # Final summary
    sermon_count = get_sermon_count(conn)
    paragraph_count = get_paragraph_count(conn)
    
    print(f"\n{'='*70}")
    print("DATABASE SUMMARY")
    print(f"{'='*70}")
    print(f"Total sermons: {sermon_count}")
    print(f"Total paragraphs: {paragraph_count}")
    print(f"Database: {args.db_path}")
    
    conn.close()


if __name__ == "__main__":
    main()

