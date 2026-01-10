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

# Paragraph number pattern: starts with digit(s) followed by space or end of line
# Matches: "2 ", "3 ", "10 ", "123 " at start of line (inline with text)
# OR: "2", "3", "10" on their own line (PyMuPDF often extracts them this way)
PARA_NUM_PATTERN_INLINE = re.compile(r"^(\d+)\s+")  # "2 Text starts..."
PARA_NUM_PATTERN_STANDALONE = re.compile(r"^(\d+)$")  # "2" on its own line

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
            
            # Check if next line is a standalone number (page number)
            if i + 1 < len(lines):
                next_stripped = lines[i+1].strip()
                if re.match(r'^\d{1,3}$', next_stripped):
                    skip_indices.add(i+1)
        
        # Pattern B: Sermon title (all caps) followed by page number
        # Example: "FAITH IS THE SUBSTANCE" on one line, "23" on next line
        if (stripped and len(stripped) < 60 and 
            stripped.isupper() and len(stripped) > 5):
            # Check if it's mostly letters (sermon title pattern)
            alpha_ratio = sum(c.isalpha() or c.isspace() for c in stripped) / len(stripped)
            if alpha_ratio > 0.80:
                # This looks like a title - check if next line is a page number
                if i + 1 < len(lines):
                    next_stripped = lines[i+1].strip()
                    if re.match(r'^\d{1,3}$', next_stripped):
                        # This is a page number after a title
                        skip_indices.add(i)  # Remove title
                        skip_indices.add(i+1)  # Remove page number
    
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


def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from PDF using PyMuPDF (fitz)
    PyMuPDF provides excellent word spacing and text extraction quality
    """
    texts = []
    doc = fitz.open(pdf_path)
    
    for page in doc:
        text = page.get_text()
        if text:
            texts.append(text)
    
    doc.close()
    
    full_text = "\n".join(texts)
    
    # Remove page headers and footers
    full_text = remove_page_headers_footers(full_text)
    
    # Remove copyright section
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
        
        # Stop if we hit a paragraph number (inline or standalone)
        if PARA_NUM_PATTERN_INLINE.match(stripped) or PARA_NUM_PATTERN_STANDALONE.match(stripped):
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


def is_page_number(lines: list[str], line_idx: int) -> bool:
    """
    Check if a standalone number is a page number (not a paragraph number)
    
    Page numbers appear with "THE SPOKEN WORD" header:
    - Pattern 1: Number + "THE SPOKEN WORD" on next line
    - Pattern 2: "THE SPOKEN WORD" on previous line + Number
    - Pattern 3: Title + Number + "THE SPOKEN WORD" (less common)
    
    Returns True if this is a page number, False if it's a paragraph number
    """
    # Pattern 1: Check if next line is "THE SPOKEN WORD"
    if line_idx + 1 < len(lines):
        next_line = lines[line_idx + 1].strip()
        if next_line == "THE SPOKEN WORD":
            return True
    
    # Pattern 2: Check if previous line is "THE SPOKEN WORD"
    if line_idx > 0:
        prev_line = lines[line_idx - 1].strip()
        if prev_line == "THE SPOKEN WORD":
            return True
    
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
        current_num = int(lines[line_idx].strip())
        
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
    current_para_text = []
    
    # Track if we've seen any numbered paragraphs
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
        
        # Check for inline paragraph number: "2 Text..."
        match_inline = PARA_NUM_PATTERN_INLINE.match(stripped)
        # Check for standalone paragraph number: "2" on its own line
        match_standalone = PARA_NUM_PATTERN_STANDALONE.match(stripped)
        
        if match_inline:
            # Found an inline numbered paragraph
            para_no = int(match_inline.group(1))
            seen_numbered = True
            
            # Save previous paragraph if exists
            if current_para_no is not None and current_para_text:
                para_text = ' '.join(current_para_text)
                if para_text:  # Non-empty
                    paragraphs.append(Paragraph(
                        date_id=date_id,
                        paragraph_no=current_para_no,
                        text=para_text
                    ))
            
            # Start new paragraph
            current_para_no = para_no
            # Remove the number prefix from the line
            text_after_number = stripped[match_inline.end():]
            current_para_text = [text_after_number] if text_after_number else []
            i += 1
        
        elif match_standalone:
            # Check if this is a page number (not a paragraph number)
            if is_page_number(lines, i):
                # Skip page numbers
                i += 1
                continue
            
            # Found a standalone paragraph number
            para_no = int(match_standalone.group(1))
            seen_numbered = True
            
            # Save previous paragraph if exists
            if current_para_no is not None and current_para_text:
                para_text = ' '.join(current_para_text)
                if para_text:  # Non-empty
                    paragraphs.append(Paragraph(
                        date_id=date_id,
                        paragraph_no=current_para_no,
                        text=para_text
                    ))
            
            # Start new paragraph
            current_para_no = para_no
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
                text=para_text
            ))
    
    # Add paragraph 1 (unnumbered intro) if we found any numbered paragraphs
    if seen_numbered and para_1_lines:
        para_1_text = ' '.join(para_1_lines)
        if para_1_text:
            paragraphs.insert(0, Paragraph(
                date_id=date_id,
                paragraph_no=1,
                text=para_1_text
            ))
    
    return sorted(paragraphs, key=lambda p: p.paragraph_no)


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
    
    paragraphs = []
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
                text=cleaned_text
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
            text TEXT,
            PRIMARY KEY (date_id, paragraph_no)
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
            INSERT OR REPLACE INTO paragraphs (date_id, paragraph_no, text)
            VALUES (?, ?, ?)
        """, (para.date_id, para.paragraph_no, para.text))
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
        sermon, paragraphs = parse_pdf_to_sermon(pdf_path)
        
        if not paragraphs:
            return False, "No paragraphs extracted"
        
        # Insert into database
        insert_sermon(conn, sermon)
        insert_paragraphs(conn, paragraphs)
        
        return True, f"{len(paragraphs)} paragraphs"
    
    except Exception as e:
        return False, str(e)


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
    
    args = parser.parse_args()
    
    # Setup
    print(f"SQLite database: {SQLITE_DB}")
    print(f"PDF directory: {args.pdf_dir}")
    
    # Initialize database
    conn = init_database(SQLITE_DB)
    
    # Determine limit
    limit = 3 if args.test else args.limit
    
    if args.test:
        print("\n[TEST MODE] Processing first 3 PDFs")
    
    # Process PDFs
    stats = process_directory(args.pdf_dir, conn, limit=limit)
    
    # Final summary
    sermon_count = get_sermon_count(conn)
    paragraph_count = get_paragraph_count(conn)
    
    print(f"\n{'='*70}")
    print("DATABASE SUMMARY")
    print(f"{'='*70}")
    print(f"Total sermons: {sermon_count}")
    print(f"Total paragraphs: {paragraph_count}")
    print(f"Database: {SQLITE_DB}")
    
    conn.close()


if __name__ == "__main__":
    main()

