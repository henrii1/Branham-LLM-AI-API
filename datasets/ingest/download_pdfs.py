"""
Download VGR sermon PDFs from branham.org

High-level approach (public + reliable):
A) Discover sermon IDs (date_id) by year via: https://branham.org/messageaudio/{yy}
B) For each date_id, fetch sermon page: https://branham.org/en/messagestream/ENG={date_id}
C) Parse the "Download PDF" anchor href (CloudFront URL) and stream-download it

IMPORTANT: Respect VGR copyright/terms. Use rate limiting and descriptive User-Agent.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Config
# ----------------------------

BASE_DIR = Path(__file__).parent.parent.parent / "data" / "raw" / "pdfs"
LANG = "ENG"

# Year range for WMB sermons (1947-1965)
YY_START = 47
YY_END = 65

# Rate limiting - be polite
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_REQUESTS_SEC = 0.5  # conservative default
USER_AGENT = "BranhamDatasetBuilder/0.1 (research/educational use)"


# ----------------------------
# URLs
# ----------------------------


def messageaudio_year_url(yy: int) -> str:
    """Example: https://branham.org/messageaudio/65"""
    return f"https://branham.org/messageaudio/{yy:02d}"


def messagestream_url(date_id: str, lang: str = "ENG") -> str:
    """Example: https://branham.org/en/messagestream/ENG%3D47-0412"""
    return f"https://branham.org/en/messagestream/{lang}%3D{date_id}"


# ----------------------------
# Data structures
# ----------------------------


@dataclass(frozen=True)
class SermonDownload:
    year_yyyy: int
    date_id: str
    pdf_url: str
    local_path: Path
    sha256: str


# ----------------------------
# HTTP helpers
# ----------------------------


def http_get(url: str, retries: int = 3) -> str:
    """HTTP GET with retries and exponential backoff"""
    headers = {"User-Agent": USER_AGENT}
    
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
            r.raise_for_status()
            return r.text
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            print(f"  [RETRY] Attempt {attempt + 1} failed: {e}. Waiting {wait}s...")
            time.sleep(wait)
    
    raise RuntimeError("Unreachable")


def stream_download(url: str, out_path: Path) -> None:
    """Stream download with atomic write (temp file + rename)"""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT}
    
    tmp_path = out_path.with_suffix(out_path.suffix + ".part")
    
    try:
        with requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT, stream=True) as r:
            r.raise_for_status()
            with open(tmp_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=1024 * 256):
                    if chunk:
                        f.write(chunk)
        
        # Atomic rename
        os.replace(tmp_path, out_path)
    except Exception:
        # Clean up partial download
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def sha256_file(path: Path) -> str:
    """Compute SHA256 hash of file"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


# ----------------------------
# Stage A: Discover date_ids
# ----------------------------

# Matches date_id format: dd-mm-yy<T> or dd-mm-yy
DATE_ID_RE = re.compile(r"\b\d{2}-\d{4}[SMABEX]?\b")


def discover_date_ids_for_year(yy: int) -> list[str]:
    """
    Fetch https://branham.org/messageaudio/{yy} and extract date_ids
    Returns sorted list of unique date_ids
    """
    url = messageaudio_year_url(yy)
    print(f"  Fetching year page: {url}")
    
    try:
        html = http_get(url)
        ids = sorted(set(DATE_ID_RE.findall(html)))
        print(f"  Found {len(ids)} sermons for year {yy_to_yyyy(yy)}")
        return ids
    except Exception as e:
        print(f"  [ERROR] Failed to fetch year {yy}: {e}")
        return []


# ----------------------------
# Stage B: Extract PDF URL
# ----------------------------


def extract_pdf_url_from_messagestream(html: str) -> Optional[str]:
    """
    Parse sermon page and return direct PDF download URL
    Tries multiple strategies for robustness
    """
    soup = BeautifulSoup(html, "html.parser")
    
    # Strategy 1: Find anchor with "Download PDF" text
    for a in soup.find_all("a"):
        txt = (a.get_text() or "").strip().lower()
        if "download pdf" in txt or "pdf" in txt:
            href = a.get("href")
            if href and href.endswith(".pdf"):
                return href
    
    # Strategy 2: Find any link ending with .pdf
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if href.endswith(".pdf"):
            return href
    
    # Strategy 3: Regex search for CloudFront PDF URLs
    m = re.search(
        r"https?://[^\"'\s]+\.cloudfront\.net/[^\"'\s]+\.pdf",
        html,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(0)
    
    # Strategy 4: Any PDF URL in the HTML
    m = re.search(
        r"https?://[^\"'\s]+\.pdf",
        html,
        flags=re.IGNORECASE
    )
    if m:
        return m.group(0)
    
    return None


def resolve_pdf_url(date_id: str) -> Optional[str]:
    """Resolve PDF URL for given date_id"""
    url = messagestream_url(date_id, LANG)
    
    try:
        html = http_get(url)
        pdf_url = extract_pdf_url_from_messagestream(html)
        
        if not pdf_url:
            print(f"  [WARN] No PDF URL found on page: {url}")
            return None
        
        return pdf_url
    except Exception as e:
        print(f"  [ERROR] Failed to resolve PDF URL for {date_id}: {e}")
        return None


# ----------------------------
# Stage C: Download
# ----------------------------


def yy_to_yyyy(yy: int) -> int:
    """Convert 2-digit year to 4-digit (47 -> 1947)"""
    return 1900 + yy


def target_pdf_path(year_yyyy: int, date_id: str, lang: str = "en") -> Path:
    """
    Returns: data/raw/pdfs/{lang}/{year}/{date_id}.pdf
    Example: data/raw/pdfs/en/1947/47-0412M.pdf
    """
    return BASE_DIR / lang / str(year_yyyy) / f"{date_id}.pdf"


def download_sermon(date_id: str, year_yyyy: int, manifest_path: Path) -> bool:
    """
    Download single sermon PDF and append to manifest
    Returns True if successful, False otherwise
    """
    out_path = target_pdf_path(year_yyyy, date_id)
    
    # Skip if already downloaded
    if out_path.exists():
        print(f"  [SKIP] {date_id} - already exists")
        return True
    
    print(f"  [DOWNLOAD] {date_id}")
    
    try:
        # Resolve PDF URL
        pdf_url = resolve_pdf_url(date_id)
        if not pdf_url:
            return False
        
        # Download
        stream_download(pdf_url, out_path)
        
        # Verify
        if not out_path.exists() or out_path.stat().st_size == 0:
            print(f"  [ERROR] {date_id} - download produced empty file")
            return False
        
        # Compute hash
        digest = sha256_file(out_path)
        
        # Write manifest record
        record = {
            "year": year_yyyy,
            "date_id": date_id,
            "lang": LANG,
            "source": "branham.org",
            "messagestream_url": messagestream_url(date_id, LANG),
            "pdf_url": pdf_url,
            "path": str(out_path.relative_to(BASE_DIR.parent.parent)),
            "sha256": digest,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        
        with open(manifest_path, "a", encoding="utf-8") as mf:
            mf.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"  [SUCCESS] {date_id} - {out_path.stat().st_size} bytes")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {date_id} - {e}")
        if out_path.exists():
            out_path.unlink()  # Clean up failed download
        return False


def download_year(yy: int, manifest_path: Path, limit: Optional[int] = None) -> dict:
    """
    Download all sermons for a given year
    Returns stats dict
    """
    year_yyyy = yy_to_yyyy(yy)
    print(f"\n{'='*60}")
    print(f"Year: {year_yyyy} (yy={yy})")
    print(f"{'='*60}")
    
    date_ids = discover_date_ids_for_year(yy)
    
    if not date_ids:
        print("  No sermons found")
        return {"total": 0, "success": 0, "failed": 0, "skipped": 0}
    
    if limit:
        date_ids = date_ids[:limit]
        print(f"  [TEST MODE] Limiting to first {limit} sermons")
    
    stats = {"total": len(date_ids), "success": 0, "failed": 0, "skipped": 0}
    
    for i, date_id in enumerate(date_ids, 1):
        print(f"\n[{i}/{len(date_ids)}] Processing {date_id}:")
        
        out_path = target_pdf_path(year_yyyy, date_id)
        if out_path.exists():
            stats["skipped"] += 1
            print(f"  [SKIP] Already exists")
            continue
        
        success = download_sermon(date_id, year_yyyy, manifest_path)
        
        if success:
            stats["success"] += 1
        else:
            stats["failed"] += 1
        
        # Rate limiting
        time.sleep(SLEEP_BETWEEN_REQUESTS_SEC)
    
    print(f"\n{'='*60}")
    print(f"Year {year_yyyy} complete: {stats}")
    print(f"{'='*60}")
    
    return stats


# ----------------------------
# Main
# ----------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Download VGR sermon PDFs")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Test mode: download only first 3 sermons from 1947"
    )
    parser.add_argument(
        "--year",
        type=int,
        help="Download specific year only (e.g., 47 for 1947)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of sermons per year (for testing)"
    )
    
    args = parser.parse_args()
    
    # Setup
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = BASE_DIR / "download_manifest.jsonl"
    
    print(f"Download location: {BASE_DIR}")
    print(f"Manifest: {manifest_path}")
    
    # Determine years to download
    if args.test:
        years = [47]  # 1947 only
        limit = 3
        print("\n[TEST MODE] Downloading first 3 sermons from 1947")
    elif args.year:
        years = [args.year]
        limit = args.limit
    else:
        years = range(YY_START, YY_END + 1)
        limit = args.limit
    
    # Download
    all_stats = {}
    for yy in years:
        stats = download_year(yy, manifest_path, limit=limit)
        all_stats[yy] = stats
    
    # Final summary
    print(f"\n{'='*60}")
    print("FINAL SUMMARY")
    print(f"{'='*60}")
    
    total_success = sum(s["success"] for s in all_stats.values())
    total_failed = sum(s["failed"] for s in all_stats.values())
    total_skipped = sum(s["skipped"] for s in all_stats.values())
    total_all = sum(s["total"] for s in all_stats.values())
    
    print(f"Total sermons: {total_all}")
    print(f"  Success: {total_success}")
    print(f"  Failed: {total_failed}")
    print(f"  Skipped (already downloaded): {total_skipped}")
    print(f"\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()

