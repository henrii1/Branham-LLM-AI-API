#!/usr/bin/env python3
"""
Audit a stress run against the system_prompt.txt contract + FE postprocess
regexes. Reads the latest data/logs/chat_flow/stress/*_cases/ directory and
prints a per-case scorecard plus an aggregate summary.

Checks:
  - mode matches case_spec.expected.mode
  - refusal text verbatim (off_topic vs insufficient_context)
  - required sections present when expected (Quotes / References / Reader Note / Unverified)
  - no standalone "Evidence:" line (must be inline within Quotes)
  - Reader Note is a heading (### Reader Note:), not plain text
  - citation pill regex matches every bracketed citation
  - tool calls fired when case_spec.expected.should_call lists tools
  - language gate: non_english_ascii_only answers do NOT include sermon sections

Usage:
    uv run python scripts/audit_stress_run.py [path/to/cases_dir]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

# FE regexes (kept literally identical to chatPostprocess.ts + citations.ts).
SECTION_HEADING_RE = re.compile(
    r"^(#{2,3}\s+(?:Quotes|References|Unverified\s*\/?\s*External\s+Information)[s]?[:\s]?.*)",
    re.IGNORECASE | re.MULTILINE,
)
READER_NOTE_RE = re.compile(
    r"^(#{2,3})\s+(Reader\s*Note)[:\s]?(.*)", re.IGNORECASE | re.MULTILINE
)
CITATION_RE = re.compile(
    r"\[([^\]]+?\s[—–\-]{1,3}\s\d{2}-\d{4}[A-Z]?(?:\d)?:\s*¶\d+[a-z]?(?:[—–\-]+¶?\d+[a-z]?)?(?:[;,]\s*¶\d+[a-z]?(?:[—–\-]+¶?\d+[a-z]?)?)*)\]"
)
# Standalone Evidence: a line starting with "Evidence:" with no preceding `>` quote.
# Bad pattern = "Evidence:" on its own line *outside* a blockquote / outside a Quotes section.
STANDALONE_EVIDENCE_RE = re.compile(r"(?m)^Evidence:\s*\[")

# Refusal messages — must match config/default.yaml exactly.
REFUSAL_OFF_TOPIC = (
    "I can only answer questions based on William Branham's sermons. "
    "I don't have enough relevant information to answer your question."
)
REFUSAL_INSUFFICIENT = (
    "I searched through the sermon archives but couldn't find enough relevant context "
    "to give you a reliable answer on this. If you can provide more details — such as a "
    "sermon title, approximate date, or related topic — I'd be happy to search again."
)
ENGLISH_ONLY_MARKERS = ["English", "ingles", "Englisch", "inglese", "Anglais"]


def find_latest_cases_dir() -> Path:
    base = Path("data/logs/chat_flow/stress")
    if not base.exists():
        print(f"[err] {base} does not exist", file=sys.stderr)
        sys.exit(2)
    dirs = sorted(
        [d for d in base.iterdir() if d.is_dir() and d.name.endswith("_cases")],
        reverse=True,
    )
    if not dirs:
        print(f"[err] no *_cases dirs under {base}", file=sys.stderr)
        sys.exit(2)
    return dirs[0]


def audit_case(case_dir: Path) -> dict[str, Any]:
    report_path = case_dir / "run_report.json"
    answer_path = case_dir / "answer.txt"
    if not report_path.exists() or not answer_path.exists():
        return {"name": case_dir.name, "fatal": "missing artifacts"}

    report = json.loads(report_path.read_text(encoding="utf-8"))
    answer = answer_path.read_text(encoding="utf-8")
    spec = report.get("case_spec", {})
    name = spec.get("name") or case_dir.name
    category = spec.get("category", "")
    expected = spec.get("expected", {}) or {}
    actual_mode = (report.get("final", {}) or {}).get("mode")
    tool_counts = report.get("tool_call_counts", {}) or {}

    issues: list[str] = []
    passes: list[str] = []

    # --- 1) mode match ---
    expected_mode = expected.get("mode")
    if expected_mode:
        if expected_mode == "refusal_or_partial":
            if actual_mode in {"refusal", "answer"}:
                passes.append(f"mode={actual_mode} (either accepted)")
            else:
                issues.append(f"mode={actual_mode} expected refusal_or_partial")
        elif actual_mode == expected_mode:
            passes.append(f"mode={actual_mode}")
        else:
            issues.append(f"mode={actual_mode} expected {expected_mode}")

    # --- 2) refusal text verbatim ---
    if expected.get("refusal_type") == "off_topic":
        if REFUSAL_OFF_TOPIC in answer:
            passes.append("refusal: OFF_TOPIC verbatim")
        else:
            issues.append("refusal text NOT verbatim OFF_TOPIC")
    if expected.get("refusal_type") == "insufficient_context":
        if REFUSAL_INSUFFICIENT in answer:
            passes.append("refusal: INSUFFICIENT_CONTEXT verbatim")
        else:
            issues.append("refusal text NOT verbatim INSUFFICIENT_CONTEXT")

    # --- 3) sections present when expected ---
    found_sections = set()
    for m in SECTION_HEADING_RE.finditer(answer):
        line = m.group(1).lower()
        if "quote" in line:
            found_sections.add("Quotes")
        elif "reference" in line:
            found_sections.add("References")
        elif "unverified" in line or "external" in line:
            found_sections.add("Unverified")
    if READER_NOTE_RE.search(answer):
        found_sections.add("Reader Note")

    for want in expected.get("sections", []):
        if want in found_sections:
            passes.append(f"section: {want}")
        else:
            issues.append(f"MISSING section: {want}")

    for nope in expected.get("no_sections", []):
        if nope in found_sections:
            issues.append(f"UNEXPECTED section: {nope}")
        else:
            passes.append(f"no_section ok: {nope}")

    # --- 4) standalone Evidence ---
    if STANDALONE_EVIDENCE_RE.search(answer):
        issues.append("standalone 'Evidence:' line (must be inline within Quotes)")
    else:
        passes.append("no standalone Evidence")

    # --- 5) Reader Note must be a heading, not plain "Reader Note:" inline ---
    if "Reader Note" in answer and not READER_NOTE_RE.search(answer):
        # Allow it to be absent entirely; only flag if present-but-not-heading.
        if re.search(r"(?<!#\s)(?<!\*\*)Reader\s+Note:?", answer):
            issues.append("Reader Note present but not as ## or ### heading")

    # --- 6) citation pill regex sanity: every [TITLE — DATE_ID: ¶X] should match ---
    bracketed = re.findall(r"\[[^\]]+?:\s*¶[^\]]+\]", answer)
    bad = [b for b in bracketed if not CITATION_RE.search(b)]
    if bad:
        issues.append(f"{len(bad)} citation(s) don't match FE pill regex: {bad[:2]}")
    elif bracketed:
        passes.append(f"{len(bracketed)} citation(s) match pill regex")

    # --- 7) tool calls ---
    for tool in expected.get("should_call", []):
        if tool_counts.get(tool, 0) > 0:
            passes.append(f"tool: {tool} called")
        else:
            issues.append(f"tool: {tool} NOT called")

    # --- 8) language gate (non-English ASCII-only must have no sermon sections) ---
    if expected.get("language_gate") == "model":
        # Should be a polite English-only refusal in the user's language with no sections.
        if found_sections:
            issues.append(f"non-English query produced sections: {sorted(found_sections)}")
        else:
            passes.append("no sections (correct for non-English)")
        # And the answer should be short
        if len(answer) > 600:
            issues.append(f"non-English answer too long ({len(answer)} chars)")

    return {
        "name": name,
        "category": category,
        "expected_mode": expected_mode,
        "actual_mode": actual_mode,
        "tool_counts": tool_counts,
        "issues": issues,
        "passes": passes,
        "answer_chars": len(answer),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dir", nargs="?", help="path to a *_cases dir (default: latest)")
    args = p.parse_args()

    cases_dir = Path(args.dir) if args.dir else find_latest_cases_dir()
    print(f"Auditing: {cases_dir}")
    print("=" * 80)

    case_dirs = sorted(d for d in cases_dir.iterdir() if d.is_dir())
    if not case_dirs:
        print("No case dirs found.")
        sys.exit(1)

    audits = [audit_case(cd) for cd in case_dirs]

    # Per-case report
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in audits:
        by_category[a.get("category", "")].append(a)

    for cat in sorted(by_category):
        print(f"\n── Category: {cat} ──")
        for a in by_category[cat]:
            n_iss = len(a.get("issues", []))
            badge = "✓" if n_iss == 0 else f"✗ {n_iss} issue(s)"
            print(f"  [{badge}] {a['name']}  (mode={a.get('actual_mode')})")
            for iss in a.get("issues", []):
                print(f"      ⚠  {iss}")

    # Aggregate
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    total = len(audits)
    clean = sum(1 for a in audits if not a.get("issues"))
    print(f"Cases: {total} | clean: {clean} | with issues: {total - clean}")

    issue_freq: Counter[str] = Counter()
    for a in audits:
        for iss in a.get("issues", []):
            # Bucket by issue head
            key = iss.split(":")[0].split("(")[0].strip()
            issue_freq[key] += 1
    if issue_freq:
        print("\nIssue frequency:")
        for issue, count in issue_freq.most_common():
            print(f"  {count:3d} × {issue}")

    # Tool call summary
    tool_total: Counter[str] = Counter()
    for a in audits:
        for t, n in (a.get("tool_counts") or {}).items():
            tool_total[t] += n
    print("\nTool calls across run:")
    for t, n in tool_total.most_common():
        print(f"  {n:3d} × {t}")


if __name__ == "__main__":
    main()
