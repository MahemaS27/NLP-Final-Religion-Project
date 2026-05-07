"""

  - Normalize to UTF-8
  - Strip verse numbers, footnotes, chapter headings, metadata
  - Filter non-English passages via langdetect
  - Remove duplicates
  - Write cleaned files to data/cleaned/<religion>_clean.txt
"""

import re
import sys
from pathlib import Path
from langdetect import detect, LangDetectException
from tqdm import tqdm

RAW_DIR = Path("data/raw")
CLEAN_DIR = Path("data/cleaned")
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]

# ── Text cleaning regexes ──────────────────────────────────────────────────────

# Verse/chapter markers like "1:1", "1.", "[1]", "(1)", "verse 1", "Chapter 1"
VERSE_NUMBER_RE = re.compile(
    r"^\s*(\d+[:\.]\d+|\[\d+\]|\(\d+\)|\d+\.)\s*",  # leading markers
)
CHAPTER_HEADING_RE = re.compile(
    r"^\s*(chapter|verse|book|surah|sura|juz|hadith|ruku|section|part|psalm|proverbs?)\s*[\d\w]*[:\.\-]?\s*$",
    re.IGNORECASE,
)
FOOTNOTE_RE = re.compile(r"\[\^?\d+\]|\{.*?\}|\(fn\..*?\)", re.IGNORECASE)
HTML_TAG_RE = re.compile(r"<[^>]+>")
ARABIC_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]+")
HEBREW_RE = re.compile(r"[\u0590-\u05FF\uFB1D-\uFDFF]+")
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]+")
EXTRA_WHITESPACE_RE = re.compile(r"\s{2,}")


def clean_line(line: str) -> str:
    """Apply all cleaning rules to a single line. Returns empty string to discard."""
    line = line.strip()
    if not line:
        return ""

    # Strip HTML tags
    line = HTML_TAG_RE.sub("", line)

    # Remove footnote markers
    line = FOOTNOTE_RE.sub("", line)

    # Skip pure chapter/verse header lines
    if CHAPTER_HEADING_RE.match(line):
        return ""

    # Strip leading verse number
    line = VERSE_NUMBER_RE.sub("", line)

    # Remove non-Latin script characters (Arabic, Hebrew, Devanagari)
    # These appear in bilingual sources — we want English only
    line = ARABIC_RE.sub("", line)
    line = HEBREW_RE.sub("", line)
    line = DEVANAGARI_RE.sub("", line)

    # Normalize whitespace
    line = EXTRA_WHITESPACE_RE.sub(" ", line).strip()

    # Discard very short lines (likely headers or artifacts)
    if len(line.split()) < 4:
        return ""

    return line


def is_english(text: str) -> bool:
    try:
        return detect(text) == "en"
    except LangDetectException:
        return False


def clean_religion(religion: str):
    raw_path = RAW_DIR / f"{religion}_raw.txt"
    clean_path = CLEAN_DIR / f"{religion}_clean.txt"

    if not raw_path.exists():
        print(f"[{religion.capitalize()}] SKIP: raw file not found at {raw_path}")
        return

    print(f"[{religion.capitalize()}] Reading {raw_path} ...")
    with open(raw_path, encoding="utf-8", errors="replace") as f:
        raw_lines = f.readlines()

    print(f"[{religion.capitalize()}] Raw lines: {len(raw_lines)}")

    cleaned = []
    non_english = 0
    duplicate_count = 0
    seen = set()

    for line in tqdm(raw_lines, desc=f"  Cleaning {religion}", unit="lines", ncols=80):
        line = clean_line(line)
        if not line:
            continue

        # Dedup
        key = line.lower()
        if key in seen:
            duplicate_count += 1
            continue
        seen.add(key)

        # Language verification
        if not is_english(line):
            non_english += 1
            continue

        cleaned.append(line)

    print(f"[{religion.capitalize()}] Removed {non_english} non-English passages")
    print(f"[{religion.capitalize()}] Removed {duplicate_count} duplicates")
    print(f"[{religion.capitalize()}] Clean passages remaining: {len(cleaned)}")

    if not cleaned:
        print(f"[{religion.capitalize()}] WARNING: no passages survived cleaning!", file=sys.stderr)
        return

    with open(clean_path, "w", encoding="utf-8") as f:
        for passage in cleaned:
            f.write(passage + "\n")

    size_kb = clean_path.stat().st_size // 1024
    print(f"[{religion.capitalize()}] Saved to {clean_path} ({size_kb} KB)\n")


def main():
    for religion in RELIGIONS:
        clean_religion(religion)
    print("[All] Cleaning complete.")


if __name__ == "__main__":
    main()