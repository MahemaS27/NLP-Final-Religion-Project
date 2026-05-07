import sys
import random
import requests
from pathlib import Path
from collections import Counter
from tqdm import tqdm

RAW_OUT = Path("data/raw/judaism_raw.txt")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

BOOKS_JSON_URL = "https://raw.githubusercontent.com/Sefaria/Sefaria-Export/master/books.json"
TARGET_CATEGORIES = {"Torah", "Talmud"}

TALMUD_TRACTATES = {
    "Berakhot", "Shabbat", "Yoma", "Sukkah",
    "Taanit", "Megillah", "Sanhedrin", "Avot",
}
MAX_PASSAGES = 10000
RANDOM_SEED = 42

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "research-script/1.0"})

def sanitize_text(text: str) -> str:
    """Removes internal line breaks to keep the passage intact."""
    if not isinstance(text, str):
        return ""
    # Standardizing newlines to spaces to ensure single-line output
    return text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def fetch_books_index():
    print("[Judaism] Fetching books.json index ...")
    resp = SESSION.get(BOOKS_JSON_URL, timeout=30)
    resp.raise_for_status()
    return resp.json()

def filter_books(books_data):
    """Return books matching English merged files in target categories/tractates."""
    selected = []
    for book in books_data.get("books", []):
        if book.get("language") != "English":
            continue
        if book.get("versionTitle") != "merged":
            continue
        if not book.get("txt_url"):
            continue

        categories = set(book.get("categories", []))
        title = book.get("title", "")

        if not TARGET_CATEGORIES.intersection(categories):
            continue

        if "Talmud" in categories and title not in TALMUD_TRACTATES:
            continue

        selected.append({
            "title": title,
            "categories": list(categories),
            "txt_url": book["txt_url"],
        })

    return selected

def fetch_txt(url: str) -> list:
    """Download a merged.txt file, sanitize lines, and return non-blank passages."""
    try:
        resp = SESSION.get(url, timeout=30)
        if resp.status_code != 200:
            return []
        # Sanitize each line immediately upon fetching
        lines = [sanitize_text(l) for l in resp.text.splitlines()]
        return [l for l in lines if l] 
    except Exception as e:
        print(f"[Judaism] WARNING: fetch failed for {url}: {e}")
        return []

def main():
    books_data = fetch_books_index()
    selected = filter_books(books_data)

    print(f"[Judaism] Selected {len(selected)} books to download")
    if not selected:
        print("[Judaism] ERROR: no matching books found.", file=sys.stderr)
        sys.exit(1)

    # Show breakdown
    cat_counts = Counter(b["categories"][0] for b in selected if b["categories"])
    for cat, count in sorted(cat_counts.items()):
        print(f"[Judaism]   {cat}: {count} books")
    print(f"[Judaism]   Titles: {[b['title'] for b in selected]}")

    all_texts = []
    for book in tqdm(selected, desc="  Downloading", unit="book", ncols=80):
        lines = fetch_txt(book["txt_url"])
        print(f"[Judaism]   {book['title']}: {len(lines)} passages")
        all_texts.extend(lines)

    print(f"[Judaism] Total passages before cap: {len(all_texts)}")

    # Downsample if over the cap
    if len(all_texts) > MAX_PASSAGES:
        rng = random.Random(RANDOM_SEED)
        all_texts = rng.sample(all_texts, MAX_PASSAGES)
        print(f"[Judaism] Downsampled to {MAX_PASSAGES} passages (seed={RANDOM_SEED})")

    if not all_texts:
        print("[Judaism] ERROR: no text collected.", file=sys.stderr)
        sys.exit(1)

    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for t in all_texts:
            f.write(t + "\n")

    size_kb = RAW_OUT.stat().st_size // 1024
    print(f"[Judaism] Raw text saved to {RAW_OUT} ({size_kb} KB)")

if __name__ == "__main__":
    main()