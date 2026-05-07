import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path

RAW_OUT = Path("data/raw/christianity_raw.txt")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "oswinrh/bible"

def check_kaggle_credentials():
    import os
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_json = Path("~/.kaggle/kaggle.json").expanduser().exists()
    if not has_env and not has_json:
        print("[Christianity] ERROR: No Kaggle credentials found.", file=sys.stderr)
        print("[Christianity]   Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or", file=sys.stderr)
        print("[Christianity]   place your kaggle.json at ~/.kaggle/kaggle.json", file=sys.stderr)
        sys.exit(1)

def download_dataset(tmp_dir: Path):
    print(f"[Christianity] Downloading {KAGGLE_DATASET} from Kaggle ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "--unzip", "-p", str(tmp_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[Christianity] ERROR: kaggle download failed:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    print(f"[Christianity] Download complete.")

def find_csv(tmp_dir: Path):
    csv_files = sorted(tmp_dir.glob("**/*.csv"))
    if not csv_files:
        print(f"[Christianity] ERROR: no CSV files found in {tmp_dir}", file=sys.stderr)
        sys.exit(1)
    # Prioritize files based on naming conventions often found in Bible datasets
    for f in csv_files:
        if "key" not in f.name.lower() and "meta" not in f.name.lower():
            return f
    return csv_files[0]

def extract_verses(csv_path: Path):
    print(f"[Christianity] Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    
    # Identify the text column
    text_col = next(
        (c for c in df.columns if c.lower() in ["text", "verse_text", "verse", "content", "t"]),
        None
    )
    
    if text_col is None:
        str_cols = df.select_dtypes(include="object").columns.tolist()
        text_col = max(str_cols, key=lambda c: df[c].dropna().str.len().mean())
        print(f"[Christianity] WARNING: guessed text column as '{text_col}'")
    else:
        print(f"[Christianity] Using text column: '{text_col}'")

    # EXTRACT AND SANITIZE:
    # 1. Drop NaNs and convert to string
    # 2. Replace internal newlines with spaces to keep the verse intact on one line
    # 3. Strip whitespace
    verses = (
        df[text_col]
        .dropna()
        .astype(str)
        .str.replace('\r\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.strip()
        .loc[lambda s: s.str.len() > 0]
        .tolist()
    )
    return verses

def main():
    check_kaggle_credentials()

    tmp_dir = Path(tempfile.mkdtemp()) / "kjv"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    download_dataset(tmp_dir)

    csv_path = find_csv(tmp_dir)
    verses = extract_verses(csv_path)

    print(f"[Christianity] Total verses collected: {len(verses)}")
    if not verses:
        print("[Christianity] ERROR: no verses extracted.", file=sys.stderr)
        sys.exit(1)

    # Write to file: Each verse is guaranteed to be one clean line
    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for v in verses:
            f.write(v + "\n")

    size_kb = RAW_OUT.stat().st_size // 1024
    print(f"[Christianity] Raw text saved to {RAW_OUT} ({size_kb} KB)")

if __name__ == "__main__":
    main()