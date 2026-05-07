import sys
import subprocess
import tempfile
import pandas as pd
from pathlib import Path

RAW_OUT = Path("data/raw/islam_raw.txt")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

KAGGLE_DATASET = "imrankhan197/the-quran-dataset"
CSV_NAME = "The Quran Dataset.csv"
TEXT_COL = "ayah_en"


def check_kaggle_credentials():
    import os
    has_env = os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")
    has_json = Path("~/.kaggle/kaggle.json").expanduser().exists()
    if not has_env and not has_json:
        print("[Islam] ERROR: No Kaggle credentials found.", file=sys.stderr)
        print("[Islam]   Set KAGGLE_USERNAME and KAGGLE_KEY env vars, or", file=sys.stderr)
        print("[Islam]   place your kaggle.json at ~/.kaggle/kaggle.json", file=sys.stderr)
        sys.exit(1)


def main():
    check_kaggle_credentials()

    tmp_dir = Path(tempfile.mkdtemp()) / "quran"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Islam] Downloading {KAGGLE_DATASET} from Kaggle ...")
    result = subprocess.run(
        ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
         "--unzip", "-p", str(tmp_dir)],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"[Islam] ERROR: kaggle download failed:\n{result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)

    csv_path = tmp_dir / CSV_NAME
    if not csv_path.exists():
        found = list(tmp_dir.glob("**/*.csv"))
        print(f"[Islam] ERROR: expected '{CSV_NAME}' not found.", file=sys.stderr)
        print(f"[Islam]   Files downloaded: {[f.name for f in found]}", file=sys.stderr)
        sys.exit(1)

    print(f"[Islam] Reading {csv_path.name} ...")
    df = pd.read_csv(csv_path)
    print(f"[Islam] Shape: {df.shape}  |  Columns: {list(df.columns)}")
    print(f"[Islam] Sample:\n{df[[TEXT_COL]].head(3).to_string()}\n")

    # EXTRACT AND SANITIZE:
    # 1. Drop NaNs and convert to string
    # 2. Flatten internal newlines to keep the ayah intact
    # 3. Strip whitespace
    verses = (
        df[TEXT_COL]
        .dropna()
        .astype(str)
        .str.replace('\r\n', ' ', regex=False)
        .str.replace('\n', ' ', regex=False)
        .str.strip()
        .loc[lambda s: s.str.len() > 0]
        .tolist()
    )

    print(f"[Islam] Total ayahs collected: {len(verses)}")
    if not verses:
        print("[Islam] ERROR: no verses extracted.", file=sys.stderr)
        sys.exit(1)

    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for v in verses:
            f.write(v + "\n")

    size_kb = RAW_OUT.stat().st_size // 1024
    print(f"[Islam] Raw text saved to {RAW_OUT} ({size_kb} KB)")


if __name__ == "__main__":
    main()