import sys
import os
import subprocess
import tempfile
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import random

# --- Configuration ---
RAW_OUT = Path("data/raw/hinduism_raw.txt")
RAW_OUT.parent.mkdir(parents=True, exist_ok=True)

# Comparison target (~150k-200k tokens)
MAX_PASSAGES = 15000 
RANDOM_SEED = 42

def sanitize_text(text: str) -> str:
    """Removes internal line breaks to keep the verse intact."""
    if not isinstance(text, str):
        return ""
    # Replace carriage returns and newlines with a single space
    return text.replace('\r\n', ' ').replace('\n', ' ').replace('\r', ' ').strip()

def collect_from_kaggle_gita() -> list:
    """Fetches the small, reliable Bhagavad Gita dataset."""
    passages = []
    tmp_dir = Path(tempfile.mkdtemp())
    print("[Hinduism] Fetching Bhagavad Gita from Kaggle...")
    try:
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", "yashnarnaware/bhagavad-gita-versewise", "--unzip", "-p", str(tmp_dir)],
            capture_output=True, check=True, timeout=120
        )
        for fpath in tmp_dir.glob("*.csv"):
            df = pd.read_csv(fpath)
            str_cols = df.select_dtypes(include=['object', 'string']).columns
            target_col = max(str_cols, key=lambda c: df[c].astype(str).str.len().mean())
            
            # Sanitize while collecting
            cleaned = [sanitize_text(val) for val in df[target_col].dropna().astype(str)]
            passages.extend(cleaned)
    except Exception as e:
        print(f"[Warning] Kaggle Gita fetch failed: {e}")
    return passages

def collect_from_huggingface_vedas() -> list:
    """Fetches foundational Rig Veda text from a stable HF source."""
    passages = []
    print("[Hinduism] Fetching Vedas from Hugging Face (siddharthjadhav6565/vedas)...")
    try:
        ds = load_dataset("siddharthjadhav6565/vedas", split="train")
        
        for item in ds:
            raw_text = item.get('text') or item.get('content') or ""
            # Sanitize and check length
            text = sanitize_text(raw_text)
            if len(text.split()) > 10:
                passages.append(text)
                
    except Exception as e:
        print(f"[Warning] Hugging Face Vedas fetch failed: {e}")
    return passages

def main():
    all_passages = []
    
    # 1. Gita (Narrative/Philosophy)
    all_passages.extend(collect_from_kaggle_gita())
    
    # 2. Vedas (Foundational Scripture)
    all_passages.extend(collect_from_huggingface_vedas())

    # 3. Deduplicate (Already sanitized)
    unique_passages = list(set([p for p in all_passages if len(p.split()) > 10]))
    
    if len(unique_passages) > MAX_PASSAGES:
        print(f"[Info] Downsampling Hinduism to {MAX_PASSAGES} passages...")
        rng = random.Random(RANDOM_SEED)
        unique_passages = rng.sample(unique_passages, MAX_PASSAGES)

    if not unique_passages:
        print("[Error] No Hinduism data collected. Check network/Kaggle credentials.", file=sys.stderr)
        sys.exit(1)

    with open(RAW_OUT, "w", encoding="utf-8") as f:
        for line in unique_passages:
            f.write(line + "\n")
            
    approx_tokens = sum(len(p.split()) for p in unique_passages)
    print(f"\n[Success] Hinduism collection complete: {len(unique_passages)} passages (~{approx_tokens:,} tokens).")

if __name__ == "__main__":
    main()