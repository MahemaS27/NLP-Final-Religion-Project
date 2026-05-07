"""
  - Lowercase all text
  - Split into 80/20 train/val per religion
  - Target token window: 50k–200k tokens per religion
    (uses whitespace tokenization as a fast proxy; ~1.3 words ≈ 1 token)
  - Shuffle before split (reproducible via fixed seed)
"""

import json
import random
import sys
from pathlib import Path

CLEAN_DIR = Path("data/cleaned")
FINAL_DIR = Path("data/final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

RELIGIONS = ["christianity", "islam", "hinduism",  "judaism"]

RANDOM_SEED = 42
TRAIN_RATIO = 0.80

# Approximate token targets play with this limt
MIN_WORDS = int(50_000 * 0.75)  
MAX_WORDS = int(200_000 * 0.75)


def word_count(lines):
    return sum(len(line.split()) for line in lines)


def approx_tokens(lines):
    """Rough token estimate: words / 0.75"""
    return int(word_count(lines) / 0.75)


def sample_to_target(lines, max_words):
    """If corpus exceeds max_words, randomly sample down."""
    if word_count(lines) <= max_words:
        return lines
    sampled = []
    total = 0
    for line in lines:
        wc = len(line.split())
        if total + wc > max_words:
            break
        sampled.append(line)
        total += wc
    return sampled


def compile_religion(religion: str, manifest: dict):
    clean_path = CLEAN_DIR / f"{religion}_clean.txt"
    if not clean_path.exists():
        print(f"[{religion.capitalize()}] SKIP: cleaned file not found at {clean_path}")
        manifest[religion] = {"status": "missing"}
        return

    with open(clean_path, encoding="utf-8") as f:
        lines = [line.strip().lower() for line in f if line.strip()]

    original_count = len(lines)
    original_tokens = approx_tokens(lines)
    print(f"[{religion.capitalize()}] Loaded {original_count} passages (~{original_tokens:,} tokens)")

    # Warn if under minimum
    if original_tokens < 50_000:
        print(f"[{religion.capitalize()}] WARNING: only ~{original_tokens:,} tokens; target is 50k–200k")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(lines)

    # Cap at max token target- for judiasm and then after new hinduism source this got huge
    if original_tokens > 200_000:
        lines = sample_to_target(lines, MAX_WORDS)
        print(f"[{religion.capitalize()}] Sampled down to {len(lines)} passages (~{approx_tokens(lines):,} tokens)")

    # Train / val split
    split_idx = int(len(lines) * TRAIN_RATIO)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Write outputs
    train_path = FINAL_DIR / f"{religion}_train.txt"
    val_path = FINAL_DIR / f"{religion}_val.txt"

    with open(train_path, "w", encoding="utf-8") as f:
        f.write("\n".join(train_lines) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        f.write("\n".join(val_lines) + "\n")

    train_tokens = approx_tokens(train_lines)
    val_tokens = approx_tokens(val_lines)

    print(f"[{religion.capitalize()}] Train: {len(train_lines)} passages (~{train_tokens:,} tokens) → {train_path}")
    print(f"[{religion.capitalize()}] Val:   {len(val_lines)} passages (~{val_tokens:,} tokens) → {val_path}\n")

    manifest[religion] = {
        "status": "ok",
        "original_passages": original_count,
        "original_tokens_approx": original_tokens,
        "final_passages": len(lines),
        "final_tokens_approx": approx_tokens(lines),
        "train": {
            "passages": len(train_lines),
            "tokens_approx": train_tokens,
            "path": str(train_path),
        },
        "val": {
            "passages": len(val_lines),
            "tokens_approx": val_tokens,
            "path": str(val_path),
        },
    }


def print_summary(manifest: dict):
    print("=" * 60)
    print("CORPUS SUMMARY")
    print("=" * 60)
    total_train = 0
    total_val = 0
    for religion, info in manifest.items():
        if info.get("status") != "ok":
            print(f"  {religion.capitalize():15s}  MISSING")
            continue
        t = info["train"]["tokens_approx"]
        v = info["val"]["tokens_approx"]
        total_train += t
        total_val += v
        in_range = "✓" if 50_000 <= (t + v) <= 200_000 else "⚠ out of range"
        print(f"  {religion.capitalize():15s}  train ~{t:>8,} tok  val ~{v:>7,} tok  {in_range}")
    print("-" * 60)
    print(f"  {'TOTAL':15s}  train ~{total_train:>8,} tok  val ~{total_val:>7,} tok")
    print("=" * 60)


def main():
    manifest = {}
    for religion in RELIGIONS:
        compile_religion(religion, manifest)

    manifest_path = FINAL_DIR / "corpus_manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[Compile] Manifest saved to {manifest_path}")

    print_summary(manifest)

    # Exit non-zero if any religion is missing
    missing = [r for r, v in manifest.items() if v.get("status") != "ok"]
    if missing:
        print(f"\n[Compile] WARNING: missing data for: {', '.join(missing)}", file=sys.stderr)
        sys.exit(1)

    print("\n[Compile] All done.")


if __name__ == "__main__":
    main()