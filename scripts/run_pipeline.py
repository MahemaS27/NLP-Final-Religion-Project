import argparse
import subprocess
import sys
from pathlib import Path

ALL_RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]

STAGES = {
    "christianity": ("scripts/collect_christianity.py", Path("data/raw/christianity_raw.txt")),
    "islam":        ("scripts/collect_islam.py",        Path("data/raw/islam_raw.txt")),
    "hinduism":     ("scripts/collect_hinduism.py",     Path("data/raw/hinduism_raw.txt")),
    "judaism":      ("scripts/collect_judaism.py",      Path("data/raw/judaism_raw.txt")),
    "clean":        ("scripts/clean_all.py",            Path("data/cleaned")),
    "compile":      ("scripts/compile_corpus.py",       Path("data/final/corpus_manifest.json")),
}


def run(script: str, force: bool, output_path: Path, label: str):
    if not force and output_path.exists():
        print(f"[Pipeline] SKIP {label} — output already exists at {output_path}")
        return True
    print(f"\n[Pipeline] ── Running: {label} ──────────────────────────")
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        print(f"[Pipeline] ERROR in {label} (exit {result.returncode})", file=sys.stderr)
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description="Run Phase 1 data pipeline.")
    parser.add_argument(
        "--religions", nargs="+", choices=ALL_RELIGIONS, default=ALL_RELIGIONS,
        help="Which religions to collect (default: all)"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run stages even if output already exists"
    )
    parser.add_argument(
        "--skip-collect", action="store_true",
        help="Skip collection scripts, go straight to clean+compile"
    )
    args = parser.parse_args()

    success = True

    if not args.skip_collect:
        for religion in args.religions:
            script, output = STAGES[religion]
            ok = run(script, args.force, output, f"collect_{religion}")
            if not ok:
                print(f"[Pipeline] Stopping: collection failed for {religion}.", file=sys.stderr)
                sys.exit(1)

    clean_script, clean_output = STAGES["clean"]
    ok = run(clean_script, args.force, clean_output, "clean_all")
    if not ok:
        print("[Pipeline] Stopping: cleaning step failed.", file=sys.stderr)
        sys.exit(1)

    compile_script, compile_output = STAGES["compile"]
    ok = run(compile_script, args.force, compile_output, "compile_corpus")
    if not ok:
        print("[Pipeline] Stopping: compilation step failed.", file=sys.stderr)
        sys.exit(1)

    print("\n[Pipeline] ✓ Phase 1 complete. Outputs in data/final/")


if __name__ == "__main__":
    main()