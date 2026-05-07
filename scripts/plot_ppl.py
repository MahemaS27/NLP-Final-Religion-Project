"""
plot_ppl.py
barchat in the results folder
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("results/metrics")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RELIGIONS = ["christianity", "islam", "hinduism", "judaism"]

COLORS = {
    "christianity": "#4E79A7",
    "islam":        "#59A14F",
    "hinduism":     "#F28E2B",
    "judaism":      "#B07AA1",
}

STYLE = {
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
}

def main():
    ppl_data = {}
    for rel in RELIGIONS:
        path = RESULTS_DIR / f"{rel}_ppl.json"
        if path.exists():
            with open(path) as f:
                data = json.load(f)
                ppl_data[rel] = data["perplexity"]
        else:
            print(f"[Plot] WARNING: No PPL data found for {rel}")

    if not ppl_data:
        print("[Plot] No PPL data found. Run evaluate_ppl.py first.")
        return

    # Sort data for the plot
    sorted_ppl = dict(sorted(ppl_data.items(), key=lambda item: item[1]))

    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(8, 5))
        
        # Create bars using the specific colors
        bar_colors = [COLORS.get(r, "#333333") for r in sorted_ppl.keys()]
        bars = ax.bar(sorted_ppl.keys(), sorted_ppl.values(), color=bar_colors, alpha=0.8)
        
        ax.set_title("Model Perplexity by Religion (Lower is Better)", fontsize=14, fontweight='bold')
        ax.set_ylabel("Perplexity")
        ax.set_xlabel("Religious Corpus")

        # Label the bars
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, yval + 0.05, 
                    f"{yval:.2f}", ha='center', va='bottom', fontweight='bold')

        plt.tight_layout()
        out = FIGURES_DIR / "perplexity_comparison.png"
        plt.savefig(out)
        print(f"[Plot] Saved perplexity comparison to {out}")

if __name__ == "__main__":
    main()