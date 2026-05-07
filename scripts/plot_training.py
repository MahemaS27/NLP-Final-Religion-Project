"""
plot_training.py
Reads logs/<religion>/train_log.json and generates performance figures.
Adds a 'Best Model' marker to the plots based on minimum validation loss.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOGS_DIR = Path("logs")
FIGURES_DIR = Path("results/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

RELIGIONS = ["christianity", "islam", "hinduism", "buddhism", "judaism"]

COLORS = {
    "christianity": "#4E79A7",
    "islam":        "#59A14F",
    "hinduism":     "#F28E2B",
    "buddhism":     "#E15759",
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

def load_log(religion: str) -> list | None:
    log_path = LOGS_DIR / religion / "train_log.json"
    if not log_path.exists():
        print(f"[Plot] WARNING: no log found for {religion} at {log_path}")
        return None
    with open(log_path) as f:
        return json.load(f)

def extract_series(records: list, key: str) -> tuple[list, list]:
    steps, values = [], []
    for r in records:
        if key in r:
            steps.append(r["step"])
            values.append(r[key])
    return steps, values

def mark_best_point(ax, steps, losses, color):
    """Finds and marks the index of the minimum loss."""
    min_loss = min(losses)
    min_idx = losses.index(min_loss)
    ax.scatter(steps[min_idx], min_loss, color="gold", edgecolor="black", 
               s=100, marker="*", zorder=5, label="Best Model")

def plot_individual(religion: str, records: list):
    with plt.rc_context(STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(f"{religion.capitalize()} — Training Curves", fontsize=14, fontweight="bold")
        color = COLORS.get(religion, "#333333")

        # Train loss
        steps, losses = extract_series(records, "train_loss")
        if steps:
            axes[0].plot(steps, losses, color=color, linewidth=1.5, label="Train loss")
            axes[0].set_xlabel("Step")
            axes[0].set_ylabel("Loss")
            axes[0].legend()
        
        # Val loss
        steps_v, losses_v = extract_series(records, "eval_loss")
        if steps_v:
            axes[1].plot(steps_v, losses_v, color=color, linewidth=1.5, linestyle="--", marker="o", markersize=4, label="Val loss")
            mark_best_point(axes[1], steps_v, losses_v, color)
            axes[1].set_xlabel("Step")
            axes[1].set_ylabel("Loss")
            axes[1].legend()

        plt.tight_layout()
        out = FIGURES_DIR / f"training_loss_{religion}.png"
        plt.savefig(out, bbox_inches="tight")
        plt.close()

def plot_combined(all_records: dict, loss_key: str, title: str, filename: str, mark_best=False):
    with plt.rc_context(STYLE):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_title(title, fontsize=14, fontweight="bold")
        
        for religion, records in all_records.items():
            steps, losses = extract_series(records, loss_key)
            if steps:
                ax.plot(steps, losses, color=COLORS.get(religion, "#333333"), linewidth=1.8, label=religion.capitalize())
                # If marking best, only mark it for the Validation Loss plot
                if mark_best and loss_key == "eval_loss":
                    mark_best_point(ax, steps, losses, COLORS.get(religion, "#333333"))
        
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Loss")
        ax.legend()
        plt.tight_layout()
        plt.savefig(FIGURES_DIR / filename, bbox_inches="tight")
        plt.close()

def main():
    all_records = {}
    for religion in RELIGIONS:
        records = load_log(religion)
        if records:
            all_records[religion] = records
            plot_individual(religion, records)

    if not all_records:
        print("[Plot] No logs found. Check logs/ directory.")
        return

    plot_combined(all_records, "train_loss", "All Training Losses", "training_loss_all.png", mark_best=False)
    plot_combined(all_records, "eval_loss", "All Validation Losses", "validation_loss_all.png", mark_best=True)
    plot_lr(all_records)
    print(f"\n[Plot] Figures saved to {FIGURES_DIR}/")

if __name__ == "__main__":
    main()