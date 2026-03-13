"""Pipeline comparison diagram: correction-based vs rollout-based pair construction."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = Path("/Users/duy/Documents/build/cs224n-submission/figures/pipeline_diagram.png")


def rounded_box(ax, x, y, w, h, text, color, fontsize=9, text_color="white", alpha=1.0):
    box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08",
                          facecolor=color, edgecolor="white", linewidth=1.5, alpha=alpha, zorder=2)
    ax.add_patch(box)
    ax.text(x + w/2, y + h/2, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, fontweight="bold", zorder=3)


def arrow(ax, x1, y1, x2, y2, color="#555"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8),
                zorder=1)


def main():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    # --- Left: Correction pipeline ---
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Correction-based pairs", fontsize=13, fontweight="bold",
                 color="#c44e52", pad=12)

    # Problem
    rounded_box(ax, 1, 8.5, 8, 0.9, "Math problem", "#777")
    arrow(ax, 5, 8.5, 5, 7.9)

    # Rollouts
    rounded_box(ax, 0.5, 6.8, 4, 0.9, "8 rollouts (vLLM)", "#4c72b0")
    arrow(ax, 2.5, 6.8, 2.5, 6.2)

    # Error identification
    rounded_box(ax, 0.2, 5.1, 4.6, 0.9, "DEEP-GRPO error ID", "#6a4c93")
    arrow(ax, 2.5, 5.1, 2.5, 4.5)

    # Truncate + correct
    rounded_box(ax, 0.2, 3.3, 4.6, 0.9, "Truncate + \"Wait...\"", "#c44e52")
    arrow(ax, 2.5, 3.3, 2.5, 2.7)

    # Pair
    rounded_box(ax, 0.5, 1.5, 4, 0.9, "chosen: correction\nrejected: original", "#c44e52",
                fontsize=8, alpha=0.85)

    # Side annotation
    ax.text(7.5, 5.5, "69% contain\nself-repair\nphrases", ha="center", va="center",
            fontsize=9, color="#c44e52", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#c44e52", alpha=0.1, edgecolor="#c44e52"))
    ax.annotate("", xy=(5.0, 3.7), xytext=(6.3, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#c44e52", lw=1.2, linestyle="--"))

    # Result
    ax.text(5, 0.5, "Result: −0.8 to −3.8pp", ha="center", fontsize=10,
            color="#c44e52", fontweight="bold")

    # --- Right: Rollout pipeline ---
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("Rollout-based pairs", fontsize=13, fontweight="bold",
                 color="#4c72b0", pad=12)

    # Problem
    rounded_box(ax, 1, 8.5, 8, 0.9, "Math problem", "#777")
    arrow(ax, 5, 8.5, 5, 7.9)

    # Rollouts
    rounded_box(ax, 0.5, 6.8, 4, 0.9, "8 rollouts (vLLM)", "#4c72b0")
    arrow(ax, 2.5, 6.8, 2.5, 6.2)

    # Filter
    rounded_box(ax, 0.2, 5.1, 4.6, 0.9, "Mixed-outcome filter", "#6a4c93")

    # Split into correct/incorrect
    arrow(ax, 1.5, 5.1, 1.5, 4.5)
    arrow(ax, 3.5, 5.1, 3.5, 4.5)

    rounded_box(ax, 0.2, 3.5, 2.2, 0.8, "Correct", "#2a9d8f", fontsize=9)
    rounded_box(ax, 2.8, 3.5, 2.2, 0.8, "Incorrect", "#e76f51", fontsize=9)

    arrow(ax, 1.3, 3.5, 2.5, 2.7)
    arrow(ax, 3.9, 3.5, 2.5, 2.7)

    # Pair
    rounded_box(ax, 0.5, 1.5, 4, 0.9, "chosen: correct rollout\nrejected: incorrect rollout", "#4c72b0",
                fontsize=8, alpha=0.85)

    # Side annotation
    ax.text(7.5, 5.5, "Both from\nmodel's own\ndistribution", ha="center", va="center",
            fontsize=9, color="#4c72b0", fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#4c72b0", alpha=0.1, edgecolor="#4c72b0"))
    ax.annotate("", xy=(5.0, 3.9), xytext=(6.3, 5.0),
                arrowprops=dict(arrowstyle="-|>", color="#4c72b0", lw=1.2, linestyle="--"))

    # Result
    ax.text(5, 0.5, "Result: +0.6 to +3.2pp", ha="center", fontsize=10,
            color="#4c72b0", fontweight="bold")

    fig.tight_layout(w_pad=2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
