"""Diagram showing shared-prefix gradient cancellation in DPO vs SFT."""
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

OUT = Path("/Users/duy/Documents/build/cs224n-submission/figures/gradient_cancellation.png")


def main():
    fig, axes = plt.subplots(2, 1, figsize=(8, 4.0), gridspec_kw={"height_ratios": [1, 1]})

    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 2)
        ax.axis("off")

    # --- Top: SFT ---
    ax = axes[0]
    ax.text(0.1, 1.7, "SFT gradient", fontsize=11, fontweight="bold", va="top")

    # Shared prefix (large)
    prefix = FancyBboxPatch((0.1, 0.4), 7.0, 1.0, boxstyle="round,pad=0.05",
                             facecolor="#e8e8e8", edgecolor="#999", linewidth=1)
    ax.add_patch(prefix)
    ax.text(3.6, 0.9, "Shared prefix (~90% of tokens)", ha="center", va="center",
            fontsize=9, color="#666")

    # Suffix (small)
    suffix = FancyBboxPatch((7.2, 0.4), 2.5, 1.0, boxstyle="round,pad=0.05",
                             facecolor="#2a9d8f", edgecolor="#1a7a6e", linewidth=1, alpha=0.7)
    ax.add_patch(suffix)
    ax.text(8.45, 0.9, "Suffix\n(answer)", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Gradient arrows pointing up from both
    for x in np.linspace(0.5, 6.7, 12):
        ax.annotate("", xy=(x, 1.55), xytext=(x, 1.45),
                    arrowprops=dict(arrowstyle="-|>", color="#c44e52", lw=1.5))
    for x in np.linspace(7.5, 9.3, 4):
        ax.annotate("", xy=(x, 1.55), xytext=(x, 1.45),
                    arrowprops=dict(arrowstyle="-|>", color="#2a9d8f", lw=1.5))

    ax.text(3.6, 1.65, "gradient updates all tokens equally", ha="center",
            fontsize=8, color="#c44e52", style="italic")

    # --- Bottom: DPO ---
    ax = axes[1]
    ax.text(0.1, 1.7, "DPO gradient  (∇log π(y_w) − ∇log π(y_l))", fontsize=11, fontweight="bold", va="top")

    # Shared prefix (cancelled)
    prefix = FancyBboxPatch((0.1, 0.4), 7.0, 1.0, boxstyle="round,pad=0.05",
                             facecolor="#f5f5f5", edgecolor="#ccc", linewidth=1, linestyle="--")
    ax.add_patch(prefix)
    ax.text(3.6, 0.9, "Shared prefix: cancels exactly", ha="center", va="center",
            fontsize=9, color="#bbb")

    # X marks over prefix
    for x in np.linspace(0.5, 6.7, 8):
        ax.text(x, 0.55, "×", ha="center", va="center", fontsize=14, color="#ddd", fontweight="bold")

    # Suffix (active)
    suffix = FancyBboxPatch((7.2, 0.4), 2.5, 1.0, boxstyle="round,pad=0.05",
                             facecolor="#4c72b0", edgecolor="#3a5a8e", linewidth=1.5)
    ax.add_patch(suffix)
    ax.text(8.45, 0.9, "Suffix\n(diverges)", ha="center", va="center",
            fontsize=8, color="white", fontweight="bold")

    # Gradient arrows only from suffix
    for x in np.linspace(7.5, 9.3, 4):
        ax.annotate("", xy=(x, 1.55), xytext=(x, 1.45),
                    arrowprops=dict(arrowstyle="-|>", color="#4c72b0", lw=2.0))

    ax.text(8.45, 1.65, "gradient targets divergence only", ha="center",
            fontsize=8, color="#4c72b0", style="italic")

    fig.tight_layout(h_pad=0.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
