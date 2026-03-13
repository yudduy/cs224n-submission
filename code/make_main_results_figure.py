"""Main results figure: delta pass@1 for all 11 conditions, grouped by experiment type."""
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUT = Path("/Users/duy/Documents/build/cs224n-submission/figures/main_results.png")


def main() -> None:
    # --- Data ---
    # Group 1: Original correction (base 77.4%)
    correction_orig = [
        ("Random", -3.8),
        ("Late div (>10%)", -2.4),
        ("Early div (≤10%)", -0.8),
    ]
    # Group 2: Matched correction (base 76.8%)
    correction_matched = [
        ("Random", -1.0),
        ("Late div (>10%)", -2.2),
        ("Early div (≤10%)", +0.6),
    ]
    # Group 3: Rollout (base 74.2%)
    rollout = [
        ("Prefix 228", +0.6),
        ("Full 150, 1ep", +1.0),
        ("Full 150, 3ep", +1.4),
        ("Prefix 150", +2.4),
        ("Full 228, 1ep", +3.2),
    ]

    # --- Colors ---
    c_corr = "#c44e52"       # muted red
    c_match = "#dd8452"      # muted orange
    c_roll = "#4c72b0"       # muted blue

    fig, ax = plt.subplots(figsize=(7.0, 5.2))

    # Build y positions with group spacing
    y = 0
    positions = []
    labels_out = []
    values_out = []
    colors_out = []
    group_spans = []  # (y_start, y_end, color, label)

    def add_group(items, color, group_label):
        nonlocal y
        y_start = y - 0.4
        for label, val in items:
            positions.append(y)
            labels_out.append(label)
            values_out.append(val)
            colors_out.append(color)
            y += 1
        y_end = y - 0.6
        group_spans.append((y_start, y_end, color, group_label))
        y += 0.8  # gap between groups

    add_group(correction_orig, c_corr, "Correction (original)")
    add_group(correction_matched, c_match, "Correction (matched)")
    add_group(rollout, c_roll, "Rollout")

    # --- Background shading ---
    for y_start, y_end, color, _ in group_spans:
        ax.axhspan(y_start, y_end, color=color, alpha=0.06, zorder=0)

    # --- Bars ---
    bar_height = 0.62
    for pos, val, color in zip(positions, values_out, colors_out):
        bar = ax.barh(pos, val, color=color, height=bar_height,
                      edgecolor="white", linewidth=0.5, zorder=2)
        # Value labels
        offset = 0.15
        x_text = val + offset if val >= 0 else val - offset
        ha = "left" if val >= 0 else "right"
        sign = "+" if val > 0 else ""
        ax.text(x_text, pos, f"{sign}{val:.1f}", va="center", ha=ha,
                fontsize=9.5, fontweight="bold", color="#333333", zorder=3)

    # --- Zero line ---
    ax.axvline(0.0, color="#555555", linestyle="-", linewidth=1.2, zorder=1)

    # --- Group labels on right side ---
    for y_start, y_end, color, label in group_spans:
        y_mid = (y_start + y_end) / 2
        ax.text(4.2, y_mid, label, va="center", ha="left", fontsize=9,
                fontweight="bold", color=color, style="italic")

    # --- Formatting ---
    ax.set_xlim(-5.0, 4.0)
    ax.set_yticks(positions)
    ax.set_yticklabels(labels_out, fontsize=9.5)
    ax.invert_yaxis()
    ax.set_xlabel("Δ pass@1 vs. within-experiment base (pp)", fontsize=10.5)

    # Light grid
    ax.grid(axis="x", linestyle=":", alpha=0.25, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.5)
    ax.spines["bottom"].set_linewidth(0.5)

    # Title
    ax.set_title(
        "Rollout pairs consistently improve; correction pairs consistently hurt",
        fontsize=12, fontweight="bold", pad=10, loc="left"
    )

    # Subtitle
    ax.text(0, 1.02, "Qwen3-8B on MATH-500  ·  each delta measured against its own experiment's base",
            transform=ax.transAxes, fontsize=8.5, color="#666666", va="bottom")

    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=250, bbox_inches="tight", facecolor="white")
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
