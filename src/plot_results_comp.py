
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

ALPHA = 0.5
BETA  = 0.1
GAMMA = 100

RUN_LABELS = [
    "Greedy",
    "CTC Beam (no lex)",
    "CTC Beam + lexicon",
    f"CTC+KenLM\nα={ALPHA} β={BETA} γ={GAMMA}",
    f"CTC+KenLM + lexicon\nα={ALPHA} β={BETA} γ={GAMMA}",
]

# Values read directly from the chart
CER_VALS = [21.1, 21.1, 25.2, 12.3, 18.4]
WER_VALS = [69.9, 69.9, 186.1, 33.4, 64.9]

# ── Colour palette ────────────────────────────────────────────────────────────
PALETTE = {
    "blue":   "#378ADD",
    "green":  "#1D9E75",
    "amber":  "#BA7517",
    "purple": "#7F77DD",
    "coral":  "#D85A30",
    "gray":   "#888780",
}

RUN_COLORS = [
    PALETTE["blue"],
    PALETTE["green"],
    PALETTE["amber"],
    PALETTE["purple"],
    PALETTE["coral"],
]

def plot():
    sns.set_theme(style="whitegrid", font_scale=1.05)

    tick_labels = [l.replace("\n", " ") for l in RUN_LABELS]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    fig.suptitle("ASR Decoding Strategy Comparison", fontsize=14, fontweight="normal")

    x     = np.arange(len(RUN_LABELS))
    bar_w = 0.6

    for ax, vals, metric in zip(
        axes,
        [CER_VALS, WER_VALS],
        ["Character Error Rate (CER)", "Word Error Rate (WER)"],
    ):
        bars = ax.bar(x, vals, width=bar_w, color=RUN_COLORS,
                      edgecolor="white", linewidth=0.8, zorder=3)

        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.1f}%",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color=PALETTE["gray"],
            )

        ax.set_xticks(x)
        ax.set_xticklabels(tick_labels, fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(f"{metric} (%)", fontsize=10)
        ax.set_title(metric, fontsize=11, fontweight="normal")
        ax.set_ylim(0, max(vals) * 1.28)
        ax.yaxis.grid(True, alpha=0.4, zorder=0)
        ax.set_axisbelow(True)

    legend_patches = [
        mpatches.Patch(color=c, label=l.replace("\n", " "))
        for c, l in zip(RUN_COLORS, RUN_LABELS)
    ]
    fig.legend(
        handles=legend_patches,
        loc="lower center",
        ncol=5,
        fontsize=8,
        bbox_to_anchor=(0.5, -0.08),
    )

    plt.savefig("final_comparison.png", dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.show()
    print("Saved: final_comparison.png")


if __name__ == "__main__":
    plot()
