import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path

sns.set_theme(style="whitegrid", font_scale=1.05)
PALETTE = {
    "blue":   "#378ADD",
    "green":  "#1D9E75",
    "amber":  "#BA7517",
    "purple": "#7F77DD",
    "coral":  "#D85A30",
    "red":    "#E24B4A",
    "gray":   "#888780",
}

ssl  = json.loads(Path("ssl_metrics.json").read_text())
asr  = json.loads(Path("asr_metrics.json").read_text())

# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 — SSL Phase 1 + Phase 2: contrast loss, active dims, ABX
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=False)
fig.suptitle("SSL Pretraining — Training Metrics", fontsize=14, fontweight="normal", y=0.98)

p1 = ssl["phase1"]
p2 = ssl["phase2"]

all_steps   = p1["steps"] + p2["steps"]
all_contrast = p1["contrast_loss"] + p2["contrast_loss"]
all_active  = p1["active_dims_pct"] + p2["active_dims_pct"]
all_abx     = p1["abx_proxy"] + p2["abx_proxy"]
all_iso     = p1["isotropy"] + p2["isotropy"]
all_norm    = p1["norm_mean"] + p2["norm_mean"]
collapse_idx = p2["steps"].index(60000) + len(p1["steps"])

# ── Axes 0: Contrast loss ─────────────────────────────────────────────────────
ax = axes[0]
valid = [(s, v) for s, v in zip(all_steps, all_contrast) if v is not None]
xs, ys = zip(*valid)
ax.plot(xs, ys, color=PALETTE["blue"], linewidth=2, marker="o", markersize=5, label="contrast loss")
ax.axvline(x=40000, color=PALETTE["gray"], linewidth=1, linestyle="--", alpha=0.7, label="phase 2 start")
ax.axvline(x=60000, color=PALETTE["red"],  linewidth=1.5, linestyle="--", alpha=0.9, label="collapse (α=1.0)")
ax.axvspan(40000, 60000, alpha=0.04, color=PALETTE["amber"])
ax.set_ylabel("contrastive loss")
ax.set_xlabel("training step")
ax.legend(fontsize=9, loc="upper right")
ax.set_title("contrastive loss (lower = better)", fontsize=11, fontweight="normal")
ax.annotate("variance penalty\n+ norm fix", xy=(30000, 3.03), xytext=(18000, 3.5),
            fontsize=8, color=PALETTE["gray"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["gray"], lw=0.8))
ax.annotate("collapse", xy=(60000, 2.94), xytext=(62000, 3.4),
            fontsize=8, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.8))

# ── Axes 1: Active dimensions ─────────────────────────────────────────────────
ax = axes[1]
ax.plot(all_steps, all_active, color=PALETTE["green"], linewidth=2, marker="o", markersize=5, label="active dims %")
ax.axvline(x=40000, color=PALETTE["gray"],  linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(x=60000, color=PALETTE["red"],   linewidth=1.5, linestyle="--", alpha=0.9)
ax.axhspan(0, 20, alpha=0.06, color=PALETTE["red"], label="collapse zone")
ax.axvspan(40000, 60000, alpha=0.04, color=PALETTE["amber"])
ax.set_ylim(0, 110)
ax.set_ylabel("active dimensions (%)")
ax.set_xlabel("training step")
ax.legend(fontsize=9, loc="lower right")
ax.set_title("active dimensions (variance > 0.01 threshold)", fontsize=11, fontweight="normal")
ax.annotate("collapse to 8%\n(28k steps)", xy=(28000, 8), xytext=(15000, 25),
            fontsize=8, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.8))
ax.annotate("var_penalty fix:\n39% → 98%", xy=(33000, 89), xytext=(18000, 75),
            fontsize=8, color=PALETTE["green"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=0.8))

# ── Axes 2: ABX proxy ─────────────────────────────────────────────────────────
ax = axes[2]
valid_abx = [(s, v) for s, v in zip(all_steps, all_abx) if v is not None]
xs_a, ys_a = zip(*valid_abx)
ax.plot(xs_a, ys_a, color=PALETTE["amber"], linewidth=2, marker="o", markersize=5, label="ABX proxy")
ax.axvline(x=40000, color=PALETTE["gray"],  linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(x=60000, color=PALETTE["red"],   linewidth=1.5, linestyle="--", alpha=0.9)
ax.axhline(y=0.887, color=PALETTE["amber"], linewidth=0.8, linestyle=":", alpha=0.6)
ax.axvspan(40000, 60000, alpha=0.04, color=PALETTE["amber"])
ax.set_ylim(0.6, 0.95)
ax.set_ylabel("ABX proxy score")
ax.set_xlabel("training step")
ax.legend(fontsize=9)
ax.set_title("ABX proxy (phoneme discriminability, higher = better)", fontsize=11, fontweight="normal")
ax.annotate("best: 0.887\n(step 55k)", xy=(55000, 0.887), xytext=(42000, 0.91),
            fontsize=8, color=PALETTE["amber"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["amber"], lw=0.8))
ax.annotate("collapse\n0.878→0.679", xy=(60000, 0.878), xytext=(62000, 0.81),
            fontsize=8, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.8))

phase1_patch = mpatches.Patch(color=PALETTE["gray"],   alpha=0.3, label="Phase 1 (CNN targets)")
phase2_patch = mpatches.Patch(color=PALETTE["amber"],  alpha=0.1, label="Phase 2 (quantizer blend)")
fig.legend(handles=[phase1_patch, phase2_patch], loc="lower center", ncol=2, fontsize=9, bbox_to_anchor=(0.5, 0.01))

plt.tight_layout(rect=[0, 0.04, 1, 0.97])
plt.savefig("ssl_training_curves.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: ssl_training_curves.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 — SSL Geometry: isotropy and norm_mean
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("SSL Pretraining — Representation Geometry", fontsize=14, fontweight="normal", y=0.98)

all_iso_steps  = [s for s, v in zip(all_steps, all_iso)  if v is not None]
all_iso_vals   = [v for v in all_iso  if v is not None]
all_norm_steps = [s for s, v in zip(all_steps, all_norm) if v is not None]
all_norm_vals  = [v for v in all_norm if v is not None]

ax = axes[0]
ax.plot(all_iso_steps, all_iso_vals, color=PALETTE["purple"], linewidth=2, marker="o", markersize=5)
ax.axvline(x=40000, color=PALETTE["gray"], linewidth=1, linestyle="--", alpha=0.7, label="phase 2 start")
ax.axvline(x=60000, color=PALETTE["red"],  linewidth=1.5, linestyle="--", alpha=0.9, label="collapse")
ax.axhline(y=0.3, color=PALETTE["green"],  linewidth=0.8, linestyle=":", alpha=0.6, label="healthy threshold (~0.3)")
ax.axvspan(40000, 60000, alpha=0.04, color=PALETTE["amber"])
ax.set_ylabel("isotropy (σ_min / σ_max)")
ax.legend(fontsize=9)
ax.set_title("isotropy — ratio of smallest to largest singular value (higher = more isotropic)", fontsize=11, fontweight="normal")
ax.annotate("0.361 at step 55k\n(used for ASR init)", xy=(55000, 0.361), xytext=(40000, 0.43),
            fontsize=8, color=PALETTE["purple"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["purple"], lw=0.8))

ax = axes[1]
ax.plot(all_norm_steps, all_norm_vals, color=PALETTE["coral"], linewidth=2, marker="o", markersize=5)
ax.axvline(x=40000, color=PALETTE["gray"], linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(x=60000, color=PALETTE["red"],  linewidth=1.5, linestyle="--", alpha=0.9)
ax.axhline(y=2.0, color=PALETTE["red"],   linewidth=0.8, linestyle=":", alpha=0.6, label="warning threshold (2.0)")
ax.axvspan(40000, 60000, alpha=0.04, color=PALETTE["amber"])
ax.set_ylabel("norm mean (L2)")
ax.set_xlabel("training step")
ax.legend(fontsize=9)
ax.set_title("representation norm mean — watch for explosion above 2.0", fontsize=11, fontweight="normal")
ax.annotate("explosion: 0.501 → 3.278\nat step 60k", xy=(60000, 3.278), xytext=(50000, 7),
            fontsize=8, color=PALETTE["red"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["red"], lw=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("ssl_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: ssl_geometry.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 — ASR: all runs CER comparison
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(11, 6))
fig.suptitle("ASR Fine-Tuning — CER Across All Runs", fontsize=14, fontweight="normal")

runs = [
    ("run1_unicode_norm_cascade",   "Run 1 — unicode, norm cascade",         PALETTE["red"],    "-",  "o"),
    ("run2_cosine_decay_collapse",  "Run 2 — cosine decay collapse",          PALETTE["amber"],  "--", "s"),
    ("run3_flat_lr_plateau",        "Run 3 — flat LR, data split mismatch",   PALETTE["purple"], ":",  "^"),
    ("run4_final",                  "Run 4 — grapheme + honest split (final)", PALETTE["green"],  "-",  "D"),
]

for key, label, color, ls, marker in runs:
    d = asr[key]
    ax.plot(d["steps"], d["cer"], color=color, linewidth=2, linestyle=ls,
            marker=marker, markersize=5, label=label)

ax.axhline(y=21.83, color=PALETTE["green"], linewidth=0.8, linestyle=":", alpha=0.5)
ax.annotate("best: 21.83% CER\n(step 58k, run 4)", xy=(58000, 21.83), xytext=(35000, 18),
            fontsize=8, color=PALETTE["green"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["green"], lw=0.8))

ax.set_xlabel("training step")
ax.set_ylabel("CER (%)")
ax.set_ylim(0, 105)
ax.legend(fontsize=9, loc="upper right")
ax.set_title("character error rate — runs 1–3 plateaued due to data split mismatch and tokenizer issues", fontsize=10, fontweight="normal")

plt.tight_layout()
plt.savefig("asr_cer_all_runs.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: asr_cer_all_runs.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 — ASR Run 4: CER, WER, val loss, space fraction
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)
fig.suptitle("ASR Fine-Tuning — Run 4 (Final) Detailed Metrics", fontsize=14, fontweight="normal", y=0.98)

r4 = asr["run4_final"]

ax = axes[0]
ax.plot(r4["steps"], r4["cer"], color=PALETTE["blue"],  linewidth=2, marker="o", markersize=4, label="CER")
ax.plot(r4["steps"], r4["wer"], color=PALETTE["coral"], linewidth=2, marker="s", markersize=4, label="WER")
ax.axvline(x=30000, color=PALETTE["gray"], linewidth=1, linestyle="--", alpha=0.7, label="LR decay (×0.3) at step 30k")
ax.axvline(x=40000, color=PALETTE["amber"], linewidth=1, linestyle="--", alpha=0.7, label="hardware resume at step 40k")
ax.axhspan(0, 25, alpha=0.04, color=PALETTE["green"])
ax.set_ylabel("error rate (%)")
ax.legend(fontsize=9, loc="upper right")
ax.set_title("CER and WER — both declining consistently from step 14k", fontsize=11, fontweight="normal")
ax.annotate("CER 21.83%\nWER 67.28%", xy=(58000, 21.83), xytext=(48000, 30),
            fontsize=8, color=PALETTE["blue"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["blue"], lw=0.8))

ax = axes[1]
valid_loss = [(s, v) for s, v in zip(r4["steps"], r4["val_loss"]) if v is not None]
xs_l, ys_l = zip(*valid_loss)
ax.plot(xs_l, ys_l, color=PALETTE["purple"], linewidth=2, marker="o", markersize=4, label="val loss")
ax.axvline(x=30000, color=PALETTE["gray"],   linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(x=40000, color=PALETTE["amber"],  linewidth=1, linestyle="--", alpha=0.7)
ax.set_ylabel("validation loss")
ax.legend(fontsize=9)
ax.set_title("validation CTC loss", fontsize=11, fontweight="normal")

ax = axes[2]
valid_sf = [(s, v) for s, v in zip(r4["steps"], r4["space_frac"]) if v is not None]
xs_sf, ys_sf = zip(*valid_sf)
ax.plot(xs_sf, ys_sf, color=PALETTE["amber"], linewidth=2, marker="o", markersize=4, label="space fraction (greedy argmax)")
ax.axhline(y=0.13, color=PALETTE["green"], linewidth=0.8, linestyle=":", alpha=0.7, label="target lower bound (0.13)")
ax.axhline(y=0.16, color=PALETTE["green"], linewidth=0.8, linestyle=":", alpha=0.7, label="target upper bound (0.16)")
ax.axhspan(0.13, 0.16, alpha=0.06, color=PALETTE["green"])
ax.axvline(x=30000, color=PALETTE["gray"],  linewidth=1, linestyle="--", alpha=0.7)
ax.axvline(x=40000, color=PALETTE["amber"], linewidth=1, linestyle="--", alpha=0.7)
ax.set_ylabel("space token fraction")
ax.set_xlabel("training step")
ax.legend(fontsize=9)
ax.set_title("space fraction — frames where argmax == <space> (target: 0.13–0.16)", fontsize=11, fontweight="normal")

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("asr_run4_detail.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: asr_run4_detail.png")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5 — ASR Run 4: Representation geometry (norm, isotropy)
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
fig.suptitle("ASR Fine-Tuning — Run 4 Representation Geometry", fontsize=14, fontweight="normal", y=0.98)

geom = r4["geometry"]

ax = axes[0]
ax.plot(geom["steps"], geom["norm_mean"], color=PALETTE["coral"], linewidth=2, marker="o", markersize=5, label="norm mean")
ax.fill_between(geom["steps"],
    [m - s for m, s in zip(geom["norm_mean"], geom["norm_std"])],
    [m + s for m, s in zip(geom["norm_mean"], geom["norm_std"])],
    alpha=0.15, color=PALETTE["coral"], label="±1 std")
ax.set_ylabel("L2 norm (mean ± std)")
ax.legend(fontsize=9)
ax.set_title("representation norm — stable 58–60 throughout (LayerNorm decouples head from scale)", fontsize=11, fontweight="normal")
ax.annotate("norm_mean ~58–60\n(stable, no explosion)", xy=(50000, 59.24), xytext=(42000, 53),
            fontsize=8, color=PALETTE["coral"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["coral"], lw=0.8))

ax = axes[1]
ax.plot(geom["steps"], geom["isotropy"], color=PALETTE["purple"], linewidth=2, marker="o", markersize=5, label="isotropy")
ax.axhline(y=0.361, color=PALETTE["gray"], linewidth=0.8, linestyle=":", alpha=0.7, label="SSL baseline (0.361)")
ax.set_ylabel("isotropy (σ_min / σ_max)")
ax.set_xlabel("training step")
ax.set_ylim(-0.001, 0.02)
ax.legend(fontsize=9)
ax.set_title("isotropy — collapsed to 0.003 (CTC specialization expected, differs from SSL baseline)", fontsize=11, fontweight="normal")
ax.annotate("CTC specialization:\n0.361 → 0.003", xy=(40000, 0.003), xytext=(45000, 0.008),
            fontsize=8, color=PALETTE["purple"],
            arrowprops=dict(arrowstyle="->", color=PALETTE["purple"], lw=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("asr_geometry.png", dpi=150, bbox_inches="tight")
plt.show()
print("saved: asr_geometry.png")

print("\nAll figures saved:")
print("  ssl_training_curves.png")
print("  ssl_geometry.png")
print("  asr_cer_all_runs.png")
print("  asr_run4_detail.png")
print("  asr_geometry.png")
