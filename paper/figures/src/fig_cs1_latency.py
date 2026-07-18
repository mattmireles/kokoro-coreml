#!/usr/bin/env python3
"""Generate fig-cs1-latency.pdf: CS1 headline latency, Config F (Surgical) vs MLX.

One panel per machine (M2 Ultra / M2 Air / M1 Mini), grouped bars per bucket
(3s-30s), comparing warm-median end-to-end wall time. MLX 3s cells failed with a
broadcast-shape error and are drawn as hatched "error" markers. Speedup (F vs MLX)
is annotated over each completed pair.

data: surgical-inference.md Section 5.3 "vs MLX" table. Numbers are milliseconds,
warm medians, June 2026 external bakeoff. MLX = mlx-audio 0.4.3 @862dfbe,
mlx-community/Kokoro-82M-bf16. Config F = full Swift + Core ML surgical pipeline.

Run: paper/.figvenv/bin/python paper/figures/src/fig_cs1_latency.py
"""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# --- Embedded fonts, legible sizes at single-column width ------------------
plt.rcParams.update({
    "pdf.fonttype": 42,          # embed TrueType (no Type-3), avoids font subset issues
    "ps.fonttype": 42,
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# --- Data (surgical-inference.md Section 5.3). ms, warm medians. -----------
BUCKETS = ["3s", "7s", "10s", "15s", "30s"]
# NaN marks the failed MLX 3s cells (broadcast-shape error).
NAN = float("nan")
DATA = {
    #                F (Surgical)                         MLX
    "M2 Ultra": ([50.6,  96.1, 126.2, 185.6, 379.3],  [NAN, 223.9, 288.8, 376.3,  762.7]),
    "M2 Air":   ([148.0, 330.7, 466.0, 693.6, 1404.8], [NAN, 685.6, 835.8, 1521.0, 2600.3]),
    "M1 Mini":  ([233.6, 492.7, 685.5, 1014.9, 1959.4], [NAN, 824.0, 1124.3, 1589.5, 3077.9]),
}

# Okabe-Ito colorblind-safe palette. Redundant encoding via hatch for grayscale.
C_F   = "#0072B2"   # blue  -> Config F
C_MLX = "#E69F00"   # orange -> MLX
H_F   = ""          # solid
H_MLX = "///"       # hatched, survives grayscale

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
x = np.arange(len(BUCKETS))
w = 0.38

for ax, (machine, (fvals, mvals)) in zip(axes, DATA.items()):
    fvals = np.array(fvals, dtype=float)
    mvals = np.array(mvals, dtype=float)

    ax.bar(x - w/2, fvals, w, color=C_F, edgecolor="black", linewidth=0.5,
           hatch=H_F, label="Surgical (F)", zorder=3)
    ax.bar(x + w/2, np.nan_to_num(mvals), w, color=C_MLX, edgecolor="black",
           linewidth=0.5, hatch=H_MLX, label="MLX", zorder=3)

    ax.set_yscale("log")
    ax.set_title(machine, fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS)
    ax.set_xlabel("Input bucket", fontsize=8)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=7)

    # Speedup annotations over each completed pair; "error" for failed MLX cells.
    top = np.nanmax(np.concatenate([fvals, mvals]))
    for i in range(len(BUCKETS)):
        if np.isnan(mvals[i]):
            # MLX failed -> mark the (empty) MLX slot.
            ax.annotate("MLX\nerror", (x[i] + w/2, fvals[i]),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=6.0,
                        color=C_MLX, fontweight="bold")
        else:
            speed = mvals[i] / fvals[i]
            ymax = max(fvals[i], mvals[i])
            ax.annotate(f"{speed:.1f}x", (x[i], ymax),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_ylim(top=top * 2.2)

axes[0].set_ylabel("Warm median wall time (ms, log)", fontsize=8)

# Single shared legend (no in-figure title).
handles = [
    Patch(facecolor=C_F, edgecolor="black", hatch=H_F, label="Surgical pipeline (F)"),
    Patch(facecolor=C_MLX, edgecolor="black", hatch=H_MLX, label="MLX"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.06, 1, 1))
out = "paper/figures/fig-cs1-latency.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
