#!/usr/bin/env python3
"""Generate fig-cs1-latency.pdf: CS1 headline latency, Surgical pipeline vs MLX.

One panel per device in a 2x2 grid (M2 Ultra, M1 Mini, iPhone 15 Pro Max / A17 Pro,
iPhone 12 Pro / A14), grouped bars per bucket (3s-30s), warm-median end-to-end wall
time on a log axis. Failed MLX cells are drawn as labeled markers: "error" (pinned
mlx-audio broadcast-shape error on every 3 s Mac input) and "OOM" (memory watchdog
killed kokoro-ios on the 4 GB A14 at 30 s). Speedup (MLX / Surgical) is annotated
over each completed pair.

data: paper/main.tex Tables tab:cs1-mac and tab:cs1-iphone (plan 014 rerun).
Milliseconds, warm medians of ten calls. Surgical = shipped pipeline, hybrid
generator default, folded below 7 s. MLX = mlx-audio 0.4.3 @862dfbe on Macs,
mlalma/kokoro-ios 1.0.8 on iPhones. The M2 Air is excluded: it was the noisiest
host and the paper uses none of its ratios. e5rt (private API) is not plotted.

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

# --- Data (paper/main.tex tab:cs1-mac, tab:cs1-iphone). ms, warm medians. ----
BUCKETS = ["3s", "7s", "10s", "15s", "30s"]
NAN = float("nan")   # failed MLX cell; reason in FAIL
DATA = {
    #                              Surgical                              MLX
    "M2 Ultra":                  ([73.6, 53.7, 70.6, 88.9, 158.9],   [NAN, 189.1, 272.5, 340.4, 769.8]),
    "M1 Mini":                   ([94.9, 98.9, 123.8, 168.5, 308.5], [NAN, 800.5, 1102.9, 1542.3, 3014.6]),
    "iPhone 15 Pro Max (A17 Pro)": ([61.8, 103.2, 155.9, 223.8, 318.5], [551.0, 1226.8, 1717.1, 2319.3, 4797.3]),
    "iPhone 12 Pro (A14)":       ([98.2, 144.8, 185.7, 260.9, 479.9], [911.1, 2050.3, 2814.4, 3950.8, NAN]),
}
# Failure label per NaN MLX cell, by (device, bucket index).
FAIL = {("M2 Ultra", 0): "error", ("M1 Mini", 0): "error", ("iPhone 12 Pro (A14)", 4): "OOM"}

# Okabe-Ito colorblind-safe palette. Redundant encoding via hatch for grayscale.
C_F   = "#0072B2"   # blue  -> Surgical
C_MLX = "#E69F00"   # orange -> MLX
H_F   = ""          # solid
H_MLX = "///"       # hatched, survives grayscale

fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), sharey=True)
axes = axes.ravel()
x = np.arange(len(BUCKETS))
w = 0.38

for ax, (machine, (fvals, mvals)) in zip(axes, DATA.items()):
    fvals = np.array(fvals, dtype=float)
    mvals = np.array(mvals, dtype=float)

    ax.bar(x - w/2, fvals, w, color=C_F, edgecolor="black", linewidth=0.5,
           hatch=H_F, label="Surgical", zorder=3)
    ax.bar(x + w/2, np.nan_to_num(mvals), w, color=C_MLX, edgecolor="black",
           linewidth=0.5, hatch=H_MLX, label="MLX", zorder=3)

    ax.set_yscale("log")
    ax.set_title(machine, fontsize=9, pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKETS)
    if ax in axes[2:]:
        ax.set_xlabel("Input bucket", fontsize=8)
    ax.grid(axis="y", which="both", linewidth=0.3, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=7)

    # Speedup annotations over each completed pair; "error" for failed MLX cells.
    top = np.nanmax(np.concatenate([fvals, mvals]))
    for i in range(len(BUCKETS)):
        if np.isnan(mvals[i]):
            # MLX failed -> mark the (empty) MLX slot.
            ax.annotate(f"MLX\n{FAIL[(machine, i)]}", (x[i] + w/2, fvals[i]),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", va="bottom", fontsize=6.0,
                        color=C_MLX, fontweight="bold")
        else:
            speed = mvals[i] / fvals[i]
            ymax = max(fvals[i], mvals[i])
            ax.annotate(f"{speed:.1f}\u00d7", (x[i], ymax),
                        textcoords="offset points", xytext=(0, 3),
                        ha="center", va="bottom", fontsize=6.8, fontweight="bold")
    ax.set_ylim(bottom=20, top=top * 2.2)

for ax in axes[::2]:
    ax.set_ylabel("Warm median wall time (ms, log)", fontsize=8)

# Single shared legend (no in-figure title).
handles = [
    Patch(facecolor=C_F, edgecolor="black", hatch=H_F, label="Surgical pipeline (hybrid; folded < 7 s)"),
    Patch(facecolor=C_MLX, edgecolor="black", hatch=H_MLX, label="MLX"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, -0.02))

fig.tight_layout(rect=(0, 0.04, 1, 1))
out = "paper/figures/fig-cs1-latency.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
