#!/usr/bin/env python3
"""Generate fig-ane-ladder.pdf: the Section 6.3 falsification ladder (money figure).

Horizontal grouped bars: p99 frame time per rung (7 rungs) x three compute-unit
policies (ANE / CPU-only / CPU+GPU). Every rung is ANE-clean (ANE cost share 1.000);
the story to make legible is that even the full 12-layer stack with 48 caches-in /
48 one-token-updates-out compiles ANE-resident and lands at 15.0 ms, beating both
CPU-only and CPU+GPU -- provided in-graph cache mutation is absent.

data: surgical-inference.md Section 6.3 ladder table. iPhone 12 Pro (A14), FP16,
p99 ms over warmed runs, MLComputePlan placement evidence. Every rung: ANE cost
share 1.000.

Run: paper/.figvenv/bin/python paper/figures/src/fig_ane_ladder.py
"""
import matplotlib
matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 8,
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

# --- Data (surgical-inference.md Section 6.3). p99 ms, A14, FP16. ----------
# Order top->bottom follows the ladder's build-up; the headline full stack is last.
RUNGS = [
    "12-layer FFN stack, stateless",
    "12-layer attention math, no state",
    "1 layer + cache reads as inputs",
    "1 layer + K/V update outputs",
    "4 layers, caches in / updates out",
    "8 layers, caches in / updates out",
    "12 layers, 48 caches in / 48 updates out",
]
P99_ANE     = [7.3, 6.4, 1.9, 1.9,  5.2, 10.0, 15.0]
P99_CPU     = [8.9, 6.5, 2.1, 2.2,  6.5, 12.1, 15.4]
P99_CPU_GPU = [14.9, 6.8, 11.9, 10.6, 11.6, 20.1, 26.7]

# Okabe-Ito. ANE is the hero (blue); CPU green; CPU+GPU vermillion. Hatches redundant.
C_ANE = "#0072B2"; H_ANE = ""
C_CPU = "#009E73"; H_CPU = "...."
C_CGP = "#D55E00"; H_CGP = "xxx"

y = np.arange(len(RUNGS))[::-1]   # reverse so first rung is at top
h = 0.26

fig, ax = plt.subplots(figsize=(7.2, 4.0))

ax.barh(y + h, P99_ANE, h, color=C_ANE, edgecolor="black", linewidth=0.5,
        hatch=H_ANE, label="ANE", zorder=3)
ax.barh(y, P99_CPU, h, color=C_CPU, edgecolor="black", linewidth=0.5,
        hatch=H_CPU, label="CPU-only", zorder=3)
ax.barh(y - h, P99_CPU_GPU, h, color=C_CGP, edgecolor="black", linewidth=0.5,
        hatch=H_CGP, label="CPU+GPU", zorder=3)

# Value labels at bar ends.
for yi, va, vc, vg in zip(y, P99_ANE, P99_CPU, P99_CPU_GPU):
    for off, val in [(h, va), (0, vc), (-h, vg)]:
        ax.annotate(f"{val:.1f}", (val, yi + off), textcoords="offset points",
                    xytext=(3, 0), ha="left", va="center", fontsize=6.2)

ax.set_yticks(y)
ax.set_yticklabels(RUNGS, fontsize=7.2)
ax.set_xlabel("p99 frame time (ms) -- iPhone 12 Pro (A14), FP16", fontsize=8)
ax.set_xlim(0, 30)
ax.grid(axis="x", linewidth=0.3, alpha=0.35, zorder=0)
ax.tick_params(labelsize=7)

# Every rung ANE-clean: annotate the shared invariant once, top-left open space.
ax.annotate("ANE cost share = 1.000 on every rung", (0.02, 0.98),
            xycoords="axes fraction", ha="left", va="top",
            fontsize=6.8, style="italic", color="#333333")

# Highlight the headline full-stack rung (bottom) with a callout into open space.
full_y = y[-1]
ax.annotate("full 12-layer stack:\n15.0 ms, ANE-resident",
            xy=(11.5, full_y + h), xytext=(21.0, full_y + 1.35),
            ha="left", va="center", fontsize=6.9, fontweight="bold", color=C_ANE,
            arrowprops=dict(arrowstyle="->", color=C_ANE, lw=0.9))

handles = [
    Patch(facecolor=C_ANE, edgecolor="black", hatch=H_ANE, label="ANE"),
    Patch(facecolor=C_CPU, edgecolor="black", hatch=H_CPU, label="CPU-only"),
    Patch(facecolor=C_CGP, edgecolor="black", hatch=H_CGP, label="CPU+GPU"),
]
ax.legend(handles=handles, loc="upper right", frameon=False, fontsize=7.6,
          bbox_to_anchor=(0.995, 0.90))

fig.tight_layout()
out = "paper/figures/fig-ane-ladder.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
