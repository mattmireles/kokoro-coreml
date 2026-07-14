#!/usr/bin/env python3
"""Generate fig-power.pdf: Section 6.7 paired power comparison on A14.

Four small paired-bar panels (one per metric, native units) contrasting the
all-ANE temporal policy against the temporal-GPU control. Identical graphs and
workload; only the temporal stage's compute policy varied; 60 s capture. The
contrast to make unmissable: GPU impact goes to zero and CPU instructions roughly
halve under the ANE policy.

data: surgical-inference.md Section 6.7 "Power" paragraph. iPhone 12 Pro (A14),
Power Profiler paired capture.
  gpuImpact:            ANE 0.000  vs  GPU 2.23
  cpuImpact:            ANE 1.38   vs  GPU 2.63
  CPU instructions:     ANE 48.1 B vs  GPU 110.3 B (over 60 s)
  producer duty cycle:  ANE 57%    vs  GPU 93%   (ANE sleeps 43% under backpressure)

Run: paper/.figvenv/bin/python paper/figures/src/fig_power.py
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

# metric label, unit label, ANE value, GPU value, value format
PANELS = [
    ("GPU impact",        "process gpuImpact",       0.000, 2.23,  "{:.2f}"),
    ("CPU impact",        "process cpuImpact",       1.38,  2.63,  "{:.2f}"),
    ("CPU instructions",  "billions, 60 s",          48.1,  110.3, "{:.1f}"),
    ("Producer duty",     "% wall-clock active",     57.0,  93.0,  "{:.0f}%"),
]

# Okabe-Ito. ANE hero blue, GPU control vermillion. Hatch redundant for grayscale.
C_ANE = "#0072B2"; H_ANE = ""
C_GPU = "#D55E00"; H_GPU = "xxx"

fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.5))
x = np.array([0, 1])

for ax, (title, unit, ane, gpu, fmt) in zip(axes, PANELS):
    bars = ax.bar(x, [ane, gpu], width=0.62,
                  color=[C_ANE, C_GPU], edgecolor="black", linewidth=0.5,
                  hatch=[H_ANE, H_GPU], zorder=3)
    ax.set_title(title, fontsize=8.5, pad=3)
    ax.set_xticks(x)
    ax.set_xticklabels(["ANE", "GPU"], fontsize=7.5)
    ax.set_xlabel(unit, fontsize=6.6)
    ax.grid(axis="y", linewidth=0.3, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=6.8)
    top = max(ane, gpu)
    ax.set_ylim(0, top * 1.30 if top > 0 else 1)
    for xi, val in zip(x, [ane, gpu]):
        ax.annotate(fmt.format(val), (xi, val), textcoords="offset points",
                    xytext=(0, 2), ha="center", va="bottom", fontsize=7.0,
                    fontweight="bold")
    # Percent-reduction callout on the metrics that drop under ANE.
    if gpu > 0 and ane < gpu:
        if title == "GPU impact":
            note = "eliminated"
        else:
            note = f"-{(1 - ane / gpu) * 100:.0f}%"
        ax.annotate(note, (0.5, 0.90), xycoords="axes fraction", ha="center",
                    va="top", fontsize=6.8, color=C_ANE, fontweight="bold")

handles = [
    Patch(facecolor=C_ANE, edgecolor="black", hatch=H_ANE, label="all-ANE policy"),
    Patch(facecolor=C_GPU, edgecolor="black", hatch=H_GPU, label="temporal-GPU control"),
]
fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False,
           fontsize=8, bbox_to_anchor=(0.5, -0.03))

fig.tight_layout(rect=(0, 0.08, 1, 1))
out = "paper/figures/fig-power.pdf"
fig.savefig(out, bbox_inches="tight")
print("wrote", out)
