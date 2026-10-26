# Paper Figures — *Surgical Inference*

Publication figures for `paper/main.tex`. Every number in every
figure is transcribed directly from the paper's own tables; the section citation
is in a `data:` comment at the top of each generator/source file. Do not edit the
figures to disagree with the draft — if a table changes, change the figure's data
array and regenerate.

## Build

**TikZ figures** (sources committed alongside compiled PDFs):

```bash
cd paper/figures
tectonic fig-kokoro-pipeline.tex
tectonic fig-mrt2-pipeline.tex
```

**matplotlib figures** (dedicated venv, never the repo `.venv`):

```bash
python3 -m venv paper/.figvenv
paper/.figvenv/bin/pip install matplotlib
paper/.figvenv/bin/python paper/figures/src/fig_cs1_latency.py
paper/.figvenv/bin/python paper/figures/src/fig_ane_ladder.py
paper/.figvenv/bin/python paper/figures/src/fig_power.py
```

All outputs are vector PDF with embedded, subsetted fonts (Type42 Helvetica for
matplotlib; NimbusSanL/Helvetica + CM math for TikZ). Colorblind-safe Okabe–Ito
palette throughout, with hatching / border-dash as a redundant (grayscale-safe)
encoding.

## Figures

| File | Type | Data source | Generator |
|------|------|-------------|-----------|
| `fig-kokoro-pipeline.pdf` | TikZ | §4.1 + Table `tab:cs1-placement` (shipped placement) + §`sec:cs1-hybrid` (hybrid generator) | `fig-kokoro-pipeline.tex` |
| `fig-mrt2-pipeline.pdf` | TikZ | §5 (streaming loop) + §5.5 / Table `tab:cs2-sustain` (placement, per-stage ms) | `fig-mrt2-pipeline.tex` |
| `fig-cs1-latency.pdf` | matplotlib | Tables `tab:cs1-mac` + `tab:cs1-iphone` (plan 014 rerun; Surgical vs MLX) | `src/fig_cs1_latency.py` |
| `fig-ane-ladder.pdf` | matplotlib | §5.1 (falsification ladder table) | `src/fig_ane_ladder.py` |
| `fig-power.pdf` | matplotlib | §5.5 (A14 paired power) | `src/fig_power.py` |

## Draft captions

Captions live in LaTeX, not inside the figures. Drafts below — each states what the
reader should conclude.

**fig-kokoro-pipeline** — *The shipped Kokoro-82M pipeline: four Core ML model
families and three native Swift stages, zero Python at inference time. Color shows
placement; a split fill marks a stage that runs on two engines. Duration runs its BERT
front on the ANE and its LSTM back on the CPU; F0Ntrain runs a CPU front and an ANE
back; DecoderPre runs on the ANE concurrently with the Swift hn-nsf harmonic source.
For inputs of 7 s and longer the generator is one stage on two engines: 53 convolutions
on the ANE and AdaIN, Snake and branch joins in Metal on the GPU, sharing IOSurface
memory without copies. Below 7 s, and as the fallback if Metal fails, a folded all-ANE
generator runs instead. The iSTFT tail and trim run in Swift.*

**fig-mrt2-pipeline** — *The MRT2 streaming loop on the A17 Pro: temporal step on
the ANE, depth rollout (FP16) on the CPU, codec decoder on the ANE (the A14 requests an
FP32 decoder instead). The host (Swift and a C++ render core) owns the K/V ring,
sampling noise, codebook summation and iSTFT. Stage costs are iteration-normalized p50s over a 610 s run, in ms per frame
(decoder amortized over one 25-frame call per 12 frames). The producer must average at
least 25 frames/s; the render callback never blocks on the model.*

**fig-cs1-latency** — *Kokoro-82M end-to-end latency, the shipped Surgical pipeline
(hybrid generator, folded below 7 s) vs. MLX, warm medians across five input buckets on
two Macs and two iPhones (log axis; Tables~\ref{tab:cs1-mac}
and~\ref{tab:cs1-iphone}). The Surgical pipeline is faster on every completed cell:
3.5–4.8× on the M2 Ultra, 8.1–9.8× on the M1 Mini, 8.9–15.1× on the A17 Pro and
9.3–15.2× on the A14 (annotated). The pinned mlx-audio fails every 3 s Mac
input with a broadcast-shape error, and the memory watchdog kills kokoro-ios at 30 s on
the 4 GB A14.*

**fig-ane-ladder** — *The falsification ladder (A14, FP16, p99 per call; §5.1). Every
rung is ANE-clean, and the ANE matches or beats CPU-only on every rung and beats CPU+GPU
by 1.8–6.3×, except on the attention-only rung, whose CPU+GPU arm Core ML ran on the
CPU; this holds up to the complete 12-layer stack with all caches read in and all
updates written out. The one thing that must be absent is in-graph cache mutation.*

**fig-power** — *Paired power comparison on the A14, all-ANE temporal policy vs a
temporal-GPU control, identical graphs, 60 s (§5.5). The ANE policy removes process GPU
impact (2.231 → 0.000) and cuts CPU instructions by 56%; producer duty comes from a
separate 60 s run of the same build. The build predates the correctness fixes, and the
A17 Pro pair, which may have run a stale app bundle, did not separate.*

## Notes / deviations from spec

- **fig-cs1-latency** uses a **log** y-axis: the cells span 54 ms → 4797 ms, so a
  linear axis would crush the short buckets. Speedup labels and the "MLX error" / "MLX
  OOM" markers preserve the per-pair story. The M2 Air is **excluded**: it was the
  noisiest host and `tab:cs1-mac` marks its ratios as not comparable. The e5rt
  (private-API) column is not plotted, because no claim uses it.
- **fig-kokoro-pipeline** uses a **split fill** for stages placed on two engines
  (Duration, F0Ntrain, Generator), with a dashed divider below each centered title.
  The Generator's GPU half is Metal, not Core ML, so the legend has a separate
  "Metal (GPU)" entry. No Core ML stage ships on CPU+GPU any more.
- **fig-mrt2-pipeline** annotates each Core ML stage with its A17 Pro
  iteration-normalized p50 per frame from the 610 s context-12 run (temporal 11.2 /
  depth 9.8 / decoder 2.3 ms; Table `tab:cs2-sustain`), and shows the July 2026
  placement: stateless one-frame temporal step and decoder on the ANE, depth on CPU.
- **fig-power** is rendered as four native-unit panels rather than one normalized chart,
  because the four metrics have incommensurable units (impact score, billions of
  instructions, percent); normalizing would hide the absolute magnitudes the paper quotes.
