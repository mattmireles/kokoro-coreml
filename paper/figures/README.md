# Paper Figures — *Surgical Inference*

Publication figures for `Scratchpad/surgical-inference.md`. Every number in every
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
| `fig-kokoro-pipeline.pdf` | TikZ | §4.1 (pipeline diagram + staged policy) | `fig-kokoro-pipeline.tex` |
| `fig-mrt2-pipeline.pdf` | TikZ | §6.2 (per-frame loop) + §6.7 (placement, per-stage ms) + §6.1 (40 ms deadline) | `fig-mrt2-pipeline.tex` |
| `fig-cs1-latency.pdf` | matplotlib | §5.3 (Config F vs MLX) | `src/fig_cs1_latency.py` |
| `fig-ane-ladder.pdf` | matplotlib | §6.3 (falsification ladder table) | `src/fig_ane_ladder.py` |
| `fig-power.pdf` | matplotlib | §6.7 (A14 paired power) | `src/fig_power.py` |

## Draft captions

Captions live in LaTeX, not inside the figures. Drafts below — each states what the
reader should conclude.

**fig-kokoro-pipeline** — *The decomposed Kokoro-82M pipeline: four Core ML model
families and three native Swift stages, zero Python at inference time. Each stage is
classified by computational motif and placed on the SoC engine that measurement
selects (color = placement); only DecoderPre is admitted to the ANE, while the
geometry-rejected GeneratorFromHar and the sequential Duration model run Core ML on
CPU+GPU. The decomposition also exposes cross-unit parallelism a monolithic graph
cannot express: DecoderPre (ANE) and the hn-nsf harmonic source (background CPU
thread) execute concurrently.*

**fig-mrt2-pipeline** — *The Magenta RealTime 2 per-frame streaming loop. Every frame
must complete within the 40 ms (25 Hz) audio deadline, which never pauses. Audio-rate
tensors never enter a Core ML graph — the codec decoder stops at STFT-rate features and
the host synthesizes PCM — and the three Core ML stages ship on three different engines
(temporal on GPU, depth rollout FP16 on CPU, NCHW codec decoder on ANE; per-stage p50
on A17 Pro shown). Generation runs ahead of playback through a lock-free SPSC ring feeding
an `AVAudioSourceNode`; the render callback never blocks on the model.*

**fig-cs1-latency** — *Kokoro-82M end-to-end latency, the Surgical pipeline (Config F)
vs. MLX, warm medians across three Macs and five input buckets (log axis; §5.3). The
Surgical pipeline is faster on every completed cell, by 1.6–2.3× (annotated), with the
gap widest on the newest silicon. The pinned MLX build fails every 3 s input with a
broadcast-shape error, so no 3 s comparison exists.*

**fig-ane-ladder** — *The falsification ladder that located the ANE admission cliff for
streaming transformers (p99 per frame, iPhone 12 Pro / A14, FP16; §6.3). Every rung is
ANE-clean (ANE cost share 1.000) and the ANE placement beats both CPU-only and CPU+GPU —
including the complete 12-layer stack with all 48 K/V caches read in and 48 one-token
updates written out, at 15.0 ms. The one thing that must be absent is in-graph cache
mutation: attention, softmax, concatenation, and update outputs are all admissible; the
state write is the cliff.*

**fig-power** — *Paired power comparison on the A14 (iPhone 12 Pro), all-ANE temporal
policy vs. a temporal-GPU control running identical graphs over 60 s (§6.7). The ANE
policy eliminates process-attributed GPU impact entirely (2.23 → 0.000) and cuts CPU
instructions by 56% (110.3 B → 48.1 B), while the producer's duty cycle drops from 93%
to 57% — it finishes each second of audio fast enough to sleep 43% of the run under
backpressure.*

## Notes / deviations from spec

- **fig-cs1-latency** uses a **log** y-axis (spec left this to judgment): the buckets
  span 50 ms → 3078 ms, so linear would crush the short buckets illegibly. Speedup labels
  and the "MLX error" markers preserve the per-pair story.
- **fig-mrt2-pipeline** annotates each Core ML stage with its **shipped** per-frame p50
  from §6.7 (temporal 12.2 / depth 8.4 / decoder 8.3 ms, A17 Pro) and labels temporal
  placement as CPU+GPU (shipped) with "ANE provable, §6.3" — matching the paper's honest
  "what ships vs. what was proven" framing rather than implying temporal ships on ANE.
- **fig-power** is rendered as four native-unit panels rather than one normalized chart,
  because the four metrics have incommensurable units (impact score, billions of
  instructions, percent); normalizing would hide the absolute magnitudes the paper quotes.
