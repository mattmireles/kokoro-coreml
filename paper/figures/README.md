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
| `fig-mrt2-pipeline.pdf` | TikZ | §6.2 (per-frame loop) + §6.7 (placement, per-stage ms) + §6.1 (40 ms deadline) | `fig-mrt2-pipeline.tex` |
| `fig-cs1-latency.pdf` | matplotlib | Tables `tab:cs1-mac` + `tab:cs1-iphone` (plan 014 rerun; Surgical vs MLX) | `src/fig_cs1_latency.py` |
| `fig-ane-ladder.pdf` | matplotlib | §6.3 (falsification ladder table) | `src/fig_ane_ladder.py` |
| `fig-power.pdf` | matplotlib | §6.7 (A14 paired power) | `src/fig_power.py` |

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

**fig-mrt2-pipeline** — *The Magenta RealTime 2 per-frame streaming loop. Every frame
must complete within the 40 ms (25 Hz) audio deadline, which never pauses. Audio-rate
tensors never enter a Core ML graph — the codec decoder stops at STFT-rate features and
the host synthesizes PCM — and the three Core ML stages ship on three different engines
(temporal on GPU, depth rollout FP16 on CPU, NCHW codec decoder on ANE; per-stage p50
on A17 Pro shown). Generation runs ahead of playback through a lock-free SPSC ring feeding
an `AVAudioSourceNode`; the render callback never blocks on the model.*

**fig-cs1-latency** — *Kokoro-82M end-to-end latency, the shipped Surgical pipeline
(hybrid generator, folded below 7 s) vs. MLX, warm medians across five input buckets on
two Macs and two iPhones (log axis; Tables~\ref{tab:cs1-mac}
and~\ref{tab:cs1-iphone}). The Surgical pipeline is faster on every completed cell:
3.5–4.8× on the M2 Ultra, 8.1–9.8× on the M1 Mini, 8.9–15.1× on the A17 Pro and
9.3–15.2× on the A14 (annotated). The pinned mlx-audio fails every 3 s Mac
input with a broadcast-shape error, and the memory watchdog kills kokoro-ios at 30 s on
the 4 GB A14.*

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

- **fig-cs1-latency** uses a **log** y-axis: the cells span 54 ms → 4797 ms, so a
  linear axis would crush the short buckets. Speedup labels and the "MLX error" / "MLX
  OOM" markers preserve the per-pair story. The M2 Air is **excluded**: it was the
  noisiest host and `tab:cs1-mac` marks its ratios as not comparable. The e5rt
  (private-API) column is not plotted, because no claim uses it.
- **fig-kokoro-pipeline** uses a **split fill** for stages placed on two engines
  (Duration, F0Ntrain, Generator), with a dashed divider below each centered title.
  The Generator's GPU half is Metal, not Core ML, so the legend has a separate
  "Metal (GPU)" entry. No Core ML stage ships on CPU+GPU any more.
- **fig-mrt2-pipeline** annotates each Core ML stage with its **shipped** per-frame p50
  from §6.7 (temporal 12.2 / depth 8.4 / decoder 8.3 ms, A17 Pro) and labels temporal
  placement as CPU+GPU (shipped) with "ANE provable, §6.3" — matching the paper's honest
  "what ships vs. what was proven" framing rather than implying temporal ships on ANE.
- **fig-power** is rendered as four native-unit panels rather than one normalized chart,
  because the four metrics have incommensurable units (impact score, billions of
  instructions, percent); normalizing would hide the absolute magnitudes the paper quotes.
