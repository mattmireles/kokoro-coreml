# Surgical Inference — 20-Minute Talk Outline

Every number below comes from `paper/main.tex` (October 2026 build). If the
paper changes, update this file from it, never the other way around.

## Thesis (say it in the first minute and the last)

Deploy a generative model as a set of fixed-shape, per-stage graphs, each placed
on the CPU, GPU or Neural Engine by measurement — not as one model on one
backend. On Apple silicon that makes Kokoro TTS about 2x faster than MLX, and it
shows the Neural Engine accepts or rejects graphs by their structure (state,
layout, geometry), not by which ops they use.

## Slides

| # | Title | One message | Show |
| --- | --- | --- | --- |
| 1 | One chip, three engines | GPU-only deployment leaves most of the SoC idle | SoC block diagram (CPU / GPU / ANE, shared DRAM) |
| 2 | Surgical Inference in one slide | Cut at dataflow seams; choose placement by measurement, not intuition | Motif → candidate engine table (Sequential-Dynamic, Data-Dependent, Dense-Static) |
| 3 | Kokoro, decomposed | 4 Core ML model families + 3 Swift stages; only DecoderPre lands on the ANE | `figures/fig-kokoro-pipeline.pdf` |
| 4 | We could not build a valid monolith | Conversion fails at the data-dependent alignment; the best partial monolith correlates at 0.27 | Two-seam diagram (new) |
| 5 | Faster than MLX on every Mac | 1.8–2.4x vs MLX; a 27 s clip in 287 ms on M2 Ultra, 1.6 s on an M1 Mini; PyTorch MPS fails 15 s and 30 s clips | `figures/fig-cs1-latency.pdf` |
| 6 | On the phone | 2.6–3.2x vs the MLX Swift port on an A17 Pro; on a 4 GB A14, MLX is killed for memory at 30 s while we finish in 4.0 s | iPhone bars with the OOM cell (new) |
| 7 | Music in real time: a 40 ms clock that never stops | Streaming changes the metric from median to p99 | `figures/fig-mrt2-pipeline.pdf` |
| 8 | State mutation is the ANE cliff | The 12-layer stack with caches as inputs and updates as outputs is ANE-clean (p99 15.0 ms vs 26.7 ms on CPU+GPU, A14); graphs that mutate state in-graph fail to compile | `figures/fig-ane-ladder.pdf` |
| 9 | Layout decides NaN vs. music | Channels-first decoder: ANE 6.7 ms and finite; the same artifact on CPU or GPU is non-finite | Per-compute-unit bar + finite/NaN marker (new) |
| 10 | Bandwidth sets the graph boundary | Twelve sampling calls became one in-graph rollout: 37 → 12.7 ms on the A14 | Weight bytes vs latency with a DRAM-bandwidth line (new) |
| 11 | Proving placement | Instruments + paired power: zero app GPU intervals, −56% CPU instructions — measured on the earlier all-ANE configuration | `figures/fig-power.pdf` + one trace screenshot |
| 12 | What's still open | Thermal soak fails on the phone; the A14 misses the 40 ms deadline; no per-factor ablation yet | Short list, no chart |

## Honesty rules for Q&A

- The shipped Kokoro pipeline puts one stage on the ANE. The speedup is from
  decomposition and placement, not "running on the Neural Engine".
- The CS2 power and residency evidence predates the correctness fixes; the
  corrected streaming pipeline runs temporal on GPU and depth on CPU.
- The monolith result is "we could not", with coremltools 8.3 — not
  "impossible".
- On macOS 27.0.1 the flexible generator can abort inside MPSGraph; the Mac
  tables use fixed buckets for that reason.
- MLX baseline is mlx-audio 0.4.3 (pinned); it fails every 3 s input.

## Figures still to make

1. iPhone F vs MLX bars with the OOM cell (data: paper Table "iPhone").
2. Decoder per-compute-unit latency with a finite/non-finite marker.
3. Weight bytes vs per-call latency with a DRAM-bandwidth line.
4. Two-seam monolith diagram.
