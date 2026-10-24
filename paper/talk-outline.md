# Surgical Inference — 20-Minute Talk Outline

Every number below comes from `paper/main.tex`. If the paper changes, update
this file from it, never the other way around. Numbers are from the frozen
rerun (plan 014).

## Thesis (say it in the first minute and the last)

The Neural Engine accepts shapes, not models. Split a generative model at its
dataflow seams, reshape each stage, without changing its math, until the right
engine accepts it, and place every stage by measurement. On Kokoro TTS, that
put the vocoder on the Neural Engine and made 27 s of speech take 309 ms on an M1 Mac Mini,
9.8× faster than MLX on the same machine.

## Slides

| # | Title | One message | Show |
| --- | --- | --- | --- |
| 1 | One chip, three engines | GPU-only inference leaves most of the SoC idle | SoC block diagram (CPU / GPU / ANE, shared DRAM) |
| 2 | Surgical Inference in one slide | Cut at seams, reshape what's rejected, place by measurement | Motif → candidate engine table |
| 3 | The wall | The vocoder is 89% of Kokoro's FLOPs, and the ANE rejects it: a 720,000-sample axis against a 16,384 limit | Compiler error text + axis-length bar |
| 4 | Fold it | Cut time into rows with 8-frame halos; five exact op rewrites; 1,927 of 1,928 ops on the ANE; 68 dB exact in fp32 | Fold diagram (new) |
| 5 | Admitted is not fast | On the M2 Ultra the all-ANE pipeline took 706 ms at 30 s, against 373 ms for the old GPU pipeline under similar load (another session). The ANE is bound by elementwise ops, the GPU by convolutions | Op-class cost bars, ANE vs GPU (new) |
| 6 | One stage, two engines | Convolutions on the ANE, AdaIN and Snake on the GPU, shared IOSurface memory, both busy at once | A14 trace: ANE 292 ms and GPU 268 ms inside a 375 ms span, both busy for 192 ms |
| 7 | The result | 27 s of speech in 309 ms on an M1 Mac Mini (9.8× MLX) and 319 ms on an iPhone 15 Pro Max; MLX is killed for memory at 30 s on the iPhone 12 Pro | Regenerated `fig-cs1-latency.pdf` |
| 8 | Where the speed came from | A14 at 30 s: folding gives 63% of the gain, the hybrid split 32%; on the M2 Air the split gives 57% | Ablation ladder bars R0→R4 (new) |
| 8b | Half the memory, no heat | Mac: 1.9-4.4x less peak memory than MLX. iPhone 15 Pro Max: our runs end thermally nominal; MLX on the same 14 clips ends at serious (throttling). Same work, not same time; no power data yet | Memory table + thermal table (tab:cs1-memory, tab:cs1-thermal) |
| 9 | Music in real time: a 40 ms clock that never stops | Streaming changes the metric from median to p99 | `figures/fig-mrt2-pipeline.pdf` |
| 10 | State mutation is the ANE cliff | The full 12-layer stack with caches in and updates out is ANE-clean; mutating state in-graph fails to compile | `figures/fig-ane-ladder.pdf` |
| 11 | Layout decides NaN vs music | Same FP16 decoder: finite on the ANE, NaN on CPU and GPU | Per-engine bar + finite marker (new) |
| 12 | What didn't work, and the ceiling | Private API saves ~0.2 ms per call in lab tests and is not faster in the rerun; a dozen rejected ideas; lab estimate: the exact model tops out near 75–85% of the hardware floor | Short list + floor-vs-measured bars |

## Honesty rules for Q&A

- Every headline number uses public Core ML APIs. The private e5rt path is one
  labeled column and buys almost nothing.
- The ablation, not intuition, says how much of the gain folding delivers.
  Quote the ladder.
- Phone numbers are the sustained regime; a single request after idle is
  slower.
- The ceiling figures are estimates, not measurements. Never claim 90%.
- MRT2's power and residency traces predate its correctness fixes; the
  shipped MRT2 pipeline runs the temporal stage on the GPU.
- The monolith result is "we could not", with coremltools 8.3, not
  "impossible".
- Quality is SNR plus listening: model-judged blind lineups during development,
  one unblinded author listen on the frozen build. Not a human MOS study.

## Figures still to make

1. Redrawn `fig-kokoro-pipeline` with the generator on the ANE (folded) or
   ANE+GPU (hybrid).
2. Fold diagram: rows, halos, stitched cores.
3. Op-class cost bars on the ANE vs the GPU.
4. Ablation ladder bars, R0→R4, M2 Air and A14.
5. `fig-cs1-latency` regenerated from the rerun.
6. Decoder per-engine latency with a finite/non-finite marker.
