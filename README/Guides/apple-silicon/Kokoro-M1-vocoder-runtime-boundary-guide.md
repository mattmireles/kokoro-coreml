# Kokoro M1 Vocoder Runtime Boundary Guide

This guide ingests the restarted external report on runtime-boundary strategies
for lower-end Apple Silicon. Treat the raw report as research input, not
canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-vocoder-runtime-boundary-strategies-for-lower-end-apple-silicon/2026-06-06T23-29-40-654Z/raw-report.md`

## Executive Summary

The report agrees with the strongest repo evidence: multi-package Core ML
splits are the trap unless they remove more boundary and synchronization cost
than they add. The fastest strict path is likely a single hot prediction with a
better internal graph surface, not a larger number of cleaner subgraphs.

## Current Boundary Model

Think about each candidate in three layers:

| Layer | Question |
| --- | --- |
| Runtime boundary | Does this add or remove a Core ML prediction call, host copy, or sync point? |
| Graph surface | Does this remove costly reductions, tiles, broadcasts, transposed conv surfaces, or fallback-prone ops? |
| Placement | Does MLComputePlan plus warmed timing show useful CPU/GPU/ANE behavior? |

Candidates need to win all three in practice. Winning only graph surface or
placement has not been enough on M1 short buckets.

## Current Best Strict Candidate Family

The production upsample ConvTranspose rewrite is the first strict local
candidate in this sequence with a material warmed CPU+GPU win across buckets.
It should be promoted to Irvine M1 only under quiet-host warmed conditions and
then combined with independent savings.

The next boundary candidates should be lower risk:

- single-package body reshaping only inside the existing GeneratorFromHar
  surface; the measured DecoderPre+Generator merge made `3s` slower and failed
  CPU+NE scheduling;
- source/body compact representation only inside a runtime-positive boundary;
- HAR/STFT learned adapter only after PyTorch parity evidence.

The generator `outputBackings` ablation is now implemented and valid, but local
CPU+GPU warmed `3s`/`7s` timing did not clear the `1 ms` promotion gate. Keep it
as a harness option for future device checks, not as the current fastest path.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Keep the hot path to one prediction unless measurement says otherwise. | Splitting because a submodel plan looks prettier. |
| Make one graph-surface change at a time and preserve strict metrics. | Combining boundary, dtype, layout, and quality changes in one branch. |
| Promote local wins to Irvine only with quiet-host proof. | Treating local M2 Studio deltas as lower-end Mac proof. |
| Track `3s` fixed cost separately from duration-scaled cost. | Optimizing only the `30s` path and expecting a short-bucket paper win. |
| Keep DecoderPre separate unless a new graph surface changes the scheduler result. | Merging DecoderPre into GeneratorFromHar just to remove one Swift-visible prediction call. |

## Related Documentation

- [Kokoro M1 vocoder partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
- [Kokoro Irvine M1 3s/7s paper frontier guide](Kokoro-Irvine-M1-3s-7s-paper-frontier-guide.md)
- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Kokoro M1 vocoder boundary research brief](../../Notes/Kokoro-M1-vocoder-boundary-research-brief.md)
- [Core ML vs MLX vocoder scheduling](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
- [Core ML split graphs and multifunction packaging](CoreML-split-graphs-multifunction-packaging-guide.md)
- [Core ML ANE temporal escape hatches](CoreML-ANE-temporal-escape-hatches-guide.md)
