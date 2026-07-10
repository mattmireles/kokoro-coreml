# Core ML Split Graphs And Multifunction Packaging Guide

This guide ingests the external report on Core ML split graphs and
multifunction packaging. Treat the raw report as research input, not canonical
truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-split-graphs-and-multifunction-packaging-for-streaming-inference-retry/2026-06-06T23-25-30-108Z/raw-report.md`

## Executive Summary

Splitting a Core ML graph can make a package compile or make a subgraph look
cleaner. That is not the same as making Kokoro faster. The repo evidence so far
is that extra hot-path prediction boundaries are dangerous on lower-end Macs,
especially in `3s` and `7s` buckets.

Use split graphs when they are required for compiler viability or when they
remove more cost than they add. Use multifunction packaging primarily as an
artifact-size and shared-weight tool until a warmed benchmark proves a runtime
benefit.

## Decision Table

| Candidate | When it is worth testing | Promotion gate |
| --- | --- | --- |
| Separate packages | A monolithic package cannot compile or one boundary is known to remove a larger cost. | Parity plus warmed wall-clock win after boundary cost. |
| Multifunction package | Multiple functions share large constants or model variants. | Smaller artifact or load win; runtime win must be measured separately. |
| Layer-group split | A specific graph region blocks compile or placement. | Boundary tensor cost below compute saved. |
| Source/body split | Strict representation repair makes source/HAR cheap enough to avoid an extra prediction tax. | Strict audio parity plus lower-end Mac warmed win. |

## Kokoro Boundary Rule

The default assumption remains one hot generator prediction. Any split must
carry its own proof:

1. strict parity against the current output;
2. warmed timing for every bucket it touches;
3. quiet-host proof before lower-end Mac rows become publishable;
4. no compile/cache timing in the speed row.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Measure the bytes crossing each boundary. | Splitting because the subgraph plan is aesthetically cleaner. |
| Keep package-load, compile, and prediction timing separate. | Reporting first-run compile behavior as inference speed. |
| Use multifunction packaging to deduplicate shared constants when useful. | Assuming shared disk weights imply shared runtime allocations. |
| Treat boundary count as a first-order performance variable. | Optimizing only per-op placement. |

## Claims Left As Heuristics

The raw report includes fixed dispatch-cost and memory-limit numbers. Treat
them as starting estimates only; Kokoro promotion requires repo measurements on
the target hardware.

## Related Documentation

- [Kokoro M1 vocoder partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)
- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
