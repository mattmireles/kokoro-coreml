# Core ML ANE Temporal Escape Hatches Guide

This guide ingests the external report on stateful temporal Core ML escape
hatches. Treat the raw report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-temporal-escape-hatches-for-stateful-streaming-transformers-on-iphon/2026-06-06T22-48-22-802Z/raw-report.md`

## Executive Summary

The report targets stateful streaming transformers, not Kokoro. Its useful
lesson for this repo is narrower: when the model has temporal state or repeated
small predictions, the cost of host-managed state, dispatch frequency, and
graph compilation can dominate compute.

For Kokoro, use this guide mainly when evaluating iPhone execution or any future
streaming/burst mode. Do not use it to justify adding more Core ML package
boundaries to the current warmed full-utterance benchmark path.

## Kokoro-Relevant Lessons

| Area | Practical reading |
| --- | --- |
| Burst sizing | Batching work can amortize dispatch, but fixed buckets remain the benchmark contract. |
| Host-owned state | Moving dynamic updates to Swift can help compiler viability but adds memory traffic. |
| Multifunction packaging | Can reduce artifact duplication; it is not automatically a runtime win. |
| Foreground/device behavior | iPhone proof must separate app/device state from model speed. |
| p99 latency | Streaming paths need p99 and thermal stability, not only median warmed rows. |

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Use temporal escape hatches only when a stateful path really exists. | Importing Crossfade transformer assumptions into Kokoro vocoder timing. |
| Measure dispatch count and state-copy size. | Hiding host state updates inside “model inference” timing. |
| Preserve warmed-only full-bucket comparisons for the paper bakeoff. | Mixing streaming p99 goals with full-utterance median goals. |

## Related Documentation

- [Core ML split graphs and multifunction packaging](CoreML-split-graphs-multifunction-packaging-guide.md)
- [iPhone Core ML device lab runbook](iPhone-CoreML-device-lab-runbook.md)
- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
