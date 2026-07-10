# Core ML ANE Transformer Layout And Op Compatibility Guide

This guide ingests the external report on ANE-friendly transformer layouts and
operator compatibility. Treat the raw report as research input, not canonical
truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-transformer-layout-and-op-compatibility-retry/2026-06-06T23-25-22-946Z/raw-report.md`

## Executive Summary

The report is transformer-focused, but it reinforces the same rule we keep
seeing in Kokoro: graph shape and runtime boundary dominate nominal accelerator
choice. Static shapes, fewer transposes, fewer dynamic gathers/slices, and
channel-first 4D surfaces are plausible ANE strategies only when warmed timing
improves.

For Kokoro vocoder work, this guide is not a direct recipe. It is a checklist
for deciding whether a proposed layout rewrite is worth testing.

## Transferable Checks For Kokoro

| Question | Why it matters |
| --- | --- |
| Does the rewrite preserve one hot prediction boundary? | Extra Core ML calls have repeatedly eaten subgraph gains. |
| Does it remove dynamic slicing, gather/scatter, concat, tile, or transpose-heavy regions? | These surfaces are common fallback and sync suspects. |
| Does it keep tensor layout stable across adjacent blocks? | Layout churn can dominate short buckets. |
| Does MLComputePlan show better placement without worse transfer cost? | Placement alone has not predicted wall-clock wins. |
| Does warmed `3s` and `7s` timing improve on lower-end Macs? | The paper fight is shortest buckets first. |

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Convert layout advice into small isolated probes. | Applying transformer ANE rules wholesale to the vocoder. |
| Measure CPU+GPU and `.all` warmed timings after each layout change. | Assuming rank-4 means faster. |
| Keep static runtime buckets: `3s`, `7s`, `10s`, `15s`, `30s`. | Flexible-shape convenience in paper-facing benchmarks. |
| Inspect fallback-prone ops before chasing ANE residency. | Treating ANE residency as the goal instead of latency. |

## Claims Left As Heuristics

The raw report includes specific ANE SRAM, dispatch, and layout-threshold
claims. Keep those as hypotheses unless reproduced with Kokoro packages on the
target device generation.

## Related Documentation

- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Core ML vs MLX vocoder scheduling](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Kokoro M1 vocoder partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
