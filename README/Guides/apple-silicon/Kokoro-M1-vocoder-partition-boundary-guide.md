# Kokoro M1 Vocoder Partition and Boundary Guide

This guide ingests the restarted external report on Core ML partition and
boundary mechanics for strict Kokoro vocoder bodies. Treat the raw report as
research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-m1-partition-and-boundary-mechanics-for-strict-1d-kokoro-vocoder-bodies/2026-06-06T23-29-25-644Z/raw-report.md`

## Executive Summary

The raw report correctly points at boundary cost as a first-class optimization
target. The repo evidence is stricter: a package can gain laishere-like
Neural Engine preferred ops and still lose warmed runtime. Therefore the
implementation target is not "more ANE." It is fewer expensive handoffs, less
buffer movement, and a graph surface that wins wall-clock time on warmed M1.

## Verified API Facts

The following API facts were checked against current Core ML/Core ML Tools
documentation or SDK headers during triage:

| API | Verified use |
| --- | --- |
| `MLPredictionOptions.outputBackings` | Allows client-provided output buffers for named outputs. |
| `MLMultiArray(pixelBuffer:shape:)` | Creates an IOSurface-backed FP16 multi-array from a compatible pixel buffer and can avoid buffer copies. |
| `coremltools.models.utils.save_multifunction` / `MultiFunctionDescriptor` | Can package multiple functions and deduplicate shared constants. |
| `convert_to="mlprogram"` and `compute_precision` | Control ML Program export target and precision. |
| `coremltools.optimize.coreml.palettize_weights` | Applies LUT palettization to MLProgram weights. |

These facts do not prove a speedup. They only justify targeted experiments.

## Boundary Experiments Worth Running

### Output Backings

The harness-only ablation exists in the Swift benchmark as
`kokoro-bench --generator-input-dump ... --generator-output-backing`.

Local M2 Studio CPU+GPU results did not clear the promotion gate:

| Bucket | Baseline median | Output-backed median | Delta |
| --- | ---: | ---: | ---: |
| `3s` | `30.032 ms` | `30.109 ms` | `-0.077 ms` |
| `7s` | `60.870 ms` | `60.456 ms` | `+0.414 ms` |

The `3s` dumped waveform was bit-identical (`max_abs=0.0`) between paths, so
the implementation is valid but not a current local speed win.

If revisiting on another device:

1. Preallocate the generator waveform output with the exact existing shape and
   dtype.
2. Call `prediction(from:options:)` with `MLPredictionOptions.outputBackings`.
3. Compare output metrics against the current `prediction(from:)` path.
4. Benchmark warmed generator-only `3s` and `7s`.
5. Promote only if local median improves by at least `1 ms` without drift.

### Multi-Function Packages

Multi-function packaging is plausible for memory footprint and packaging
cleanliness. It is not automatically a short-bucket latency win. Only use it as
a speed candidate if it removes package load churn, deduplicates real constants,
or keeps a hot sequence inside one runtime-positive call boundary.

### DecoderPre + Generator Merge

The direct DecoderPre+Generator boundary-collapse probe is rejected. Local 3s
M2 Studio timing showed the merged CPU+GPU package slower than the current
two-prediction path (`33.589 ms` vs `31.929 ms`), `.all` was also slower, and
`cpuAndNeuralEngine` produced an ANE compiler failure with `1568.424 ms`
median latency. The waveform stayed close to the current baseline, so the loss
is scheduler/runtime behavior, not a gross correctness failure.

Do not retry this exact merge as a promotion path. Future boundary work needs
either a smaller generator-internal graph surface, a source/HAR representation
change, or a removed handoff that does not absorb DecoderPre into the generator
MLProgram.

### Layout And Rank Changes

Rank-4 `[B, C, 1, T]`, `[B, C, T]`, and packed time-major variants should be
treated as scheduler hypotheses. Measure actual plans and warmed timings before
canonicalizing any layout rule.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Measure wall-clock warmed latency after every placement change. | Assuming NE-preferred op counts imply a win. |
| Keep experiments harness-only until parity and speed both pass. | Rewriting production input buffers after output backings failed to clear the local short-bucket gate. |
| Prefer single-package reshaping over new hot-path predictions. | Adding package boundaries to chase subgraph speed. |
| Inspect generated MIL and MLComputePlan before claiming placement. | Treating `.all` or `.cpuAndNeuralEngine` as proof of device use. |

## Claims Left As Heuristics

The raw report contains several statements that should not be used as
paper-facing facts until verified on the actual packages:

- arbitrary tensor zero-copy through `CVPixelBuffer`;
- rank-4 tensors being required for ANE;
- a fixed dispatch-floor number for Core ML predictions;
- replacing all `linear` operators with `conv1d` guaranteeing ANE placement.

## Related Documentation

- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Kokoro M1 source/body Core ML guide](Kokoro-M1-source-body-coreml-guide.md)
- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Core ML split graphs and multifunction packaging](CoreML-split-graphs-multifunction-packaging-guide.md)
- [Core ML ANE transformer layout and op compatibility](CoreML-ANE-transformer-layout-op-compatibility-guide.md)
- [Kokoro M1 kernel partition prompt](../../Notes/Kokoro-M1-kernel-partition-deep-research-prompt.md)
