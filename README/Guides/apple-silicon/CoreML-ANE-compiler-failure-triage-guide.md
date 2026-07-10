# Core ML ANE Compiler Failure Triage Guide

This guide ingests the external report on Core ML / ANE compiler failures.
Treat the raw report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-compiler-failure-triage-for-large-ml-program-models-retry/2026-06-06T23-25-15-016Z/raw-report.md`

## Executive Summary

Core ML conversion success is not ANE execution proof. A model can convert to
ML Program cleanly and still fail later when the target device tries to build an
execution plan for `.cpuAndNeuralEngine` or `.all`.

For Kokoro, this matters most when moving from macOS warmed benchmarks to
iPhone proof. We must separate:

- host-side conversion and package generation;
- first-load compile and execution-plan build;
- steady-state warmed prediction;
- silent fallback to CPU/GPU;
- hard ANE plan failure.

Paper-facing speed claims should use warmed prediction only. Device-lab failure
claims should preserve compile/load logs separately.

## Failure Taxonomy

| Symptom | Likely meaning | Kokoro action |
| --- | --- | --- |
| Conversion succeeds, `.cpuAndGPU` loads, `.cpuAndNeuralEngine` fails | ANE lowering problem | Capture package, compute unit matrix, and load logs before rewriting graph. |
| `.all` succeeds but timing regresses | Silent fallback or bad partition | Prove placement with MLComputePlan/Instruments before calling it an ANE path. |
| First prediction is much slower than later predictions | Compile/cache or specialization pollution | Exclude from warmed inference rows. |
| Repro only on one device generation | Device-specific compiler or memory cliff | Record exact device/OS/Xcode and keep a compatibility matrix. |

## Triage Loop

1. Run a compute-unit matrix: `.cpuOnly`, `.cpuAndGPU`,
   `.cpuAndNeuralEngine`, then `.all`.
2. Keep load/compile timing separate from prediction timing.
3. Preserve exact Core ML error text and unified logs around model load.
4. Inspect ML Program structure before assuming a code-signing or app-launch
   problem.
5. Promote a rewrite only after parity and warmed timing both pass.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Treat `.cpuAndNeuralEngine` failure as a graph-placement clue. | Treat `.all` success as proof of ANE residency. |
| Capture the compiled package and device metadata with every failure. | Averaging compile failures into runtime speed rows. |
| Use small isolated probes to identify the failing op family. | Rewriting the full pipeline blindly after one opaque Core ML error. |
| Keep iPhone launch/signing failures separate from model compiler failures. | Calling a locked-device run a model failure. |

## Claims Left As Heuristics

The raw report includes useful ANE/compiler folklore, but the exact thresholds
for graph depth, SRAM cliffs, and state alignment must be treated as
device-version hypotheses until reproduced against the actual Kokoro packages.

## Related Documentation

- [Kokoro A14 iPhone generator execution guide](Kokoro-A14-iPhone-generator-execution-guide.md)
- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [iPhone Core ML device lab runbook](iPhone-CoreML-device-lab-runbook.md)
- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Kokoro M1 vocoder partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
