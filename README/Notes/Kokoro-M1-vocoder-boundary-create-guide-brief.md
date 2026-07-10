# Kokoro M1 Vocoder Boundary Create-Guide Brief

June 6, 2026

Create an advanced developer field guide on runtime-positive Core ML vocoder
boundaries for lower-end Apple Silicon. This is not a generic Core ML
optimization guide.

## Context

We are optimizing a first-party Swift + Core ML Kokoro/CoPro-style TTS pipeline
against popular MLX and Core ML implementations. Runtime buckets are fixed:
`3s`, `7s`, `10s`, `15s`, and `30s`. Only warmed inference counts; Core ML
compile/cache time is excluded.

Corrected warmed evidence says MLX is not the current Mac blocker. Config F
beats or ties MLX on validated Mac rows. The remaining strict competitor is
`laishere/kokoro-coreml` on Irvine M1 short/medium buckets.

For Irvine M1 `3s`, warmed profile gap:

| Runtime | Warm median |
| --- | ---: |
| Config F | `233.6 ms` |
| laishere | `195.0 ms` |
| Gap | `38.5 ms` |

After strict local keepers are projected:

- HAR-post upsample ConvT rewrite;
- `decoder_pre`/HnSF runtime overlap;

remaining profile gaps are about `27.0 ms` at `3s`, `28.0 ms` at `7s`,
`12.8 ms` at `10s`, and closed at `15s`.

## Core Observation

Compute-unit flags and visible graph cleanup are not enough.

Placement evidence:

| Plan | Units | Ops | Preferred counts | NE cost |
| --- | --- | ---: | --- | ---: |
| First-party HAR-post | CPU+NE | `2207` | `cpu=1038`, `unknown=1169` | `0.0%` |
| First-party HAR-post | CPU+GPU | `2207` | `gpu=1041`, `unknown=1166` | `0.0%` |
| laishere vocoder | CPU+NE | `1534` | `cpu=58`, `neuralEngine=597`, `unknown=879` | `47.5%` |
| First-party exact decoder+vocoder body | CPU+NE | `1546` | `cpu=64`, `neuralEngine=599`, `unknown=883` | `48.7%` |

The last row is the trap: first-party can already create a strict body with
laishere-like NE placement, but it is slower. The target is not merely "more NE
ops"; the target is a runtime-positive package boundary and partition that
avoids synchronization, copy, layout, and launch overhead.

The first-party fused graph has manual AdaIN/tile/reduce surfaces; laishere has
native-looking instance norm and LUT/palettization surfaces. Prior probes that
only matched these surfaces did not produce enough M1 runtime gain.

## Primary Research Goal

Identify Core ML graph boundaries, tensor layouts, model splits, and scheduling
patterns that make a strict Kokoro vocoder body fast on M1 and iPhone without
the measured split-boundary penalty.

## Questions To Answer

1. Why can a Core ML package with laishere-like NE preferred ops still run
   slower than laishere?
2. What boundaries typically trigger CPU/GPU/ANE synchronization penalties in
   MLProgram vocoder graphs?
3. Which tensors should cross the Swift/Core ML boundary: `har_source`, HAR
   features, `x_source_*`, decoder pre features, body activations, or waveform?
4. How do static bucket shapes affect Core ML partitioning and command
   scheduling?
5. What tensor layouts are best for 1D vocoder bodies on Apple hardware:
   `[B,C,T]`, `[B,T,C]`, `[B,C,1,T]`, or packed variants?
6. Which ops should be fused into one package, and which should stay in Swift,
   Accelerate, Metal, or a separate package?
7. What `MLComputePlan` signatures distinguish runtime-positive partitioning
   from deceptive NE placement?
8. How should `cpuAndNeuralEngine`, `cpuAndGPU`, and `.all` be chosen for this
   graph family?
9. What profiling workflow should prove whether the loss is launch overhead,
   memory copies, synchronization, ANE fallback, GPU scheduling, or CPU scalar
   work?
10. What first two implementation experiments should we run locally before
    quiet Irvine M1 promotion?

## Output Format

Start with an executive summary of why laishere-like NE placement can still be
slower. Then provide:

- exact boundary recommendations;
- tensor layout recommendations;
- MLProgram/Core ML conversion recipes;
- `MLComputePlan` and Instruments/xctrace profiling workflow;
- do-this / avoid-this tables;
- known misleading benchmark patterns;
- first two experiments with stop/go criteria;
- clearly mark speculative ideas separately from evidence-backed guidance.
