# Surgical Inference CS1 Rerun on the ANE Pipeline

**Date:** 2026-10-23
**Status:** Complete
**Progress model:** Phase task checkboxes only

## Executive Summary

Re-measure every Kokoro (case study 1) number in `paper/main.tex` on the
October ANE pipeline (split duration and F0Ntrain, ANE-layout DecoderPre,
folded and hybrid generators), on the paper's five devices, under plan 013's
protocol. The ablation ladder (one change at a time), placement proof and
quality gate let the paper attribute the 4-8x to specific changes. Requested by
the paper session; the owner approved the M2 Ultra and the start
(2026-10-22 PT).

## Problem Statement

- **Symptom:** The paper's Config F rows (plan 013, 2026-10-20) measure the
  bucketed GPU-generator pipeline: 30 s A14 4,017 ms, A17 1,721, M1 1,635.5,
  M2 Air 1,175.5. Tonight's pipeline runs 30 s in ~460-490 / ~320 / ~300 /
  ~220 ms (`README/Notes/ane-folded-generator.md`).
- **Root Cause:** Five changes landed after plan 013 and no frozen
  measurement covers them.
- **Impact:** The paper understates the system by 4-8x, and cannot say which
  change gave what.

## Goals and Non-Goals

### Goals

- Every device: folded and hybrid Config F (public Core ML), hybrid on e5rt as
  an aside column, MLX, at every bucket.
- Ablation ladder on the M2 Air and A14 (3 s and 30 s).
- Placement proof for the folded and hybrid generators.
- Quality gate per clip, plus a listening pass on the final F.
- A recorded shipping decision by the owner's rule.

### Non-Goals

- No pipeline changes once Phase 0 freezes. A failure is recorded, not fixed.
- The October pipeline is rerun only as ablation rung R0 (same session, no mixed-session ratios).
- No HF publish of the frozen bundle without the owner.

## Frozen Protocol (Do Not Change After Phase 0)

| Item | Value |
| --- | --- |
| Source | `main` at the Phase 0 commit, no local diff in `swift/`, `swift-tts/`, `ios-bench/` |
| Packages | Frozen set listed with SHA-256 in `outputs/frozen_rerun_014/manifest.json`: HF `dadebca4` base (`coreml/`) plus `coreml-dur-ane/`, `coreml-f0n-split/`, `coreml-dpre-ane/`, `coreml-fold/`, `coreml-hybrid/` |
| Inputs | `3s`, `7s`, `10s`, `15s`, `30s` (`outputs/swift_bench_inputs`), voice `af_heart`, speed 1.0 |
| Timing | 3 preflight calls discarded, 1 cold call recorded, median of 10 warm calls (and IQR) |
| Arms | `folded` (split stages + folded generator), `hybrid` (split stages + hybrid generator, Core ML), `hybrid-e5` (aside: `KOKORO_HYBRID_E5=1`), `mlx` (mlx-audio 0.4.3 on Macs, kokoro-ios 1.0.8 on phones) |
| Macs | `kokoro-bench --compute-units staged --batch`; quiet gate 1-min load ≤ 1.5; Botnet paused, prior mode restored |
| iPhones | Release `ios-bench`, one launch per arm per bucket, arm order alternated per bucket, ≥ 10 min cooldown before each launch, nominal thermal state at start |
| Record per cell | raw per-call timings, cold call, commit, OS, load or thermal state, peak footprint |

## Quality Gate (Set Before Results)

- Every clip, every bucket, both generators: audio SNR ≥ 40 dB against the
  PyTorch fp32 reference, and total predicted frames equal. Report the worst
  clip, not only the median.
- Hard inputs as well: numbers, heavy punctuation, the shortest and longest
  buckets.
- `audio-judge` listening pass on the final F outputs.

## Folded vs Hybrid Decision (Owner Rule)

- Hybrid becomes the shipped default only if, on every device, it is faster
  than folded, quality is equal, and it never crashes in the matrix.
  Otherwise folded stays the default and hybrid is its own row.
- **Tie:** on a device and bucket, warm medians within 5%, or overlapping
  run-to-run ranges. Report ties per device and bucket, with the median
  difference (ms and %), peak RAM of each, and GPU share if cheap.
- **Noise-floor tie (reported alongside, not replacing the 5% rule):** at 3 s
  both arms run the folded generator (hybrid serves ≥ 7 s), so the 3 s gap is
  the device's launch-to-launch noise. A 30 s folded-vs-hybrid gap smaller
  than that device's 3 s gap is also called a tie; report both numbers.
- GPU busy share: one extra `KOKORO_HYBRID_TIMELINE=1` launch per device at
  10 s and 30 s after the matrix, labeled as instrumented.

## Ablation Ladder (M2 Air and A14, 3 s and 30 s)

| Rung | Change | Switch |
| --- | --- | --- |
| R0 | October bundle on the current runtime (host-code changes included; no switch) | GPU-layout DecoderPre, GPU generator, `KOKORO_SPLIT_DURATION=0 KOKORO_SPLIT_F0N=0` |
| R1 | + split duration and F0Ntrain | as R0 with the splits on |
| R2 | + ANE-layout DecoderPre | same, with `coreml-dpre-ane` |
| R3 | + folded generator | `--folded-generator` |
| R4 | hybrid generator instead | `--hybrid-generator` |

SNR at every rung.

## Implementation Phases

### Phase 0: Freeze

- [x] Record commit, OS per host, and the frozen package manifest with hashes.
- [x] Build `ios-bench` Release from the frozen set (main app) and an R0/R1 app
  (GPU-layout DecoderPre); record the bundle hashes.
- [x] Sync source and packages to m2-air and irvine-m1; build `kokoro-bench`.
- [x] Smoke: one call per arm and bucket on each host passes the quality gate.

### Phase 1: Placement and quality

- [x] MLComputePlan for the folded and hybrid packages on the M2 Air, and
  through `tools/ane-lab` on the A14.
- [x] Per-clip SNR on every bucket and hard input; worst clip reported.
- [x] `audio-judge` pass on the final F.

### Phase 2: iPhone matrix

- [x] A17 Pro and A14: folded, hybrid, hybrid-e5, MLX × 5 buckets.
- [x] A14 ablation R0-R4 at 3 s and 30 s.

### Phase 3: Mac matrix

- [x] m2-studio, m2-air, irvine-m1: folded, hybrid, hybrid-e5, MLX × 5 buckets.
- [x] M2 Air ablation R0-R4 at 3 s and 30 s.

### Phase 4: Record

- [x] Restore every host's Botnet mode; verify with `SupervisorCLI status`.
- [x] Write `README/Notes/paper-cs1-hybrid-rerun-2026-10-23.md` (plan 013
  note style): tables, ties, ablation, placement, quality, shipping decision.
- [x] Send the paper session results, manifest path and hashes.

## Kill and Stop Gates

- A quality-gate failure stops that device's matrix for that arm; record it.
- A crash counts against hybrid in the decision rule; record it, do not retry
  around it.
- A host that cannot pass the quiet gate is run and labeled `upper bound (load X)`.

## Success Criteria

- Every Kokoro number in paper CS1 has a cell here with its raw JSON path.
- The shipping decision follows the owner's rule, with ties stated.
- Botnet back in each host's prior mode.
