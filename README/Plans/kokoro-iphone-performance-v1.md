# Kokoro iPhone Performance v1 Plan

**Date:** 2026-06-10
**Status:** Planned

## Executive Summary

Close the iPhone-vs-Mac gap for the Kokoro Core ML pipeline. Today the
iPhone 12 Pro (A14) runs warmed RTF ~0.45 and the iPhone 15 Pro Max (A17 Pro)
~0.23, versus Irvine M1 Config F at ~0.072. The plan instruments the iPhone
bench to name the `.all` ANEF reject and measure per-stage time, recovers the
GPU path toward RTF ~0.2 on A14, then runs a cheap ANE-admittance gate that
decides whether to build the ANE-resident re-chunked generator that
Mac-class RTF requires.

## Problem Statement

- **Symptom:** Both test iPhones reject the `.all` compute plan at first
  predict (`MILCompilerForANE ... ANECCompile() FAILED`, surfaced as
  `error code: -9`); the bench falls back to the staged policy and runs
  5.9-6.3x slower than Irvine M1.
- **Root Cause (two distinct problems):** (1) the generator is ~82-90% of
  phone wall and runs on a GPU 2-2.7x weaker than M1's yet measures 6.8x
  slower — an unexplained platform-scaling anomaly; (2) the generator is
  structurally inadmissible to the ANE at every bucket (per-axis tensors
  28,801-720,000 vs the 16,384 ANE limit), so the one accelerator where A14
  matches M1 (same 16-core 11-TOPS ANE generation) sits idle.
- **Impact:** iPhone TTS is ~2x real time at 3s on A14 (unusable for
  interactive playback start), and background-safe synthesis (ANE-only) is
  impossible.

## Goals and Non-Goals

### Goals

- [ ] Name the stage(s) whose ANEF compile fails under `.all` on device, with
      verbatim device-log evidence.
- [ ] Replace fitted per-stage iPhone numbers with measured ones
      (`SynthesisResult.timings` serialized by the bench).
- [ ] Remove the duration-package confound (exact-native `lstm` packages on
      device, as on the Mac frontier rows).
- [ ] Recover measurable GPU-path time on A14 (target: 30s bucket wall under
      ~8 s, RTF <= 0.30; stretch ~0.2) with strict waveform parity.
- [ ] Produce a GO/STOP verdict on ANE admittance for the generator conv
      stack (stripped, all axes < 16,384, fp16, CPU+NE).
- [ ] If GO: a working overlap-add re-chunked generator prototype with parity
      and placement evidence.

### Non-Goals

- Retraining the model (Vocos-style transposed-conv-free generator) — flagged
  speculative in the guide corpus; out of scope for v1.
- Adopting MLX for the iPhone path — no ANE access; 30s jetsams the 4 GB
  phone ([iphone-debug-notes.md](../Notes/iphone-debug-notes.md)).
- Changing the Mac production policy or Mac bakeoff rows.
- Streaming/incremental synthesis API design (re-chunking here is an internal
  execution detail of one bucket, not a product streaming feature).

## Scope and Constraints

- **Scope:** `ios-bench/` app, `swift/Sources/KokoroPipeline` (timing/copy
  internals only), `export_synth/` probe exports, new `scripts/` probes,
  `README/Notes` evidence updates.
- **Constraints:** Physical-device steps need an unlocked, tethered, charged
  iPhone — they cannot run in CI and may be unavailable in a given session.
  Host `xcodebuild` may stall at CreateBuildDescription; the probe-reaper
  workaround is documented in
  [iphone-debug-notes.md](../Notes/iphone-debug-notes.md).
- **Guardrails:** Mac Config F warmed medians must not regress; the staged
  policy remains the shipping default until device evidence says otherwise;
  no RangeDim/flexible shapes anywhere
  ([coreml-compute-unit-ablation.md](../Notes/coreml-compute-unit-ablation.md)).

### Device-Gated Task Protocol

Tasks marked **[device]** require the physical iPhone. If no device is
available when a phase executes: implement and verify everything local,
mark the [device] checkboxes as deferred in this plan with a dated note, and
do NOT fabricate device numbers. A deferred [device] verification does not
block committing the phase's code; it blocks only the claims that depend on
it.

## Ground Truth Contracts (Do Not Violate)

- **Strict parity gate:** any export or runtime change that can touch the
  waveform must be compared against the current staged baseline output for
  the same frozen inputs before promotion. Compute-unit changes alone have
  broken parity before (corr 0.69 —
  [stage-compute-policy-ablation.md](../Notes/stage-compute-policy-ablation.md)).
- **Warmed-only timing:** discard >= 2 warmups, medians over >= 5 iterations,
  record thermal state, one bench arm per process on the 4 GB phone.
- **Evidence before claims:** ANE placement claims need a compute plan and/or
  Instruments trace, never latency alone
  ([CoreML-ANE-compiler-failure-triage-guide.md](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md)).
- **Bucket geometry is fixed:** 3s/7s/15s/30s exports with explicit shapes.

## Already Shipped (Do Not Re-Solve)

- **ios-bench app + policy ladder:** `ios-bench/Sources/BenchApp.swift` runs
  both arms headless with `all → staged → cpuAndGPU → cpuOnly` fallback and
  jetsam-safe result flushing.
- **Per-stage timing plumbing:** `SynthesisResult.timings: StageTimings`
  (`swift/Sources/KokoroPipeline/KokoroPipeline.swift:77-138`) is populated
  unconditionally on every call — the bench just drops it.
- **Exact-native duration packages:** `kokoro_duration_exact_t{N}` (780 ops,
  native MIL `lstm`) already exported under `coreml/`; the Mac frontier rows
  use them via `KOKORO_USE_EXACT_DURATION_MODELS=1`.
- **Upsample ConvTranspose rewrite:** `--rewrite-ups-conv-transpose` in
  `export_synth/main.py` (zero-insert + conv1d via `ZeroInsertConvTranspose1d`
  in `export_synth/wrappers.py`), the only strict-parity speed win in the
  ledger (+2.2-4.5%).
- **Rejected-experiment ledger:** do not re-run anything in
  [coreml-compute-unit-ablation.md](../Notes/coreml-compute-unit-ablation.md)
  "rejected" entries (splits, palettization, surface-matching, fp16-input,
  style specialization, dual-anchor at the HAR-post boundary, etc.).

## Fresh Baseline (Current State)

- **A14 staged warmed medians:** 3s 1.383 s / 7s 2.966 s / 15s 6.250 s /
  30s 12.301 s (RTF 0.44-0.49). A17 Pro: 0.702 / 1.492 / 3.272 / 6.374 s
  (RTF ~0.23). Irvine M1 Config F: 0.234 / 0.493 / 1.015 / 1.959 s
  ([iphone-performance-notes.md](../Notes/iphone-performance-notes.md)).
- **Fitted (unmeasured) generator share on A14:** ~82-90% of wall.
- **Known gaps:** no per-stage device timings; failing `.all` stage unnamed;
  decoder-pre's ANE residency on device unverified; padded-vs-exact duration
  confound in the iPhone bundle.

## Solution Overview

```
Phase 1 (code)      Phase 2 [device]      Phase 3 (exports)     Phase 4 (gate)      Phase 5 (build)
instrument bench -> measure + name the -> GPU-path recovery  -> ANE admittance  -> overlap-add
exact duration      ANEF reject           with parity gates     GO/STOP on A14     re-chunked generator
packages                                                            |                (only on GO)
                                                                  STOP -> accept GPU ceiling,
                                                                          close plan at Phase 3 result
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Prerequisites

**Goal:** Build environment and artifacts confirmed.

**Tasks:**

- [x] Confirm `xcodegen generate` + `xcodebuild` build of
      `ios-bench/KokoroIPhoneBench.xcodeproj` succeeds on this host (apply the
      probe-reaper workaround from
      [iphone-debug-notes.md](../Notes/iphone-debug-notes.md) if SwiftBuild
      stalls). 2026-06-10: BUILD SUCCEEDED, no stall; reaper ran idle.
- [x] Confirm `swift build -c release` of `swift/` package passes.
- [x] Confirm `kokoro_duration_exact_t44/t105/t219/t476` packages exist under
      `coreml/` for the four bench inputs (44/105/219/476 tokens); export any
      missing size with `export_duration.py`. 2026-06-10: all four present.
- [x] Run root `pytest` to baseline green. 2026-06-10: 104 passed via
      `.venv/bin/python -m pytest` (system `python3` lacks repo deps; pytest
      was bootstrapped into `.venv`).

**Verification:** All builds green; package inventory listed in the phase
commit message.

---

### Phase 1: Bench Instrumentation + Exact Duration Packages

**Goal:** The bench measures per-stage time, names ladder failures, supports
single-stage compute-unit flips, and uses exact-native duration models.

**Tasks:**

- [x] `ios-bench/Sources/BenchApp.swift`: serialize `result.timings` per
      iteration into the record dict (per-stage arrays + medians for every
      `StageTimings` field) plus `bucketSeconds`, `duration_cache_key`, and
      `ProcessInfo.processInfo.thermalState` per iteration.
- [x] `BenchApp.swift`: on every failed ladder rung, persist a record with
      the failing policy name, NSError domain/code/localizedDescription, the
      full error chain, and a `last_vended_stage` attribution hint from
      `BundleModelCache` — never just the surviving policy.
- [x] `BenchApp.swift`: add `--mode matrix` — staged baseline plus exactly
      one stage flipped per cell (`duration=ne`, `f0n=ne`, `decoderPre=gpu`
      inverse probe, `generator=ne`) and a `cpuOnly` reference; 3s bucket
      for all cells, 30s additionally for `generator=ne` (tests 16,384
      enforcement granularity: the 3s post-upsample body axis 14,401 fits).
      No ladder fallback in matrix mode — a failed cell is the data point.
- [x] Bundle `kokoro_duration_exact_t{44,105,219,476}` (via
      `ios-bench/prepare_resources.sh`, which populates the gitignored
      `Resources/coreml`) and add a `--exact-duration 1` launch flag that
      builds `DurationModelChoice` entries for them (`allowsPadding: false`);
      padded packages remain the default so the A/B is explicit.
- [x] Keep jetsam-safety invariants: flush after every record, `--arms` /
      `--keys` process splitting untouched, fresh cache per matrix cell.

2026-06-10: iOS build SUCCEEDED with the instrumented BenchApp (compile is
the field-coverage check — `stageDict` names every `StageTimings` field).
One SwiftBuild `CreateBuildDescription` stall hit during verification and was
cleared with the documented probe-reaper workaround.

**Verification:** iOS build green; a macOS unit or compile-level check that
the record dict serializes all `StageTimings` fields (no device needed);
`pytest` still green.

---

### Phase 2: Device Evidence Run [device]

**Goal:** Replace every fitted number and unnamed failure with measurement.

**Tasks:**

- [ ] **[device]** Run the instrumented bench on the iPhone 12 Pro (and
      A17 Pro if available): default ladder run + `--matrix` run +
      `--exact-duration` A/B, one arm per process.
- [ ] **[device]** During a failing `.all` first predict, capture
      `log stream --device --predicate '(subsystem IN {"com.apple.espresso","com.apple.coreml"})' --info --debug`
      from the tethered Mac; archive the verbatim failure lines.
- [ ] **[device]** Dump per-op compute plans for decoder_pre (CPU+NE) and the
      3s generator (CPU+GPU and CPU+NE) on-device via
      `coremltools.models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device`
      (new script `scripts/dump_device_compute_plan.py`); settles whether
      decoder-pre's ANE pin is real on the phone and which generator ops the
      GPU plan maps to CPU.
- [ ] Record all results in
      [iphone-performance-notes.md](../Notes/iphone-performance-notes.md)
      (timings) and
      [iphone-debug-notes.md](../Notes/iphone-debug-notes.md) (failure
      attribution), including the named ANEF-reject stage(s).

**Verification:** Notes updated with measured per-stage tables and the named
failing stage; raw JSON archived under `outputs/iphone_bench/`. If no device:
script + notes scaffolding committed, [device] boxes marked deferred with
date.

---

### Phase 3: GPU-Path Recovery (Parity-Gated)

**Goal:** Cut A14 generator wall on the GPU path; every change strict-parity
gated on Mac before any device promotion.

**Tasks:**

- [ ] Export probe: trim `x_source` to the receptive-field-padded target
      length BEFORE `noise_res[i]` runs (`export_synth/wrappers.py:148-158`,
      kernel-11/dilation-5 margin); verify strict parity on Mac via the
      existing generator-isolation harness, then warmed A/B
      (`kokoro-bench --generator-input-dump ... --warmup 5 --iterations 20`).
- [ ] Promote the `--rewrite-ups-conv-transpose` export into the iPhone
      bundle set and re-verify parity on Mac.
- [ ] `swift/Sources/KokoroPipeline/KokoroSynthesisExecutor.swift`: move the
      `zeroPad3D` MLMultiArray allocations (lines ~360-374) out of the timed
      generator window — preallocate per bucket and reuse; bit-identical
      output required.
- [ ] If Phase 2's compute plan shows CPU-mapped generator ops under
      CPU+GPU: rewrite exactly those subgraphs (new probe script following
      the `scripts/probe_generator_*.py` pattern); parity gate; do NOT
      re-run ledger-rejected rewrites.
- [ ] **[device]** Re-run the bench A/B on the A14 with the new export set.

**Verification:** Mac parity reports (corr/SNR/max-abs vs staged baseline)
recorded; Mac Config F medians not regressed (M2-class same-host A/B);
`pytest` green. Device delta recorded or deferred.

---

### Phase 4: ANE Admittance Gate

**Goal:** A cheap GO/STOP verdict on whether the generator conv stack can
ever be ANE-resident on A14-generation hardware.

**Tasks:**

- [ ] New `scripts/probe_generator_ane_admittance.py`: export a stripped
      fp16 generator body (real weights, hardcoded small T, every tensor
      axis < 16,384, no fp32 noise/tail branches, no over-limit I/O).
- [ ] Verify it loads and predicts under `.cpuAndNeuralEngine` on the local
      M-series Mac first (M1-generation ANE compiler signal), with
      MLComputePlan NE-preferred op counts recorded.
- [ ] **[device]** Load + one predict on the A14 pinned `.cpuAndNeuralEngine`;
      record GO (predicts, NE-resident ops in the on-device compute plan) or
      STOP (ANECCompile failure) with the verbatim error.
- [ ] If STOP and the compute plan indicts a specific op family (e.g. AdaIN
      lowering): one rewrite iteration of that family, then retry once.
      Two STOPs = verdict stands.
- [ ] Record the verdict and evidence in
      [iphone-debug-notes.md](../Notes/iphone-debug-notes.md) and update the
      [A14 guide triage note](../Notes/kokoro-a14-iphone-guide-triage-2026-06-10.md).

**Verification:** Probe script committed with its Mac evidence; device verdict
recorded or deferred. **STOP verdict ends the plan after Phase 6** (the A14
ceiling is the Phase 3 GPU path); GO authorizes Phase 5.

---

### Phase 5: Overlap-Add Re-Chunked Generator (Conditional on Phase 4 GO)

**Goal:** A prototype generator package set whose ANE-targeted stage has
every axis < 16,384, fed by Swift-side windowed overlap-add, meeting strict
parity.

**Tasks:**

- [ ] Design note first (`README/Notes/`): chunk length, receptive-field
      overlap margin (account for dilated convs and the k20/s10 + k12/s6
      upsample stack), AdaIN statistics policy (per-chunk vs precomputed
      full-utterance statistics passed as inputs), and seam cross-fade
      length. AdaIN statistics policy is the parity-critical decision —
      prototype both if cheap.
- [ ] Export: chunked generator body (fp16, CPU+NE target) + fp32 tail
      (conv_post + exp/sin + iSTFT) as separate packages, laishere-boundary
      style; fixed per-bucket chunk geometry, no RangeDim.
- [ ] Swift: chunk scheduler + overlap-add cross-fade in `KokoroPipeline`
      behind a runtime flag (staged path remains default).
- [ ] Parity gate on Mac vs staged baseline (corr/SNR/max-abs + listening
      spot-check); warmed Mac A/B.
- [ ] **[device]** A14 run: warmed medians, on-device compute plan
      (zero-CPU-fallback for the ANE stage), Instruments Core ML trace,
      clean espresso log; record against the promotion gates in the
      [A14 guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md).

**Verification:** Strict parity met on Mac; device evidence stack collected
or deferred; flag-off path bit-identical to current production.

---

### Phase 6: Validation and Cleanup

**Goal:** Evidence consolidated, memory layer healthy, scaffolding removed.

**Tasks:**

- [ ] Consolidate results into
      [iphone-performance-notes.md](../Notes/iphone-performance-notes.md) and
      cross-link from the wiki if runtime beliefs changed
      (`README/Wiki/runtime-boundary.md`).
- [ ] Update this plan's checkboxes/status; record the final A14 RTF table.
- [ ] Remove any temporary probe scaffolding not worth keeping; keep probes
      that document negative results (ledger convention).
- [ ] `node scripts/memory-health.mjs --write-coverage` and `--strict` pass.
- [ ] Root `pytest` green.

**Verification:** Memory-health green; notes tell the full story without this
chat's context.

## Executable Memory

- Regression test: `swift build -c release && python3 -m pytest`
- Bench build: `cd ios-bench && xcodegen generate && xcodebuild -project KokoroIPhoneBench.xcodeproj -scheme KokoroIPhoneBench -destination generic/platform=iOS CODE_SIGNING_ALLOWED=NO build`
- Not testable in CI: all **[device]** tasks — proof is the measured tables
  and archived device logs in `README/Notes/iphone-*.md` plus raw JSON under
  `outputs/iphone_bench/`.

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] Failing `.all` stage named with verbatim device log (or explicitly
      deferred for device access, never guessed).
- [ ] Every promoted export/runtime change has a recorded strict-parity
      result vs the staged baseline.
- [ ] Mac Config F warmed medians not regressed by any promoted change.
- [ ] No new RangeDim/flexible-shape inputs anywhere.
- [ ] Phase 4 verdict recorded with evidence before any Phase 5 work.

### Definition of Done

- [ ] Phases 0-4 complete (Phase 5 conditional on GO; Phase 6 always).
- [ ] A14 RTF improvement quantified against the 0.44-0.49 baseline.
- [ ] All evidence in `README/Notes`, plan checkboxes current, CI green.

## Open Questions

### Resolved

- **Q:** Is the `.all` rejection A14-generation-specific?
- **A:** No — A17 Pro fails identically; it is iOS-side ANEF specialization
  behavior ([iphone-debug-notes.md](../Notes/iphone-debug-notes.md)).
- **Q:** Does Mac-like RTF require ANE residency for the generator?
- **A:** Yes — budget arithmetic in the
  [triage note](../Notes/kokoro-a14-iphone-guide-triage-2026-06-10.md); the
  A14 GPU ceiling is ~0.19-0.24.

### Unresolved

- **Q:** Which stage rejects ANEF under `.all`?
- **Options:** duration unrolled-LSTM (prime suspect — predicts first, 17k-134k
  ops), f0ntrain, generator, or several. Phase 2 answers.
- **Q:** Is the 16,384 limit enforced per-segment or whole-program on device?
- **Options:** per-segment (re-chunking viable) vs whole-program (Phase 4
  STOP likely). Phase 4 answers.
- **Q:** AdaIN statistics under chunking: per-chunk drift vs precomputed
  full-utterance statistics as extra inputs.
- **Options:** precomputed statistics is the parity-safe lean; decided in the
  Phase 5 design note.

## References

### Internal

- [Kokoro A14 iPhone generator execution guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md)
- [A14 guide ingest triage](../Notes/kokoro-a14-iphone-guide-triage-2026-06-10.md)
- [iPhone performance notes](../Notes/iphone-performance-notes.md)
- [iPhone debug notes](../Notes/iphone-debug-notes.md)
- [Compute-unit ablation ledger](../Notes/coreml-compute-unit-ablation.md)
- [Stage compute policy ablation](../Notes/stage-compute-policy-ablation.md)
- [ANE compiler failure triage guide](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md)
- [ConvTranspose/ISTFTNet scheduling guide](../Guides/apple-silicon/Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
- [iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)

## Modules

### Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `ios-bench/Sources/BenchApp.swift` | Modify | Stage timings, failure capture, `--matrix`, `--exact-duration` |
| `ios-bench/project.yml` | Modify | Bundle exact-duration packages |
| `swift/Sources/KokoroPipeline/KokoroSynthesisExecutor.swift` | Modify | Preallocate generator input buffers out of timed window |
| `export_synth/wrappers.py` | Modify | `x_source` pre-trim before `noise_res` |
| `scripts/dump_device_compute_plan.py` | Create | On-device compute plan dump (coremltools experimental) |
| `scripts/probe_generator_ane_admittance.py` | Create | Phase 4 stripped-body probe |
| `README/Notes/iphone-performance-notes.md` | Modify | Measured per-stage tables |
| `README/Notes/iphone-debug-notes.md` | Modify | Failure attribution, Phase 4 verdict |

### Risks and Mitigations

- **No device available during execution:** [device] tasks defer per the
  protocol above -> code and Mac evidence still land; plan stays honest.
- **Host xcodebuild stall:** known SWBBuildService probe hang -> reaper loop
  from [iphone-debug-notes.md](../Notes/iphone-debug-notes.md).
- **Parity break from chunked AdaIN:** prototype precomputed-statistics
  variant first; kill Phase 5 if neither variant passes the strict gate.
- **ANECompilerService jetsam during device probes:** run probes one model
  per process, charged and idle, per the
  [A14 guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md).
- **Scope creep into ledger-rejected ideas:** every new probe must cite the
  ablation ledger section proving it is not a re-run.

### Degradation and Rollback

- **Rollback:** all export changes are additive artifacts; runtime changes
  sit behind flags with the staged path as default. Revert = drop the flag
  or the phase commit.
- **Kill switch:** Phase 4 STOP verdict ends ANE work; the plan still ships
  Phase 1-3 instrumentation and GPU-path wins.

---

## Critical Reminder

> SIMPLER IS BETTER. The cheap probes come first; the expensive build is
> gated behind a measured GO. Do not start Phase 5 on hope.
