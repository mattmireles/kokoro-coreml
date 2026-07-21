# LFM2.5 A17 Pro ANE Parity Probe Plan

**Date:** 2026-07-20
**Status:** Complete — terminal `KILL` at Gate P2

> This is one final, deliberately small diagnostic after
> [plan 011](./011-lfm2-selective-surgical-prefill-plan.md) terminated at G0d.
> It does not reopen plan 010 or plan 011, and it cannot produce a performance
> claim. Its only question is whether the fixed-512 ANE numerical failure seen
> on M2 Ultra reproduces on an A17 Pro iPhone.

## Executive Summary

Run the exact three fixed-512 convolution-pair twins from plan 011 on the
physical iPhone 15 Pro Max (`iPhone16,2`, A17 Pro) under two Core ML policies:

1. `.cpuAndGPU`, the within-phone numerical control; and
2. `.cpuAndNeuralEngine`, the candidate whose A17 ANE behavior is in question.

Start with `C6-7`. On M2 Ultra, that pair changed the final token from `941` to
`509`, so it is the cheapest and strongest falsification probe. Stop if it
fails. Test `C0-1` and `C3-4` only if `C6-7` passes every frozen correctness and
dispatch check.

This plan records first-use and five warmed correctness comparisons, but no
latency, energy, or thermal-performance samples. A result is one of:

- **`KILL`** — at least one admitted A17 pair violates the frozen numerical
  contract; stop work on this partitioning approach;
- **`A17 PARITY PASS`** — all three pairs pass on a proven A17 ANE path; this
  justifies a new preregistered performance plan, not performance work inside
  this plan; or
- **`INCONCLUSIVE`** — signing, device readiness, or dispatch evidence does
  not prove that the candidate actually exercised the A17 ANE. Repair only the
  evidence path and rerun the same frozen probe.

## Problem Statement

Plan 011 established that all three fixed twins were structurally correct and
fully ANE-preferred on M2 Ultra, but numerically invalid when loaded with
`.cpuAndNeuralEngine`. The failures were too large to treat as fp16 noise:

| Pair | Boundary `max_abs` | Suffix-logit `max_abs` | GPU token | ANE token |
| --- | ---: | ---: | ---: | ---: |
| `C0-1` | 0.288086 | 7.826172 | 941 | 941 |
| `C3-4` | 0.154297 | 9.095703 | 941 | 941 |
| `C6-7` | 0.073975 | 3.377930 | 941 | 509 |

That result killed the planned selective split before timing. It did not tell
us whether the defect is specific to the M2-generation ANE/compiler path or is
portable to the A17 Pro path. One physical-phone parity probe answers that
narrow question at low cost.

## Relationship to Plans 010 and 011

The following conclusions remain terminal:

- Plan 010 killed the 13-segment performance experiment because 12 public
  boundaries consumed 37.43% of segmented prefill latency.
- Plan 011 killed the six-piece selective experiment because all three
  ANE-permitted fixed pairs failed correctness before timing.
- Neither result may be relabeled as positive if this A17 probe passes.
- This plan is a new preregistered device-generation diagnostic, not a repair,
  continuation phase, or post-hoc threshold change.

A positive result supports only this statement:

> Under the recorded iPhone and OS build, the three tested fixed-512 packages
> retained GPU-versus-ANE parity on the A17 Pro despite failing on M2 Ultra.

It does not establish a speedup, useful end-to-end partition, cross-device
generalization, or production path.

## Goals and Non-Goals

### Goals

- [x] Prove that each tested `.cpuAndNeuralEngine` package actually admits its
  heavy operations to the A17 Pro ANE on the same phone and OS build.
- [x] Compare all named pair outputs under `.cpuAndGPU` and
  `.cpuAndNeuralEngine` using real layer-entry activations from the frozen
  prompt.
- [x] Propagate both outputs through the same untimed GPU suffix and compare
  final logits and top token.
- [x] Preserve exact package hashes, app revision, signing identity, device
  identity, OS build, raw per-repetition results, and dispatch evidence.
- [x] Stop at the first failed pair and publish the negative result as useful
  evidence.

### Non-Goals

- No latency, throughput, energy, memory, sustained thermal, or battery study.
- No tail export, six-piece assembly, or end-to-end selective benchmark.
- No alternate partition, additional layer pair, or partition search.
- No iPad, A14, Mac rerun, simulator run, second phone, or multi-device claim.
- No bucket other than fixed 512.
- No alternate prompt, synthetic hidden state, tolerance sweep, or repetition
  sweep.
- No precision change, model surgery, reconversion, quantization, retraining,
  custom operator, Metal kernel, private API, or Core ML version sweep.
- No decode or KV-cache experiment.
- No timing after a parity pass. Performance requires a separate plan.

## Scope and Constraints

- Experiment orchestration, the iOS probe app, runtime code, generated results,
  and public report live in
  `/Users/mm/Documents/GitHub/lfm2-surgical-coreml`.
- This plan and the final interpretation note live in `kokoro-coreml`.
- Generated model resources, Xcode build products, signing material, and raw
  results remain ignored and uncommitted.
- The physical target is the phone named `Commas?`, identified by CoreDevice
  identifier `8A12AEE8-0136-50BE-8EB3-91650E467F15` and hardware UDID
  `00008130-000600660EA2001C`.
- The executor must revalidate both identifiers immediately before every
  install or run. A name match alone is insufficient.
- The phone is a daily-driver device. Creating this plan does not authorize an
  install or run. Execution begins only after an explicit `execute-plan`
  request and a live unlocked-device check.
- The existing iPad provisioning profile is not valid evidence for this phone.
  Phase 0 must prove that the selected development profile contains this exact
  phone UDID.
- Never print, commit, or copy Apple credentials, private keys, session tokens,
  provisioning payloads, or `.env` contents into result artifacts.

## Frozen Scientific Contract

### Checkpoint and Input

- Checkpoint:
  `LiquidAI/LFM2.5-350M@b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`.
- Bucket: fixed `512` only.
- Prompt:
  `Explain why fixed tensor shapes can help a compiler target an accelerator.`
- Precision: existing fp16 packages and tensors; no reconversion.
- Final GPU-oracle token: `941`.
- Pair-output and final-logit tolerance: `max_abs <= 1e-2`.
- Comparisons cover every named output, not only `hidden_out`.

### Exact Pair Artifacts

The fixed twins are immutable inputs. Their package-tree SHA-256 values must be
verified before resource staging, after app staging, and in every result row.

| Order | Pair | Fixed-twin SHA-256 | Named state inputs | Named outputs |
| ---: | --- | --- | --- | --- |
| 1 | `segment_04_conv_6_7` | `fd270492253c5f653ead28ff0368ae912278ae21fb201a9910d97c337be29b5b` | `conv_state_6_in`, `conv_state_7_in` | `hidden_out`, `conv_state_6_out`, `conv_state_7_out` |
| 2 | `segment_00_conv_0_1` | `dfbc10cad5d42c26b97b69d1ce74f01a28a07f170323529871aa4a4fb8576218` | `conv_state_0_in`, `conv_state_1_in` | `hidden_out`, `conv_state_0_out`, `conv_state_1_out` |
| 3 | `segment_02_conv_3_4` | `db264232357277efdc5d0efca961c50ba1971e53018da3efade7b29583e7133a` | `conv_state_3_in`, `conv_state_4_in` | `hidden_out`, `conv_state_3_out`, `conv_state_4_out` |

The source enumerated package must still match its fixed twin at bucket 512
under `.cpuAndGPU` within `1e-2` on the phone before an ANE comparison is
interpreted. Plan 011's exact Mac materialization result is necessary evidence,
but it is not substituted for this cheap same-phone check.

### Real Activation and Common Suffix

- Load embedding, head, and all 13 frozen source segments with
  `.cpuAndGPU`.
- Run the registered prompt through the all-GPU oracle to obtain real
  layer-entry activations and explicit convolution states.
- Require the oracle final top token to equal `941` before testing a twin.
- For a pair, replace only that source segment with the fixed twin. Keep the
  preceding path, following suffix, states, mask, RoPE inputs, head, and all
  other package hashes unchanged.
- Run the fixed twin once with `.cpuAndGPU` and once with
  `.cpuAndNeuralEngine` from independently copied identical inputs.
- Snapshot or reduce every output before the next prediction. Do not retain
  mutable `MLMultiArray` views across model calls.
- Feed each candidate's outputs into an independent copy of the same
  `.cpuAndGPU` suffix state. The suffix is outside any measured region because
  this plan records no time.

### Correctness Repetitions

For each admitted pair:

1. Record one first-use comparison after model load, in frozen order
   GPU then ANE.
2. Record exactly five warmed comparisons with candidate order alternating:
   `GPU/ANE`, `ANE/GPU`, `GPU/ANE`, `ANE/GPU`, `GPU/ANE`.
3. Recreate candidate inputs and suffix state for every comparison.
4. Record per-output maximum absolute error, aggregate pair-output maximum,
   final-logit maximum, GPU token, ANE token, and oracle token for all six
   comparisons.
5. Require all six comparisons to pass. There is no median, majority vote,
   tolerance adjustment, or discarded outlier.

The first-use row is correctness evidence, not a cold-start benchmark. The
five warmed rows are repeatability evidence, not latency samples.

### Same-Phone Dispatch Gate

Before running a pair, capture compute plans from the exact phone for the exact
fixed-twin hash under both policies.

An ANE candidate is admitted only when:

- provenance matches the frozen CoreDevice identifier, hardware UDID,
  `iPhone16,2`, and the runtime OS version/build;
- requested compute units are `CPU_AND_NE`;
- at least 80% of costed operations prefer `neuralEngine`; and
- every convolution and matrix-multiplication operation reported by the plan
  prefers `neuralEngine`.

The paired control must record `CPU_AND_GPU` and contain no neural-engine
preferred operation. Package hashes in the two compute-plan files and the app
result must be identical.

If dispatch cannot satisfy these rules, the result is `INCONCLUSIVE`, not a
numerical pass or failure. Do not loosen the placement gate to obtain an
answer.

## Verdict State Machine

```text
device/signing/hash/dispatch not proven
    -> INCONCLUSIVE; repair evidence path; rerun same pair only

C6-7 admitted
    -> any of 6 comparisons fails: KILL; stop
    -> all 6 pass: continue to C0-1

C0-1 admitted
    -> any of 6 comparisons fails: KILL; stop
    -> all 6 pass: continue to C3-4

C3-4 admitted
    -> any of 6 comparisons fails: KILL; stop
    -> all 6 pass: A17 PARITY PASS; stop without timing
```

### Numerical Failure

One comparison fails if any of the following is true:

- the source package and fixed twin under GPU differ by more than `1e-2` on
  any named pair output;
- GPU-versus-ANE `max_abs` exceeds `1e-2` on any named pair output;
- suffix-logit `max_abs` exceeds `1e-2`;
- either substituted path's top token differs from the GPU oracle token; or
- the GPU oracle token differs from the frozen token `941`.

NaN, infinity, missing output, shape mismatch, state mismatch, or a result row
count other than six is an automatic numerical failure once artifact and
dispatch provenance are valid.

## Already Shipped (Do Not Re-Solve)

The public experiment repo already contains:

- the real checkpoint equations and 13 frozen source segment definitions;
- embedding, source-segment, head, monolith, and fixed-pair artifact generation;
- the registered prompt and bucket-512 Stage 1 input;
- package-tree hashing and exact artifact provenance;
- direct fp16 Swift/Core ML handoffs and real pair-entry generation in
  `SelectiveRuntime.swift`;
- common untimed GPU-suffix validation for each pair;
- `scripts/dump_device_compute_plan.py`, including exact-device provenance,
  mobile signing, remote compute-plan loading, and package-hash binding;
- terminal Stage 1 and selective reports; and
- Python tests and Release Swift build gates.

Reuse these mechanisms. Do not create a second checkpoint loader, tokenizer,
segment registry, package-hash format, device-provenance schema, or numerical
tolerance.

The main repo also has a proven iOS XcodeGen convention in
`ios-bench/project.yml`: a headless SwiftUI app, local Swift package dependency,
generated Info.plist, development team `6ETYBAJKY8`, iPhone-only target, and a
durable JSON result under the app Documents directory. Follow that convention
instead of checking in a hand-edited project file.

## Fresh Baseline at Plan Creation

- `kokoro-coreml` is clean at `7b704428` before this plan file is added.
- `lfm2-surgical-coreml` is clean at `b29b2ff`.
- The iPhone is physically connected over a wired CoreDevice tunnel, paired,
  booted, and Developer Mode is enabled.
- Live identity at plan creation:
  - name: `Commas?`;
  - product: iPhone 15 Pro Max, `iPhone16,2`, A17 Pro;
  - storage: 512 GB;
  - CoreDevice identifier:
    `8A12AEE8-0136-50BE-8EB3-91650E467F15`;
  - hardware UDID: `00008130-000600660EA2001C`;
  - iOS: 27.0 beta, build `24A5380h`;
  - connection: paired, wired, tunnel connected;
  - DDI services: available.
- This is only a creation-time observation. Phase 0 records fresh live state
  and does not assume the beta build or connection remains unchanged.
- There is no repo-owned LFM2 iOS parity app or parity-specific device command.
- `dump_device_compute_plan.py` proves placement but does not execute arbitrary
  model predictions or retrieve numerical outputs.
- `ios-bench` uses XcodeGen, but `xcodegen` is not currently available on the
  shell PATH. Phase 1 must resolve the executable explicitly or install the
  pinned tool before generating the app project.

## Proposed Solution

Add one minimal headless SwiftUI app to the public repo. It imports the existing
`LFM2SurgicalRuntime` Swift package, loads only ignored build-staged resources,
runs one explicitly named pair, and atomically writes a versioned pair-specific
JSON result under `Documents/` before exiting its work loop.

A narrow host-side command performs the mechanical workflow:

1. verify exact artifact hashes;
2. stage only the resources needed for the requested pair plus the common GPU
   oracle path;
3. generate and build the Xcode project in Release;
4. install and foreground-launch it on the exact CoreDevice identifier;
5. wait for the app's durable completion marker;
6. copy the JSON result from the app data container;
7. validate schema, pair name, row count, package hashes, device identity, and
   frozen thresholds; and
8. return the state-machine verdict only after evidence is safely on the Mac.

Do not turn the app into a general benchmark framework. It accepts one pair
name and an output filename through a build-time resource manifest. There are
no timing APIs, dynamic model downloads, server, UI controls, or arbitrary
compute-unit matrices.

## Performance and Latency Budget

The permitted performance-sample count is exactly zero. The app and result
schema do not record prediction duration, model-load duration, throughput,
energy, or candidate-order elapsed time. Host orchestration may use wall-clock
timeouts only to detect a hung install, launch, or result transfer; those
timeouts are operational safeguards and never enter the scientific result.

This constraint prevents a correctness probe from becoming an unregistered
performance fishing expedition. Even an obvious visual difference in run time
is not a result.

## Implementation Phases

### Phase 0: Device and Signing Readiness

**Goal:** Prove that the exact phone can run and expose evidence without
touching model code.

**Required skills:** `botnet` only if CoreDevice connectivity fails; `debug`
for signing or install failures; `phase-audit` before the phase commit.

**Tasks:**

- [x] Re-query CoreDevice details by the frozen identifier and verify physical
  `iPhone16,2`, exact UDID, booted state, Developer Mode, DDI, pairing, and
  connected tunnel.
- [x] Verify the phone is unlocked immediately before build/install work.
- [x] Verify an Apple Development signing identity for team `6ETYBAJKY8`
  without logging private material.
- [x] Verify the phone is enabled in the developer portal and select or create
  one explicit development provisioning profile containing the exact hardware
  UDID and the parity-app bundle identifier.
- [x] Decode the installed profile locally and record only its UUID, name,
  team, bundle identifier, expiration, and whether the exact UDID is present.
- [x] Do not reuse the iPad profile unless its decoded device list independently
  contains this iPhone UDID.
- [x] Write ignored `device-readiness.json` with nonsensitive live provenance
  and a Boolean readiness decision.

**Gate P0:** Continue only if exact device identity, unlock state, signing
identity, nonexpired explicit profile, app identifier, and device membership
all pass. Otherwise stop `INCONCLUSIVE` without installing the probe app.

**Verification:** A reviewer can bind the selected profile to the exact phone
and bundle identifier without access to credentials or a raw profile payload.

**Executed 2026-07-21:** Gate P0 passed. CoreDevice reported the exact physical
`iPhone16,2` and hardware UDID, booted, unlocked, paired over a connected wired
tunnel, with Developer Mode and DDI services enabled on iOS 27.0 beta build
`24A5380h`. The portal already contained the exact enabled phone. Execution
created only the explicit bundle identifier and `LFM2 A17 Parity Development`
profile, UUID `173c72a0-1f63-4748-95bd-bd195ca1580f`, expiring
2027-07-21, bound to team `6ETYBAJKY8`, the matching local development
certificate, and the exact phone UDID. Nonsensitive ignored evidence is at
`outputs/lfm2_surgical/a17_parity/device-readiness.json`; its canonical payload
SHA-256 is
`cb10cca6a88d8e16a0bb63bd9f9809043cd9645718f27c15a6a7b36a94a0e578`.

---

### Phase 1: Minimal iOS Parity Harness

**Goal:** Build a deterministic, untimed phone runner without changing the
frozen model artifacts or Stage 1 behavior.

**Required skills:** `coreml`, `debug`, `documentation`, `phase-audit`.

**Tasks:**

- [x] Add a versioned `A17ParityResult` schema with device, app, checkpoint,
  prompt, artifact, dispatch-reference, oracle, per-output, suffix-logit, token,
  repetition, and verdict fields.
- [x] Extract only the minimal package loading, real pair-entry, output-copy,
  max-absolute-error, and common-suffix helpers from `SelectiveRuntime.swift`
  for reuse. Preserve the plan-011 executable's output byte-for-byte under its
  regression replay.
- [x] Add a parity-only runtime that accepts exactly one of the three frozen
  pair names and enforces the first-use plus five-warmed sequence internally.
- [x] Add the headless SwiftUI iPhone app and XcodeGen spec, using a local
  package dependency on `LFM2SurgicalRuntime` and iOS 18 or later.
- [x] Add a resource-staging script that copies rather than mutates the source
  packages, verifies all package-tree hashes, and refuses unregistered pair or
  bucket input.
- [x] Add a host orchestrator that targets the frozen CoreDevice identifier,
  builds Release with the explicit profile, installs, launches, retrieves the
  durable JSON, and validates it before reporting a verdict.
- [x] Keep generated `.xcodeproj`, staged `.mlmodelc`/`.mlpackage` resources,
  DerivedData, `.app`, and result payloads ignored.
- [x] Add tests for manifest validation, frozen pair order, hash mismatch,
  result row count, threshold boundary, NaN/infinity, output-name completeness,
  token mismatch, dispatch mismatch, and state-machine stop behavior.
- [x] Replay the old Stage 1 and plan-011 Phase 0 commands to prove the shared
  helper refactor did not change their terminal classifications.

**Gate P1:** Continue only if tests pass, Release app builds and signs for the
exact phone, generated resources match frozen hashes, and both historical
replays retain their prior results. Do not launch the app in this phase.

**Verification:** The app has no timing code, the orchestrator can request only
one frozen pair, and all generated or credential-bearing material is ignored.

**Executed 2026-07-21:** Gate P1 passed without installing or launching the
app. The public repo now contains a versioned untimed result contract, the
single-pair state machine, a foreground SwiftUI runner, a checked-in XcodeGen
spec, deterministic resource staging, and a fail-closed exact-device host
orchestrator. The build-only C6-7 manifest staged and rehashed all 16 frozen
packages. The Release iPhone app built with Xcode 26.6, passed strict codesign
verification, embedded profile UUID
`173c72a0-1f63-4748-95bd-bd195ca1580f`, and retained the exact application
identifier `6ETYBAJKY8.com.mattmireles.LFM2A17Parity`. Its ignored build
provenance is
`outputs/lfm2_surgical/a17_parity/app-build-provenance.json` (file SHA-256
`2db55da2d79fda4ebf3952c5ca1b4b7855a14d351c7388a1a4386468aa72b204`).

All 27 Python tests passed, the Release Swift package built, `git diff
--check` passed, and static scans found no A17 clock or prohibited performance
field. Generated resources, the generated project, DerivedData, signed app,
profile copy, and result payloads are ignored. The Stage 1 replay retained
bit-exact fp16 logits (`outputMaxAbs = 0`) and terminal `KILL`; ignored evidence
is `outputs/lfm2_surgical/a17_parity/regression/stage1_replay.json` (file
SHA-256 `50d85a478a50d46203a3adc8cc1c561ed12a1831ac25289632677dbce1995830`).
The plan-011 Phase 0 replay retained terminal G0d `KILL`, the C6-7 token change
`941 -> 509`, and zero pair timing rows; ignored evidence is
`outputs/lfm2_surgical/a17_parity/regression/plan011_phase0_replay.json` (file
SHA-256 `32f515043926a133bc32618d0654b94626c38d7137333b9ff7325c0f4070275c`).
The latter replay ran only after terminating a leaked set of 1,133 probe
children from `Usage Helper`, waiting for the frozen quiet-host gate to pass,
briefly pausing that exact helper, and resuming it immediately afterward.
The audited public implementation is commit `a30820b`. The final independent
phase audit found no remaining issues and graded Architecture A, Correctness
risk A, and Complexity debt A. A post-commit replay of all 27 Python tests
passed and the public tracked tree was clean.

---

### Phase 2: C6-7 Falsification Probe

**Goal:** Ask the highest-information question and stop early on failure.

**Required skills:** `coreml`, `debug`, `phase-audit`.

**Tasks:**

- [x] Capture same-phone `CPU_AND_GPU` and `CPU_AND_NE` compute plans for the
  exact `C6-7` fixed-twin hash.
- [x] Validate the frozen dispatch gate and persist the two hash-bound JSON
  files before launching the parity app.
- [x] Run only `segment_04_conv_6_7` with the registered prompt and exact six
  correctness comparisons.
- [x] Retrieve and validate the result before uninstalling, replacing, or
  rebuilding the app.
- [x] Write the canonical C6-7 result envelope and update the public report
  with either
  `INCONCLUSIVE`, `KILL`, or authorization for Phase 3.

**Gate P2:**

- Dispatch/setup failure: stop `INCONCLUSIVE`.
- Any numerical failure: stop terminal `KILL`; do not capture plans or run apps
  for the other pairs.
- All six comparisons pass: continue to Phase 3.

**Verification:** The result binds exact device, OS/build, app revision,
package hash, dispatch files, all named outputs, all six comparisons, suffix
logits, and tokens.

**Executed 2026-07-21:** Gate P2 returned terminal `KILL` on the exact physical
iPhone 15 Pro Max (`iPhone16,2`, A17 Pro), iOS 27.0 beta build `24A5380h`.
The C6-7 GPU control preferred 54/56 costed operations on GPU, two on CPU, and
zero on ANE; its dispatch file SHA-256 is
`6855d7ec08195b63a05a5b566f666b187ef254d596f810f0961d9e1371f9a05d`.
The ANE-permitted candidate preferred all 56/56 operations on ANE, including
12/12 convolutions; its dispatch file SHA-256 is
`1a3bb1f7143501fc8971bd0abb9f4e9142414e5bbc4e32032b8f9d32bd571470`.

The GPU oracle passed with token `941`, and the enumerated source matched its
fixed twin exactly in all six rows. Every GPU-versus-ANE row then reproduced
the same error: pair-output `max_abs = 0.070068359375`, suffix-logit
`max_abs = 3.6102294921875`, and token `941 -> 509`. First-use, warmed, and
reversed-call-order rows were identical. The immutable result is
`outputs/lfm2_surgical/a17_parity/pairs/segment_04_conv_6_7.result.json`;
file SHA-256
`6878e45d085ea0683c5f8ee864e13eb5a25f228f7a3245155c28b74ea44f5568`,
canonical payload SHA-256
`53fcae5d2502d6e9aae93fe1c85d0523ae378ae126a6ea877611308e574f6c20`.
It records `performance_samples_recorded = 0`.

---

### Phase 3: Remaining Registered Pairs

**Goal:** Determine whether the A17 result generalizes across the complete
three-pair set already selected by plan 011.

**Required skills:** `coreml`, `debug`, `phase-audit`.

**Tasks:**

- [ ] Repeat the exact dispatch and six-comparison procedure for `C0-1`.
  **Cancelled:** C6-7 returned terminal `KILL` at Gate P2.
- [x] Stop immediately on `INCONCLUSIVE` or `KILL`.
- [ ] Only after `C0-1` passes, repeat the procedure for `C3-4`.
  **Cancelled:** C0-1 was not authorized.
- [x] Preserve separate raw result and dispatch files for each pair; do not
  overwrite a passing earlier result during a later app build.
- [ ] Produce a machine-readable aggregate that references rather than copies
  the three immutable pair-result hashes.
  **Cancelled:** the aggregate is defined only for three passing pairs.

**Gate P3:**

- Any admitted pair fails: terminal `KILL`.
- Any pair lacks valid dispatch or device provenance: `INCONCLUSIVE`.
- All three pairs pass all six comparisons: terminal `A17 PARITY PASS`.

No outcome authorizes timing inside this plan.

**Verification:** The aggregate verdict is mechanically derivable from three
pair results and six compute-plan files, with no manually entered metric.

**Execution outcome:** Phase 3 was correctly skipped. No C0-1 or C3-4 phone
dispatch or result file exists, and the orchestrator rejects either command
because the predecessor result is not `PAIR PASS`.

---

### Phase 4: Closeout and Publication

**Goal:** Preserve the result and make the next decision unambiguous.

**Required skills:** `markdown`, `write-notes`, `david-ogilvy`,
`phase-audit`.

**Tasks:**

- [x] Complete `docs/a17-parity-report.md` in the public repo with the frozen
  hypothesis, device/OS, hashes, commands, raw-result paths, dispatch tables,
  all correctness rows, terminal gate, and limitations.
- [x] Add `README/Notes/lfm2-a17-parity-result.md` in `kokoro-coreml` with the
  local interpretation and links to plans 010, 011, and this plan.
- [x] Update this plan's status, checkboxes, actual revisions, executed
  commands, evidence paths, and terminal verdict.
- [x] Link the report from the public README without weakening the terminal
  Stage 1 or selective reports.
- [x] Run public Python tests, Release Swift build, historical replays,
  Markdown checks, memory health, and secret scans over both diffs.
- [x] Keep every result phrased as device- and OS-specific.

**Interpretation:**

- `KILL`: Give up on this public-boundary selective-preload direction unless a
  materially new mechanism is proposed and preregistered. Do not test more
  devices merely to search for a pass.
- `A17 PARITY PASS`: The M2 numerical failure does not reproduce on the tested
  A17 configuration. Write a separate performance plan before measuring even
  one latency sample.
- `INCONCLUSIVE`: The scientific question remains unanswered. Only repair the
  failed readiness/dispatch/evidence mechanism; do not change model variables.

**Verification:** Public report, raw JSON, kokoro note, and this plan all state
the same terminal verdict and explicitly say that no performance measurement
occurred.

**Executed 2026-07-21:** All 27 public Python tests passed, the public Swift
package built in Release mode, `py_compile` and `git diff --check` passed, and
both historical classifications reproduced: Stage 1 remained terminal `KILL`
with bit-exact fp16 logits, while plan 011 Phase 0 remained terminal G0d
`KILL` with zero pair timing rows. The final ignored replay files are
`outputs/lfm2_surgical/a17_parity/regression/final_stage1_replay.json`
(SHA-256
`4821b226bd832f903ea7882797c024fe0dec8928fe57e1d2eb27adcca0abaece`)
and
`outputs/lfm2_surgical/a17_parity/regression/final_plan011_phase0_replay.json`
(SHA-256
`de86cfce83959613acf6fdc4b59d025bac6ee803dd37451d39986fb2e50b119c`).
Markdown lint passed for both closeout documents and their routing pages.
Memory health passed strict Grade A at 87/87 canonical sources, with the new
note added to the generated coverage index. Scoped TruffleHog scans over both
plan diffs found zero verified or unknown secrets. Mechanical envelope
validation reproduced `KILL`, returned no authorized next pair, and confirmed
that no later-pair result or dispatch artifact exists.

The public closeout is commit
`0fcfd70df7636c66cffd9775c3842b05a9d23fef`. The independent terminal phase
audit found no remaining issues and graded Architecture A, Correctness risk A,
and Complexity debt A.

## Intended Files

Exact names may change only to match adjacent code conventions; responsibilities
may not expand.

### `lfm2-surgical-coreml`

- Add `Sources/LFM2SurgicalRuntime/A17ParityRuntime.swift`.
- Minimally modify
  `Sources/LFM2SurgicalRuntime/SelectiveRuntime.swift` and, only if required,
  `BenchmarkRuntime.swift` to expose shared internal helpers without changing
  existing public behavior.
- Add `ios-a17-parity/project.yml`.
- Add `ios-a17-parity/Sources/A17ParityApp.swift`.
- Add `ios-a17-parity/prepare_resources.sh`.
- Add `scripts/run_a17_parity.py`.
- Extend `tests/test_lfm2_surgical_tools.py` and add focused fixture files only
  when needed.
- Modify `.gitignore` for staged resources, generated Xcode files, DerivedData,
  app products, and raw A17 outputs.
- Add `docs/a17-parity-report.md` during closeout.
- Modify `README.md` during closeout.
- Generated only: `outputs/lfm2_surgical/a17_parity/**`.

### `kokoro-coreml`

- This plan.
- Add `README/Notes/lfm2-a17-parity-result.md` only during closeout.
- Update wiki routing only when memory-health requires it.

## Result Artifact Layout

```text
outputs/lfm2_surgical/a17_parity/
  device-readiness.json
  app-build-provenance.json
  dispatch/
    segment_04_conv_6_7.cpu_and_gpu.json
    segment_04_conv_6_7.cpu_and_ne.json
    segment_00_conv_0_1.cpu_and_gpu.json       # only after C6-7 passes
    segment_00_conv_0_1.cpu_and_ne.json        # only after C6-7 passes
    segment_02_conv_3_4.cpu_and_gpu.json       # only after C0-1 passes
    segment_02_conv_3_4.cpu_and_ne.json         # only after C0-1 passes
  pairs/
    segment_04_conv_6_7.result.json
    segment_00_conv_0_1.result.json             # only after C6-7 passes
    segment_02_conv_3_4.result.json              # only after C0-1 passes
  aggregate.json                                 # only after all three run
```

Every JSON file receives a schema version and SHA-256. The aggregate stores the
three pair-result hashes and six dispatch-file hashes so later edits cannot
silently change the evidence set.

## Executable Memory

These are the intended stable commands. The executor must update this section
if implementation changes a CLI spelling, but not the experiment variables.

```bash
cd /Users/mm/Documents/GitHub/lfm2-surgical-coreml

# Phase 0: nonsensitive live readiness evidence.
.venv/bin/python scripts/run_a17_parity.py readiness \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --device-udid 00008130-000600660EA2001C \
  --development-team 6ETYBAJKY8 \
  --bundle-identifier com.mattmireles.LFM2A17Parity \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out outputs/lfm2_surgical/a17_parity/device-readiness.json

# Phase 1: mechanical gates before touching the phone.
.venv/bin/python -m pytest -q
swift build -c release

# Generate from the checked-in spec. Resolve xcodegen explicitly if it is not
# on PATH; do not commit the generated project.
xcodegen generate --spec ios-a17-parity/project.yml

# Phase 2: C6-7 dispatch, then parity. Both compute plans must pass before run.
.venv/bin/python scripts/run_a17_parity.py dispatch \
  --pair segment_04_conv_6_7 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out-dir outputs/lfm2_surgical/a17_parity/dispatch

.venv/bin/python scripts/run_a17_parity.py run \
  --pair segment_04_conv_6_7 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out outputs/lfm2_surgical/a17_parity/pairs/segment_04_conv_6_7.result.json

# Phase 3: issued in this exact order and only after the prior pair passes.
.venv/bin/python scripts/run_a17_parity.py dispatch --pair segment_00_conv_0_1 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out-dir outputs/lfm2_surgical/a17_parity/dispatch
.venv/bin/python scripts/run_a17_parity.py run --pair segment_00_conv_0_1 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out outputs/lfm2_surgical/a17_parity/pairs/segment_00_conv_0_1.result.json

.venv/bin/python scripts/run_a17_parity.py dispatch --pair segment_02_conv_3_4 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out-dir outputs/lfm2_surgical/a17_parity/dispatch
.venv/bin/python scripts/run_a17_parity.py run --pair segment_02_conv_3_4 \
  --device-id 8A12AEE8-0136-50BE-8EB3-91650E467F15 \
  --provisioning-profile-uuid "$LFM2_A17_PROFILE_UUID" \
  --out outputs/lfm2_surgical/a17_parity/pairs/segment_02_conv_3_4.result.json

.venv/bin/python scripts/run_a17_parity.py aggregate \
  --root outputs/lfm2_surgical/a17_parity \
  --out outputs/lfm2_surgical/a17_parity/aggregate.json
```

`LFM2_A17_PROFILE_UUID` is an environment-local identifier, not a credential.
The command must fail closed if it is absent or the decoded installed profile
does not contain the exact phone UDID and bundle identifier.

## Test Strategy

### Static and Unit Tests

- Registered pair set and order are exactly C6-7, C0-1, C3-4.
- Only bucket 512 and the registered prompt are accepted.
- Package-tree hashes match the frozen twins before any device action.
- Result validator requires every named output and exactly six comparisons.
- `1e-2` passes at equality and fails above equality.
- NaN, infinity, shape drift, state-name drift, token drift, device drift,
  package drift, and dispatch drift fail closed.
- The state machine cannot issue a later pair after an earlier `KILL` or
  unresolved `INCONCLUSIVE`.
- No result schema contains timing, duration, throughput, energy, or benchmark
  summary fields.

### Regression Tests

- Python test suite passes.
- Release Swift library and existing executables build.
- Stage 1 replay retains exact fp16 segmented-versus-monolith parity and its
  terminal boundary-cost result.
- Plan-011 Phase 0 replay retains terminal G0d and does not begin pair timing.
- Package hashes for all existing artifacts remain unchanged.

### Physical-Device Tests

- Exact device identifier and hardware UDID are checked immediately before
  every install, launch, compute-plan capture, and result retrieval.
- Same-phone compute plans precede prediction for each pair.
- App result records actual OS version/build at execution, not a configured
  expectation.
- First-use and all five warmed comparisons are retained.
- JSON is retrieved and hash-verified before any next build or pair.
- A phone disconnect after app execution but before result retrieval is
  `INCONCLUSIVE` until the same result file is safely recovered; it is not
  rerun blindly.

## Success Criteria

### Hard Requirements

- [x] Exact checkpoint, prompt, bucket, precision, and fixed-twin hashes.
- [x] Exact physical `iPhone16,2` identity and live OS/build provenance.
- [x] Same-phone compute plans for both policies for every executed pair.
- [x] Heavy-operation ANE admission satisfies the frozen dispatch gate.
- [x] Real layer-entry activations and explicit states from the GPU oracle.
- [ ] Every named output and final logits stay within `1e-2` for all six
  comparisons. **Failed:** C6-7 exceeded the threshold in every row.
- [ ] GPU oracle and both substituted paths retain token `941`.
  **Failed:** the ANE-permitted path returned token `509` in every row.
- [x] Stop at the first failed or uninterpretable pair.
- [x] Record no latency or performance claim.
- [x] Publish positive, negative, and inconclusive outcomes with equal rigor.

### Definition of Done

The plan is complete when exactly one terminal state is durably reported:

1. `KILL`, with the first valid failing pair and all required evidence;
2. `A17 PARITY PASS`, with all three valid passing pair results; or
3. `INCONCLUSIVE`, with the exact readiness, signing, retrieval, or dispatch
   condition that prevented interpretation and no model-variable change.

In all three cases, this plan, the public report, raw JSON, and the kokoro Notes
pointer agree. Cancelled pairs and all uncollected performance work are marked
explicitly rather than left ambiguous.

## Rollback and Kill-Switch Strategy

- The app, runtime, and orchestrator are additive. Existing executables and
  reports remain the regression oracle.
- Generated model resources and Xcode products can be deleted and regenerated;
  source packages are copied, never moved or overwritten.
- The orchestrator refuses later pair commands unless it verifies the prior
  signed result hash and passing verdict.
- A `--force`, `--ignore-gate`, alternate-tolerance, alternate-pair, or timing
  escape hatch is prohibited. Debug-only manual runs cannot enter the report.
- If shared-helper extraction changes an existing replay, revert the
  extraction and duplicate only the minimal parity support.
- App uninstall or resource cleanup happens only after the result is recovered
  and hashed. Do not uninstall unrelated apps or remove broad device data.

## Risks and Mitigations

| Risk | Mitigation |
| --- | --- |
| A policy label is mistaken for actual ANE execution. | Require same-phone per-op compute plans and a frozen heavy-op admission rule. |
| The phone agrees only because both candidates accidentally use the same backend. | Require `CPU_AND_GPU` to show no ANE preference and `CPU_AND_NE` to admit heavy ops to ANE. |
| A fixed twin differs from its source on the phone. | Compare source versus twin under GPU before interpreting GPU-versus-ANE parity. |
| Reused mutable output buffers hide or fabricate a difference. | Snapshot every named output before the next prediction and recreate inputs/state per comparison. |
| First-use behavior differs from warmed behavior. | Retain one first-use and five warmed comparisons; require every row to pass. |
| OS beta behavior is mistaken for A17-wide behavior. | Bind the result to exact OS/build and state only a tested-configuration conclusion. |
| Signing repair silently uses the wrong device or profile. | Decode the profile locally and verify exact UDID, bundle ID, team, and expiration before build. |
| Daily-driver phone is disrupted. | Explicit execution authorization, one headless app, foreground run, no soak, no timing, and stop at first failure. |
| A pass creates pressure to collect “just one” latency number. | Runtime and schema contain no timing fields; performance is a separate preregistered plan. |
| A negative result triggers device shopping. | Terminal `KILL` explicitly forbids testing more devices without a materially new mechanism. |

## Open Questions

### Resolved by This Plan

- **Should we give up immediately?** Give the approach one final cheap A17
  correctness probe, then stop on the first failure.
- **Which pair runs first?** C6-7, because it changed the M2 final token.
- **How many phone runs?** One first-use plus five warmed comparisons per pair.
- **What is the tolerance?** The unchanged `1e-2` maximum-absolute threshold.
- **Does a pass authorize timing?** No. It authorizes only a new plan.
- **Does a setup/dispatch failure count as model failure?** No; it is
  `INCONCLUSIVE` and only the evidence path may be repaired.
- **Should the iPad profile be reused?** Only if decoded evidence shows it
  already contains the exact iPhone UDID; otherwise create/select a phone-valid
  explicit profile.

### Resolved During Execution

- The installed profile was
  `173c72a0-1f63-4748-95bd-bd195ca1580f`, expiring 2027-07-21.
- The phone ran iOS 27.0 beta build `24A5380h`.
- C6-7 passed the same-phone dispatch gate: the control used no ANE-preferred
  operations, while the ANE candidate placed 56/56 costed operations on ANE.
- C6-7 failed all six correctness rows and changed token `941 -> 509`; C0-1
  and C3-4 were cancelled before phone dispatch.
- The final verdict is terminal `KILL`. No performance sample was collected.

None of these execution-derived values permits changing the experiment design.

## References

- [Plan 010: LFM2 Surgical Prefill](./010-lfm2-surgical-prefill-plan.md)
- [Plan 011: LFM2 Selective Six-Piece Prefill](./011-lfm2-selective-surgical-prefill-plan.md)
- [Selective Split Result Note](../Notes/lfm2-selective-split-result.md)
- [iPhone Core ML Device Lab Runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)
- [LFM2 Surgical Prefill Core ML Guide](../Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md)
- [Core ML Enumerated Shape and Compute-Plan Specialization Guide](../Guides/apple-silicon/CoreML-enumerated-shape-compute-plan-specialization-guide.md)
- [Apple Silicon Warmed Inference Benchmark Hygiene Guide](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/docs/selective-split-report.md`
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/Sources/LFM2SurgicalRuntime/SelectiveRuntime.swift`
- `/Users/mm/Documents/GitHub/lfm2-surgical-coreml/scripts/dump_device_compute_plan.py`
- `/Users/mm/Documents/GitHub/kokoro-coreml/ios-bench/project.yml`
