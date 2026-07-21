# LFM2.5 Surgical Prefill Experiment Plan

**Date:** 2026-07-20
**Status:** In Progress (Phases 0-2 complete; Phase 3 next)

> Plan-of-record for the spec
> `README/Notes/lfm2-surgical-experiment-spec-v1.1.md` (checked in alongside
> this plan; source: Matt's v1.1 draft, 2026-07-20). The spec is the scientific contract;
> this plan is the execution order. Where they conflict, the spec's hypotheses,
> gates, and pre-registered criteria win; this plan's sequencing wins.

## Executive Summary

Decompose LFM2.5-350M (8 double-gated LIV conv blocks + 6 GQA blocks at the
documented 230M layout; 350M interleaving must be read from the checkpoint)
into per-block-class Core ML segments and measure whether surgical compute-unit
placement (conv→ANE, GQA→GPU) beats every homogeneous placement on prefill
latency and energy. Decode is measured only as a bandwidth-wall control
(expected null). This is the third case study for the Surgical Inference paper
(`Scratchpad/surgical-inference.md`), and the first on a model we did not port
ourselves.

Two standing decisions, made 2026-07-20:

1. **Repo lifecycle:** the plan, Stage 0 code, and Stage 0 report live in
   `kokoro-coreml` (precedent: the MoE prefetch experiment,
   [009](./009-moe-ssd-dram-prefetch-plan.md), died at Stage 1 without ever
   needing a repo). A new **public-from-first-commit** GitHub repo
   `mattmireles/lfm2-surgical-coreml` is created only after Stage 0 passes.
2. **Device sequencing:** the iPhone 15 Pro Max is Matt's daily-driver phone.
   All Mac Studio, M1 Mini, and iPad Pro work runs first; the phone phase is
   **last**, designed to run unattended overnight.

## Problem Statement

- **Symptom:** The Surgical Inference paper has two case studies (Kokoro-82M,
  MRT2), both our own ports. A reviewer will discount the thesis until it
  survives contact with a third-party model we did not shape.
- **Root Cause:** No prior experiment tests per-block-class placement on a
  hybrid conv/attention LLM where the architecture itself predicts which blocks
  admit to the ANE.
- **Impact:** Without this, the paper's central claim ("decompose and place
  per-stage") generalizes from n=2 self-selected examples. LFM2.5 makes a
  falsifiable prediction (conv→ANE wins, GQA and decode don't); either outcome
  is publishable.

## Mode Definitions

These are placement configurations, not runtime modes. They are the independent
variable of the whole experiment and are **frozen before any Stage 2 run**.

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| `C1` | Conv and GQA segments both `.cpuOnly`. | Floor; llama.cpp-comparable reference point. |
| `C2` | Both `.cpuAndGPU`. | Homogeneous GPU baseline; also the G1a decomposition-tax control. |
| `C3` | Both `.cpuAndNeuralEngine`. | Homogeneous ANE baseline; approximates the CoreML-LLM monolith. |
| `C4` | Both `.all`. | "Let Core ML decide" — the do-nothing baseline any win must beat. |
| **`C5`** | **Conv → ANE, GQA → GPU.** | **The thesis.** Per-block-class placement matching each class's shape/state behavior. |
| `C6` | Conv → GPU, GQA → ANE. | Inversion control. Must lose to C5, or the mechanism is not what we claim. |

External baselines run alongside these, same prompts and device: llama.cpp
Q4_0 GGUF (different precision — reported as context, never head-to-head) and
the CoreML-LLM `lfm2.5-350m` monolith.

## Goals and Non-Goals

### Goals

- [ ] **H1 (admission):** determine whether LFM2.5's double-gated short-conv
      blocks compile to and are scheduled on the ANE under enumerated shapes at
      fp16, with per-op dispatch evidence per device.
- [ ] **H2 (prefill win):** find or refute a prompt-length regime where
      surgical placement (C5) beats every homogeneous config on prefill
      latency and/or energy per prompt token on A17 Pro.
- [ ] **H3 (decode null, control):** show batch=1 decode tok/s is
      placement-invariant within ±5%; anything larger gets a root-cause.
- [ ] **H4 (sustained load):** measure degradation under ≥10 min continuous
      prefill on iPad Pro M2 (primary) and iPhone 15 Pro Max (secondary).
- [ ] Produce `stage0-report.md`, `stage1-report.md`, CSV results + plots, a
      public `lfm2-surgical-coreml` repo, and a case-study-3 section for the
      paper. A gate-triggered kill report satisfies this goal.

### Non-Goals

- No decode optimization, speculative decoding, or KV-cache tricks.
- No training or fine-tuning; instruct checkpoint used as-is.
- No quantization in v1 — fp16 weights end-to-end. Quantization changes
  bytes-moved and ANE admission simultaneously; one variable at a time.
- No Core ML stateful models for conv state in v1 — explicit I/O tensors only.
  Stateful models are a follow-up ablation.
- No Android/Snapdragon comparison; no Core AI (`.aimodel`) port in v1.
- No iPhone 12 Pro (A14) run unless a mobile-OS 1st-gen-ANE datapoint proves
  necessary (M1 Mini covers the ANE generation).

## Scope and Constraints

- **Scope (pre-spin-out):** new tooling under `scripts/lfm2_surgical/`,
  generated artifacts under `outputs/lfm2_surgical/` (uncommitted), reports
  under `README/Notes/`.
- **Scope (post-spin-out):** conversion scripts, segment models, Swift
  harness, and Stage 1+ reports live in `mattmireles/lfm2-surgical-coreml`
  (public from first commit). Stage 0 scripts are copied there so the public
  repo reproduces the whole chain. The paper and program memory stay here.
- **Constraints:** iPhone 15 Pro Max is unavailable until every other phase is
  complete; phone runs happen overnight, unattended where possible.
- **Constraints:** LFM2.5 weights are under LFM Open License v1.0 (fine for
  research); our code is MIT. The public repo README must carry the
  attribution split.
- **Guardrails:** No Kokoro export, bakeoff, or Swift runtime path in this
  repo is modified. This is a parallel experiment lane.

## Ground Truth Contracts (Do Not Violate)

- **No dispatch table, no claim.** Every reported number is accompanied by a
  per-op dispatch table captured **on the device that produced it** (Xcode
  Performance Report or `scripts/dump_device_compute_plan.py`). macOS and
  iPadOS/iOS schedulers differ per OS build; an admission result on one device
  certifies nothing about another.
- **Real checkpoint only.** Never a `tiny-random-*` fixture — randomly
  initialized weights make every downstream metric meaningless (MoE Stage 1
  postmortem, [009](./009-moe-ssd-dram-prefetch-plan.md)).
- **Enumerated shapes only.** Range-flexible shapes demote ops off the ANE.
  Buckets are frozen at {128, 256, 512, 1024, 2048} prompt tokens.
- **Cold and warm are separate.** First-run ANE compilation is reported
  separately, never averaged into steady-state numbers (benchmark hygiene
  guide).
- **Release builds only for cross-framework ratios.** Debug tax is asymmetric
  across frameworks (`README/Notes/iphone-release-build-mlx-comparison.md`).
- **Run order is a thermal treatment on fanless devices.** Counterbalance
  config order; 10-min cooldown between thermal runs; log
  `ProcessInfo.thermalState` continuously; discard runs that enter `.serious`.
- **The port is not the contribution.** CoreML-LLM already runs LFM2.5 on the
  ANE. Our contribution is per-block-class placement, the prefill crossover
  curve, and the decode null. If their monolith dominates everywhere, that is
  the publishable result — record it, don't hide it.
- **Pre-registered criteria are frozen** (§ Success Criteria) before any
  Stage 2 run. No post-hoc goalpost moves.
- **A kill decision is a valid successful outcome.** Any gate failure ends in
  a written negative-result report, not silent abandonment.

## Already Shipped (Do Not Re-Solve)

- **Dispatch evidence tooling:** `scripts/dump_device_compute_plan.py`,
  `scripts/inspect_coreml_compute_plan.m`, and the `coreml-profile` skill.
- **Compute-unit scheduling knowledge:**
  `README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`.
- **ANE layout/op compatibility:**
  `README/Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md`
  and `CoreML-ANE-compiler-failure-triage-guide.md`.
- **Split-graph packaging patterns:**
  `README/Guides/apple-silicon/CoreML-split-graphs-multifunction-packaging-guide.md`.
- **Benchmark hygiene + device lab protocol:**
  `Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md`,
  `iPhone-CoreML-device-lab-runbook.md`.
- **powermetrics discipline:**
  `apple-silicon-nvme-energy-measurement-guide.md` (within-machine
  comparisons only — never compare joules across devices).
- **Decomposition-tax control precedent:**
  `README/Notes/monolithic-coreml-control-experiment.md` (Kokoro monolith
  control; the G1a measurement mirrors it).

## Fresh Baseline (At Plan Creation, 2026-07-20)

- **Architecture:** No LFM2.5 code, exports, or measurements exist in this
  repo. CoreML-LLM (john-rocky) has a monolithic LFM2.5-350M ANE port with
  published power/thermal data — unread as of this writing.
- **Metrics:** None. Liquid's published numbers are CPU llama.cpp Q4_0; their
  sub-100 ms TTFT target is the external reference line.
- **Known gaps (Phase 0 closes these):** LFM2.5-350M layer interleaving
  (230M layout is documented; 350M must be read from the checkpoint config),
  CoreML-LLM's conv-state and KV-cache strategy, exact device/OS inventory.

## Solution Overview

```text
Stage 0 (this repo)              Stage 1+ (new public repo)
+---------------------+          +----------------------------+
| single conv block   |  gates   | embed | conv-run | GQA-run |
| single GQA block    | G0a-c    |  seg  |  seg(s)  |  seg(s) | ... | lm_head
| admission + parity  | ------>  |  per-segment computeUnits  |
+---------------------+          |  Swift orchestrator        |
     Mac Studio + iPad           +----------------------------+
                                   Mac Studio -> M1 Mini -> iPad -> iPhone (last)
```

Stage 2 and later measurement phases compare the same six placement configs
(see Mode Definitions). Stage 0 instead isolates one conv block and one GQA
block to decide whether the full matrix is worth building. The experiment is
one variable at a time: placement at fp16 first, with quantization, stateful
models, and larger checkpoints deferred to follow-ups.

## Implementation Phases

> Do one phase at a time. Verify before proceeding. Phases 0–4 never touch the
> iPhone.

### Required Skills

Use the workflow skills explicitly when executing this plan:

- **Whole plan:** `execute-plan` for normal phase-by-phase implementation. Use
  `execute-plan-hardcore` only if the user explicitly asks for the
  post-execution audit-to-A loop.
- **Every phase:** `phase-audit` before moving on. If delegated review is
  unavailable, run the local rubric in `README/Skills/phase-audit-rubric.md`.
- **Plan and docs edits:** `markdown` for this plan, `write-notes` for
  measurement evidence under `README/Notes/`, `documentation` for docstrings
  on the new scripts, and `david-ogilvy` only for reader-facing README copy in
  the public repo.
- **Git side effects:** `git-commit` per phase; `git-push` and `deploy` only
  when the user authorizes push/release.

| Phase | Required skills | Why |
| --- | --- | --- |
| Phase 0 | `coreml`, `guide-ingest`, `write-notes`, `phase-audit` | Read CoreML-LLM's conversion path and the LFM2.5 config, then land it as durable prior-art knowledge rather than a chat summary. |
| Phase 1 | `coreml`, `coreml-profile`, `coreml-validate`, `ilya-sutskever`, `debug`, `phase-audit` | The admission gate is exactly `coreml-profile`'s job (dispatch, silent fallback) and `coreml-validate`'s job (fp16 vs fp32 parity); `ilya-sutskever` keeps the block isolation simple. |
| Phase 2 | `markdown`, `david-ogilvy`, `documentation`, `git-commit`, `phase-audit` | The repo is public from the first commit, so the README is reader-facing copy and the scripts need real docstrings before strangers read them. |
| Phase 3 | `ilya-sutskever`, `coreml`, `coreml-validate`, `debug`, `documentation`, `phase-audit` | Segment boundaries and state I/O are architecture judgment calls; token-exact equivalence is a parity check. |
| Phase 4 | `coreml-profile`, `coreml`, `debug`, `write-notes`, `phase-audit` | Per-device dispatch tables plus rail attribution are profiling work; each device's results become a note. |
| Phase 5 | `coreml-profile`, `coreml`, `debug`, `write-notes`, `phase-audit` | On-phone G0a confirmation is a dispatch question before it is a latency question. |
| Phase 6 | `markdown`, `write-notes`, `david-ogilvy`, `documentation`, `deploy`, `phase-audit` | Figures, the case-study section, public reproduction docs, and release artifacts. |

**Skills that do not apply here, despite looking relevant:**

- `bakeoff` and `audio-judge` are Kokoro-specific (TTS harness, audio quality).
  This experiment has no audio and its own harness. Do not reach for them
  during Phases 4–5 just because they are the repo's benchmarking skills.
- The `coreml` router detects workspace by repo; from Phase 2 onward the work
  happens in `lfm2-surgical-coreml`, which it will treat as an unknown
  workspace. Use it for its Core ML guide links, not its repo routing.

### Phase 0: Prior-Art Diff and Inventory Freeze (½–1 day, Mac Studio)

**Goal:** Freeze the facts needed to interpret Stage 0; kill re-derivation.

**Required skills:** `coreml`, `guide-ingest`, `write-notes`, `phase-audit`.

**Tasks:**

- [x] Check the spec into the repo:
      `README/Notes/lfm2-surgical-experiment-spec-v1.1.md` (done at plan
      creation, 2026-07-20).
- [x] Read CoreML-LLM's LFM2.5 conversion path (conv state handling, KV
      strategy, shape strategy, quantization). Write a one-page diff summary
      into `README/Notes/lfm2-stage0-report.md` (started now, finished in
      Phase 1) with a reuse-vs-rewrite decision for the converter.
- [x] Pull `LiquidAI/LFM2.5-350M` config from HF and record the **actual**
      layer interleaving (conv vs GQA order) plus hidden dim, kernel size k,
      GQA head layout. If 350M isolation is awkward, record the fallback
      decision to `LFM2.5-230M`.
- [x] Record the device/OS inventory: Mac Studio (M2 Ultra), M1 Mini, iPad
      Pro 11" M2, iPhone 15 Pro Max — exact OS builds, coremltools and Xcode
      versions. Confirm the M1 Mini and iPad are physically available and on
      the OS builds we will measure on.
- [x] Record the license posture: LFM Open License v1.0 for weights, MIT for
      our code.

**Verification:** `lfm2-stage0-report.md` contains the converter decision, the
interleaving table, and the device inventory. No export code written yet.

---

### Phase 1: Stage 0 — Single-Block Admission Gate (1–2 days, Mac Studio + iPad)

**Goal:** Kill cheaply. Prove or refute ANE admission for the LIV conv block
before any decomposition work.

**Required skills:** `coreml`, `coreml-profile`, `coreml-validate`,
`ilya-sutskever`, `debug`, `phase-audit`.

**Tasks:**

- [x] `scripts/lfm2_surgical/extract_blocks.py` — load the real checkpoint,
      extract one double-gated LIV conv block and one GQA block (prefill
      form: no cache read, full-sequence attention) as standalone
      `nn.Module`s with flat tensor I/O.
- [x] `scripts/lfm2_surgical/export_blocks.py` — trace and convert each block
      with coremltools ≥ 8.x, fp16, `convert_to="mlprogram"`, **enumerated
      shapes** over {128, 256, 512, 1024, 2048}. ANE-friendly layout per the
      transformer layout guide (last axis = sequence).
- [x] Admission check on **Mac Studio ANE** (valid here: admission is a
      toolchain property) and **iPad Pro M2** (mobile-OS scheduler proxy):
      per-op dispatch per bucket via Xcode Performance Report and
      `scripts/dump_device_compute_plan.py`. Tables into the report.
- [x] `scripts/lfm2_surgical/check_numerics.py` — fp16 Core ML block outputs
      vs fp32 PyTorch on 32 real prompts; report max-abs and cosine per
      bucket.
- [x] Write go/no-go into `README/Notes/lfm2-stage0-report.md`.

**Kill gates (any one → stop and write the negative-result report):**

- **G0a (provisional):** kill if **either** the Mac Studio or iPad Pro places
  <80% of conv-block ops on the ANE at any gate bucket {128, 256, 512}. Both
  proxies must pass all three buckets. The spec defines G0a on A17; since the
  phone runs last, Mac+iPad is the provisional early gate and **G0a is
  confirmed on-phone at the start of Phase 5** before any headline
  measurement. If the phone contradicts the proxies, the phone verdict
  governs and Phase 5 stops at admission.
- **G0b:** converter rejects enumerated shapes / forces range shapes for the
  conv block.
- **G0c:** fp16 divergence > 1e-2 max-abs on block outputs with no
  identifiable fixable op.

**Verification:** report contains per-op dispatch tables per bucket per
device, the numerics table, and an explicit go/no-go line.

**Phase audit:** Grade A. All 12 iPad plans match the current package hashes
and their Mac counterparts; G0a/G0b/G0c pass with no blocking finding.

---

### Phase 2: Public Repo Spin-Out (½ day; only after Phase 1 passes)

**Goal:** Create `mattmireles/lfm2-surgical-coreml`, public from the first
commit, so every Stage 1+ artifact is born reproducible.

**Required skills:** `markdown`, `david-ogilvy`, `documentation`,
`git-commit`, `phase-audit`.

**Tasks:**

- [x] `gh repo create mattmireles/lfm2-surgical-coreml --public` with MIT
      LICENSE, README carrying the LFM Open License v1.0 weights attribution,
      and links back to the Surgical Inference paper repos (kokoro-coreml,
      magenta-realtime-2-iphone HF).
- [x] Copy `scripts/lfm2_surgical/` (Stage 0 scripts) into the new repo;
      kokoro-coreml keeps the originals frozen as the Stage 0 record.
- [x] Adapt this repo's `CLAUDE.md` (PyTorch→Core ML field guide) for the new
      repo; link back to this repo's `README/Guides/apple-silicon/` rather
      than duplicating them.
- [x] Copy the spec and `lfm2-stage0-report.md` into the new repo's docs.

**Verification:** fresh clone of the public repo reproduces the Stage 0
exports from the HF checkpoint with documented commands.

**Phase audit:** Grade A. Public repository
[`mattmireles/lfm2-surgical-coreml`](https://github.com/mattmireles/lfm2-surgical-coreml)
is live on `main` at `34f1b7b`. A fresh Python 3.11 environment installed the
checked-in pins, all 10 tests passed, all 12 packages rebuilt from the pinned
checkpoint, all 25 numerical cells passed, and the Transformers 5.5 oracle
matched exactly through layers 0-2. Core ML regenerated package UUIDs, but all
12 `weight.bin` payloads are byte-identical to the frozen Phase 1 artifacts.
Architecture, correctness risk, and complexity debt are all grade A.

---

### Phase 3: Stage 1 — Full Decomposition + Correctness (3–5 days, Mac Studio, new repo)

**Goal:** Working segmented pipeline, token-exact against PyTorch, with the
decomposition tax measured.

**Required skills:** `ilya-sutskever`, `coreml`, `coreml-validate`, `debug`,
`documentation`, `phase-audit`.

**Tasks:**

- [ ] Partition the layer stack into contiguous same-class segments per the
      Phase 0 interleaving table. Each segment: hidden states
      `[1, L_bucket, d]` fp16 in/out; conv rolling buffer `[1, d, k-1]` as
      explicit I/O; GQA prefill segments emit K/V as outputs (seed the decode
      cache), no cache inputs at prefill. Embedding and LM head as separate
      small models.
- [ ] Export all segments with enumerated shapes over the frozen buckets
      (`export_segments.py` in the new repo).
- [ ] Swift orchestrator (macOS/iOS shared core): per-segment
      `MLModelConfiguration.computeUnits`, prompt → embedding → segments →
      logits, signpost timing per segment, end-to-end TTFT hook.
- [ ] **Equivalence test:** greedy-decode 64 tokens on 32 prompts;
      token-exact vs fp32 PyTorch; every mismatch inspected at logit level and
      either attributed to fp16 or treated as a bug.
- [ ] **Decomposition tax measurement (G1a):** monolithic-GPU export vs
      segmented-all-GPU, same bucket 512, same prompts.

**Kill gates:**

- **G1a:** segment-boundary I/O overhead > 30% of total prefill at bucket 512
  (all-GPU vs monolithic-GPU) — the tax swamps any possible placement win.
- **G1b:** end-to-end token mismatch not attributable to fp16.

**Verification:** `stage1-report.md` in the new repo with equivalence results
and the measured tax; harness runs all six configs on the Mac.

---

### Phase 4: Stage 2a — Non-Phone Measurement (3–4 days, M1 Mini + iPad Pro M2)

**Goal:** Everything measurable without the daily-driver phone: mechanism
evidence (power rails), the H4 primary result, and full protocol shakedown so
the phone phase is a solved, scriptable procedure.

**Required skills:** `coreml-profile`, `coreml`, `debug`, `write-notes`,
`phase-audit`. Not `bakeoff` — that harness is Kokoro's.

**Tasks:**

- [ ] Fixed protocol for all runs: same prompts, OS build recorded, airplane
      mode / network quiesced, minimum brightness, battery 60–80% where
      applicable, no case, 10-min cooldowns, ambient logged, counterbalanced
      config order, thermal state logged continuously.
- [ ] **iPad Pro M2 dispatch tables** for every segment × config actually
      reported from the iPad (per-device dispatch rule).
- [ ] **iPad prefill + decode matrix:** C1–C6 × buckets {128…2048}, N=20,
      median + IQR, cold/warm separated; decode 128 tokens (H3 datapoint).
- [ ] **H4 primary (iPad):** 10-min continuous prefill loop at bucket 512,
      throughput per 30 s, configs C2/C3/C5, `thermalState` logged.
- [ ] **M1 Mini rail attribution:** C2/C3/C5/C6 under
      `sudo powermetrics --samplers cpu_power,gpu_power,ane_power,thermal`;
      report ANE/GPU/CPU package power per config. Appendix-only figure —
      evidence that C5's delta is watts moving between rails, not a scheduler
      artifact. Within-machine comparison only.
- [ ] **External baselines built + shaken down:** llama.cpp Q4_0 GGUF and the
      CoreML-LLM monolith running on iPad/Mac with the same prompt set.
- [ ] Freeze the Phase 5 run script end-to-end (one command per overnight
      batch, results to JSON/CSV) and rehearse it fully on the iPad.

**Verification:** iPad results tables + H4 curves + M1 rail figure exist under
the new repo's results dir; the phone-phase script has completed a full
dry-run on iPad with zero manual intervention between runs.

---

### Phase 5: Stage 2b — iPhone 15 Pro Max, Overnight (LAST; 1–3 nights)

**Goal:** All headline numbers. Runs only when Phases 0–4 are complete, on
Matt's schedule (phone on the desk while he sleeps).

**Required skills:** `coreml-profile`, `coreml`, `debug`, `write-notes`,
`phase-audit`.

**Tasks:**

- [ ] **Night 0 setup (minutes, attended):** install Release build, trust
      profile, Developer Mode, airplane mode, min brightness, no case,
      battery 60–80%, per the device-lab runbook.
- [ ] **Admission confirmation first:** per-op dispatch tables on the phone
      for every segment × bucket. This confirms G0a on A17. If <80% conv-op
      ANE dispatch at bucket ≤512, stop here and write the divergence finding
      (Mac/iPad admitted, phone didn't — that per-OS-build delta is itself a
      result).
- [ ] **Headline matrix:** C1–C6 × buckets, N=20, median + IQR, cold/warm
      separated; TTFT vs the 100 ms line; decode 128 tokens per config (H3).
- [ ] **Energy:** fixed 500-prompt batches per config → battery-drain deltas
      and Instruments Energy Log → J/prompt-token.
- [ ] **H4 secondary:** 10-min sustained prefill, C2/C3/C5, thermal state
      logged; compare degradation vs the iPad curves.
- [ ] **External baselines on-phone:** llama.cpp GGUF (context) and
      CoreML-LLM monolith, same prompts.
- [ ] Runs that hit `.serious` thermal state are auto-discarded and re-queued
      by the run script (this logic was rehearsed in Phase 4).

**Verification:** every phone number in the results CSV has a same-device
dispatch table, thermal log, and cold/warm labeling. Foreground/unlock policy
per the runbook honored for GPU-containing configs (screen-lock behavior for
overnight runs is validated during the Phase 4 rehearsal — see Open
Questions).

---

### Phase 6: Analysis, Write-Up, Publication (2 days)

**Goal:** Turn measurements into the case study and the public artifact.

**Required skills:** `markdown`, `write-notes`, `david-ogilvy`,
`documentation`, `deploy`, `phase-audit`.

**Tasks:**

- [ ] The crossover figure: prefill latency vs bucket, all configs, one plot.
- [ ] Per-segment attribution: which segments moved, by how much (mechanism,
      not just aggregate).
- [ ] H3 check: decode across configs; anything outside ±5% root-caused
      before write-up.
- [ ] Honest decomposition-tax accounting (G1a number) inside every C5/C6
      total.
- [ ] Evaluate the pre-registered criteria verbatim; write strong / weak /
      negative verdict accordingly.
- [ ] HF repo `mattmireles/lfm2.5-350m-surgical-coreml`: segment models,
      conversion scripts, reproduction README, license notes.
- [ ] Case-study-3 section drafted into `Scratchpad/surgical-inference.md`;
      results notes + pointers recorded here in `README/Notes/`.

**Verification:** a stranger can go from the public repos to every figure.

## Executable Memory

- Stage 0 proof: `python scripts/lfm2_surgical/export_blocks.py --all-buckets`
  then `python scripts/dump_device_compute_plan.py --package
  <block.mlpackage> --compute-units <CPU_AND_NE|CPU_AND_GPU> --out
  <dispatch.json>` — dispatch tables match the report. Mobile runs additionally
  pass `--device-type ipad --device-name iDesk`.
- Numerics proof: `python scripts/lfm2_surgical/check_numerics.py --prompts 32`
  — max-abs ≤ 1e-2 per gate G0c.
- Not testable by command: iPhone overnight protocol — proven by the Phase 4
  iPad dry-run log plus the phone-phase artifacts (dispatch tables + thermal
  logs per run).

## Success Criteria

### Pre-Registered Outcome Criteria (frozen before any Stage 2 run)

- **Strong:** C5 beats the best homogeneous config by ≥15% prefill latency or
  ≥20% energy at ≥2 buckets on A17 Pro, **and** C6 does not.
- **Weak / publishable negative:** C3 monolith wins everywhere → "monolithic
  ANE placement suffices for hybrid conv/GQA models at 350M scale;
  decomposition tax exceeds placement benefit" + the decode null. Written up
  with the same rigor.
- **H3 control:** decode within ±5% across configs; violations investigated,
  not celebrated.

### Hard Requirements (Must Pass)

- [ ] Every reported number has a same-device per-op dispatch table.
- [ ] Real trained checkpoint only; no random-weight fixtures anywhere.
- [ ] Enumerated shapes on every exported model; no range shapes.
- [ ] Cold vs warm never mixed; Release builds for all cross-framework rows.
- [ ] iPhone 15 Pro Max untouched until Phases 0–4 are verified complete.
- [ ] Any gate failure produces a written kill report in the repo.

### Definition of Done

- [ ] One of: full results through Phase 6, or a gate-triggered kill report
      (`lfm2-stage0-report.md` / `stage1-report.md`) with the negative finding.
- [ ] Public GitHub + HF repos live (if past Phase 2) with reproduction docs.
- [ ] Paper case-study section drafted or explicitly cancelled with reasons.

## Open Questions

### Resolved

- **Q:** This repo or a new one?
- **A:** Plan + Stage 0 here; new public repo at Phase 2, only after
  admission passes. Precedent: 009 died at Stage 1 with zero repo overhead.
- **Q:** Public or private new repo?
- **A:** Public from first commit (decided 2026-07-20) — MRT2's retroactive
  public-sync pain (`README/Notes/mrt2-public-repo-sync-2026-07-14.md`) is
  the cautionary tale.
- **Q:** GitHub repo name vs the spec's HF name?
- **A:** GitHub: `lfm2-surgical-coreml` (survives a 230M fallback). HF model
  artifacts keep the spec's `lfm2.5-350m-surgical-coreml`.
- **Q:** How does the A17-defined gate G0a work if the phone runs last?
- **A:** Split: Mac Studio + iPad Pro provisional kill in Phase 1; on-phone
  confirmation as the first step of Phase 5, before any headline run. Phone
  verdict governs.
- **Q:** 350M or 230M?
- **A:** 350M. The frozen checkpoint isolates cleanly and contains 16 layers:
  10 conv and 6 GQA in the measured `C C A C C A C C A C A C A C A C`
  order.
- **Q:** Reuse CoreML-LLM's converter or write ours?
- **A:** Use a small direct `coremltools` path informed by its shape and
  layout choices. CoreML-LLM's monolithic converter is not a clean
  per-block admission harness.

### Unresolved

- **Q:** Can GPU-containing configs (C2, C4, C5, C6) run overnight with the
  screen locked, given the runbook's foreground-Metal policy?
- **Options:** (a) keep display on at min brightness with autolock off —
  changes the energy baseline, must be constant across configs; (b) run
  ANE/CPU configs locked and GPU configs in an attended evening block.
  Current lean: (a), validated during the Phase 4 iPad rehearsal; whichever
  is chosen must be identical for every config so deltas stay meaningful.

## References

### Internal

- [Spec v1.1 (checked in at Phase 0)](../Notes/lfm2-surgical-experiment-spec-v1.1.md)
- [MoE prefetch plan — staging/kill-gate precedent](./009-moe-ssd-dram-prefetch-plan.md)
- [Compute-unit scheduling guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [ANE transformer layout & op compatibility](../Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md)
- [ANE compiler failure triage](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md)
- [Split graphs & multifunction packaging](../Guides/apple-silicon/CoreML-split-graphs-multifunction-packaging-guide.md)
- [Warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [iPhone device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)
- [powermetrics / energy measurement](../Guides/apple-silicon/apple-silicon-nvme-energy-measurement-guide.md)
- [Monolithic control experiment (Kokoro)](../Notes/monolithic-coreml-control-experiment.md)
- [Release-build / thermal confound findings](../Notes/iphone-release-build-mlx-comparison.md)
- [Plan workflow skills guide](../Skills/plan-workflow-skills-guide.md)

### External

- [LFM2 Technical Report, arXiv:2511.23404](https://arxiv.org/abs/2511.23404)
- [LFM2.5-230M architecture blog (8 LIV conv + 6 GQA)](https://www.liquid.ai/blog/lfm2-5-230m)
- [LiquidAI on Hugging Face (weights, LFM Open License v1.0)](https://huggingface.co/LiquidAI)
- [CoreML-LLM (john-rocky) — prior LFM2.5 ANE port](https://github.com/john-rocky/CoreML-LLM)
- [coremltools flexible input shapes (enumerated vs range)](https://apple.github.io/coremltools/docs-guides/source/flexible-inputs.html)
- [Apple: Deploying Transformers on the ANE](https://machinelearning.apple.com/research/neural-engine-transformers)
- [HeteroInfer, arXiv:2501.14794 (related work, must-cite)](https://arxiv.org/abs/2501.14794)
- [Liquid AI sub-100 ms TTFT target (job post)](https://jobs.ashbyhq.com/liquid-ai/1ed0e32c-11f4-4f93-bfab-bdfac37f0b1b)

## Performance and Latency Budget

Targets, not predictions. "Current" is empty by design — filling it is the
experiment. Prefill buckets are prompt-token counts on A17 Pro, C5.

| Operation | Target | Source of target | Current |
| --- | --- | --- | --- |
| TTFT (prefill + first decode step) | <100 ms | Liquid AI's published edge target | Unmeasured |
| Prefill @ bucket 512, C5 vs best homogeneous | ≥15% faster | Pre-registered strong criterion | Unmeasured |
| Energy per prompt token, C5 vs best homogeneous | ≥20% lower | Pre-registered strong criterion | Unmeasured |
| Decode tok/s spread across C1–C6 | within ±5% | H3 bandwidth-wall control | Unmeasured |
| Segment-boundary I/O overhead @ bucket 512 | <30% of prefill | Gate G1a | Unmeasured |
| Conv-block ops dispatched to ANE @ bucket ≤512 | ≥80% | Gate G0a | Mac Studio: 100% (28/28); iPad M2: 100% (27/27), each at 128/256/512; A17 confirmation pending Phase 5 |

## Degradation and Rollback

- **If G0a/G0b/G0c fails:** stop; `lfm2-stage0-report.md` becomes the
  negative admission finding; no repo is created; paper gains a paragraph,
  not a case study.
- **If G1a/G1b fails:** stop; `stage1-report.md` records that decomposition
  tax or numerics kill the approach at 350M scale — the "weak/publishable
  negative" framing applies.
- **If the phone contradicts the iPad on admission:** the per-OS-build
  scheduler divergence is written up as a finding; headline claims restrict
  to devices with confirming dispatch tables.
- **Rollback (this repo):** delete `scripts/lfm2_surgical/`,
  `outputs/lfm2_surgical/`, and the two Notes files. No Kokoro path is
  touched.

## Monitoring and Observability

**Metrics to Track:**

- `conv_ops_ane_dispatch_fraction` / `gqa_ops_ane_dispatch_fraction` — per
  bucket, per device; the G0a gate and the evidence behind every claim.
- `prefill_latency_ms_median` / `_iqr` — per config, per bucket, cold and warm
  recorded separately.
- `ttft_ms` — prefill plus first decode step, against the 100 ms line.
- `decode_tokens_per_second` — per config; the H3 control.
- `joules_per_prompt_token` — iPhone headline, from fixed 500-prompt batches.
- `ane_power_watts` / `gpu_power_watts` / `cpu_power_watts` — M1 Mini rail
  attribution only; within-machine comparison, never cross-device.
- `sustained_throughput_by_30s_bucket` — H4 degradation curves.
- `thermal_state` — sampled continuously; `.serious` invalidates the run.
- `peak_memory_bytes` — per config.
- `segment_boundary_overhead_fraction` — the G1a decomposition tax, carried
  inside every C5/C6 total.

**Artifacts to Preserve:**

- `outputs/lfm2_surgical/stage0/dispatch_<device>_<block>_<bucket>.json`
- `outputs/lfm2_surgical/stage0/numerics.json`
- `outputs/lfm2_surgical/stage1/equivalence.json`
- `outputs/lfm2_surgical/stage1/decomposition_tax.json`
- `outputs/lfm2_surgical/stage2/<device>_results.csv`
- `outputs/lfm2_surgical/stage2/powermetrics_m1_<config>.txt`
- `outputs/lfm2_surgical/stage2/thermal_<device>_<config>.jsonl`

## Phase Dependencies

```text
Phase 0 (inventory, prior-art diff)
   |
   v
Phase 1 (Stage 0 admission)  --[G0a/G0b/G0c]--> KILL: negative-result report
   |
   v
Phase 2 (public repo spin-out)
   |
   v
Phase 3 (Stage 1 decomposition)  --[G1a/G1b]--> KILL: negative-result report
   |
   v
Phase 4 (M1 Mini rails || iPad matrix + H4 + phone-protocol rehearsal)
   |
   v
Phase 5 (iPhone 15 Pro Max, overnight)  <-- LAST, blocked on Phase 4 complete
   |
   v
Phase 6 (analysis, figures, write-up, publication)
```

Strictly serial except inside Phase 4, where M1 Mini rail attribution and the
iPad matrix are independent and can interleave. The Phase 4 → Phase 5 edge is
a hard barrier: the phone is a daily driver and its protocol must already be a
rehearsed, one-command script before the first night.

## Files Likely to Change

Pre-spin-out (this repo). Everything from Phase 3 onward lands in
`mattmireles/lfm2-surgical-coreml` instead.

| File | Change Type | Notes |
| --- | --- | --- |
| `README/Plans/010-lfm2-surgical-prefill-plan.md` | Modify | Phase checkboxes and Debug Notes as work proceeds. |
| `README/Notes/lfm2-surgical-experiment-spec-v1.1.md` | Created | Frozen scientific contract; edit only via a version bump. |
| `README/Notes/lfm2-stage0-report.md` | Create | Prior-art diff, dispatch tables, numerics, go/no-go. |
| `scripts/lfm2_surgical/extract_blocks.py` | Create | Isolates one LIV conv block and one GQA block from the real checkpoint. |
| `scripts/lfm2_surgical/export_blocks.py` | Create | Traces and converts each block, fp16, enumerated shapes. |
| `scripts/lfm2_surgical/check_numerics.py` | Create | fp16 Core ML vs fp32 PyTorch on 32 real prompts. |
| `tests/test_lfm2_surgical_tools.py` | Create | Guards schema and bucket-freeze invariants. |
| `outputs/lfm2_surgical/**` | Create | Generated artifacts; stays uncommitted. |
| `Scratchpad/surgical-inference.md` | Modify | Phase 6 only: case-study-3 section. |

## Risks and Mitigations

- **LIV op lowering:** the double-gated conv may lower to op sequences the ANE
  compiler rejects or splits → Stage 0 per-op dispatch tables are the
  diagnostic. Manual re-expression of the gating is permitted only if it is
  numerics-preserving and documented.
- **Scheduler opacity:** `.cpuAndNeuralEngine` is a permission, not a command;
  Core ML may silently fall back → no dispatch table, no claim. This is a
  ground-truth contract, not a best effort.
- **Enumerated-shape memory blow-up:** each shape may precompile a variant →
  watch model load time and package size at Stage 0, before five buckets
  become five segments' worth of variants.
- **Thermal confounds on the phone:** overnight batches on a fanless daily
  driver → enforced cooldowns, continuous `thermalState` logging, auto-discard
  and re-queue of `.serious` runs, counterbalanced config order.
- **Daily-driver availability:** the phone is needed for real life → Phase 5
  is last, scripted, and rehearsed end-to-end on the iPad in Phase 4 so a
  night costs setup minutes, not debugging hours.
- **Prior art dominates:** CoreML-LLM's monolith may beat every surgical
  config → that is the pre-registered weak/publishable-negative outcome.
  Record it; do not search for a configuration that rescues the thesis.
- **Decomposition tax swamps the win:** segment-boundary I/O may exceed any
  placement benefit → measured explicitly as G1a and carried inside every
  C5/C6 total rather than reported separately.

## Debug Notes

Append real issues encountered during implementation with fixes.

### 2026-07-20 - Plan Skill Routing

**Problem:** The first draft of this plan named phases, gates, and files but
never told the executor which repo skills to invoke, so `execute-plan` would
have had to guess at profiling, validation, notes, and docs routing.
**Root Cause:** Same gap [008](./008-kokoro-drop-in-sdk-plan.md) hit on
2026-06-28; the canonical template has no Required Skills section, so it is
easy to omit when following the template literally.
**Fix:** Added a `Required Skills` section under `Implementation Phases` with
a per-phase table, a `Required skills:` line on every phase, and an explicit
do-not-use note for `bakeoff` / `audio-judge` (Kokoro-specific, and tempting
during the measurement phases).
**Files:** `README/Plans/010-lfm2-surgical-prefill-plan.md`

### 2026-07-20 - CPU-Only FP16 Runtime Trap

**Problem:** Core ML CPU-only validation of the fp16 block package terminated
with `SIGTRAP` before producing parity evidence.
**Root Cause:** The CPU-only runtime path on this host/toolchain does not
provide a usable fp16 execution oracle for this package.
**Fix:** Kept the exported graph and precision unchanged and ran validation
with `CPU_AND_GPU`, comparing every output against the fp32 PyTorch reference.
The gate is numerical parity, not CPU placement.
**Files:** `scripts/lfm2_surgical/check_numerics.py`,
`README/Notes/lfm2-stage0-report.md`

### 2026-07-20 - RMSNorm Re-expression

**Problem:** Reusing a LayerNorm-based RMS diagnostic exceeded the numerical
budget for the conv block.
**Root Cause:** The doubled `[x, -x]` LayerNorm rewrite is equivalent to
RMSNorm in exact arithmetic, but its compiled fp16 reduction and normalization
path accumulated enough rounding to push the live conv state over G0c.
**Fix:** Exported the simpler direct
`x * rsqrt(mean(x^2) + eps) * weight` composite. The final MIL operations
remain fp16; no unsupported fp32-accumulation claim is retained.
**Files:** `scripts/lfm2_surgical/blocks.py`,
`scripts/lfm2_surgical/export_blocks.py`

### 2026-07-20 - Actual-Activation GQA Drift

**Problem:** The isolated GQA block passed a synthetic probe but its exposed
key output drifted on the real layer-2 activation path.
**Root Cause:** K RMSNorm and RoPE rounding compounded after the K projection
and before the cache-format transpose.
**Fix:** Kept only three named TorchScript module scopes in fp32
(`operator_norm`, `attention_positioning`, `ffn_norm`) and inserted explicit
fp16 boundaries after each island. Attention scores, mask, softmax,
projections, residuals, and MLP remain fp16.
**Files:** `scripts/lfm2_surgical/blocks.py`,
`scripts/lfm2_surgical/export_blocks.py`,
`tests/test_lfm2_surgical_tools.py`

### 2026-07-20 - CoreDevice Tunnel-State Drift

**Problem:** coremltools 8.3 reported the paired iPad as disconnected after a
CoreDevice usage assertion ended, despite a usable live tunnel.
**Root Cause:** `Device.get_connected_devices()` consumed stale cached tunnel
state from `devicectl list devices`.
**Fix:** Reacquire identifier-specific `devicectl device info details`, then
fail closed on identifier, UDID, type, Developer Mode, DDI, pairing, and
tunnel before repairing only the frozen in-memory connection state.
**Files:** `scripts/dump_device_compute_plan.py`,
`tests/test_lfm2_surgical_tools.py`

### 2026-07-20 - Xcode SwiftBuild Clang Probe Deadlock

**Problem:** Xcode 26.6 stalled while creating the model-runner build
description at `clang -v -E -dM`.
**Root Cause:** The `-v -E -dM` probe path deadlocked during build-description
generation; pipe saturation is the leading inference, not a directly observed
Xcode diagnosis.
**Fix:** Added a narrow compiler wrapper that removes `-v` only for the
`-E -dM` probe. That single change advances the build to Apple signing. iDesk
was then registered through the existing App Store Connect key after explicit
authorization. A new wildcard iOS development profile includes only iDesk and
the two existing development certificates; no certificate was created,
revoked, or rotated. The runner accepts that explicit profile UUID and uses
manual signing, which Xcode 26 requires for this headless branch. The signed
runner produced all 12 iPad compute plans: conv 27/27 ANE and GQA 84/84 GPU for
the canonical enumerated packages and every fixed diagnostic bucket. Its local
cache is keyed by team, bundle, and profile, so later signing changes cannot
silently reuse an app built with different credentials.
**Files:** `scripts/xcode_clang_probe_wrapper.zsh`,
`scripts/dump_device_compute_plan.py`

### 2026-07-20 - Public Fresh-Clone Core ML UUID Drift

**Problem:** A fresh clone reproduced every export, but the regenerated
`.mlpackage` tree hashes did not equal the Phase 1 package hashes.
**Root Cause:** coremltools assigns new package item and root-model UUIDs on
each conversion. Those UUIDs appear in `Manifest.json` and the serialized
model, so a semantically identical rebuild is not container-byte-identical.
**Fix:** Verified the independent invariants instead of pretending the package
container is reproducible: exact checkpoint/config/tokenizer identities, all
12 `weight.bin` files byte-identical, all 25 numerical cells passing, and the
Transformers 5.5 oracle exact through layers 0-2. The public report also now
creates its own `.venv-hf`; it no longer references the private repo's ignored
Transformers checkout.
**Files:** public `README.md`, public `docs/stage0-report.md`

## Critical Reminder

> SIMPLER IS BETTER. The experiment's whole design is one variable at a time:
> placement at fp16 first; quantization, stateful models, and bigger
> checkpoints only as follow-ups.
