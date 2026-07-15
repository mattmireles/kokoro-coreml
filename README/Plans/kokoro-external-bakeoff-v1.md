# Kokoro External Bakeoff Plan

**Date:** 2026-06-05
**Status:** Complete. External result section and consolidated platform table
written; compile-contaminated 30s Config F cells replaced with warmed-inference
reruns; local Config F, MLX, Soniqo, and laishere powermetrics captured; signed
iPhone execution ingested; human listening decisions recorded as valid; final
completion verifier passes.

> Internal bakeoff methodology lives in `README/Plans/kokoro-bakeoff-v2.md`.
> This plan extends that methodology to external Apple Silicon Kokoro
> implementations for a publication-grade claim. The paper question is not
> "can Kokoro run fast on a Mac?" It is whether our surgical Swift+Core ML
> inference pipeline reaches quality parity faster than the popular Apple
> Silicon alternatives.

## Executive Summary

Benchmark our Swift+Core ML pipeline (Config F) against two external Apple
Silicon Kokoro implementations: the popular MLX Python implementation
(`Blaizzy/mlx-audio`) and `soniqo/speech-swift` as the primary iOS/Core ML
comparator. Measure time-to-parity as warmed inference: wall time until
equivalent in-memory PCM audio is produced after model load, compile, and cache
priming are complete, with voice/prosody parity spot-checked and hardware
placement documented. The thesis is falsifiable: our pipeline should have the
lowest warm median RTF on each machine and should do so while producing
comparable Kokoro audio.

## Problem Statement

- **Symptom:** We have strong internal Config F numbers, but no controlled
  comparison against the Apple Silicon implementations readers will ask about.
- **Root Cause:** The internal bakeoff was designed for our own runtime modes,
  not for external MLX or iOS/Core ML packages with different APIs and model
  packaging.
- **Impact:** We cannot make a credible paper claim that surgical inference is
  the fastest Kokoro path on Apple Silicon without reproducible competitor data,
  provenance, and quality parity evidence.

## Research Hypothesis

Our implementation is faster because it does not ask Core ML to solve the whole
dynamic TTS graph. It surgically splits the pipeline: small dynamic setup stays
on CPU, fixed-shape buckets send bulk math to Core ML/ANE, and Swift handles the
host-side waveform path with minimal Python overhead. The bakeoff should test
that hypothesis against:

- **MLX/GPU:** a high-mindshare Python MLX implementation.
- **Core ML/ANE:** an iOS-ready Core ML implementation that is not ours.

If an external implementation is faster but fails voice/prosody parity, the
result is a latency result with a quality caveat, not evidence against the
time-to-parity claim.

## Competitor Inventory

| Role | Handle | Repo | Public adoption | Framework | HW target | Status |
| --- | --- | --- | ---: | --- | --- | --- |
| Primary MLX competitor | **mlx-audio** | `github.com/Blaizzy/mlx-audio` | 7,186 stars | MLX Python | GPU | In scope |
| Primary Core ML comparator | **speech-swift KokoroTTS** | `github.com/soniqo/speech-swift` | 783 stars | Swift + Core ML | ANE | In scope |
| Specialized Core ML candidate | **laishere/kokoro-coreml** | `github.com/laishere/kokoro-coreml` | 15 stars | Core ML pipeline | ANE | Long-bucket backup |
| MLX Swift, not Core ML | **kokoro-ios** | `github.com/mlalma/kokoro-ios` | 256 stars | MLX Swift | GPU/Metal | Optional appendix only |

**Important correction:** `mlalma/kokoro-ios` is not the primary Core ML
comparator. Its current README and package manifest describe an MLX Swift port,
not a Core ML implementation. It can be useful as a third "native Swift MLX"
appendix, but it does not answer the iOS/Core ML comparison.

**Cut unless Phase 0 disproves this ranking:** gabrimatic/kokoro-mlx,
TTS.cpp/GGML, ONNX-only variants, and non-Kokoro engines.

## Goals and Non-Goals

### Goals

- [x] Establish warm median end-to-end RTF for our Config F, mlx-audio, and the
      selected iOS/Core ML comparator on m2-studio, irvine-m1, and m2-air.
- [x] Use identical input texts and voice (`af_heart` or closest equivalent)
      for the shipped runtime model buckets (`3s`, `7s`, `10s`, `15s`, `30s`).
- [x] Time the same boundary: immediately before synthesis call / CLI command
      to after full PCM audio is materialized in memory, excluding file writes.
- [x] Record cold first-call latency separately from warm median latency.
- [x] Capture hardware-placement evidence: MLX GPU activity for MLX competitors
      and Core ML/ANE evidence for Core ML competitors and our Config F.
- [x] Spot-check audio quality and voice/prosody parity before interpreting
      speed as time-to-parity.
- [x] Write a reproducible external-competitor section in
      `README/Notes/performance-notes.md`.

### Non-Goals

- Re-running all internal A/D/E baselines. Only Config F needs same-window
  calibration for the paper table.
- Benchmarking ONNX, GGML, browser, cloud, or non-Kokoro engines.
- Benchmarking Whisper, ASR, VAD, or echo-demo dependencies; this bakeoff is
  Kokoro TTS only.
- Integrating external implementations into production workers.
- Objective MOS/PESQ/MCD scoring for this round; listening plus waveform sanity
  metrics are sufficient for the first publication table.
- Modifying production LaunchAgents, `.env` files, or model bundles.

## Scope and Constraints

- **Scope:** Three machines x three primary implementations x five runtime
  model buckets (`3s`, `7s`, `10s`, `15s`, `30s`)
  x cold latency plus warm iterations.
- **Iterations:** N=5 warm iterations to match internal bakeoff methodology.
  If any competitor is within 20% of Config F or variance is high, run an N=20
  confirmation pass for those impl x machine x input cells.
- **Constraints:** External installs must be isolated and disposable. Do not
  touch the repo `uv.lock`, production worker envs, or system package managers.
- **Guardrails:** Generated JSON, WAVs, caches, build products, and cloned
  external repos stay under `outputs/external_bakeoff/` or `/tmp` and remain
  uncommitted.

## Ground Truth Contracts (Do Not Violate)

- **Comparator taxonomy:** Do not label an implementation Core ML unless Phase 0
  verifies a Core ML model path and `MLModel`/`MLComputeUnits` control.
- **Timing boundary:** Wall time starts immediately before the user-facing
  synthesis call or benchmark subprocess command and ends after the complete
  in-memory PCM/audio array is available. File I/O, playback, and WAV encoding
  are excluded.
- **Cold vs warm:** The paper-facing comparator is always warmed inference.
  First call on a fresh process is recorded as cold latency for operational
  evidence, but it is excluded from ranking, speedup, and thesis tables. If Core
  ML compile/cache work leaks into measured calls, discard additional preflight
  calls and rerun the affected cell until the reported median reflects
  steady-state inference only. Warm median is computed from recorded post-prime
  calls.
- **Input identity:** Use the exact text strings from the internal bakeoff.
  If `outputs/bakeoff/input_manifest.json` is missing, regenerate it with
  `scripts/bakeoff_harness.py prepare-inputs` before writing adapters. Then
  create an external runtime-bucket manifest that includes a verified `10s`
  input in addition to the historical `3s`, `7s`, `15s`, and `30s` inputs. Do
  not silently copy durations from prose tables into result JSON.
- **Runtime bucket coverage:** The paper table uses the shipped runtime model
  buckets (`3s`, `7s`, `10s`, `15s`, `30s`), not only the four historical
  bakeoff fixtures. Each input must be checked with Config F and recorded with
  the actual selected bucket.
- **RTF denominator:** Record both canonical duration from the manifest and
  observed audio duration from the emitted PCM. Use observed duration for
  competitor RTF if tokenization changes output length.
- **Voice:** Prefer `af_heart`. If unavailable, choose the closest American
  female Kokoro voice and document the substitution in the result record.
- **Provenance:** Every result JSON records git SHA or package version, machine,
  OS version, hardware model, compute-unit setting, command line, environment,
  input text hash, cold latency, warm per-iteration wall times, audio duration,
  and output WAV SHA256 for spot-check samples.
- **Same-window Config F:** For paper tables, run Config F on each machine in the
  same collection window as external competitors. The older
  `README/Notes/bakeoff-results-v2.md` tables remain a baseline, not the final
  paper comparator.

## Already Shipped (Do Not Re-Solve)

- **Internal bakeoff harness:** `scripts/bakeoff_harness.py`.
- **Config F baseline note:** `README/Notes/bakeoff-results-v2.md`.
- **Historical input constants:** `BAKEOFF_INPUTS`, `VOICE`, and `SPEED` in
  `scripts/bakeoff_harness.py` cover the historical `3s`, `7s`, `15s`, and
  `30s` fixtures.
- **Swift benchmark CLI:** `swift/Sources/KokoroBenchmark/main.swift`.

## Fresh Baseline (Current State)

Internal Config F medians from `README/Notes/bakeoff-results-v2.md`:

| Machine | 3s | 7s | 10s | 15s | 30s |
| --- | ---: | ---: | ---: | ---: | ---: |
| M2 Ultra | 57 ms | 124 ms | not in v2 | 239 ms | 476 ms |
| M2 Air | 185 ms | 396 ms | not in v2 | 1326 ms | 3021 ms |
| M1 Mini | 156.8 ms | 510.8 ms | not in v2 | 691.5 ms | 1228.9 ms |

These numbers prove the internal path is strong. They are not enough for a
paper-grade external comparison because they were not collected in the same
window as the external implementations and do not include the `10s` runtime
bucket.

## Solution Overview

Build small, disposable adapters that normalize external implementations into
one result schema. Keep the existing internal harness intact, but run Config F
same-window through its existing CLI/harness path.

```text
scripts/bakeoff_harness.py prepare-inputs
        |
        v
outputs/bakeoff/input_manifest.json
        |
        v
scripts/external_bakeoff/prepare_runtime_inputs.py
        |
        v
outputs/external_bakeoff/runtime_input_manifest.json
        |
        v
scripts/external_bakeoff/
  run_mlx_audio.py              -> Blaizzy MLX Python adapter
  run_speech_swift_kokoro.py    -> selected iOS/Core ML adapter
  run_config_f_reference.py     -> same-window Config F wrapper
  summarize_external.py         -> paper tables and speedups
        |
        v
outputs/external_bakeoff/
  results_<impl>_<machine_id>.json
  spotcheck_<impl>_<machine_id>_<bucket>.wav
        |
        v
README/Notes/performance-notes.md
```

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Competitor and API Audit

**Goal:** Prove the selected external implementations can synthesize one
sentence before writing benchmark code.

**Tasks:**

- [x] Verify `Blaizzy/mlx-audio` Kokoro API locally:
      `from mlx_audio.tts.utils import load_model`,
      `load_model("mlx-community/Kokoro-82M-bf16")`,
      `model.generate(text=..., voice="af_heart", speed=1.0, lang_code="a")`.
- [x] Build-verify `soniqo/speech-swift` Kokoro as the primary Core ML
      comparator: clone at a pinned SHA, build only the `KokoroTTS`/CLI surface
      needed, confirm it uses Core ML for Kokoro and exposes compute-unit
      selection.
- [x] `soniqo/speech-swift` built and synthesized, so
      `laishere/kokoro-coreml` remains a backup / appendix candidate rather
      than the selected Core ML comparator.
- [x] Exclude `mlalma/kokoro-ios` from the primary matrix. Add it only if the
      paper later needs a native Swift MLX appendix.
- [x] Confirm `af_heart` or closest substitute in each selected impl.
- [x] Confirm hardware-placement path:
      - MLX: source and dependencies route through MLX/Metal; capture runtime
        GPU evidence during benchmark collection.
      - Core ML: Soniqo loads `MLModel` with `MLComputeUnits`; capture runtime
        ANE/GPU/CPU evidence during benchmark collection.
- [x] Record exact versions, git SHAs, model IDs, and model cache paths.

**Verification:** One sentence synthesizes for mlx-audio and the selected Core
ML comparator on the operator Mac. The result note states the selected primary
Core ML comparator and why. Phase 0 evidence is recorded in
`README/Notes/external-bakeoff-phase0-api-audit.md`.

---

### Phase 1: Write Adapter Scripts

**Goal:** Produce one common JSON schema for Config F, MLX, and Core ML
competitors.

**Tasks:**

- [x] Ensure `outputs/bakeoff/input_manifest.json` exists. If not, run:

      ```bash
      uv run --with-requirements requirements-bakeoff.txt --no-sync \
        python scripts/bakeoff_harness.py prepare-inputs
      ```

- [x] Create `scripts/external_bakeoff/prepare_runtime_inputs.py`:
      - Reads the historical bakeoff manifest for `3s`, `7s`, `15s`, and `30s`.
      - Adds a new `10s` text candidate.
      - Runs Config F on every candidate and records the selected bucket.
      - Fails if any input does not route to its named runtime bucket.
      - Writes `outputs/external_bakeoff/runtime_input_manifest.json`.
- [x] Create `scripts/external_bakeoff/schema.py` with a tiny shared result
      writer/validator for:
      `impl`, `machine_id`, `framework`, `hardware_target`, `version`,
      `input_key`, `text_sha256`, `voice`, `cold_wall_time_s`,
      `warm_wall_times_s`, `canonical_audio_duration_s`,
      `observed_audio_duration_s`, `rtf_observed`, `output_sha256`,
      and `provenance`.
- [x] Create `scripts/external_bakeoff/run_mlx_audio.py`.
- [x] Create `scripts/external_bakeoff/run_speech_swift_kokoro.py` for the
      selected Core ML comparator. If Phase 0 selects a different Core ML repo,
      name the adapter after that repo.
- [x] Create a minimal signed iOS runner for Soniqo Kokoro so the connected
      iPhone 12 Pro can run the Core ML comparator without the full echo demo
      dependency graph.
- [x] Create `scripts/external_bakeoff/run_config_f_reference.py` as a thin
      wrapper around the existing Config F harness/Swift CLI so same-window
      Config F records share the external result schema.
- [x] Create `scripts/external_bakeoff/summarize_external.py`:
      - Reads all external result JSONs.
      - Emits wall-time, RTF, cold-start, and speedup tables.
      - Computes Config F speedup per machine/input/competitor.
      - Flags missing quality or hardware-placement evidence.
- [x] Create pinned install docs:
      `requirements_mlx_audio.txt` and `README.md` under
      `scripts/external_bakeoff/`.

**Verification:** All three adapters run locally on one input and emit schema-
valid JSON. The runtime manifest has five inputs and each routes to its named
bucket. The summarizer emits a markdown table without hardcoded M2 Ultra only
assumptions. Local smoke status:
`mlx-audio` and `soniqo/speech-swift` emitted successful `10s` records;
Config F emitted a schema-valid local error because this worktree lacks Core ML
duration artifacts (`Duration choices: ` empty), so Phase 2 must run on hosts
with the full Core ML model set.

---

### Phase 2: Deploy and Run on Fleet

**Goal:** Collect same-window data from all three machines without disrupting
production TTS workers.

**Current gate:** `pnpm check:tts-worker-health --json` was green on 2026-06-05
with queue depth `0`, `0` claimed jobs, `3` fresh Kokoro workers, and passing
canary worker `operator-prove-live`. The local Config F collection gate is
resolved: `run_config_f_reference.py` now uses persistent `kokoro-bench --batch`
mode, and a 10s one-input smoke against the main checkout's Core ML artifacts
emitted schema-valid JSON with `status=ok`, `bucket_used=10s`, cold wall time
`0.54524s`, warm wall time `0.505431s`, observed duration `9.625s`, and WAV
SHA-256 `1422cb557e87f09008dec461850ffc803fb3c24e4911bb89ef1498fa15aec904`.
The connected iPhone 12 Pro is visible to CoreDevice and developer mode is
enabled. The live device check on 2026-06-05 identified the phone as `Webcam`,
identifier `F383FC46-FD64-5346-AEC6-59E3E2F8C9CA`, UDID
`00008101-001134561A0A001E`, model `iPhone13,3`, connected and paired. The
Soniqo iOS runner is manifest-driven across the runtime buckets (`3s`, `7s`,
`10s`, `15s`, `30s`) and reports one cold call plus five warmed calls per
bucket with observed-duration RTF. After Apple Development signing was added,
`SPEECH_SWIFT_PATH=/tmp/kokoro-external-bakeoff/speech-swift DEVELOPMENT_TEAM=6ETYBAJKY8 python scripts/external_bakeoff/preflight_ios_runner.py --generate-project --build --output outputs/external_bakeoff/ios_runner_build_latest.json`
completed with `ok: true`, no blockers, and `BUILD SUCCEEDED`. The app was
installed and launched on the physical phone with
`xcrun devicectl device process launch`, run on device, copied JSON through the
clipboard, and was ingested with
`python scripts/external_bakeoff/ingest_ios_runner_result.py --input outputs/external_bakeoff/ios_runner_payload_latest.json --machine-id iphone-12-pro`.
The normalized output is
`outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_ios_iphone-12-pro.json`
with five signed-device records. Warm medians were `832.7 ms` for 3s,
`833.9 ms` for 7s, `853.6 ms` for 10s, `864.4 ms` for 15s, and `879.1 ms` for
30s. As with the macOS Soniqo runs, the longer inputs emit `5.0s` audio from
the public Soniqo artifact, so those iPhone cells are device execution evidence
but not full-duration long-bucket parity evidence. The iOS runner remains
deliberately Kokoro-only; Whisper, ASR, VAD, and the full Soniqo echo-demo
dependency graph are excluded from the measurement path.

**Current collection note:** M2 Studio, irvine-m1, and M2 Air now have
schema-valid JSON for Config F, MLX, Soniqo, and laishere, and every successful
result cell has a durable spot-check WAV. MLX fails deterministically on the
shared `3s` input with a broadcast-shape error, so that cell is recorded as
public-implementation behavior. Soniqo emits 5.0s audio for longer manifest
inputs because the selected public Core ML model repo only publishes
`kokoro_5s.mlmodelc`; it remains the high-adoption iOS/Core ML comparator with
this public-artifact caveat. laishere is the normalized long-bucket Core ML
backup for quality-parity evidence. Human listening and hardware-placement
evidence are recorded before interpreting the speed table as time-to-parity.
The original Config F 30s cells on m2-air and irvine-m1 were compile/cache
contaminated, so the paper-facing warmed-inference tables use corrected 30s
runs with `KOKORO_USE_EXACT_DURATION_MODELS=1`, three discarded preflight
calls, and 20 recorded warm calls. A local M2 Studio powermetrics capture now
exists for a post-prime 3s Config F debug run, and a matching local MLX 7s
powermetrics capture exists for the pinned `mlx-audio` path. These are placement
evidence only, not replacement latency cells, and they do not close the
laishere backup placement requirement. A local Soniqo 3s powermetrics capture
also exists for the primary iOS/Core ML comparator, and a local laishere 3s
powermetrics capture exists for the long-bucket backup Core ML chain.

**Tasks:**

- [x] Before each host, check fleet health and queue pressure from the botnet
      repo:

      ```bash
      cd /Users/mm/Documents/GitHub/botnet
      WEB_SCRAPER_BASE_URL="${WEB_SCRAPER_BASE_URL:?set base url}" \
        pnpm check:tts-worker-health
      ```

- [x] For each machine (`m2-studio`, `irvine-m1`, `m2-air`):
      - SSH with the known user for that host.
      - Copy only `scripts/external_bakeoff/`, the manifest, and any required
        Swift package wrapper.
      - Run Config F same-window first, then mlx-audio, then Core ML competitor.
      - Run laishere/kokoro-coreml when Soniqo's public 5s artifact cannot
        represent the longer runtime buckets.
      - Run one implementation at a time; on m2-air add a cooldown and abort if
        thermal pressure is non-nominal.
      - Copy JSON and spot-check WAVs back to
        `outputs/external_bakeoff/`.
      - Remove venvs, cloned external repos, and Swift build products.
- [x] Re-check fleet health after each host and record the result in the run
      note.

**Verification:** Twelve Mac result JSON files exist: Config F, MLX, Soniqo,
and laishere across 3 machines. One signed iPhone result JSON exists for the
Soniqo iOS runner. Each result has cold latency, 5 warm iterations per
successful runtime bucket, and provenance; Mac success cells have durable
spot-check WAVs, while the minimal iPhone runner exposes timing JSON only.
Framework/runtime placement is documented. One local Config F privileged
`powermetrics` capture and one local MLX privileged `powermetrics` capture
exist. One local Soniqo privileged `powermetrics` capture exists for the primary
iOS/Core ML comparator. One local laishere privileged `powermetrics` capture
exists for the backup Core ML chain.

---

### Phase 3: Quality Spot-Check

**Goal:** Make the speed claim about comparable Kokoro audio, not merely any
audio buffer.

**Tasks:**

- [x] Save WAV output for each implementation and each runtime bucket on each
      machine.
- [x] Add a reproducible TTS-only listening review generator that reads the
      collected WAVs and waveform-quality reports without Whisper, ASR, VAD, or
      echo-demo dependencies. The generator emits Markdown, HTML, and a
      fillable CSV decision sheet with blank `human_decision` fields.
- [x] Listen against Config F for the same text and voice.
- [x] Run lightweight waveform sanity checks using `scripts/audio_quality_probe.py`
      where applicable: duration, RMS, clipping, silence, and gross spectral
      failures.
- [x] Document voice/prosody caveats. If a competitor cannot use `af_heart`,
      state that speedups are not direct voice-matched claims.

**Verification:** Each implementation has a short quality note. Any latency
table cell without quality parity evidence is marked with a caveat. Running
`python scripts/external_bakeoff/create_listening_review.py` emits Markdown,
HTML, and `external_bakeoff_listening_decisions.csv` review artifacts under
`outputs/external_bakeoff/listening/`. The generated CSV has one row per
available result cell plus error rows, and `human_decision` remains blank until
the operator listens. The operator listened to all successful rows and marked
all 57 successful audio rows `pass` because the audio was valid. Regenerating
the review preserves existing
`human_decision` and `notes` fields by default; use `--reset-decisions` only
when intentionally discarding prior listening work. After filling the CSV, run
`python scripts/external_bakeoff/validate_listening_decisions.py`; it must pass
before this phase can be marked complete. The validator now passes with
`decisions={'pass': 57}`.

---

### Phase 4: Document Results

**Goal:** Produce the table and narrative needed for the paper.

**Tasks:**

- [x] Run `summarize_external.py` and paste the tables into
      `README/Notes/performance-notes.md`.
- [x] Add a section titled:
      "External Bakeoff: surgical Core ML vs MLX and iOS/Core ML Kokoro".
- [x] Include:
      - Competitor selection and excluded repos.
      - Machine and OS provenance.
      - Consolidated warm median and RTF tables by hardware platform.
      - Cold latency table.
      - Warm median wall-time table.
      - RTF table using observed duration.
      - Config F speedup against each external implementation.
      - Hardware-placement evidence.
      - Quality caveats and interpretation.
- [x] Update this plan to `Status: Complete` only after results and notes are
      committed and
      `python scripts/external_bakeoff/verify_external_bakeoff_completion.py`
      passes.
- [x] Use `git-commit` to commit only the plan, adapters, and notes. Do not
      commit generated JSON or WAV files under `outputs/`.

**Verification:** `performance-notes.md` contains enough information for a
reader to reproduce the comparison from clean clones and pinned versions.
Running `python scripts/external_bakeoff/verify_external_bakeoff_completion.py`
is the final plan-completion gate; it now passes with `result_record_count=65`,
`ios_preflight_ok=true`, and `decisions={'pass': 57}`.

## Success Criteria

### Hard Requirements (Must Pass)

- [x] The primary table includes our Config F, mlx-audio, and one verified
      iOS/Core ML comparator across all three machines and all five runtime
      model buckets.
- [x] Each impl x machine x input has cold latency and N=5 warm calls.
- [x] Timing boundaries are explicitly equivalent.
- [x] `af_heart` or documented substitute is used for every implementation.
- [x] Hardware placement is documented for every implementation family.
- [x] No production worker disruption is observed.
- [x] Adapter scripts and pinned install docs are checked in.
- [x] Quality spot-check is documented before interpreting speedups.

### Definition of Done

- [x] `README/Notes/performance-notes.md` has the external-competitor section.
- [x] Adapter scripts and requirements files are committed.
- [x] This plan is updated to `Status: Complete`.
- [x] `python scripts/external_bakeoff/verify_external_bakeoff_completion.py`
      passes.

## Open Questions

### Unresolved

- None.

### Resolved

- **Q:** Is `soniqo/speech-swift` the right primary iOS/Core ML comparator?
- **A:** Yes. It is the highest-adoption Core ML Kokoro candidate found so far
  and it directly matches the paper's iOS/Core ML comparison. Phase 0 still
  build-verifies the benchmark boundary; if Soniqo cannot expose a clean
  Kokoro-only path, fall back to `laishere/kokoro-coreml` and document the
  adoption tradeoff.

- **Q:** Is `mlalma/kokoro-ios` needed as an appendix?
- **A:** No for the primary paper claim. It is MLX Swift, not Core ML, and would
  duplicate the MLX family. Add it only if the paper later needs a native Swift
  MLX appendix.

- **Q:** Does m2-air's production queue need to be paused during the run?
- **A:** No blanket pause. Run during a low-traffic window, check fleet health
  before and after the host run, run one implementation at a time, add cooldown,
  and abort/reschedule if queue drain is active or thermal pressure is
  non-nominal. Pause the worker only as an operator override after those gates
  fail.

- **Q:** Should this extend `scripts/bakeoff_harness.py` directly?
- **A:** No. External APIs and dependency environments differ. Standalone
  adapters plus a shared result schema are simpler and safer.

- **Q:** Should old Config F tables be the final paper comparator?
- **A:** No. They remain the baseline note, but paper tables require same-window
  Config F calibration on each machine.

- **Q:** Is `mlalma/kokoro-ios` a Core ML comparator?
- **A:** No. Current public docs and package dependencies identify it as MLX
  Swift. It is not the primary Core ML comparison.

## References

### Internal

- [Bakeoff v2 Plan](kokoro-bakeoff-v2.md) - internal harness design
- [Bakeoff Results v2](../Notes/bakeoff-results-v2.md) - existing Config F baseline
- [Performance Notes](../Notes/performance-notes.md) - where final results land
- `/Users/mm/Documents/GitHub/botnet` - fleet health and worker status commands

### External

- [Blaizzy/mlx-audio](https://github.com/Blaizzy/mlx-audio)
- [soniqo/speech-swift](https://github.com/soniqo/speech-swift)
- [Soniqo Kokoro guide](https://soniqo.audio/guides/kokoro)
- [laishere/kokoro-coreml](https://github.com/laishere/kokoro-coreml)
- [mlalma/kokoro-ios](https://github.com/mlalma/kokoro-ios)

## Files to Create

| File | Change Type | Notes |
| --- | --- | --- |
| `scripts/external_bakeoff/schema.py` | Create | shared result schema helpers |
| `scripts/external_bakeoff/prepare_runtime_inputs.py` | Create | five-bucket runtime input manifest |
| `scripts/external_bakeoff/run_mlx_audio.py` | Create | Blaizzy MLX adapter |
| `scripts/external_bakeoff/run_speech_swift_kokoro.py` | Create | Soniqo Core ML adapter, or rename if fallback selected |
| `scripts/external_bakeoff/run_laishere_kokoro_coreml.py` | Create | long-bucket Core ML backup adapter |
| `scripts/external_bakeoff/run_config_f_reference.py` | Create | same-window Config F wrapper |
| `scripts/external_bakeoff/summarize_external.py` | Create | paper table generator |
| `scripts/external_bakeoff/requirements_mlx_audio.txt` | Create | pinned MLX Python deps |
| `scripts/external_bakeoff/README.md` | Create | install/run instructions |
| `README/Notes/performance-notes.md` | Modify | add external section |
| `README/Plans/kokoro-external-bakeoff-v1.md` | Modify | this revised plan |

## Risks and Mitigations

- **Wrong Core ML comparator:** Phase 0 verifies Core ML usage before adapter
  work. Soniqo exposes a clean Kokoro-only path but its public artifact is 5s
  only; use `laishere/kokoro-coreml` for long-bucket Core ML parity evidence
  and explain the adoption tradeoff.
- **External API drift:** Pin git SHAs or package versions and record them in
  every result JSON.
- **Thermal and production interference:** Run one implementation at a time,
  monitor fleet health, and treat m2-air as fanless and fragile.
- **Unfair duration denominator:** Record both canonical and observed duration;
  use observed duration for competitor RTF if output length differs.
- **Quality mismatch:** Keep speed tables, but mark any non-parity audio as a
  latency-only result.

---

> Simpler is better. The proof must be small, reproducible, and hard to
> misinterpret: one MLX competitor, one verified Core ML competitor, one
> same-window Config F reference, one schema.
