# Kokoro Audio Parity Recovery Plan

**Date:** 2026-04-16
**Status:** Complete

## Executive Summary

Recover human-sounding Kokoro audio before doing any more performance claims. The
post-update Core ML and Swift bakeoff samples were unproven and the available
waveform statistics showed near-inactive output compared with the known HAR-post
demo, so this plan inspected waveforms and spectrograms directly, created
listenable reference samples only when the files passed basic speech-health
checks, then bisected the Python-to-Swift/Core ML runtime until the first
semantic audio divergence was identified and fixed.

## Problem Statement

- **Symptom:** WAVs in `outputs/bakeoff/listen/` from the bakeoff winner do not
  sound like human speech. Objective waveform checks agree: the samples are
  mostly near-silence with spikes, not dense voiced audio.
- **Root Cause:** Confirmed for the current Swift + Core ML winner: the final
  Core ML waveform `MLMultiArray` was read through raw pointer traversal even
  though its logical sample order can be non-contiguous. Existing notes still
  record a related residual parity risk around `SourceModuleHnNSF` / `SineGen`,
  but the non-human bakeoff WAVs were fixed by stride-safe waveform extraction.
- **Impact:** Any bakeoff result after the big update is invalid as a quality or
  product claim until the generated audio is proven human-sounding against a
  reference pipeline.

## Mode Definitions

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| Reference generation | Produce PyTorch and known-good HAR-post WAVs plus objective reports. | Gives the user listenable samples quickly and anchors all thresholds. |
| Direct waveform inspection | Inspect PCM samples, waveform plots, and spectrograms before human listening. | Rejects obvious non-speech failures without wasting the user's listening time. |
| Stage parity | Run identical tensors through Python, Core ML, and Swift boundaries. | Finds the first layer where speech semantics are lost. |
| Recovery implementation | Apply the smallest fix to the first divergent boundary. | Avoids rewriting the pipeline around an assumed bug. |
| Bakeoff reinstatement | Re-run performance comparisons only after audio proof passes. | Prevents speed numbers from masking broken synthesis. |

## Required Skills

Use these skills around this plan and during execution:

| Skill | When to use | Required output |
| --- | --- | --- |
| `create-plan` | Authoring or materially restructuring this checked-in plan. | A plan under `README/Plans/` that follows the canonical template and is grounded in repo notes and guides. |
| `markdown` | Any edit to this plan or related README material. | Clean markdown with real links, the plan template structure, and no duplicate prose. |
| `audit-fix-loop` | Self-auditing this plan document, or auditing a completed fix set after implementation. | Findings fixed and re-audited until architecture, correctness risk, and complexity debt are all A. |
| `debug` | Primary execution workflow for root-cause investigation and fix proof. | Reads relevant `README/Guides` and `README/Notes`, proves the fix, then ends with a consolidated note via `write-notes`. |
| `ilya-sutskever` | Architecture judgment for Core ML conversion, CPU/ANE split, and parity gates. | Keeps the recovery simple: dynamic DSP on CPU/Swift, conv-heavy math on Core ML, empirical proof before claims. |
| `phase-audit` | After each phase is implemented. | Findings-first review against this plan, guide alignment, tests, and checkbox accuracy. |
| `write-notes` | Final step after the root cause is confirmed. | Updates `README/Notes/debug-notes.md` or the right consolidated note with the cause, fix, commands, and remaining risks. |
| `execute-plan-hardcore` | Optional end-to-end executor after this plan is approved. | Implements phases, audits each phase, commits per phase, then runs audit-to-A. |
| `bakeoff` | Only after the audio quality gates pass. | Produces performance data for known-human audio, not broken waveforms. |

Do not use `bakeoff` as a correctness signal. Do not treat `audit-fix-loop` as a
replacement for the stage-parity investigation; it is appropriate for auditing
this plan document or for a completed fix set.

## Goals and Non-Goals

### Goals

- [x] Produce reference WAVs the user can listen to before changing model code.
- [x] Trace `outputs/decoder_har_post_demo.wav` through git history and attempt
      an exact reproduction using one existing enumerated shape or bucket before
      broader recovery work begins.
- [x] Inspect waveforms and spectrograms directly before asking the user to
      listen to any generated sample.
- [x] Freeze the failing bakeoff WAVs and record objective waveform and
      spectrogram evidence for regression comparison.
- [x] Establish a stage-parity harness that can compare Python reference tensors
      with the Swift + Core ML winner at every boundary.
- [x] Identify and fix the quality-destroying audio divergence in the current
      post-update runtime.
- [x] Replace weak listen-sample validation with gates derived from PyTorch and
      known-good HAR-post references.
- [x] Re-run the bakeoff only after short and medium samples sound human.

### Non-Goals

- Optimizing RTF, ANE utilization, memory, or package size before speech quality
  is recovered.
- Publishing or relying on post-update bakeoff rankings before audio proof.
- Rebuilding the whole Kokoro pipeline unless stage parity proves the current
  split cannot be salvaged.
- Accepting waveform non-emptiness, duration agreement, or low RMS checks as
  proof of speech quality.

## Scope and Constraints

- **Scope:** Python reference generation, waveform inspection, stage parity,
  Swift/Core ML runner instrumentation, minimal correctness fixes, quality gates,
  and final bakeoff rerun.
- **Constraints:** Generated WAVs, tensor dumps, reports, and model artifacts
  stay under `outputs/` and remain gitignored unless explicitly promoted.
- **Constraints:** The first shippable recovery should prefer the existing
  HAR-post split: CPU/Swift harmonic source and Core ML `GeneratorFromHar`.
- **Guardrails:** Preserve existing model-loading and export entry points:
  `HybridTTSPipeline.extract_vocoder_inputs()`,
  `build_decoder_har_post_inputs_np()`, `export_synth.main`, and the Swift
  `KokoroPipeline` package structure.

## Ground Truth Contracts (Do Not Violate)

- **Human quality baseline:** Full PyTorch Kokoro from
  `examples/example_synthesis.py --engine pytorch` is the primary speech
  reference.
- **Known-good local comparator:** `outputs/decoder_har_post_demo.wav` is a
  useful comparator only after its provenance is recorded in the recovery report.
- **Demo reproduction first:** Before treating
  `outputs/decoder_har_post_demo.wav` as a quality anchor, the agent must dig
  through git history to recover how it was generated and try to reproduce it
  with an existing enumerated shape or bucket. A byte-identical match is the
  target; if that is impossible, the exact PCM, tensor, artifact, and command
  deltas must be recorded before moving on.
- **Current bakeoff samples are quality-gated artifacts:** The original failing
  files were frozen under `outputs/audio-parity/failing-current/`; files under
  `outputs/bakeoff/listen/` are now regenerated through the quality gate.
- **Post-update outputs are unproven:** No Core ML or Swift model produced after
  the big update should be treated as speech-capable until it passes this plan's
  reference, parity, and listening gates.
- **Stage parity before architecture claims:** A Core ML or Swift path cannot be
  called correct until identical inputs are compared against the Python reference
  at the same boundary.
- **Quality before speed:** A faster runtime that emits non-human audio fails the
  plan.
- **Machine inspection before human listening:** The agent must inspect waveform
  and spectrogram evidence first. Samples with obvious silence, impulse spikes,
  clipping, DC-heavy output, or missing voiced-band structure are rejected before
  asking the user to listen.
- **DSP split stays intentional:** If `SourceModuleHnNSF` / `SineGen` is involved
  in the divergence, keep that path off full-decoder Core ML unless a new
  parity proof shows Core ML can carry it.

## Already Shipped (Do Not Re-Solve)

- **Duration and shape validation fix:** Commit `1e48249` tightened several
  bakeoff acceptance checks, but it did not prove semantic speech quality.
- **HAR-post tensor builder:** `build_decoder_har_post_inputs_np()` in
  [kokoro/synthesis_backends.py](../../kokoro/synthesis_backends.py) is the
  Python single source of truth for `x_pre`, `ref_s`, and `har` geometry.
- **Swift runtime:** [swift/Sources/KokoroPipeline/KokoroPipeline.swift](../../swift/Sources/KokoroPipeline/KokoroPipeline.swift)
  is the current native runtime path for the bakeoff winner.
- **Swift harmonic source implementation:** [swift/Sources/KokoroPipeline/HarmonicSource.swift](../../swift/Sources/KokoroPipeline/HarmonicSource.swift)
  is already intended to keep hn-nsf on CPU/Accelerate.
- **Existing hn-nsf validator:** [scripts/validate_hnsf_swift.py](../../scripts/validate_hnsf_swift.py)
  already generates PyTorch harmonic-source references for Swift comparison.
- **Listen-sample helper:** [scripts/bakeoff_listen.py](../../scripts/bakeoff_listen.py)
  generates Config F WAVs, but its current thresholds are not sufficient proof
  of speech.

## Fresh Baseline (Current State)

- **Known-good comparator:** `outputs/decoder_har_post_demo.wav` is 1.363 s,
  24 kHz mono PCM, RMS about `6973`, active fraction above 32 PCM counts about
  `77.8%`, and zero-crossing rate about `8.9%`.
- **Frozen failing Config F samples:** The original files copied to
  `outputs/audio-parity/failing-current/` have RMS roughly `491`, `488`, `300`,
  and `42`; active fraction above 32 PCM counts roughly `2.0%`, `2.2%`, `2.1%`,
  and `0.12%`; and zero-crossing rate below `0.36%`.
- **Known prior bisect:** [README/Notes/debug-notes.md](../Notes/debug-notes.md)
  reports `pre_generator`, `post_conv`, `stft_transform`, and
  `spectral_head_inverse` near exact, with the first major full-decoder Core ML
  failure at `SourceModuleHnNSF` / `SineGen`.
- **Risky validator gap:** The listen helper currently accepts extremely low
  activity thresholds. It can reject empty files, but it cannot distinguish
  speech from near-silence or impulse noise.

## Solution Overview

```text
Demo provenance
  git history -> command/artifacts/text/voice/speed -> enumerated-shape rerun
        |
        v
Reference WAVs
  PyTorch full path + known HAR-post demo
        |
        v
Objective report
  waveform stats + spectrogram snapshots + provenance
        |
        v
Machine rejection gate
  silence / spikes / clipping / missing speech-band structure
        |
        v
Stage parity ladder
  Python vi -> Swift prefix -> DecoderPre -> hn-nsf/HAR -> GeneratorFromHar
        |
        v
Smallest fix at first divergent boundary
        |
        v
Human listen samples + objective gates
        |
        v
Bakeoff rerun
```

The recovery should be empirical and conservative: first make the broken output
measurable, then compare identical tensors, then patch the earliest bad boundary.

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Demo Provenance and Exact HAR-Post Replication

**Goal:** Recover how `outputs/decoder_har_post_demo.wav` was made and reproduce
it with one existing enumerated shape or bucket before using it as a comparator.

**Tasks:**

- [x] Search git history for the demo artifact and generation breadcrumbs:
      `git log --all --full-history -- outputs/decoder_har_post_demo.wav`,
      `git log -S decoder_har_post_demo --all`, and related searches for
      `decoder_har_post`, `demo.wav`, and HAR-post smoke commands.
- [x] Use `git show` on candidate commits to inspect old scripts, notes, command
      lines, text prompts, voice, speed, package names, and artifact hashes
      without mutating the current worktree.
- [x] Identify the enumerated shape or bucket that produced the demo. The likely
      first candidate is the `3s` HAR-post bucket because the demo is about
      `1.363s`, but the plan must follow the recovered provenance rather than
      assume that bucket.
- [x] Re-run the recovered command with the matching enumerated shape or bucket
      and write the reproduction to
      `outputs/audio-parity/replicated/decoder_har_post_demo_repro.wav`.
- [x] Compare the original and reproduction by file SHA256, WAV header, PCM
      SHA256, duration, RMS, peak, active fraction, zero-crossing rate, waveform
      plot, and spectrogram.
- [x] If byte-exact reproduction fails, compare after normalizing WAV container
      metadata and then compare intermediate tensors (`x_pre`, `ref_s`, `har`,
      and waveform) to locate the first remaining delta.
- [x] Record the exact recovered command, commit provenance, model artifact
      paths, artifact hashes, enumerated shape or bucket, text, voice, speed,
      and comparison result in `outputs/audio-parity/demo-provenance.json` and
      `outputs/audio-parity/demo-provenance.md`.

**Verification:** `outputs/audio-parity/demo-provenance.md` states whether the
reproduction is byte-exact. If not exact, it names the smallest observed delta
and explains whether the demo can still be used as a secondary comparator.

---

### Phase 1: Freeze Evidence and Produce Listen References

**Goal:** Give the user trustworthy samples immediately and preserve the failing
ones for comparison.

**Tasks:**

- [x] Create `outputs/audio-parity/` with a manifest containing git commit,
      dirty-tree status, machine info, voice, speed, text, and artifact paths.
- [x] Generate full PyTorch reference WAVs for the same short inputs used by
      `scripts/bakeoff_listen.py`.
- [x] Copy or link `outputs/decoder_har_post_demo.wav` into the report set with
      Phase 0 provenance attached. If Phase 0 could not reproduce it exactly,
      label it as a secondary comparator, not primary truth.
- [x] Snapshot current failing files from `outputs/bakeoff/listen/` into
      `outputs/audio-parity/failing-current/`.
- [x] Write `outputs/audio-parity/index.md` listing the reference and failing
      WAV paths for listening.

**Verification:** The user has at least one PyTorch reference WAV and one failing
Config F WAV for the same text, plus a manifest tying both to the repo state.

---

### Phase 2: Add Objective Audio Inspection

**Goal:** Turn "sounds like garbage" into repeatable numeric and visual evidence.

**Tasks:**

- [x] Add `scripts/audio_quality_probe.py` to report duration, sample rate, RMS,
      DC offset, peak, clipping fraction, active-sample fractions, zero-crossing
      rate, spectral centroid, voiced-band energy, and optional spectrogram PNGs.
- [x] Run the probe over PyTorch reference WAVs, `outputs/decoder_har_post_demo.wav`,
      and all files under `outputs/bakeoff/listen/`.
- [x] Inspect waveform and spectrogram outputs directly and classify each sample
      as `reject_without_listening`, `needs_listening`, or `reference_pass`.
- [x] Store reports under `outputs/audio-parity/reports/`.
- [x] Derive provisional speech-health thresholds from the reference set; do not
      hard-code thresholds from the broken samples.

**Verification:** The report clearly separates PyTorch or known HAR-post speech
from the current bakeoff outputs without using subjective listening alone, and
obvious non-speech files are rejected before the user is asked to listen.

---

### Phase 3: Build the Stage-Parity Ladder

**Goal:** Compare identical tensors across Python and Swift/Core ML boundaries.

**Tasks:**

- [x] Extend or add a Python capture script, likely
      `scripts/capture_audio_parity_tensors.py`, that dumps `tokens`, `ref_s`,
      duration output, alignment output, `asr`, `f0`, `n`, `x_pre`, `har`, and
      final waveform for one short input.
- [x] Extend `swift/Sources/KokoroBenchmark/main.swift` or add a debug subcommand
      to dump the same boundaries from `KokoroPipeline`.
- [x] Add `scripts/compare_audio_parity_tensors.py` with shape, dtype, max error,
      cosine similarity, and correlation checks.
- [x] Include the existing `scripts/validate_hnsf_swift.py generate` and Swift
      harmonic-source comparison in the ladder.

**Verification:** `uv run python scripts/run_audio_parity_ladder.py --input-key
3s` can show the first failing boundary between Python reference and the current
Swift + Core ML path for the short sample.

---

### Phase 4: Diagnose the First Divergence

**Goal:** Decide which subsystem is responsible before applying a fix.

**Tasks:**

- [x] Compare `HybridTTSPipeline.extract_vocoder_inputs()` output with Swift
      tokenization, duration, alignment, and `asr` construction.
- [x] Compare Python `F0Ntrain` outputs against Swift/Core ML `f0` and `n`.
- [x] Compare Python `DecoderPre` `x_pre` against Swift/Core ML `x_pre` with the
      same padded `asr`, `f0`, `n`, and `ref_s`.
- [x] Compare Python `build_decoder_har_post_inputs_np()` `har` against Swift
      `buildHar()` with the same padded F0 and hn-nsf weights.
- [x] Feed Python reference `x_pre`, `ref_s`, and `har` into Swift
      `GeneratorFromHar` Core ML and compare final waveform.

**Verification:** The investigation names exactly one earliest divergent boundary
or records a ranked list if two boundaries fail independently.

**Phase 4 Result:** `uv run python scripts/run_audio_parity_ladder.py --input-key
3s` reports `first_failing_boundary=har_source`: tokens, duration, alignment,
`asr`, `f0`, `n`, and `x_pre` pass shape checks with high correlation before
the Swift harmonic source diverges. A separate literal Swift generator isolation
check using `kokoro-bench --generator-input-dump outputs/audio-parity/tensors/python_3s`
reports `waveform` correlation about `0.929`, so the ranked diagnosis is: first
fix Swift `har_source`; then re-check the `GeneratorFromHar` package parity with
known-good Python `x_pre/ref_s/har`. The Phase 4 instrumentation also found that
raw-pointer reads of Core ML waveform `MLMultiArray` output can corrupt trimmed
debug/audio samples; tensor dumps now use stride-safe MLMultiArray access.

---

### Phase 5: Apply the Smallest Correctness Fix

**Goal:** Restore human speech with minimal architecture churn.

**Tasks:**

- [x] If Core ML waveform output uses non-contiguous `MLMultiArray` storage,
      replace raw pointer extraction in
      [swift/Sources/KokoroPipeline/KokoroPipeline.swift](../../swift/Sources/KokoroPipeline/KokoroPipeline.swift)
      and the benchmark WAV path with stride-safe `MLMultiArray` reads before
      trimming.
- [ ] If Swift hn-nsf or STFT diverges, fix
      [swift/Sources/KokoroPipeline/HarmonicSource.swift](../../swift/Sources/KokoroPipeline/HarmonicSource.swift)
      against PyTorch references and strengthen
      [scripts/validate_hnsf_swift.py](../../scripts/validate_hnsf_swift.py).
- [ ] If `DecoderPre` or `GeneratorFromHar` package geometry diverges, fix the
      wrapper or export path in [export_synth/wrappers.py](../../export_synth/wrappers.py)
      and [export_synth/convert.py](../../export_synth/convert.py), then
      re-export only the affected packages.
- [ ] If Swift input preparation diverges, fix
      [swift/Sources/KokoroPipeline/KokoroPipeline.swift](../../swift/Sources/KokoroPipeline/KokoroPipeline.swift)
      or [swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift](../../swift/Sources/KokoroPipeline/MLMultiArrayHelpers.swift)
      and add focused Swift tests.
- [ ] If identical known-good tensors still produce garbage from Core ML
      `GeneratorFromHar`, quarantine the current package and route through the
      last known-good HAR-post artifact while rebuilding the export.

**Verification:** The applied quality-destroying boundary now produces short and
medium samples that pass objective machine inspection and are ready for human
listening. Any remaining stage-parity divergence stays tracked as a release
risk before bakeoff reinstatement.

**Phase 5 Result:** The smallest confirmed corruption fix was the final waveform
read, not a package re-export. `GeneratorFromHar` can return a non-contiguous
Core ML `MLMultiArray`; reading it through `dataPointer` and then trimming by
linear index produced corrupted WAV output even when stride-safe tensor reads
showed high waveform correlation. The Swift runtime and benchmark WAV writer now
use stride-safe `floatValues(from:)` before trimming. Regenerated samples at
`outputs/audio-parity/fixed-attempts/stride_safe_3s.wav` and
`outputs/audio-parity/fixed-attempts/stride_safe_7s.wav` classify as
`needs_listening`, not `reject_without_listening`, with RMS, activity, and
zero-crossing rates back in the reference range. The `har_source` parity
divergence remains a residual risk for the final quality gate and the hardcore
audit loop.

---

### Phase 6: Replace Weak Listen Gates

**Goal:** Prevent future bakeoff winners from passing with non-human audio.

**Tasks:**

- [x] Update [scripts/bakeoff_listen.py](../../scripts/bakeoff_listen.py) to use
      thresholds derived in Phase 2 and to emit the full audio-quality report.
- [x] Add tests for silence, impulses, clipped output, and a known-good reference
      fixture where practical.
- [x] Add a manifest field that marks listen samples as `quality_pass: true` only
      when both duration and speech-health gates pass.
- [x] Add a manifest field that records the machine-inspection decision:
      `reject_without_listening`, `needs_listening`, or `reference_pass`.
- [x] Document that these gates are smoke tests, not replacements for human
      listening or tensor parity.

**Verification:** The current failing Config F files fail the new gate, while the
PyTorch and known-good HAR-post references pass.

**Phase 6 Result:** `scripts/bakeoff_listen.py` now derives its speech-health
gate from the Phase 2 reference WAVs, rebuilds the release `kokoro-bench` binary
when Swift sources are newer than the binary, annotates each listen metrics JSON
with `quality_pass`, `quality_decision`, `quality_reject_reasons`, and the full
`audio_quality` record, and writes a consolidated `quality/audio_quality_report.json`
plus markdown summary. Focused tests cover the listen manifest fields and
silence, impulse, clipped, and reference cases. The frozen failing Config F
samples under `outputs/audio-parity/failing-current/` classify as
`reject_without_listening`; a fresh short/medium listen-helper smoke under
`outputs/audio-parity/phase6-listen-smoke/` classifies both regenerated files as
`needs_listening`.

---

### Phase 7: Re-run Quality Proof and Bakeoff

**Goal:** Restore the benchmark only after audio quality is demonstrably sane.

**Tasks:**

- [x] Regenerate listen samples for short and medium inputs.
- [x] Have the user listen to the new samples before treating the fix as done.
- [x] Run `pytest` at the repo root and focused Swift tests under
      `swift/Tests/KokoroPipelineTests/`.
- [x] Run the `bakeoff` skill only after the samples pass listening and objective
      gates.
- [x] Update [README/Notes/debug-notes.md](../Notes/debug-notes.md) through
      `write-notes` with cause, fix, commands, and residual risks.

**Verification:** The user confirms at least short and medium outputs sound
human, objective gates pass, and the bakeoff results are regenerated from the
fixed path.

**Phase 7 Result:** All enumerated listen shapes were regenerated through
`scripts/bakeoff_listen.py --keys 3s,7s,15s,30s` with the quality gate active.
`outputs/bakeoff/listen/config_f_3s.wav`, `config_f_7s.wav`, `config_f_15s.wav`,
and `config_f_30s.wav` all have `quality_pass=true` and
`quality_decision=needs_listening`. The user confirmed the short and medium
samples sound human. The complete A/D/E/F bakeoff ran on the M2 Ultra with
`--iterations 5 --order-seed 0 --machine-id m2_ultra_v6`; all 80 records
completed with `status=ok`. After the audit tightened Config F to materialize
the trimmed waveform inside the timed path, Config F remained faster than
PyTorch CPU/MPS at every enumerated shape, was faster than Config A at 3s/7s,
and was slower than Config A at 15s/30s. Results are recorded in
[performance-notes.md](../Notes/performance-notes.md#bakeoff-v6-audio-fixed-config-f-on-m2-ultra).

## Success Criteria

### Hard Requirements (Must Pass)

- [x] At least one PyTorch reference WAV and one fixed Swift/Core ML WAV are
      available for the same text and voice.
- [x] `outputs/decoder_har_post_demo.wav` provenance is recovered from git
      history and exact reproduction has been attempted with an enumerated shape
      or bucket.
- [x] Direct waveform and spectrogram inspection is run before any user
      listening request.
- [x] The frozen failing `outputs/audio-parity/failing-current/config_f_*.wav`
      files fail the new quality gate.
- [x] The fixed output passes the confirmed waveform extraction boundary and
      residual stage-parity risks are recorded.
- [x] Short and medium fixed samples sound human to the user.
- [x] Performance bakeoff results are not regenerated or cited until quality
      gates pass.

### Definition of Done

- [x] Plan phases are checked off only after `phase-audit` verifies each phase.
- [x] `pytest` passes or any unrelated failure is documented with evidence.
- [x] Focused Swift tests pass for any touched Swift code.
- [x] `README/Notes/debug-notes.md` records the final root cause and fix.
- [x] Final bakeoff artifacts include `quality_pass: true` for listen samples.

## Open Questions

### Resolved

- **Q:** Should performance tuning continue before audio quality is fixed?
- **A:** No. Quality proof gates all further bakeoff claims.

- **Q:** Should the plan assume the current bug is the same
  `SourceModuleHnNSF` / `SineGen` Core ML failure from earlier notes?
- **A:** No. That failure is a strong prior and informs the CPU/Swift split, but
  the current Swift + Core ML winner still needs fresh stage parity.

### Unresolved

- **Q:** Is `outputs/decoder_har_post_demo.wav` exactly from the older
  `decoder_har_post_bucket_impl()` path or from another local experiment?
- **Options:** Confirm from artifact metadata if present, regenerate from the
  older path, or treat it as a useful but secondary comparator.

- **Q:** Can objective gates catch every bad speech sample?
- **Options:** Use gates as smoke tests, add mel/MCD/PESQ-style metrics if
  feasible, and keep human listening as a required final check.

## References

### Internal

- [Debug Notes](../Notes/debug-notes.md)
- [Kokoro generator rebuild notes](../kokoro-generator-rebuild.md)
- [Kokoro to Core ML conversion](../Kokoro-to-CoreML-conversion.md)
- [Core ML conversion guide](../coreml-conversion-guide.md)
- [Core ML compute-unit scheduling guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [Plan workflow skills guide](../Skills/plan-workflow-skills-guide.md)
- [Phase audit rubric](../Skills/phase-audit-rubric.md)
- [Kokoro bakeoff v2 plan](003-kokoro-bakeoff-plan.md)

### Local Artifacts

- `outputs/decoder_har_post_demo.wav`
- `outputs/bakeoff/listen/config_f_3s.wav`
- `outputs/bakeoff/listen/config_f_7s.wav`
- `outputs/bakeoff/listen/config_f_15s.wav`
- `outputs/bakeoff/listen/config_f_30s.wav`

## Degradation and Rollback

- **If Swift recovery remains blocked:** Use Python `build_decoder_har_post_inputs_np()`
  for hn-nsf/HAR generation and Core ML only for `GeneratorFromHar` until Swift
  parity is proven.
- **If current packages are corrupt:** Quarantine the package set and re-export
  from the last known-good wrapper path.
- **If objective thresholds overfit:** Keep them as smoke gates, lower their
  authority, and require tensor parity plus human listening before success.

## Monitoring and Observability

- `audio_quality.active_fraction_32` - rejects near-silent speech outputs.
- `audio_quality.rms_pcm` - catches implausibly low energy.
- `audio_quality.zero_crossing_rate` - catches spike-only or DC-heavy output.
- `audio_quality.voiced_band_energy_ratio` - catches output with no speech-band
  structure.
- `parity.correlation` and `parity.max_abs_error` per stage - locate the first
  divergent boundary.
