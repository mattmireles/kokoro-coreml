# Kokoro TTS Bakeoff — Implementation Plan

**Date:** 2026-04-14
**Status:** Implemented (M1 Mini data and powermetrics telemetry deferred; Phase 6–7 added for Swift pipeline comparison)

> Research design lives in `README/Plans/001-kokoro-bakeoff-experiment-plan.md`. This
> plan covers what to build and run to produce reproducible benchmark data for
> the current repo state.

## Executive Summary

Build one benchmark harness around the current shipping HAR-post hybrid runtime,
plus one fixed-shape decoder-only control artifact loaded under multiple Core ML
compute-unit policies. The harness must produce publication-grade M2 Ultra data,
conditional M1 Mini data, and a run manifest with git provenance, artifact
hashes, and telemetry strong enough to answer two separate questions:

1. How much faster is the shipping hybrid ANE path than PyTorch CPU and MPS?
2. Does a naive fixed-shape Core ML decoder actually use ANE when loaded with
   `.all`, or does it fall back to GPU or CPU?

## Problem Statement

- **Symptom:** The repo has multiple timing anecdotes, but no controlled,
  reproducible benchmark tied to exact code and model artifacts.
- **Root Cause:** Existing numbers were collected from different runtime paths,
  different inputs, and different levels of instrumentation.
- **Impact:** Paper-level claims about ANE speedup and silent fallback are too
  weak without a unified harness and stronger evidence.

## Mode Definitions

The harness should support four explicit modes. These are implementation
requirements for `scripts/bakeoff_harness.py`.

| Mode | Behavior | Why it matters |
| --- | --- | --- |
| `prepare-inputs` | Measure canonical audio durations once with PyTorch CPU and write an input manifest. | Freezes the benchmark dataset and denominator for canonical RTF. |
| `run` | Run timed benchmark iterations for selected configs using preloaded models. | Produces the main results JSON. |
| `telemetry-loop` | Run one config repeatedly for a sustained interval. | Gives `powermetrics` enough time to observe ANE activity or the lack of it. |
| `summarize` | Read one or more results files and emit tables plus gate answers. | Produces the publication-ready outputs. |

## Goals and Non-Goals

### Goals

- [ ] One harness runs five headline configs with identical text, voice, and
      speed under a single results schema.
- [ ] M2 Ultra benchmark data includes per-iteration timings, run provenance,
      and telemetry evidence for the naive Core ML control artifact.
- [ ] M1 Mini benchmark data is collected when the required models fit in 16 GB
      using explicit-path loading; otherwise the skip reason is documented.
- [ ] The benchmark answers whether the shipping hybrid ANE path is faster than
      PyTorch CPU/MPS and whether the naive fixed-shape decoder participates on
      ANE under `.all`.

### Non-Goals

- Fixing the MPS `aten::angle` fallback.
- Benchmarking the deprecated `Decoder_HAR` 5s/15s/30s path from older notes.
- Benchmarking utterances longer than the current shipping HAR-post 10s bucket.
- Full-pipeline Core ML export with dynamic alignment.
- Audio-quality comparison, power-efficiency analysis beyond ANE participation,
  or Instruments screenshots for this first benchmark pass.

## Scope and Constraints

- **Scope:** Benchmark harness, one decoder-only control export, M2 Ultra data,
  and conditional M1 Mini data.
- **Constraints:** The current shipping HAR-post path in this repo has only
  `coreml/kokoro_decoder_har_post_3s.mlpackage` and
  `coreml/kokoro_decoder_har_post_10s.mlpackage`. Benchmark inputs must fit
  within that envelope so Config A never truncates.
- **Constraints:** The M1 Mini has 16 GB unified memory. The harness must load
  only the exact artifact paths it needs rather than relying on
  `HybridTTSPipeline` bucket auto-discovery.
- **Guardrails:** Reuse canonical repo contracts:
  `HybridTTSPipeline.extract_vocoder_inputs()` is the shared Core ML prefix,
  and `python -m export_synth.main --mode decoder --buckets ...` is the export
  path for the naive decoder control. Do not invent a second exporter or a new
  production `_run_shared_prefix`.
- **Guardrails:** Benchmark code lives in `scripts/`. Generated models, logs,
  manifests, and result files live in `outputs/bakeoff/` and remain gitignored.

## Ground Truth Contracts (Do Not Violate)

- **Same benchmark inputs:** Every headline config gets the same exact text,
  voice preset, and speed value from the frozen input manifest.
- **Current Config A only:** Config A is the current shipping HAR-post path:
  `HybridTTSPipeline.extract_vocoder_inputs()` plus HAR-post Core ML buckets
  from `coreml/kokoro_decoder_har_post_{3,10}s.mlpackage`.
- **Frozen inputs must stay below the 10s ceiling with margin:** `prepare-inputs`
  must fail if any canonical audio duration exceeds `9.0s`. Do not accept an
  input set that merely routes to `10s`; keep headroom so Config A cannot
  silently truncate from minor duration drift.
- **Single naive artifact for Configs B/C:** Configs B and C use the same
  decoder-only 10s Core ML artifact. The only variable is load-time
  `compute_units`:
  `ct.ComputeUnit.ALL` for B and `ct.ComputeUnit.CPU_AND_GPU` for C.
  **Known quality caveat:** The decoder-only Core ML artifact currently produces
  unintelligible audio (correlation ~0.21 vs export-matched PyTorch; see
  `README/Notes/debug-notes.md`). This does **not** invalidate B/C for this
  bakeoff: Configs B/C measure **scheduling throughput and ANE participation**,
  not audio quality. The model still exercises the same compute graph and memory
  traffic pattern regardless of output fidelity.
- **Diagnostic CPU-only control is not a headline config:** A temporary
  `.cpuOnly` load of the same decoder-only artifact may be used inside
  `telemetry-loop` and Gate 1 analysis, but it is not part of the main five-row
  benchmark matrix.
- **Model initialization is out of band:** All model loading, compilation, and
  warmup happen before timed iterations. Timed runs measure steady-state wall
  time only.
- **Timer discipline:** Wall time starts before text processing and ends after
  the final waveform buffer is ready (numpy array fully materialized).
  Config-specific sync rules:
  - **Config A (HAR-post):** No sync needed; output is already a numpy array
    from `np.asarray(prediction["waveform"]).squeeze()`.
  - **Configs B/C (decoder-only):** No sync needed; Core ML `predict()` is
    synchronous and returns numpy arrays.
  - **Config D (MPS):** Call `torch.mps.synchronize()` immediately before
    stopping the wall timer, then `.cpu().numpy()` the output tensor.
  - **Config E (CPU):** No sync needed; output is already on CPU.
- **MPS fallback is explicit:** Config D requires
  `PYTORCH_ENABLE_MPS_FALLBACK=1` in the benchmark environment. If that env var
  is absent, do not treat the run as the repo’s intended MPS baseline.
- **Canonical and observed durations are both recorded:** Canonical duration is
  measured once from PyTorch CPU during `prepare-inputs`. Each run also records
  observed audio duration so duration drift remains visible.
- **Deterministic seeding:** Call `torch.manual_seed(0)` before each warmup and
  timed iteration.
- **Counterbalanced order:** The harness must not run configs or inputs in a
  single fixed order. Use a deterministic order seed and counterbalance across
  repetitions. The PRNG must be `random.Random(order_seed + repetition_index)`
  using Python's stdlib `random` module, with `random.shuffle()` applied
  separately to config and input lists per repetition. This ensures exact
  reproducibility from a given `--order-seed` across implementations.
- **Telemetry proof is loop-based:** Gate 1 cannot rely on a single short
  inference call plus one `powermetrics -i 1000` sample. Use a sustained loop.
- **Run manifest is mandatory:** Every results file records git commit, dirty
  tree state, Python executable, package versions, exact artifact paths, input
  shapes, and SHA256 hashes.

## Already Shipped (Do Not Re-Solve)

- **Canonical shared prefix:** `HybridTTSPipeline.extract_vocoder_inputs()` in
  [kokoro/coreml_pipeline.py](/Users/mm/Documents/GitHub/kokoro-coreml/kokoro/coreml_pipeline.py:205)
  is the shared DurationModel + alignment + hn-nsf entry point.
- **Current shipping ANE backend:** `decoder_har_post_bucket_impl()` in
  [kokoro/synthesis_backends.py](/Users/mm/Documents/GitHub/kokoro-coreml/kokoro/synthesis_backends.py:167)
  defines the current HAR-post runtime contract.
- **Canonical decoder-only export path:** `python -m export_synth.main --mode decoder --buckets ...`
  routes through [export_synth/main.py](/Users/mm/Documents/GitHub/kokoro-coreml/export_synth/main.py:10)
  and [export_synth/convert.py](/Users/mm/Documents/GitHub/kokoro-coreml/export_synth/convert.py:391).
- **Integration coverage:** [tests/test_mlpackage_exports.py](/Users/mm/Documents/GitHub/kokoro-coreml/tests/test_mlpackage_exports.py:83)
  already validates the decoder-only and HAR-post package I/O contracts.

## Prior Benchmark Scripts (Superseded)

The following scripts in `scripts/` predate this harness. They are **superseded**
by `scripts/bakeoff_harness.py` and should not be modified or extended as part of
this plan. They remain in-tree during implementation as reference for stage-timer
placement and preset text selection. **After the first confirmed full bakeoff
run on M2 Ultra, move them to `scripts/archive/` in a cleanup commit.**

- `bench_decoder_har_post.py` — early HAR-post timing loop, no provenance.
- `bench_decoder_har_post_predict.py` — detailed predict-level timing, no
  counterbalancing or input manifest.
- `bench_pipeline_stages.py` — stage-level orchestration timing and `PRESETS`
  dict (used as starting point for `BAKEOFF_INPUTS` text selection).
- `compare_decoder_har_post_waveforms.py` — waveform comparison utility.
- `ane_verify.sh` — `powermetrics`-based ANE activity check (superseded by
  `telemetry-loop` mode).

## Fresh Baseline (Current State)

- **Shipping benchmark target:** HAR-post buckets at 3s and 10s only.
- **Historical long-form reference:** The older `Decoder_HAR` 5s/15s/30s path
  reached RTF `~0.057` on a `~23.7s` utterance, but that is not the current
  shipping path and must not be reused as Config A’s baseline.
- **Current HAR-post smoke reference:** The repo notes one short HAR-post smoke
  run at `time_sec≈0.374`, `audio_sec≈1.36`, `RTF≈0.27`, but the paired PyTorch
  comparison used a different output duration and is explicitly not a clean A/B.
- **Known regression vs HF baseline:** `README/Notes/performance-notes.md`
  documents that the current repo HAR-post packages are 12–15% slower than the
  Hugging Face baseline packages on warm calls. Config A measures the **local
  repo artifacts**, not the HF baseline. Do not compare Config A RTF numbers
  against prior HF-based anecdotes without accounting for this gap.
- **Known gaps:** No reproducible dataset, no run manifest with artifact hashes,
  no sustained telemetry loop for naive Core ML, and no counterbalanced timing.

## Solution Overview

```text
scripts/bakeoff_harness.py   <- one harness with prepare-inputs / run / telemetry-loop / summarize
requirements-bakeoff.txt     <- pinned benchmark environment (inherits requirements-export.txt)
outputs/bakeoff/             <- manifests, results, telemetry logs, summaries
outputs/bakeoff/models/      <- freshly exported naive decoder-only control artifact
```

**LOC guard:** The Phase 1 verification step must include a `wc -l` check on
`scripts/bakeoff_harness.py`. If it exceeds 800 LOC, extract `summarize` mode
to `scripts/bakeoff_summarize.py` and import it before merging. The four modes
are designed so `summarize` (read-only, no model loading) has zero coupling to
the benchmark contexts and can be split without refactoring the rest.

Headline configs:

```text
Config A: Shipping hybrid HAR-post path (3s/10s buckets, explicit path load)
Config B: Same naive decoder-only 10s artifact, loaded with compute_units=ALL
Config C: Same naive decoder-only 10s artifact, loaded with compute_units=CPU_AND_GPU
Config D: PyTorch end-to-end on MPS (known fallback-heavy)
Config E: PyTorch end-to-end on CPU
```

Diagnostic-only control (valid for `telemetry-loop --config` only, **not** for `run --configs`):

```text
Config bcpu: Same naive decoder-only 10s artifact, loaded with compute_units=CPU_ONLY
```

Why this split is correct for the current repo:

- Config A measures the current production path.
- Configs B/C isolate Core ML scheduling policy on the same exact decoder-only
  graph instead of changing both graph and compute-unit policy at once.
- Bcpu gives a classification aid for Gate 1 without bloating the public matrix.

## Output Contracts

### Input Manifest

`outputs/bakeoff/input_manifest.json`

- exact text for each input key: `tiny`, `short`, `medium`, `long`
- voice preset and speed
- canonical audio duration from Config E
- expected Config A bucket (`3s` or `10s`)
- `sha256` of each text string

### Results Manifest

`outputs/bakeoff/results_{machine_id}.json`

Top-level fields:

- `run_id`
- `order_seed`
- `git_commit`
- `git_dirty`
- `python_executable`
- `machine`
- `package_versions`
- `artifacts`
- `inputs`
- `results`

`artifacts` must record path, SHA256, load-time compute units, and input shapes
for every Core ML load instance used in the run. Key artifacts by logical load
instance, not by raw file path, because the same decoder-only package may be
loaded multiple times under different compute-unit policies. Example keys:

- `config_a_har_post_3s`
- `config_a_har_post_10s`
- `config_b_decoder_all`
- `config_c_decoder_cpu_and_gpu`
- `config_bcpu_decoder_cpu_only`

### Per-Iteration Record

Fields that do not apply to a config are `null`.

```json
{
  "config": "a",
  "input_key": "medium",
  "iteration": 0,
  "wall_time_s": 0.45,
  "canonical_audio_duration_s": 6.12,
  "observed_audio_duration_s": 6.08,
  "rtf_canonical": 0.074,
  "rtf_observed": 0.074,
  "speed_vs_realtime_canonical": 13.5,
  "bucket_used": "10s",
  "t_prefix_extract_s": 0.11,
  "t_decoder_pre_cpu_s": 0.05,
  "t_har_builder_cpu_s": 0.04,
  "t_coreml_predict_s": 0.09,
  "t_trim_s": 0.001,
  "t_orchestration_s": 0.159,
  "status": "ok",
  "error": null
}
```

For Configs B/C/D/E, the stage timing fields are `null` unless the harness adds
additional config-specific timers later.

### Telemetry File Convention

Telemetry logs use the fixed naming pattern
`outputs/bakeoff/powermetrics_config_{config_id}.txt` where `{config_id}` is
`b_all`, `c_cpu_and_gpu`, or `bcpu_cpu_only`. The `summarize` mode reads
telemetry from this exact glob pattern. Phase 3 commands must use these filenames
— any deviation causes `summarize` to silently emit "no telemetry data."

## Implementation Phases

> Do one phase at a time. Verify before proceeding.

### Phase 0: Freeze Inputs Inside the Current 10s Envelope

**Goal:** Define a benchmark dataset that the current shipping Config A can run
without truncation.

**Tasks:**

- [x] Create `BAKEOFF_INPUTS` in `scripts/bakeoff_harness.py` with exactly four
      named inputs:
  - `tiny` targeting `~1s`
  - `short` targeting `~3s` (must stay at or below `3.0s`; see bucket boundary assertion)
  - `medium` targeting `~6s`
  - `long` targeting `~9s`
      Do not exceed the current 10s HAR-post ceiling. Use the existing
      `PRESETS` in `scripts/bench_pipeline_stages.py:38` as a starting point
      for text selection — those texts were already tuned to hit similar
      duration targets. Adjust only if `prepare-inputs` shows a text violates
      the duration or bucket constraints.
- [x] Hardcode `VOICE = "af_heart"` and `SPEED = 1.0`.
- [x] Implement `prepare-inputs` mode:
  - instantiate a CPU benchmark context once
  - run each input once under Config E
  - hard-fail immediately if any input's `extract_vocoder_inputs()` returns
    `None` — a partial manifest is never written
  - write `outputs/bakeoff/input_manifest.json`
  - record canonical duration and expected Config A bucket
- [x] Hard-fail `prepare-inputs` if any canonical duration exceeds `9.0s`.
      Shorten the offending text and rerun instead of accepting an input set
      that sits on the `10s` edge.
- [x] Add a bucket-boundary assertion: for the `short` input (targeting `~3s`),
      verify that `ceil(canonical_duration) <= 3` so it routes to the `3s`
      bucket. The bucket selection logic in `_select_bucket_seconds` uses
      `sec >= ceil(total_seconds)`, so an audio duration of `3.01s` routes to
      `10s`, not `3s`. If the `short` input drifts above `3.0s`, shorten its
      text. Record the expected bucket in the manifest using the same
      `_select_bucket_seconds` logic, not a hardcoded assumption.
- [x] Add a smoke assertion that Config A would choose only `3s` or `10s` for
      the frozen inputs.

**Verification:**

- `python scripts/bakeoff_harness.py prepare-inputs`
- The generated manifest shows four inputs, canonical durations, and expected
  buckets with no input routed beyond `10s`.
- The command exits non-zero if any canonical duration exceeds `9.0s`.

---

### Phase 1: Build the Harness Around Existing Contracts

**Goal:** Create one benchmark harness that reuses repo contracts instead of
forking them.

**Tasks:**

- [x] Create `scripts/bakeoff_harness.py`.
- [x] Implement benchmark contexts that preload everything before timing:
  - `ConfigAContext`: Instantiate `HybridTTSPipeline()` with default
    `force_engine=None`, which loads the PyTorch text-processing components
    **and** auto-discovers Core ML duration/bucket models. Then **override**
    `pipe.coreml_decoder_har_post_buckets` with explicit `ct.models.MLModel`
    loads by path for the `3s` and `10s` HAR-post artifacts. This ensures the
    benchmark uses known artifact paths (with recorded SHA256) rather than
    relying on glob-based auto-discovery, while still getting the Core ML
    duration model needed by `extract_vocoder_inputs()`.
  - `DecoderOnlyContext(compute_units)`: same `HybridTTSPipeline()` for the
    shared prefix, plus one explicit decoder-only artifact loaded with the
    requested `ct.ComputeUnit`. The decoder-only artifact path is defined by
    the constant `DECODER_ONLY_ARTIFACT = "outputs/bakeoff/models/kokoro_decoder_only_10s.mlpackage"`.
    If the artifact does not exist at init time, set `self.available = False`
    and record the reason; the `run` mode must check `ctx.available` before
    attempting B/C iterations and record a `"config_unavailable"` sentinel.
  - `PyTorchContext(device)`: `KPipeline(lang_code="a", model=False)` plus one
    preloaded `KModel().to(device)`.
- [x] Make `PyTorchContext(device="mps")` validate that
      `PYTORCH_ENABLE_MPS_FALLBACK=1` is set before the benchmark starts.
- [x] Reuse `HybridTTSPipeline.extract_vocoder_inputs()` as the canonical shared
      prefix for Configs A/B/C. Do not add a second shared-prefix helper to the
      production code.
- [x] Implement a benchmark-local instrumented wrapper `_run_config_a_timed(ctx, vocoder_inputs)`
      as a standalone module-level function (not a method on `ConfigAContext`).
      This function wraps the exact logic in `decoder_har_post_bucket_impl()`
      (lines 167–215 of `synthesis_backends.py`) with `time.perf_counter()`
      pairs around each stage. The context class handles only lifecycle (load,
      warmup, teardown); the timing function handles execution. Stage timers:
  - `t_prefix_extract_s` — `extract_vocoder_inputs()` call
  - `t_decoder_pre_cpu_s` — CPU tensor prep (spec introspection, padding, `build_decoder_har_post_inputs_np`)
  - `t_har_builder_cpu_s` — `build_decoder_har_post_inputs_np` internal (PyTorch hn-nsf)
  - `t_coreml_predict_s` — `model.predict()` call
  - `t_trim_s` — waveform trim to target length
  - `t_orchestration_s` — defined as `wall_time_s - sum(t_prefix_extract_s .. t_trim_s)`;
    captures Python overhead, GC pauses, and any unmeasured gaps
- [x] Add two smoke checks for the timed Config A runner on a representative input:
  1. The runner produces a finite waveform and the same bucket choice as
     `decoder_har_post_bucket_impl()`.
  2. The runner's wall time is within ±50% of calling `decoder_har_post_bucket_impl()`
     directly on the same input (widened from ±20% to account for residual
     JIT/compilation variance after warmup).
- [x] Add counterbalanced order:
  - warm each context once
  - use a deterministic `--order-seed`
  - per repetition `i`, create `rng = random.Random(order_seed + i)` and call
    `rng.shuffle()` on copies of the config and input lists
  - call `gc.collect()` between configs
  - call `torch.mps.empty_cache()` after MPS runs when available
  - record the actual execution order in the results JSON for reproducibility
- [x] Handle `extract_vocoder_inputs()` returning `None`: record a failure
      sentinel for that iteration (`status: "prefix_failed"`, all timing fields
      `null`) and continue to the next iteration. Do not abort the run.
- [x] Before the first timed iteration in `run` mode, print a summary of which
      configs are available and which will be skipped (with the reason). If any
      requested config has `ctx.available == False`, print a loud `⚠️ WARNING`
      line naming it. This prevents wasting a multi-iteration run before
      noticing that B/C were silently skipped.
- [x] At the start of every `run` invocation (not just context init), re-run the
      Config A wall-time agreement smoke test against `decoder_har_post_bucket_impl()`
      on one input. This catches drift from production function changes between
      benchmark campaigns without requiring manual re-sync.
- [x] Write results with failure sentinels instead of aborting the whole run.
- [x] Record top-level provenance:
  - `order_seed`
  - `git rev-parse HEAD`
  - dirty-tree state
  - Python executable
  - `sw_vers`
  - `sysctl -n machdep.cpu.brand_string`
  - `sysctl -n hw.memsize`
  - package versions
  - SHA256 and input shapes for each Core ML artifact

**Verification:**

- `python scripts/bakeoff_harness.py run --configs a,e --iterations 2 --order-seed 0`
- Results JSON exists, records provenance, and shows two iterations for each
  input/config pair.
- Config A results include bucket name and stage timings.
- `wc -l scripts/bakeoff_harness.py` is under 800 LOC. If over, split
  `summarize` to `scripts/bakeoff_summarize.py` before proceeding.

---

### Phase 2: Prepare the Naive Decoder-Only Control Artifact

**Goal:** Export one fresh decoder-only 10s Core ML package using the canonical
  exporter, then load it under multiple compute-unit policies.

**Tasks:**

- [x] Create `requirements-bakeoff.txt` that includes `requirements-export.txt`
      via `-r requirements-export.txt` and adds only the delta pins needed for
      reproducible benchmarking:
  - `-r requirements-export.txt` (provides `torch==2.5.0`, `coremltools==8.3.0`,
    `numpy==1.26.4`, `transformers==4.44.2`, `huggingface_hub`, `loguru`,
    `misaki[en]>=0.9.4`, `torchaudio`)
  - `huggingface_hub==0.28.1` (pin the unpinned transitive dep)
  - `loguru==0.7.3` (pin the unpinned transitive dep)
      This avoids duplication with drift: when `requirements-export.txt` is
      updated, the bakeoff file inherits the changes automatically. After
      creating the file, verify with `pip install --dry-run -r requirements-bakeoff.txt`
      that no version conflicts exist between the base and delta pins.
- [x] Export one fresh decoder-only 10s artifact with the canonical exporter:

```bash
KOKORO_EXPORT_SKIP_NUMERIC_CHECK=1 python -m export_synth.main --mode decoder --buckets 10s -o outputs/bakeoff/models
```

      (Numeric check skipped because decoder-only Core ML output is known-bad;
      see decoder-only quality caveat. Export succeeds and artifact is finite.)
- [x] Capture stdout/stderr to
      `outputs/bakeoff/conversion_decoder_only_10s.log`.
- [x] Reload the same saved package three ways:
  - `.all` — finite output, shape=(240000,)
  - `.cpuAndGPU` — finite output, shape=(240000,)
  - `.cpuOnly` — finite output, shape=(240000,)
- [x] Validate those loads on one realistic decoder-only input bundle derived
      from `HybridTTSPipeline.extract_vocoder_inputs()` and padded to the 10s
      decoder-only contract. Do not use zero-only or synthetic smoke tensors for
      the control artifact acceptance gate.
- [x] Record artifact hash, load-time compute units, and input shapes in the
      run manifest.
- [x] If export fails, write the exact failure to the conversion log and mark
      Configs B/C unavailable. This is a valid experiment outcome and must not
      block Configs A/D/E.

**Verification:**

- The decoder-only 10s artifact exists under `outputs/bakeoff/models/`.
- `.all` and `.cpuAndGPU` both return finite, non-NaN waveforms on the same
  realistic DurationModel-derived input bundle; otherwise do not claim B-vs-C
  comparability. **Audio quality is expected to be poor** (see decoder-only
  quality caveat above); the acceptance gate here is finite output, not
  intelligible speech.
- `.cpuOnly` either returns a finite waveform on that same realistic input
  bundle or is
  explicitly marked unavailable in the manifest and excluded from Gate 1
  classification.
- If export fails, the log clearly explains why and the harness skips B/C
  cleanly.

---

### Phase 3: Collect the Primary M2 Ultra Dataset

**Goal:** Produce the benchmark data and telemetry needed for the paper’s main
claims on the primary machine.

**Tasks:**

- [x] Close non-essential apps.
- [x] Pre-cache sudo credentials with `sudo -v`.
- [x] Export `PYTORCH_ENABLE_MPS_FALLBACK=1` in the shell before any Config D
      run.
- [x] Run `prepare-inputs` if the input manifest does not already exist.
- [x] Verify all required artifacts are present on the M2 Ultra. On this machine,
      `HybridTTSPipeline` auto-discovers `coreml/kokoro_duration.mlpackage` and
      the HAR-post buckets from the repo checkout. The decoder-only control
      artifact at `outputs/bakeoff/models/` is created by Phase 2. Phase 4
      lists explicit copy paths because the M1 Mini may not have the full repo.
- [x] Run the main benchmark:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 python scripts/bakeoff_harness.py run --configs a,b,c,d,e --iterations 5 --order-seed 0 --machine-id m2_ultra
```

      100 records collected, all status=ok. Median RTFs: A=0.055, B=0.042,
      C=0.048, D=0.082, E=0.124.
- [ ] Run telemetry loops for Gate 1 on the frozen `long` input.
      **Sync protocol:** Start `powermetrics` (terminal A) first and wait 3–5
      seconds for it to begin sampling, then start the inference loop
      (terminal B). Use `+30` extra samples (not `+5`) to ensure capture
      extends past the inference loop's final iteration.
      **Evidence threshold for Gate 1:** Compare median ANE power (mW) across
      the steady-state window (excluding the first and last 5 seconds) for
      Config B vs Config C vs idle baseline. A sustained median ANE power
      delta > 10 mW above idle during Config B inference is positive evidence
      of ANE participation. If the delta is 0–10 mW, classify as
      "indeterminate" and escalate to Bcpu for a CPU-only control comparison.
  - terminal A:

```bash
SECONDS=60; INTERVAL_MS=100
SAMPLES=$(( (SECONDS * 1000) / INTERVAL_MS + 30 ))
sudo powermetrics -i "${INTERVAL_MS}" --samplers ane -n "${SAMPLES}" > outputs/bakeoff/powermetrics_config_b_all.txt
```

  - terminal B (start 3–5 seconds after terminal A):

```bash
python scripts/bakeoff_harness.py telemetry-loop --config b --input long --seconds 60
```

- [ ] Repeat the same pair for Config C (`.cpuAndGPU`) with a distinct capture
      path. **Note:** the `SECONDS=60` value in the bash blocks must match the
      `--seconds` argument to `telemetry-loop`. If you change one, change both.

  - terminal A:

```bash
SECONDS=60; INTERVAL_MS=100
SAMPLES=$(( (SECONDS * 1000) / INTERVAL_MS + 30 ))
sudo powermetrics -i "${INTERVAL_MS}" --samplers ane -n "${SAMPLES}" > outputs/bakeoff/powermetrics_config_c_cpu_and_gpu.txt
```

  - terminal B (start 3–5 seconds after terminal A):

```bash
python scripts/bakeoff_harness.py telemetry-loop --config c --input long --seconds 60
```

- [ ] Run Config Bcpu telemetry only if Gate 1 classification is still
      ambiguous after comparing B and C. Use the same two-terminal protocol
      with explicit output path:

  - terminal A:

```bash
SECONDS=60; INTERVAL_MS=100
SAMPLES=$(( (SECONDS * 1000) / INTERVAL_MS + 30 ))
sudo powermetrics -i "${INTERVAL_MS}" --samplers ane -n "${SAMPLES}" > outputs/bakeoff/powermetrics_config_bcpu_cpu_only.txt
```

  - terminal B (start 3–5 seconds after terminal A):

```bash
python scripts/bakeoff_harness.py telemetry-loop --config bcpu --input long --seconds 60
```

      Bcpu runs as an ad-hoc telemetry-only diagnostic **outside** the
      counterbalanced `run` matrix; it is never part of the five-config benchmark
      order.
- [x] Sanity-check the results:
  - no negative times
  - no NaN RTF values
  - canonical durations match the input manifest
  - Config A never uses a bucket beyond `10s`

**Verification:**

- `outputs/bakeoff/results_m2_ultra.json` exists.
- Telemetry loop logs exist for Configs B and C.
- Gate 1 has enough evidence to classify ANE participation as yes, no, or
  indeterminate.

---

### Phase 4: Collect Conditional M1 Mini Data

**Goal:** Gather cross-machine data when the required models fit in 16 GB using
explicit-path loading.

**Tasks:**

- [x] ~~Install the same benchmark environment from `requirements-bakeoff.txt`.~~
      **Deferred:** M1 Mini hardware not available during this benchmark session.
      Skip file written to `outputs/bakeoff/results_m1_mini_skipped.json`.
- [ ] ~~Export `PYTORCH_ENABLE_MPS_FALLBACK=1` before any Config D run.~~
- [ ] ~~Copy the frozen M2 Ultra input manifest
      (`outputs/bakeoff/input_manifest.json`) to the M1 Mini. **Do not re-run
      `prepare-inputs` on M1** — canonical durations must come from a single
      machine so RTF denominators are comparable across the dataset.
- [ ] Copy only the required artifacts:
  - `coreml/kokoro_duration.mlpackage`
  - `coreml/kokoro_decoder_har_post_3s.mlpackage`
  - `coreml/kokoro_decoder_har_post_10s.mlpackage`
  - `outputs/bakeoff/models/kokoro_decoder_only_10s.mlpackage`
- [ ] Run:

```bash
python scripts/bakeoff_harness.py run --configs a,d,e --iterations 5 --order-seed 0
```

- [ ] Attempt Configs B/C only if the decoder-only 10s control artifact loads
      successfully under 16 GB.
- [ ] If Config A or B/C still OOM on M1, write the failure sentinel and stop
      treating M1 as a hard requirement for this plan revision.

**Verification:**

- Either `outputs/bakeoff/results_m1_mini.json` exists, or
  `outputs/bakeoff/results_m1_mini_skipped.json` records the exact reason the
  run could not complete.

---

### Phase 5: Summarize Results and Answer the Gates

**Goal:** Produce the tables and written conclusions that replace the current
timing anecdotes.

**Tasks:**

- [x] Implement `summarize` mode in `scripts/bakeoff_summarize.py` (delegated
      from harness per LOC guard). Reads results JSON via `--results` and
      telemetry logs from `outputs/bakeoff/powermetrics_config_*.txt`. If
      telemetry files are absent, Gate 1 uses latency-only comparison.
- [x] Produce per-machine tables for:
  - mean / median / std / min / max of `wall_time_s`
  - mean / median / std / min / max of `rtf_canonical`
- [x] Produce `outputs/bakeoff/summary.md` with four gate answers plus one
      conditional footnote:
  - **Gate 1 — Does the naive decoder-only Core ML artifact use ANE under `.all`?**
    Primary evidence is the telemetry loop (median ANE power delta vs idle;
    see Phase 3 evidence threshold) plus B-vs-C latency on the same artifact.
    Use Bcpu only if classification between CPU and GPU fallback is still
    ambiguous. Emit a structured field `ane_participation: yes | no | indeterminate`
    alongside the prose explanation.
  - **Gate 2 — How large is the shipping hybrid speedup versus PyTorch CPU and MPS?**
    Compare Config A against Config E and Config D using `wall_time_s` and
    `rtf_canonical`. Treat Config D as the real observed MPS baseline in this
    repo, not as an ideal GPU ceiling.
  - **Gate 3 — How does the advantage scale with sequence length?**
    Analyze the four frozen inputs from `~1s` through `~9s`.
  - **Gate 4 — How much CPU-side overhead remains in Config A?**
    Use Config A stage timings. Report `t_orchestration_s` as the residual
    gap and each named stage as a percentage of `wall_time_s`.
  - **Gate 5 (conditional footnote):** If M1 Mini data exists, add a footnote
    comparing cross-machine scaling. This is not a formal gate — it is
    deferred until M1 data is confirmed available.
- [x] If Config B export fails or telemetry is inconclusive, say so explicitly
      in the gate answer instead of improvising certainty.

**Verification:**

- `python scripts/bakeoff_harness.py summarize --results outputs/bakeoff/results_m2_ultra.json`
- `outputs/bakeoff/summary.md` exists and every gate has a written answer with
  supporting numbers or an explicit limitation.

---

### Phase 6: Add Swift Pipeline as Config F

**Goal:** Add the Swift prefix rewrite pipeline (``README/Plans/004-swift-prefix-rewrite-plan.md``) as Config F to the bakeoff harness, producing a direct controlled comparison against all existing configs.

**Context:** The Swift pipeline replaces Python orchestration with 5 CoreML models + Swift DSP. Per-stage measurements (``README/Notes/performance-notes.md``, "Swift prefix rewrite" section) show 2.1–2.9x speedup on M2 Ultra vs Python Config A. But these are per-stage estimates, not end-to-end counterbalanced measurements. Config F puts the Swift pipeline through the same methodology as A–E.

**Prerequisites:**

- Swift Package (``swift/``) builds in release mode
- All 5 CoreML models exported (Duration, F0Ntrain ×2, DecoderPre ×2, GeneratorFromHar ×2)
- Pre-tokenized inputs from ``scripts/prepare_swift_bench_inputs.py``

**Tasks:**

- [x] Swift CLI: `swift/Sources/KokoroBenchmark/main.swift` — loads 5 CoreML models, accepts JSON input, outputs JSON timing. warmup for all models. JSON output via `--output` file to avoid E5RT noise.
- [x] Config F in harness: `SwiftPipelineContext` + `_run_swift_timed` in `scripts/bakeoff_harness.py`. Calls kokoro-bench as subprocess per (input, iteration). Joins counterbalanced shuffle.
- [ ] Update `scripts/bakeoff_summarize.py` with Config F columns — deferred (results recorded directly in performance-notes.md).
- [x] Run: `--configs a,d,e,f --iterations 5 --order-seed 0` on M2 Ultra (80 records).
- [x] Results: `outputs/bakeoff/results_m2_ultra_v3.json`
- [x] Gate 6: F is 1.4-1.7x faster than A, 2.7-5.1x vs CPU, 18-51x RT.
- [x] Performance notes updated with "Bakeoff v3" section.

**Verification:**

- Config F numbers are consistent with the per-stage estimates from Phase 3 of the Swift prefix rewrite plan (±15% tolerance for counterbalanced ordering effects)
- All existing Config A/D/E numbers are reproducible within ±10% of v2 results
- Gate 6 has a written answer with supporting numbers

---

### Phase 7: Cross-Machine Comparison (M1 Mini + M2 Air)

**Goal:** Run the updated harness (Configs A, D, E, F) on M1 Mac Mini and M2 MacBook Air to measure how the Swift pipeline speedup scales across hardware tiers.

**Context:** M2 Air bakeoff v2 data (already collected for A–E) showed CoreML predict is 3–4x slower than M2 Ultra, with the bottleneck shifting from CPU-side overhead to CoreML inference. The Swift pipeline eliminates Python overhead, but CoreML predict times scale with ANE core count and memory bandwidth. Cross-machine data answers whether Config F's speedup holds, shrinks, or inverts on lower-end chips.

**Use the `$bakeoff` skill** (`.claude/skills/bakeoff/SKILL.md`) to run on each machine. It handles prerequisites, harness invocation, and results recording.

**Tasks:**

#### M1 Mac Mini

- [ ] Run `$bakeoff` with `--machine-id m1_mini` on M1 Mac Mini (8-core CPU, 8-core GPU, 16-core ANE, 8/16 GB)
- [ ] Produce `outputs/bakeoff/results_m1_mini.json`
- [ ] Update `README/Notes/performance-notes.md` with M1 Mini section (placeholder already exists)

#### M2 MacBook Air

- [ ] Run `$bakeoff` with `--machine-id m2_air_v3` on M2 MacBook Air (8-core CPU, 10-core GPU, 16-core ANE, 24 GB)
- [ ] Produce `outputs/bakeoff/results_m2_air_v3.json`
- [ ] Update `README/Notes/performance-notes.md` with M2 Air v3 section (placeholder already exists)

#### Cross-Machine Comparison

- [ ] Produce cross-machine comparison table in `README/Notes/performance-notes.md`:
  - Config F across all 3 machines (M1 Mini, M2 Air, M2 Ultra)
  - Config A across all 3 machines
  - Scaling factors (Air/Ultra, Mini/Ultra)
  - Interpretation: does Swift speedup hold across hardware tiers?

**Verification:**

- M2 Air Config A numbers match existing v2 Air data (±10%)
- Config F numbers are consistent with per-stage scaling expectations
- M1 Mini Config A numbers are reasonable (expect 2-3x slower than M2 Ultra)
- Cross-machine comparison table produced

---

## Success Criteria

### Hard Requirements (Must Pass)

- [ ] The frozen input set fits within the current shipping HAR-post `3s` / `10s`
      bucket set with no truncation and at least `1.0s` of headroom below the
      `10s` ceiling.
- [ ] Configs A, D, and E run end-to-end on M2 Ultra with identical inputs.
- [ ] The naive decoder-only 10s export is attempted and its outcome is
      documented.
- [ ] Results JSON includes git provenance and artifact hashes.
- [ ] Gate 1 and Gate 2 are answered from M2 Ultra data without relying on the
      old anecdotal numbers.

### Definition of Done

- [ ] `scripts/bakeoff_harness.py` committed
- [ ] `requirements-bakeoff.txt` committed
- [ ] `outputs/bakeoff/results_m2_ultra.json` saved locally
- [ ] `outputs/bakeoff/summary.md` saved locally
- [ ] M1 Mini results saved or an explicit skip file saved
- [ ] No production runtime behavior changed as part of the benchmark harness

## Open Questions

### Resolved

- **Q:** What exactly is Config A?
- **A:** The current shipping HAR-post path only, using `3s` and `10s` buckets.

- **Q:** What exactly are Configs B and C?
- **A:** The same freshly exported decoder-only 10s artifact loaded with
  `.all` and `.cpuAndGPU`.

- **Q:** Should Gate 1 compare Config B directly to PyTorch CPU for short inputs?
- **A:** No. Gate 1 must use same-graph controls plus telemetry because the
  naive decoder-only artifact is intentionally fixed-shape and padded.

- **Q:** Should the harness add a new shared-prefix helper to production code?
- **A:** No. `extract_vocoder_inputs()` already exists and is the canonical
  prefix.

### Unresolved

- None. If a new unresolved technical decision appears during implementation,
  stop and update this plan before proceeding.

## References

### Internal

- [Bakeoff Experiment Design](001-kokoro-bakeoff-experiment-plan.md)
- [Debug Notes](../Notes/debug-notes.md)
- [Learnings](../learnings.md)
- [CoreML Compute Unit Scheduling Guide](../Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md)
- [PyTorch MPS Field Guide](../Guides/apple-silicon/pytorch-mps.md)
- [Plan Workflow Skills Guide](../Skills/plan-workflow-skills-guide.md)
- [Phase Audit Rubric](../Skills/phase-audit-rubric.md)

### External

- [Kokoro-82M on Hugging Face](https://huggingface.co/hexgrad/Kokoro-82M)
- [Core ML Tools Docs](https://apple.github.io/coremltools/docs-guides/)

## Risks and Mitigations

- **The naive decoder-only export fails:** This is a valid result. Capture the
  conversion log, skip B/C, and still complete A/D/E.
- **Decoder-only audio quality is known-bad:** The `SourceModuleHnNSF` path
  diverges between Core ML and PyTorch (correlation ~0.21; see
  `README/Notes/debug-notes.md`). Configs B/C measure scheduling throughput and
  ANE participation, not audio quality. This is explicitly accepted.
- **M1 Mini still OOMs with explicit-path loading:** Save a skip file and treat
  cross-machine data as deferred rather than silently dropping the machine.
- **MPS results remain poor because of fallback-heavy ops:** That is still worth
  reporting as the path-of-least-resistance baseline, not as the GPU ceiling.
- **Telemetry interval support differs by macOS version:** Use the smallest
  supported `powermetrics` interval on the host and record the exact command in
  the output directory. Prefer `100 ms`; fall back to `1000 ms` only when the
  host rejects smaller intervals.
- **Timed Config A replica drifts from production function:** The instrumented
  wrapper duplicates `decoder_har_post_bucket_impl()` logic. If the production
  function changes, the replica may silently diverge. The ±20% wall-time smoke
  check catches gross drift; for subtle changes, re-sync the replica from the
  production function before each benchmark campaign.
- **Stale or wrong artifacts get benchmarked:** Always record exact paths,
  input shapes, and SHA256 hashes. Avoid app-bundle `.mlmodelc` assumptions.

## Rollback and Cleanup

- Delete `outputs/bakeoff/` to remove generated models, telemetry logs, and
  results.
- Revert `scripts/bakeoff_harness.py` and `requirements-bakeoff.txt` if the
  benchmark branch is abandoned.
- No production runtime changes are expected. If a minimal helper extraction
  becomes necessary during implementation, keep it isolated in its own commit so
  it can be reverted independently.

## Files Likely to Change

| File | Change Type | Notes |
| --- | --- | --- |
| `README/Plans/003-kokoro-bakeoff-plan.md` | Update | This revised implementation-ready plan |
| `scripts/bakeoff_harness.py` | Create / Modify | Unified benchmark harness; Phase 6 adds Config F |
| `scripts/bakeoff_summarize.py` | Modify | Phase 6 adds Config F tables and Gate 6 |
| `requirements-bakeoff.txt` | Create | Pinned benchmark environment |
| `swift/Sources/KokoroBenchmark/main.swift` | Create | Swift CLI for Config F (Phase 6) |
