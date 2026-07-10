# Kokoro M1 Source/Body Deep Research Prompt

June 6, 2026

Use this as the handoff prompt for the next optimization pass. The goal is not
to find another plausible Core ML tweak. The goal is to close the remaining
real warmed-inference loss against laishere on Irvine M1 while preserving the
paper-facing comparison rules.

## Objective

Make first-party Config F the fastest warmed-inference Kokoro implementation on
Apple Silicon for the runtime buckets `3s`, `7s`, `10s`, `15s`, and `30s`.
Prioritize Irvine M1 losses in this order: `3s`, `7s`, `10s`, `15s`, then
`30s`.

Current real Irvine M1 losses:

| Bucket | Config F | laishere | Gap |
| --- | ---: | ---: | ---: |
| `3s` | `233.6 ms` | `195.0 ms` | `38.5 ms / 19.75%` |
| `7s` | `492.7 ms` | `444.2 ms` | `48.4 ms / 10.90%` |
| `10s` | `685.5 ms` | `644.9 ms` | `40.6 ms / 6.29%` |
| `15s` | `1014.9 ms` | `990.6 ms` | `24.3 ms / 2.46%` |

The HAR-post upsample rewrite and the `decoder_pre`/HnSF runtime overlap are
measured strict local wins and should be kept, but they are not enough alone.
The current post-overlap+rewrite budget is tracked in
`outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md`. If both
measured local effects transfer to Irvine, the remaining warmed profile target
still needs:

| Bucket | Projected Config F after overlap+rewrite | laishere profile | Extra strict save needed |
| --- | ---: | ---: | ---: |
| `3s` | `222.0 ms` | `195.0 ms` | `27.0 ms` |
| `7s` | `472.2 ms` | `444.2 ms` | `28.0 ms` |
| `10s` | `657.6 ms` | `644.9 ms` | `12.8 ms` |
| `15s` | `976.4 ms` | `990.6 ms` | `0.0 ms` |

The stricter paper-facing target is still open after overlap+rewrite:

| Bucket | Paper frontier best | Extra strict save needed |
| --- | ---: | ---: |
| `3s` | `176.3 ms` | `45.7 ms` |
| `7s` | `394.6 ms` | `77.6 ms` |
| `10s` | `593.9 ms` | `63.7 ms` |
| `15s` | `912.0 ms` | `64.4 ms` |

Do not use cold compile/cache timings. Every claim must use warmed inference.

## Research Question

Design an M1 MLProgram source/STFT/vocoder body that preserves the current
Swift HAR/source contract, keeps strict waveform parity or earns explicit
no-ASR listening acceptance, and shifts the expensive convolution/add/mul/
instance-norm body work into a laishere-like mixed CPU/Neural Engine plan
without the existing split-boundary synchronization penalty or a warmed `3s`
regression.

## Evidence To Respect

Authoritative frontier and target files:

- `outputs/external_bakeoff/goal_frontier_status.md`
- `outputs/external_bakeoff/lower_end_mac_win_gate.md`
- `outputs/external_bakeoff/irvine_paper_frontier_path.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/irvine_next_targets.md`
- `outputs/external_bakeoff/candidate_frontier_matrix.md`
- `outputs/external_bakeoff/remote_host_quiet_latest.md`
- `README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`
- `README/Kokoro-M1-vocoder-boundary-research-brief.md`
- `README/Kokoro-M1-graph-surface-target.md`
- `README/Kokoro-M1-kernel-partition-deep-research-prompt.md`
- `README/Notes/performance-notes.md`

The apparent MLX win was a comparison bug: cold compile/cache behavior,
padding, `.all` behavior, and HAR overhead were mixed into non-equivalent
timings. Under corrected warmed full-duration inference, MLX has `0` Mac wins
against Config F, Config F beats MLX on `12` Mac rows, and MLX has `3` Mac
`3s` broadcast-shape failures. The detailed explanation is now generated at
`outputs/external_bakeoff/mlx_speed_explanation.md`. The remaining strict
competitor is laishere on Irvine M1, not MLX.

The current strict CPU+NE body split is the main warning sign. It already gets
laishere-like Neural Engine preferred-op counts, but it is slower. Therefore
the target is not "more NE placement." The target is a runtime-positive graph
boundary and synchronization pattern.
For a focused investigation of that partition problem, hand off
`README/Kokoro-M1-kernel-partition-deep-research-prompt.md`.

The graph-surface target is also specific: first-party `GeneratorFromHar` has
manual AdaIN lowering with `88` reductions and `96` tiles, while laishere's
vocoder has native `instance_norm`, no tiles, and LUT-backed weight
decompression. A useful strict candidate must change that surface without
adding a new hot-path package boundary.

The fully rewritten local probe surface
(`cos-Snake + native-IN + broadcast AdaIN + fp16 inputs + iOS17 + upsample
rewrite`) is contract-compatible as a Swift HAR-post overlay, but was rejected
as a production replacement because it does not beat the simpler production
rewrite overall. Do not treat "more visible surface matching" as a sufficient
hypothesis.

## New Critical Blocker

The source equation itself is solved, but the Swift HAR/STFT representation is
not. The five-bucket source-variant refresh says
`source_equation_is_solved=true` and `recomputed_stft_har_is_solved=false`:
`swift_like_seeded` matches dumped Swift `har_source` at SNR `138.15-140.33 dB`
across all buckets, but recomputing HAR/STFT from even the exact dumped source
stays near SNR `8.11-8.23 dB`. Dumped Nyquist phase plus padded shipping HAR
geometry repairs strict source-boundary parity, but that padded strict path
loses the speed edge.

The highest-value external guide now is narrower than "source/body": explain
how to represent the current Swift STFT/HAR contract, or a quality-equivalent
replacement, inside one Core ML package without reintroducing padded-geometry
cost or an extra prediction call. Use
`README/Kokoro-M1-HAR-STFT-contract-deep-research-prompt.md` for that handoff.

## Do Not Repeat

Treat these as measured rejections unless the proposal changes the boundary or
runtime synchronization mechanism:

- `.all` toggles.
- Palette-only changes.
- Final-waveform int4/int8 quantization.
- fp16 input-only changes for the current static body.
- iOS17/spec8 metadata-only rebuilds.
- More exact decoder+vocoder multi-package splits.
- fp16 body inputs for the exact decoder+vocoder split.
- Plain or native-InstanceNorm style specialization.
- Native-InstanceNorm style specialization plus HAR trim.
- Broad generator noise/stage splits that add Core ML calls.
- Standalone strict HAR-source fused paths that preserve padded geometry but
  lose the speed edge.

## Promising Hypotheses

### 1. Single-Package Body Reshaping

Keep the current `GeneratorFromHar` runtime call boundary but reshape the graph
surface inside the package. The candidate should change operator fusion,
layout, or partitioning enough that M1 avoids repeated CPU/GPU/NE sync while
keeping the Swift input/output contract stable.

Acceptance:

- strict waveform gate passes;
- after the overlap + upsample rewrite baseline, Irvine M1 `3s` improves by at
  least another `27.0 ms` warmed median, or combines with measured upstream
  savings to close the full remaining profile gap;
- M2 Studio does not materially regress.

### 2. In-Package HAR Source Consumption

Find a smaller Swift-produced tensor that preserves source quality but avoids
the HAR-post input cost, then consume it inside one body+tail package. Do not
add an extra Core ML call on the `3s` hot path unless the measured savings are
larger than the call/sync overhead.

Acceptance:

- strict parity or no-ASR listening acceptance;
- same five bucket contract;
- warmed Irvine M1 win against laishere on at least one real-loss bucket.

### 3. Fast F0/Source Quality Recovery

The speed-positive candidates are quality-failing F0/source simplifications.
They are useful only if quality can be recovered without losing the speed
signal, or if human listening explicitly accepts the exact generated WAVs.

Latest lower-end evidence:

- M2 Air `3s_natural_asr_cos_rsqrt`: candidate stack `106.7 ms` versus baseline
  stack `123.9 ms`, projected full `~128.5 ms`, waveform corr `0.813986`,
  SNR `5.08 dB`.
- M2 Air
  `3s_padded_native_in_ios17_nopal_cos_rsqrt_cos_rsqrt_native_in`: candidate
  stack `113.4 ms` versus baseline stack `123.5 ms`, projected full
  `~135.6 ms`, waveform corr `0.931815`, SNR `9.19 dB`.
- Both would beat the M2 Air paper-facing laishere `3s` row (`142.0 ms`) if
  accepted, but both are non-strict and require human listening.
- Dedicated no-ASR review pack:
  `outputs/f0_source_listening/m2_air_3s_source_body/README.md`.
- Decision CSV:
  `outputs/f0_source_listening/m2_air_3s_source_body/f0_source_listening_decisions.csv`
  is intentionally blank until a human listens. Do not treat these as accepted
  production rows before that CSV validates.

Latest paper-frontier correction:

- M2 Air `3s` has two paper-frontier wins if human listening accepts the
  source/body rows.
- Irvine source/body candidates can beat several newer warmed profile rows, but
  none beats the stricter paper-facing frontier by itself.
- Combining source/body with the measured HAR-post rewrite and runtime overlap
  should close Irvine `10s` and `15s` if listening accepts the saved source
  candidates, but still leaves `3s/7s` short (`27.0 ms` and `29.0 ms`
  additional saves needed).
- For the next external research pass, use
  `README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`.

Acceptance:

- strict waveform parity, or accepted rows in
  `outputs/f0_source_listening/irvine_exact_speed_branch/f0_source_listening_decisions.csv`;
- no Whisper/ASR proxy metrics;
- exact candidate WAVs and warmed timing reports are linked from the review
  dashboard.

## Required Output For A Candidate

Every candidate must leave a durable report containing:

- exact git SHA and command line;
- exported or reused model package paths;
- deployment target, precision, input dtypes, and compute units;
- warmup count, warm iteration count, cold latency, and warm median;
- waveform metrics against the same reference tensor dump or emitted WAV;
- `MLComputePlan` preferred-device counts and cost weights;
- local M2 Studio timing before any Irvine timing;
- Irvine timing only when the host is quiet enough for publishable data.
- the current `outputs/external_bakeoff/remote_host_quiet_latest.md` result for
  any lower-end Mac run;
- an updated `outputs/external_bakeoff/candidate_frontier_matrix.md` row if
  the candidate is promoted, rejected, or remains quality-fail speed evidence.

## Current External Blocks

The iPhone 12 Pro is paired and the Config F manual runner is installed, but
launch is denied while the physical phone is locked. Irvine M1 is not quiet
enough for publishable timing while `mediaanalysisd` or Spotlight consumes CPU.
