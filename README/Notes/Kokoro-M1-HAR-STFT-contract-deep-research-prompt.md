# Kokoro M1 HAR/STFT Contract Deep Research Prompt

June 6, 2026

Use this as the next external-research prompt. The task is narrow: solve the
remaining strict source/HAR representation problem that blocks a faster
source/body Core ML path on Irvine M1. Do not return generic Core ML or MLX
advice.

## Context

We are optimizing a first-party Swift + Core ML Kokoro/CoPro-style TTS pipeline
for Apple devices. Runtime buckets are fixed: `3s`, `7s`, `10s`, `15s`, and
`30s`. All benchmark claims must use warmed inference only; cold compile/cache
time is disallowed.

Corrected warmed evidence says MLX is not the current blocker. Config F beats
or ties MLX on validated Mac rows. The remaining strict competitor is
`laishere/kokoro-coreml` on Irvine M1 short/medium buckets.

Current strict candidates already kept:

- `decoder_pre`/HnSF runtime overlap: strict, hash-identical, local all-bucket
  M2 Studio speedup.
- HAR-post upsample ConvT rewrite: strict-like package parity/correlation,
  local generator speedup.

After overlap + rewrite projections, Irvine M1 profile gaps still open:

| Bucket | Projected Config F | laishere profile | Extra strict save needed |
| --- | ---: | ---: | ---: |
| `3s` | `222.0 ms` | `195.0 ms` | `27.0 ms` |
| `7s` | `472.2 ms` | `444.2 ms` | `28.0 ms` |
| `10s` | `657.6 ms` | `644.9 ms` | `12.8 ms` |
| `15s` | `976.4 ms` | `990.6 ms` | `0.0 ms` |

Paper-facing gaps are stricter and still open:

| Bucket | Paper frontier best | Extra strict save needed after overlap+rewrite |
| --- | ---: | ---: |
| `3s` | `176.3 ms` | `45.7 ms` |
| `7s` | `394.6 ms` | `77.6 ms` |
| `10s` | `593.9 ms` | `63.7 ms` |
| `15s` | `912.0 ms` | `64.4 ms` |

## Current Blocker

The sine/source equation itself is solved. The five-bucket source-variant
refresh reports `source_equation_is_solved=true`: `swift_like_seeded` matches
dumped Swift `har_source` at SNR `138.15-140.33 dB` across all buckets.

The unsolved part is the HAR/STFT representation:

- `recomputed_stft_har_is_solved=false`.
- Recomputing HAR/STFT from even the exact dumped `har_source` stays near SNR
  `8.11-8.23 dB` with max abs around `6.28`.
- Dumped Nyquist phase plus padded shipping HAR geometry repairs replacement
  quality versus the current generator in measured strict fused reports, but
  the padded path loses the speed edge.
- Natural compact geometry remains quality-failing.

Additional branch bisection:

- `README/Notes/har-stft-phase-contract.md` records the current phase-contract
  bisection.
- Standalone Core ML STFT matches PyTorch almost exactly, so this is not a Core
  ML conversion bug.
- Raw phase branch mismatches are isolated to the Nyquist bin. On the 3s dump,
  `2331` samples differ by a `2*pi` branch choice while wrapped phase error is
  tiny.
- `atan_swift` fp32 made raw phase worse (`-0.89 dB` SNR), so do not spend time
  on more direct `atan2`/`atan_manual`/`atan_swift` branch variants unless the
  proposal changes the representation consumed by the generator.
- Strict padded/Nyquist source-boundary paths are speed-negative in the measured
  probes. A direct Swift HnSF boundary timing run measured removable STFT work
  at only `0.518/1.293/1.738/2.495/5.001 ms` for `3s/7s/10s/15s/30s`.
  After crediting that removed STFT work, strict fused source candidates still
  have no net win on measured buckets: `+0.051 ms` 3s, `+1.326 ms` 7s,
  `+2.231 ms` 10s, `+14.977 ms` 30s.

This means future work should not spend time on another sine-source equation
variant. The missing strict contract is the Swift HAR/STFT representation, or a
quality-equivalent replacement that does not reintroduce padded-geometry cost.

## Primary Research Goal

Find a Core ML-friendly representation of the current Swift HAR/STFT contract
that enables a faster source/body path without adding a new hot-path Core ML
prediction call.

The ideal result is one fixed-shape per-bucket package that consumes a small
Swift-produced source representation and emits waveform or pre-tail features
with strict waveform parity. If strict parity is impossible, define the exact
no-ASR listening review required for a quality-equivalent claim.

## Questions To Answer

1. Why does recomputing HAR/STFT from exact dumped `har_source` fail while
   dumped Nyquist phase plus padded shipping geometry repairs parity?
2. Which details are load-bearing: center padding mode, Hann window definition,
   DFT basis sign, magnitude epsilon, `atan2` convention, phase wrapping,
   Nyquist-bin representation, DC/Nyquist packing, float32 vs float64, or Core
   ML op lowering?
3. Is the current `[magnitude, phase]` representation the wrong contract for
   Core ML? Would `[real, imag]`, `[magnitude, sin(phase), cos(phase)]`, or a
   wrapped/normalized phase representation preserve quality and schedule
   better?
3a. Can the first noise-conv layer be analytically weight-folded so a
    `sin/cos` or corrected Nyquist representation is equivalent to raw phase
    without retraining?
3b. Can Nyquist branch corrections be represented as a cheap Swift side input
    without reintroducing the full padded HAR geometry or an extra Core ML
    prediction boundary?
4. Can a fixed-bucket Core ML package compute the 20-point STFT exactly enough
   using Conv1d/Conv2d constants, reshape/transpose-free layout, and fp32
   precision where needed?
5. Can the package avoid padded shipping HAR geometry while preserving strict
   output, or is padded geometry mathematically load-bearing for this model?
6. Is it better to pass `har_source`, `har_source + nyquist_phase`, `mag + phase`
   with corrected Nyquist, `real + imag`, or precomputed `x_source_0/x_source_1`
   into the body?
7. Can source/STFT/noise/body/tail be kept in one package boundary without the
   split-boundary synchronization penalty already measured?
8. How should deployment target, compute precision, input dtypes, and compute
   units be chosen for this package on M1, M2 Air, M2 Studio, and iPhone?
9. What `MLComputePlan` signature would indicate a runtime-positive partition,
   not merely more Neural Engine preferred ops?
10. What local M2 Studio proof should be required before quiet Irvine M1 timing?

## Do Not Repeat

Treat these as measured dead ends unless the proposal changes the HAR/STFT
contract or removes a hot boundary:

- More `.all` or compute-unit toggles.
- More broad decoder+vocoder or noise/body multi-package splits.
- More sine-source equation variants without changing STFT/HAR representation.
- Padded strict HAR-source fused paths that recover quality but lose speed.
- Native InstanceNorm/broadcast/cos/fp16/palette surface matching by itself.
- RangeDim/flexible generator inputs.
- Final-waveform int4/int8 quantization.
- Style specialization.

## Required Implementation Plan

A useful answer must include:

- exact proposed package boundary and tensor names;
- per-bucket static shapes for `3s`, `7s`, `10s`, `15s`, `30s`;
- whether the boundary consumes `har_source`, `nyquist_phase`, real/imag STFT,
  mag/phase, or `x_source_*`;
- PyTorch/coremltools wrapper sketch with concrete ops;
- deployment target, precision, and compute-unit recommendation;
- expected MIL op surface and expected `MLComputePlan` signature;
- parity gate: waveform corr/SNR/max-abs thresholds against existing Swift
  dumps, plus which tensors should be compared before final waveform;
- warmed local M2 Studio benchmark protocol;
- quiet Irvine M1 promotion protocol;
- risk list and the first two experiments to run.

## Repo Evidence To Read

- `README/Kokoro-M1-source-body-deep-research-prompt.md`
- `README/Kokoro-M1-kernel-partition-deep-research-prompt.md`
- `README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`
- `README/Kokoro-M1-vocoder-boundary-research-brief.md`
- `README/Kokoro-M1-graph-surface-target.md`
- `README/Notes/performance-notes.md`
- `outputs/external_bakeoff/candidate_frontier_matrix.md`
- `outputs/external_bakeoff/lower_end_mac_win_attempts.md`
- `outputs/external_bakeoff/overlap_rewrite_candidate_impact.md`
- `outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md`
- `outputs/f0_source_variants/summary_3s_7s_10s_15s_30s.md`
- `outputs/nyquist_phase_contribution/summary.md`
- `README/Notes/har-stft-phase-contract.md`

## Output Format

Start with an executive summary of the most likely HAR/STFT contract bug or
representation mismatch. Then provide:

- evidence-backed diagnosis;
- exact recommended representation;
- do-this / avoid-this tables;
- Core ML conversion recipe;
- parity validation plan;
- warmed benchmark plan;
- fallback options if strict parity is impossible;
- clearly marked speculative ideas.
