# Kokoro M1 HAR/STFT Contract Repair Guide

This guide ingests the first external Deep Research report for the Kokoro M1
HAR/STFT contract problem and records the repo-verified decision. Treat the raw
report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-har-stft-contract-repair-for-strict-core-ml-source-body-vocoder-path/2026-06-06T22-37-46-144Z/raw-report.md`

## Executive Summary

The useful part of the report is its focus: the remaining strict blocker is the
raw Nyquist phase convention reaching the trained generator body. That matches
the existing repo evidence in
[HAR/STFT phase contract bisection](../../Notes/har-stft-phase-contract.md).

The report's concrete branch-free Nyquist formula was tested and rejected:

```text
phase_nyq = (pi / 2) * ((mag_nyq - real_nyq) / (mag_nyq + 1e-7))
```

It is phase-equivalent modulo wrap, but the generator consumes raw phase. The
formula still chooses the wrong raw branch thousands of times and does not
recover natural-geometry waveform quality.

## Verified Repo Evidence

The existing source/HAR bisection remains authoritative:

- The source equation is solved.
- Recomputed HAR/STFT from exact dumped `har_source` is not solved.
- The standalone Core ML STFT subgraph can match PyTorch.
- The raw phase branch mismatch is isolated to Nyquist.
- Padded geometry plus dumped Nyquist can recover replacement quality, but it is
  speed-negative after crediting removable Swift STFT work.

The branch-free formula probe is recorded at:

- `outputs/external_bakeoff/nyquist_formula_candidate_probe.json`

Short-bucket results:

| Bucket | Formula Nyquist vs dumped | Natural waveform vs dump | Padded waveform vs dump |
| --- | ---: | ---: | ---: |
| `3s` | SNR `-1.49 dB`, `2548` raw branch errors | SNR `16.21 dB` | SNR `24.52 dB` |
| `7s` | SNR `-1.03 dB`, `5323` raw branch errors | SNR `15.41 dB` | SNR `25.90 dB` |
| `10s` | SNR `-1.08 dB`, `7687` raw branch errors | SNR `15.06 dB` | SNR `25.92 dB` |

Decision: reject the formula as a strict path.

## What To Do Next

Do not spend more time on direct scalar Nyquist formulas that only preserve
phase modulo `2*pi`. The trained generator needs the raw Swift branch convention
or a representation/model change that removes raw Nyquist sensitivity.

The next researchable paths are:

- analytically fold the raw Nyquist branch correction into the first
  `noise_convs` surface;
- train or calibrate a tiny adapter for a phase-wrap-invariant representation;
- replace the raw phase channel contract with `sin/cos` or real/imag features
  only if the first body surface is transformed or retrained;
- find a runtime-positive package boundary that makes padded quality recovery
  speed-positive by removing another Core ML call or enough body cost.

The companion
[HAR/STFT strict repair and distillation guide](Kokoro-HAR-STFT-strict-repair-distillation-guide.md)
captures the follow-up adapter path. Do not move that work to Core ML until a
PyTorch activation-level probe proves the adapter can recover the body contract.

## Avoid

| Avoid | Reason |
| --- | --- |
| More direct `atan2` / `atan_manual` / `atan_swift` variants | Existing probes show they do not reproduce the Swift raw branch convention. |
| Branch-free formulas that map Nyquist to `0..pi` only | They are modulo-correct but raw-branch wrong. |
| Natural compact HAR with dumped or formula Nyquist only | Still quality-failing. |
| Padded/Nyquist direct replacement | Quality improves, but timing is speed-negative unless another boundary/body cost is removed. |

## Related Documentation

- [HAR/STFT phase contract bisection](../../Notes/har-stft-phase-contract.md)
- [HAR/STFT strict repair and distillation](Kokoro-HAR-STFT-strict-repair-distillation-guide.md)
- [Restarted Kokoro guide triage](../../Notes/kokoro-restarted-guide-triage-2026-06-06.md)
- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Core ML vs MLX vocoder scheduling](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
