# Kokoro HAR/STFT Strict Repair and Distillation Guide

This guide ingests the restarted external report on strict HAR/STFT repair and
lightweight distillation for the Core ML source/body vocoder path. Treat the raw
report as research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-har-stft-strict-repair-and-lightweight-distillation-for-core-ml-source-bo/2026-06-06T23-29-40-702Z/raw-report.md`

## Executive Summary

The first HAR/STFT report produced a concrete branch-free Nyquist formula; repo
testing rejected it as strict because the generator consumes the raw phase
branch, not just modulo-equivalent phase. The restarted report is useful because
it moves away from scalar formulas and toward representation repair or a tiny
learned adapter.

This is a parity experiment, not the next immediate speed win.

## Repair Strategy

Use this order:

1. Reproduce PyTorch generator activations from the current Swift HAR/STFT
   contract and dumped tensors.
2. Train or calibrate a tiny adapter from a branch-stable representation such as
   `[real, imag]`, `[mag, sin(phase), cos(phase)]`, or a compact equivalent.
3. Freeze the existing generator body during the first probe.
4. Gate on activation match before Core ML conversion.
5. Convert only a simple surface, preferably `Conv1d(kernel_size=1)` plus basic
   elementwise ops, if the PyTorch probe passes.

## Acceptance Gates

| Gate | Required before promotion |
| --- | --- |
| Activation parity | First affected generator activations match the dumped baseline closely enough to explain waveform recovery. |
| Waveform parity | Objective metrics recover the current strict fused-package output. |
| Runtime contract | The adapter stays inside one runtime-positive boundary. |
| Core ML surface | The exported adapter is static-shape and placement-inspectable. |
| Benchmark policy | Warmed-only timing, no Whisper ASR gate. |

## Avoid

| Avoid | Reason |
| --- | --- |
| More direct Nyquist scalar formulas | The tested formula was modulo-correct but raw-branch wrong. |
| Training a large replacement vocoder first | It changes the research question and delays strict parity evidence. |
| Converting to Core ML before PyTorch activation proof | Core ML debugging will hide whether the math or the runtime failed. |
| Using ASR as the quality gate | The current project decision is to skip Whisper ASR for this pass. |

## Related Documentation

- [Kokoro M1 HAR/STFT contract repair guide](Kokoro-M1-HAR-STFT-contract-repair-guide.md)
- [Kokoro M1 source/body Core ML guide](Kokoro-M1-source-body-coreml-guide.md)
- [HAR/STFT phase contract bisection](../../Notes/har-stft-phase-contract.md)
- [Strict source/HAR representation repair prompt](../../Notes/Kokoro-strict-source-HAR-representation-repair-deep-research-prompt.md)
- [Restarted Kokoro guide triage](../../Notes/kokoro-restarted-guide-triage-2026-06-06.md)
