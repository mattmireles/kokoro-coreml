# Kokoro HAR/STFT Strict Repair And Distillation Deep Research Prompt

June 6, 2026

Use this as an external-research prompt for strict or near-strict repair of the
Kokoro HAR/STFT representation boundary. This is a fallback/complement to the
analytical HAR/STFT contract repair guide. Do not return generic distillation or
quality-evaluation advice.

## Context

We are optimizing a first-party Swift + Core ML Kokoro/CoPro-style TTS pipeline
for Apple devices. Runtime buckets are fixed: `3s`, `7s`, `10s`, `15s`, and
`30s`. Benchmark claims must use warmed inference only.

The sine/source equation is solved: `swift_like_seeded` matches dumped Swift
`har_source` at about `138-140 dB` SNR across all buckets. The unsolved piece is
the HAR/STFT representation consumed by the generator/body:

- recomputing HAR/STFT from exact dumped `har_source` remains near `8.1 dB`
  SNR with max abs around `6.28`;
- dumped Nyquist phase plus padded shipping HAR geometry repairs quality but
  loses the speed edge;
- compact/natural geometry is faster but quality-failing;
- raw phase branch mismatches are isolated to Nyquist-bin branch choices, while
  wrapped phase error is tiny;
- direct `atan` variants have not solved the contract.

The speed target is to beat `laishere/kokoro-coreml` on lower-end Apple Silicon
short/medium buckets without relying on cold compile/cache behavior.

## Primary Research Goal

Find strict-preserving or near-strict accepted repair strategies that let a
compact source/HAR representation feed the existing or lightly adapted vocoder
body without padded-geometry cost. Prefer analytical transformations, but
research lightweight retraining/distillation only where it can be bounded,
validated, and converted to Core ML safely.

## Questions To Answer

1. Can the first convolution or first few body layers be analytically folded to
   consume `[real, imag]`, `[mag, sin phase, cos phase]`, corrected Nyquist
   channels, or compact `x_source_*` features while preserving exact output?
2. If exact folding is impossible because nonlinear layers intervene, what is
   the smallest adapter or calibration layer that can repair the representation
   mismatch?
3. What loss functions best preserve perceived TTS quality and waveform
   similarity for source/HAR boundary repair: waveform L1/L2, multi-resolution
   STFT, mel loss, feature loss, phase loss, adversarial loss, or direct body
   activation matching?
4. What data volume is needed for a small repair pass across Kokoro speakers,
   durations, phoneme distributions, and F0/noise cases?
5. Can training be bucket-specific, speaker-independent, or style-conditioned
   without hurting paper comparability?
6. How should no-ASR listening acceptance be designed if strict waveform parity
   is impossible?
7. What objective gates are appropriate before listening: waveform correlation,
   SNR, max abs, mel cepstral distortion, PESQ/STOI caveats, F0 error, voiced
   region error, or ABX preference?
8. Can a repaired adapter be represented with Core ML-friendly ops that schedule
   well on M1/iPhone, or would it erase the speed win?
9. How do we avoid overfitting to a small validation set or hiding audible
   regressions in short buckets?
10. What exact first two experiments should be run in PyTorch before any Core ML
    conversion?

## Constraints

- Strict waveform parity is preferred.
- If quality-equivalent/non-strict is proposed, clearly label it and define the
  listening protocol required before paper use.
- Do not add another hot-path Core ML prediction boundary unless the expected
  speed win exceeds the boundary cost.
- Do not change runtime buckets.
- Do not compare cold compile/cache timings.

## Output Format

Start with an executive summary of whether analytical folding or learned repair
is more likely to succeed. Then provide:

- exact representation candidates;
- analytical folding recipes where possible;
- small-adapter/distillation recipes where analytical folding fails;
- PyTorch training/evaluation sketch;
- Core ML conversion recipe and expected op surface;
- validation thresholds and no-ASR listening protocol;
- speed-risk analysis for Apple Silicon;
- do-this / avoid-this tables;
- first two experiments with stop/go criteria;
- clearly separate evidence-backed recommendations from speculation.
