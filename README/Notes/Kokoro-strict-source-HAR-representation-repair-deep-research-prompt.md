# Kokoro Strict Source/HAR Representation Repair Deep Research Prompt

Use this prompt with an external deep-research agent. The agent has no project
context beyond what is written here.

## Objective

Create an advanced implementation guide for making a Kokoro/CoPro-style Core ML
TTS pipeline faster on Apple Silicon by replacing the current strict
source/HAR/STFT contract with a cheaper representation that preserves waveform
quality and does not add a hot-path Core ML prediction boundary.

We are comparing against MLX, Soniqo, and laishere Core ML implementations on
Apple Silicon. We benchmark warmed inference only, excluding Core ML
compile/cache time. Runtime buckets are fixed: `3s`, `7s`, `10s`, `15s`, and
`30s`. We skip Whisper/ASR as a gate; quality decisions must be objective audio
metrics plus human listening when needed.

The immediate target is lower-end Macs, especially Irvine M1. After current
strict runtime improvements, Irvine still needs large source/body wins:

| Bucket | Projected Config F after overlap+rewrite | laishere profile | Extra strict save needed |
| --- | ---: | ---: | ---: |
| `3s` | `222.0 ms` | `195.0 ms` | `27.0 ms` |
| `7s` | `472.2 ms` | `444.2 ms` | `28.0 ms` |
| `10s` | `657.6 ms` | `644.9 ms` | `12.8 ms` |
| `15s` | `976.4 ms` | `990.6 ms` | `0.0 ms` |

The stricter paper frontier remains harder:

| Bucket | Paper frontier best | Extra save needed after overlap+rewrite |
| --- | ---: | ---: |
| `3s` | `176.3 ms` | `45.7 ms` |
| `7s` | `394.6 ms` | `77.6 ms` |
| `10s` | `593.9 ms` | `63.7 ms` |
| `15s` | `912.0 ms` | `64.4 ms` |

## Current Evidence

The Swift-like HnSF source equation is solved:

- source minimum SNR versus reference: `138.15 dB`;
- source equation should not be reopened without new evidence.

The recomputed HAR/STFT contract is not solved:

- standalone Core ML STFT subgraph matches PyTorch: magnitude SNR `156.10 dB`,
  phase SNR `152.00 dB`;
- PyTorch magnitude matches Swift dump: SNR `128.13 dB`;
- raw phase branch mismatch remains, isolated to the Nyquist bin;
- direct `atan2`, `atan_manual`, `atan_swift`, and branch-free Nyquist formulas
  are rejected as strict replacements because the trained generator consumes raw
  phase, not phase modulo `2*pi`.

Known strict source-boundary attempts:

| Candidate | Quality | Speed signal | Decision |
| --- | --- | --- | --- |
| natural `har_source -> fused generator` | quality fail, about `16-17 dB` SNR | speed-positive vs generator-only | reject for strict paper claim |
| `har_source + dumped Nyquist + padded HAR` fused generator | replacement quality: SNR about `48-50 dB` | after Swift STFT credit, still no net win: `+0.051 ms` 3s, `+1.326 ms` 7s, `+2.231 ms` 10s, `+14.977 ms` 30s | reject direct replacement |
| `har_source + dumped Nyquist + padded HAR` source/noise split | quality good: corr `0.999986975`, SNR `46.25 dB` | slower than decoder-pre+generator: `34.4 ms` vs `30.4 ms` 3s | reject |

Body-only counterfactual is promising if source/noise tensors become cheap:

| Machine | Fused generator | Body only | Source/noise | Full split | Body-only save | Full split delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| M2 Studio | `26.4 ms` | `17.6 ms` | `11.3 ms` | `28.9 ms` | `8.8 ms` | `-2.4 ms` |
| Irvine M1 | `168.3 ms` | `105.9 ms` | `74.0 ms` | `179.8 ms` | `62.4 ms` | `-11.5 ms` |

Quality-changing F0/source branches have large speed signals but are not strict:

| Bucket | Candidate | Projected with HAR-post rewrite | Quality |
| --- | --- | ---: | --- |
| `10s` | `10s_natural_asr_cos_resblock_natural_asr_cos_rsqrt` | `591.2 ms`, beating strict paper row by `2.6 ms` | corr `0.867`, SNR `6.55 dB` |
| `15s` | `15s_padded_cos_resblock_cos_rsqrt` | `914.7 ms`, `2.7 ms` short of strict paper row | corr `0.957`, SNR `11.00 dB` |
| `7s` | `7s_natural_asr_cos_rsqrt` | profile-positive, paper-negative | corr `0.797`, SNR `4.77 dB` |
| `3s` | `3s_natural_asr_cos_rsqrt` | not sufficient alone | corr `0.814`, SNR `5.08 dB` |

## What We Need From The Research

Find a representation and implementation path that can preserve quality while
getting most of the body-only speed benefit without adding a separate
source/noise prediction call.

Prioritize approaches that:

- stay inside one hot Core ML prediction boundary;
- preserve fixed bucket shapes;
- avoid raw Nyquist branch sensitivity;
- can be implemented as a simple static MLProgram surface, such as
  `Conv1d(kernel_size=1)`, folded weights, or basic elementwise ops;
- can be validated first in PyTorch by matching first generator-body activations
  before Core ML conversion;
- avoid dynamic shapes, RangeDim, extra package boundaries, and broad retraining.

## Research Questions

1. How can a generator trained on raw `[magnitude, phase]` HAR be converted to
   consume a branch-stable representation such as `[real, imag]`,
   `[magnitude, sin(phase), cos(phase)]`, or source/noise features without full
   retraining?
2. Can the first `noise_convs` or adjacent generator layers be analytically
   folded to absorb a representation change from raw phase to real/imag or
   sin/cos phase?
3. If analytical folding is impossible because the first layers are nonlinear,
   what is the smallest calibration or distillation adapter likely to recover
   activations?
4. What activation-level losses should be used: first noise-conv output, first
   residual block input, pre-tail waveform logits, final waveform, or a weighted
   mixture?
5. How much data is likely needed for a tiny adapter calibration if the existing
   generator body is frozen?
6. Which adapter surfaces are Core ML friendly on Apple Silicon: 1x1 conv,
   grouped 1x1 conv, low-rank linear, small depthwise conv, lookup table, or
   piecewise affine correction?
7. Can the Nyquist branch correction be represented as a deterministic side
   channel that is cheaper than full HAR/STFT padding?
8. Can the F0/source simplification candidates be quality-repaired by a small
   adapter rather than accepted by human listening?
9. What failure modes should be expected when replacing raw phase with
   sin/cos or real/imag in a trained vocoder?
10. What objective gates should be used before human listening, given that ASR
    is explicitly skipped?

## Required Output Format

Start with an executive summary: what is most likely to work and why.

Then provide:

- a practical implementation plan for PyTorch activation-matching probes;
- concrete adapter architectures to try in priority order;
- exact Core ML export surfaces and shape/layout recommendations;
- do/avoid tables;
- a section on analytical weight folding feasibility;
- a section on tiny distillation/calibration workflows;
- a section on no-ASR quality gates and listening review;
- a section on benchmark methodology for warmed inference only;
- a section separating evidence-backed recommendations from speculation;
- references to relevant papers, repos, and Apple/Core ML docs.

Be explicit about which ideas are likely to preserve strict parity and which are
quality-changing but possibly listening-acceptable.
