# LFM2.5 A17 Pro Parity Negative Result

**Date:** 2026-07-21

**Status:** Terminal `KILL` at Plan 012 Gate P2

**Public evidence:**
[`docs/a17-parity-report.md`](https://github.com/mattmireles/lfm2-surgical-coreml/blob/main/docs/a17-parity-report.md)

## Decision

Stop the public-boundary selective-prefill direction. The highest-information
C6-7 probe reproduced the M2 Ultra ANE numerical failure on the physical A17
Pro phone. Do not run C0-1, C3-4, another phone, or a performance experiment to
search for a pass.

This closes the question raised after plans 010 and 011:

- [Plan 010](../Plans/010-lfm2-surgical-prefill-plan.md) rejected the
  13-segment path because 12 public Core ML boundaries consumed 37.43% of
  segmented prefill latency despite exact GPU-permitted logits.
- [Plan 011](../Plans/011-lfm2-selective-surgical-prefill-plan.md) rejected a
  coarser six-piece path because its three fixed-512 ANE islands failed
  GPU-versus-ANE correctness on M2 Ultra before timing.
- [Plan 012](../Plans/012-lfm2-a17-parity-probe-plan.md) now shows that the
  strongest token-changing failure reproduces on the tested A17 Pro and iOS
  build after complete ANE admission.

## What the phone proved

The result came from an iPhone 15 Pro Max (`iPhone16,2`, A17 Pro) on iOS 27.0
beta build `24A5380h`. The explicit signed app used the same
`LiquidAI/LFM2.5-350M` revision, real prompt, bucket 512, fp16 tensors, fixed
C6-7 package, zero prefill state, and `1e-2` maximum-absolute tolerance frozen
by the plan.

Same-phone compute plans separated the policies:

| Policy | Preferred placement | Conv placement |
| --- | --- | --- |
| `CPU_AND_GPU` | 54 GPU, 2 CPU, 0 ANE | 12/12 GPU |
| `CPU_AND_NE` | 56/56 ANE | 12/12 ANE |

The complete GPU oracle retained token `941`. The enumerated C6-7 source and
fixed twin matched exactly under the GPU control in all six rows. The fixed
ANE-permitted execution then produced the same failure in the first-use row
and all five warmed rows, independent of candidate call order:

- `hidden_out max_abs = 0.012085`
- `conv_state_6_out max_abs = 0.000977`
- `conv_state_7_out max_abs = 0.070068`
- pair-output `max_abs = 0.070068`
- suffix-logit `max_abs = 3.610229`
- final token `941 -> 509`

Every row failed. The result envelope records zero performance samples.

## Interpretation

This is stronger than a second noisy mismatch. The error was invariant across
first use, warmup state, and reversed candidate order. Source-versus-twin
equality excludes fixed-shape materialization, and the dispatch plans exclude
accidental same-backend comparison. The exact same token-changing pathology
appeared on two ANE generations under their recorded OS/compiler paths.

The remaining plausible work is not “try another device.” It would require a
materially new mechanism: different graph formulation, precision strategy, or
runtime boundary with a new hypothesis and new frozen gates. Nothing in this
result authorizes such an experiment automatically.

## Durable local evidence

The ignored public-repo evidence is:

- `outputs/lfm2_surgical/a17_parity/dispatch/segment_04_conv_6_7.cpu_and_gpu.json`
  — file SHA-256
  `6855d7ec08195b63a05a5b566f666b187ef254d596f810f0961d9e1371f9a05d`
- `outputs/lfm2_surgical/a17_parity/dispatch/segment_04_conv_6_7.cpu_and_ne.json`
  — file SHA-256
  `1a3bb1f7143501fc8971bd0abb9f4e9142414e5bbc4e32032b8f9d32bd571470`
- `outputs/lfm2_surgical/a17_parity/pairs/segment_04_conv_6_7.result.json`
  — file SHA-256
  `6878e45d085ea0683c5f8ee864e13eb5a25f228f7a3245155c28b74ea44f5568`
  and canonical payload SHA-256
  `53fcae5d2502d6e9aae93fe1c85d0523ae378ae126a6ea877611308e574f6c20`

C0-1 and C3-4 have no phone dispatch or result files. Their absence is the
intended evidence that the stop-first state machine held.
