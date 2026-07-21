# LFM2.5 Selective Prefill Negative Result

**Date:** 2026-07-20

**Status:** Terminal `KILL` at preregistered G0d

**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`

**Public evidence:**
[`docs/selective-split-report.md`](https://github.com/mattmireles/lfm2-surgical-coreml/blob/main/docs/selective-split-report.md)

## Decision

Stop before tail export and physical-device timing. The independently
registered six-piece follow-up did not rescue plan 010's 13-segment result.
Its proposed fixed-512 ANE islands failed numerical equivalence on the Mac
before their speed could become relevant.

This is a different negative result from plan 010:

- Plan 010 proved that 12 public Core ML boundaries cost too much even with
  exact GPU-permitted logits.
- Plan 011 reduced the proposed boundary count to five, but the three ANE
  islands failed the frozen `1e-2` GPU-versus-ANE correctness gate before the
  new tail or six-piece path was built.

## Authoritative G0 evidence

The Release Swift harness used one real 512-token prompt, direct fp16
`MLMultiArray` tensors, same-host package-hash-bound compute plans, three
warmups, and 20 counterbalanced samples where timing was valid.

The zero-boundary `.all` monolith did dispatch heterogeneously: 308/784 costed
operations preferred ANE, including 78/102 convolutions, while all matmuls and
softmaxes stayed on GPU. That schedule was not useful. Its median was
`53.281 ms` versus `35.539 ms` for `.cpuAndGPU`; the paired 95% interval for
`M_GPU - M_ALL` was `[-18.694, -17.065] ms`. It also changed the top token
`941 -> 1470` with final-logit `max_abs = 10.0859375`.

The result binds all 19 packages used by timing and the correctness oracle.
The pre-warmup host-noise record passed with normalized one-minute load
`0.1962`, 59% system memory free, no competing process at or above 50% CPU,
and nominal thermal state.

Each fixed convolution pair passed the G0b placement threshold with 58/58
costed operations preferring ANE. Each materialized twin matched its enumerated
source exactly at bucket 512 (`max_abs = 0.0`). The same fixed package then
diverged under GPU- versus ANE-permitted execution:

| Pair | Boundary max_abs | Suffix-logit max_abs | GPU→ANE token |
| --- | ---: | ---: | --- |
| C0-1 | 0.288086 | 7.826172 | 941→941 |
| C3-4 | 0.154297 | 9.095703 | 941→941 |
| C6-7 | 0.073975 | 3.377930 | 941→509 |

All three exceeded the `1e-2` threshold. G0d therefore killed pair timing,
aggregate pair economics, tail export, the six-piece Mac path, and every iPad
and iPhone phase. No device performance claim exists.

The frozen Stage 1 regression replay remained exact and terminal: monolith
`35.432 ms`, 13 segments `57.449 ms`, `38.32%` decomposition tax, direct fp16
boundaries, and final-logit `max_abs = 0.0`.

## Interpretation

ANE admission is not enough. In this experiment, fixed shapes and 100% ANE
preferred placement did not make the registered convolution islands valid
substitutes for their GPU executions. The source-versus-twin equality rules
out the diagnostic materialization step as the cause.

The correct outcome is to preserve the negative result and stop. Testing
another partition, precision policy, tolerance, device generation, or fused
runtime would be a new preregistered experiment, not a repair to plan 011.

## Durable pointers

- [Plan 011 and frozen gates](../Plans/011-lfm2-selective-surgical-prefill-plan.md)
- [Plan 010 terminal experiment](../Plans/010-lfm2-surgical-prefill-plan.md)
- [Stage 1 negative-result note](lfm2-stage1-negative-result.md)
- [LFM2 surgical prefill guide](../Guides/apple-silicon/LFM2-surgical-prefill-CoreML-guide.md)
- [Bucket-specific compute-plan guide](../Guides/apple-silicon/CoreML-enumerated-shape-compute-plan-specialization-guide.md)
- Raw Max report:
  `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-mlcomputeplan-for-non-default-enumeratedshapes-specializations/2026-07-21T04-19-00-642Z/raw-report.md`
