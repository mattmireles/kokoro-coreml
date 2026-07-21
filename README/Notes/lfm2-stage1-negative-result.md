# LFM2.5 Surgical Prefill Stage 1 Negative Result

**Date:** 2026-07-20
**Status:** Terminal KILL at pre-registered gate G1a
**Checkpoint:** `LiquidAI/LFM2.5-350M` at
`b9d6e4e2d75f440b12a2b4d731c808004ecbbd89`
**Public evidence:**
[`mattmireles/lfm2-surgical-coreml`](https://github.com/mattmireles/lfm2-surgical-coreml)

## Decision

Stop before the placement and device matrices. The real 16-layer checkpoint
requires 13 maximal same-class segment predictions in its frozen
`C C A C C A C C A C A C A C A C` order. Under the same GPU-permitted
placement, those predictions and their 12 added inter-segment boundaries
consume `37.43%` of segmented 512-token prefill. The pre-registered maximum
was `30%`.

This is a negative systems result, not an admission failure. Stage 0 proved
that the representative short-convolution block prefers ANE on both the Mac
Studio and physical M2 iPad, while the representative GQA block prefers GPU.
Stage 1 shows why attractive per-block dispatch is insufficient: the model
interleaves the two classes too finely for separate Core ML predictions to be
competitive under this packaging strategy.

## Authoritative G1a Evidence

| 512-token full-prefill control | Median | IQR | Same-host dispatch |
| --- | ---: | ---: | ---: |
| Monolithic | 36.032 ms | 0.352 ms | 797/797 GPU ops |
| 13 segments | 57.591 ms | 4.367 ms | 807/807 GPU ops |
| Added tax | **21.558 ms** | — | **37.43% of segmented prefill** |

The governing result comes from a Release Swift executable, not the Core ML
Python bridge. It directly reuses fp16 `MLMultiArray` outputs as the next
model's inputs and fails closed on any non-fp16 hidden boundary. The protocol
used three untimed warmups per candidate, 20 counterbalanced AB/BA
measurements per candidate, and excluded package loading and first-use
compilation. Both controls include embedding, the complete layer stack,
last-position selection, final normalization, and the tied language-model
head.

Paired final logits match exactly (`max_abs = 0.0`). The result JSON records
the exact prompt, checkpoint revision and file digests, host and OS build,
coremltools version, compute-unit permission, and package-tree SHA-256 for all
16 timed artifacts. Each package hash matches its same-host compute-plan
provenance. An independent N=20 Swift repeat also failed, with a `37.61%` tax.

The earlier Python diagnostic measured a larger `54.75%` tax because the
bridge materialized fp16 model outputs as fp32 NumPy arrays. It is retained as
diagnostic evidence but does not govern G1a.

## What Was Cancelled

The frozen decision tree says any G1a failure ends the experiment. Therefore
the 32-prompt × 64-token G1b matrix, six placement configurations, M1 rail
study, iPad latency and thermal matrix, and iPhone overnight runs were not
performed. One real-prompt full-prefill probe matched the fp32 top token, but
is not presented as the cancelled G1b result. No iPhone performance, energy,
decode, or thermal claim exists.

The next scientifically distinct experiment would reduce prediction
boundaries through multifunction packaging or a fused runtime. That is a new
pre-registration, not a post-hoc rescue of plan 010.

## Durable Pointers

- Frozen plan and gates:
  [`010-lfm2-surgical-prefill-plan.md`](../Plans/010-lfm2-surgical-prefill-plan.md)
- Stage 0 admission report:
  [`lfm2-stage0-report.md`](lfm2-stage0-report.md)
- Public Stage 1 report:
  [`docs/stage1-report.md`](https://github.com/mattmireles/lfm2-surgical-coreml/blob/main/docs/stage1-report.md)
- Public Swift runtime:
  [`Sources/LFM2SurgicalRuntime`](https://github.com/mattmireles/lfm2-surgical-coreml/tree/main/Sources/LFM2SurgicalRuntime)
- External field guide source:
  `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/lfm2-5-full-sequence-prefill-and-surgical-core-ml-segment-export/2026-07-20T23-58-47-025Z/raw-report.md`
