# Kokoro M1 Source/Body Core ML Guide

This guide ingests the restarted external report on making the first-party
source/body path competitive with laishere on M1. Treat the raw report as
research input, not canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-source-body-core-ml-implementation-path-against-laishere/2026-06-06T23-29-25-697Z/raw-report.md`

## Executive Summary

The useful direction is not another standalone source/body split. Existing repo
evidence says broad splits can improve visible Core ML placement while losing
warmed runtime through boundary and synchronization cost.

The viable path is to change the source/body representation inside a
runtime-positive boundary:

- preserve current Swift source/HAR semantics, or explicitly label a
  quality-accepted non-strict branch;
- avoid adding hot-path Core ML predictions unless the measured saved compute is
  larger than the call/sync cost;
- remove padded or branch-sensitive HAR payload only when strict waveform gates
  still pass;
- use laishere-like graph surfaces as hypotheses, not proof.

## Verified Repo Evidence

The source/body frontier currently has these durable results:

| Result | Decision |
| --- | --- |
| Source equation is solved. | Do not reopen basic source math without new evidence. |
| Natural HAR/STFT strict replacement still fails. | Raw Nyquist phase convention remains a blocker. |
| Broad exact body split can look more ANE-like but run slower. | Placement alone is insufficient. |
| Native InstanceNorm, broadcast AdaIN, fp16 inputs, and palettization were already tested. | Do not repeat these as standalone fixes. |
| Upsample ConvTranspose rewrite is a strict local win. | Promote under quiet Irvine conditions before making a paper claim. |

## Source/Body Candidate Shape

The next useful source/body candidate should satisfy all of these constraints:

1. one hot Core ML prediction boundary unless timing proves otherwise;
2. static bucket shapes for `3s`, `7s`, `10s`, `15s`, and `30s`;
3. strict waveform comparison against the current fused package;
4. MLComputePlan evidence when claiming CPU/GPU/ANE placement changes;
5. warmed local timing before any Irvine promotion.

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Treat laishere's operator surface as a lead to test. | Assuming its visible surface is sufficient to win. |
| Fold source/body changes into an existing runtime-positive boundary. | Adding a split because a subgraph benchmark looks faster alone. |
| Keep quality-failing F0/source candidates in a separate lane. | Counting source-quality regressions as strict parity wins. |
| Use the HAR/STFT repair guides before changing phase representation. | Replacing raw phase with sin/cos or real/imag without body adaptation. |

## Related Documentation

- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Kokoro M1 partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
- [Kokoro M1 HAR/STFT contract repair guide](Kokoro-M1-HAR-STFT-contract-repair-guide.md)
- [Kokoro HAR/STFT strict repair and distillation guide](Kokoro-HAR-STFT-strict-repair-distillation-guide.md)
- [Kokoro M1 source/body research prompt](../../Notes/Kokoro-M1-source-body-deep-research-prompt.md)
