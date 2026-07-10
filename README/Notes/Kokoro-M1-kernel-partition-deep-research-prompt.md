# Kokoro M1 Kernel Partition Deep Research Prompt

June 6, 2026

Use this prompt for a focused external research pass on the remaining strict
Irvine M1 loss. This is not a generic Core ML optimization request. The task is
to explain and change the Core ML runtime partition behavior that makes
laishere's vocoder fast on M1 while first-party strict `GeneratorFromHar`
surfaces either stay GPU-only or lose when split.

## Objective

Find a strict, warmed-inference implementation path that makes first-party
Config F beat laishere on Irvine M1 for `3s`, `7s`, `10s`, and `15s` without
using cold compile/cache time, `.all` wishful thinking, or quality-changing
shortcuts.

After the HAR-post upsample rewrite plus `decoder_pre`/HnSF overlap projection,
the remaining Irvine profile budget is:

| Bucket | Projected Config F after overlap+rewrite | laishere profile | Extra strict save needed |
| --- | ---: | ---: | ---: |
| `3s` | `222.0 ms` | `195.0 ms` | `27.0 ms` |
| `7s` | `472.2 ms` | `444.2 ms` | `28.0 ms` |
| `10s` | `657.6 ms` | `644.9 ms` | `12.8 ms` |
| `15s` | `976.4 ms` | `990.6 ms` | `0.0 ms` |

The stricter paper frontier target still needs more. Use
`outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md` as the
current numeric authority.

The latest paper-frontier path is now more specific:

- `outputs/external_bakeoff/irvine_paper_frontier_path.md`
- source/body plus HAR-post rewrite plus runtime overlap should close Irvine
  `10s` and `15s` if listening accepts the saved source candidates;
- `3s` and `7s` remain the hard paper-frontier gaps, needing another `27.0 ms`
  and `29.0 ms` after the best saved source/body plus overlap+rewrite
  projections.

For an external research pass focused only on those hard short-bucket gaps, use
`README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`.
For the narrower source/HAR representation blocker, use
`README/Kokoro-M1-HAR-STFT-contract-deep-research-prompt.md`.

Do not spend this research pass explaining or optimizing against MLX. The
current generated evidence says MLX is not faster than corrected warmed Config F
on any full-duration Mac row:

- `outputs/external_bakeoff/mlx_speed_explanation.md`
- corrected warmed MLX wins: `0`
- corrected warmed Config F wins over MLX: `12`
- MLX Mac `3s` rows: deterministic broadcast-shape failures

If MLX appears faster in another table, first check whether that table uses raw
Config F rows that include Core ML compile/cache behavior or stale artifacts.
The real lower-end strict target is laishere.

## Core Observation

Visible graph cleanup is not enough.

| Package | Units | Ops | Preferred CPU | Preferred GPU | Preferred NE | NE cost |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| first-party HAR-post baseline | CPU+NE | `2207` | `1038` | `0` | `0` | `0.0%` |
| first-party native-IN+broadcast+fp16 | CPU+NE | `1533` | `677` | `0` | `0` | `0.0%` |
| laishere vocoder | CPU+NE | `1534` | `58` | `0` | `597` | `47.5%` |
| first-party exact decoder+vocoder body | CPU+NE | `1546` | `64` | `0` | `599` | `48.7%` |

This creates the main paradox:

- A first-party fused package can look very close to laishere by op count and
  surface, but Core ML still keeps it off Neural Engine.
- A first-party split body can get laishere-like Neural Engine preferred-op
  counts, but warmed runtime loses because the boundary/synchronization pattern
  is wrong.
- Therefore the target is neither "match MIL histogram" nor "get any NE
  placement." The target is a runtime-positive partition and boundary.

## Evidence To Read First

- `outputs/external_bakeoff/candidate_frontier_matrix.md`
- `outputs/external_bakeoff/lower_end_mac_win_gate.md`
- `outputs/external_bakeoff/irvine_paper_frontier_path.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md`
- `README/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md`
- `README/Kokoro-M1-graph-surface-target.md`
- `README/Kokoro-M1-vocoder-boundary-research-brief.md`
- `README/Notes/coreml-compute-unit-ablation.md`
- `README/Notes/performance-notes.md`

## Do Not Repeat

Treat these as measured unless the proposal changes kernel selection,
partitioning, memory layout, or Core ML call boundaries:

- `.all` toggles or compute-unit flags alone.
- CT8/CT9/iOS17 toolchain-only rebuilds.
- RangeDim/flexible inputs for the fused generator.
- HAR tail trimming.
- Native `instance_norm`, broadcast AdaIN, cos-Snake, fp16 inputs, or
  palettization as standalone surface changes.
- Linear quantization of the final-waveform fused generator.
- Style specialization.
- Multi-package exact decoder+vocoder or generator noise/body splits.
- Strict padded source/STFT/HAR fused paths that preserve quality but lose
  speed.

These are recorded in
`outputs/external_bakeoff/candidate_frontier_matrix.md`.

## Research Questions

1. What Core ML compiler or runtime properties cause laishere's `KokoroVocoder`
   to receive a mixed CPU/Neural Engine plan while a near-surface first-party
   fused `GeneratorFromHar` does not?
2. Which tensor ranks, layouts, input/output contracts, intermediate aliases,
   constant shapes, op groupings, or MIL patterns are likely responsible for
   the difference?
3. Can first-party `GeneratorFromHar` be rewritten as one package with a
   laishere-like body partition while preserving the current Swift HAR/source
   input contract?
4. If a split is required to get the partition, how can the split avoid the
   measured call/synchronization penalty?
5. Is there a Core ML-friendly way to pass a smaller Swift-produced source/HAR
   representation into one package so that the package is partitionable and
   strict?

## Candidate Shape Requirements

A useful proposed implementation must specify:

- exact package boundary: inputs, outputs, ranks, static bucket shapes, and
  whether `3s/7s/10s/15s/30s` need separate packages;
- deployment target and coremltools/PyTorch requirements;
- expected compute units for each stage;
- why it avoids the existing CPU/GPU/NE sync penalty;
- why it should preserve strict waveform parity or what no-ASR listening review
  would be required;
- expected effect on the post-rewrite Irvine budget.

## Acceptance Gate

Do not call a candidate successful until all of these are true:

- local M2 Studio warmed timing is positive or the proposal is specifically an
  M1-only partition hypothesis with compute-plan evidence;
- strict waveform gate passes against the same Swift dump, or the exact WAVs
  are accepted through no-ASR listening review;
- `MLComputePlan` reports a plausible runtime-positive plan, not just more
  Neural Engine preferred ops;
- quiet Irvine M1 warmed timing closes at least one real-loss bucket or proves
  a measured step toward the remaining budget;
- `outputs/external_bakeoff/candidate_frontier_matrix.md` is updated with the
  result, including failures.

## Current External Gates

Before running lower-end Mac timing, regenerate:

```bash
uv run --no-sync python scripts/external_bakeoff/check_remote_host_quiet.py
```

Run publishable timing only when the target host reports `quiet=yes`.

The connected iPhone 12 Pro is visible and paired, but Config F iPhone timing
is unavailable until the physical device is unlocked and
`com.kokoro.externalbakeoff.ConfigFIOSRunnerManual` can launch.
