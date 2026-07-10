# Kokoro M1 Paper Frontier 3s/7s Deep Research Prompt

June 6, 2026

Use this as the handoff prompt for the next external research pass. The target
is narrow: make first-party Config F beat laishere on Irvine M1 for the
paper-facing `3s` and `7s` rows. Do not optimize against MLX; corrected warmed
evidence already shows MLX has `0` Mac wins against Config F.

## Objective

Find an implementation path that removes at least the remaining Irvine M1
paper-frontier gap after combining the best saved source/body candidates with
the measured HAR-post rewrite and `decoder_pre`/HnSF overlap projections:

| Bucket | Best saved source/body | Rewrite save | Overlap save | Combined projected Config F | Paper frontier | Extra save needed |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `3s` | `3s_natural_asr_cos_rsqrt` | `7.2 ms` | `4.4 ms` | `203.3 ms` | `176.3 ms` | `27.0 ms` |
| `7s` | `7s_natural_asr_cos_rsqrt` | `12.1 ms` | `8.4 ms` | `423.6 ms` | `394.6 ms` | `29.0 ms` |

The same combined path should close Irvine `10s` and `15s` if listening accepts
the saved source/body candidates, so this research pass should not spend its
main effort there. The hard problem is still `3s/7s`.

Authoritative current report:

- `outputs/external_bakeoff/irvine_paper_frontier_path.md`

## Ground Truth To Preserve

- Warmed inference only. Cold compile/cache time is disallowed.
- Runtime buckets are `3s`, `7s`, `10s`, `15s`, `30s`.
- Paper-facing frontier rows are stricter than newer laishere stage-profile
  rows. Do not claim an Irvine win from profile-only margins.
- Current MLX result is not the problem:
  - `outputs/external_bakeoff/mlx_speed_explanation.md`
  - corrected warmed MLX wins: `0`
  - corrected warmed Config F wins over MLX: `12`
  - MLX Mac `3s` rows fail with broadcast-shape errors
- The real strict competitor is `laishere/kokoro-coreml`.

## What Existing Evidence Says

### Source/body simplification helps but is insufficient for `3s/7s`

Current saved source/body candidates:

| Bucket | Candidate | Source/body projected | Paper row | Gap |
| --- | --- | ---: | ---: | ---: |
| `3s` | `3s_natural_asr_cos_rsqrt` | `214.8 ms` | `176.3 ms` | `38.5 ms short` |
| `7s` | `7s_natural_asr_cos_rsqrt` | `444.1 ms` | `394.6 ms` | `49.5 ms short` |

After adding the measured HAR-post rewrite and runtime-overlap projections,
they still remain short by `27.0 ms` and `29.0 ms`.

### Surface matching alone has failed

First-party fused graph probes have removed the visible manual AdaIN/tile
surface but did not reproduce laishere's M1 runtime behavior:

- native `instance_norm` alone: noise-sized speedup;
- native `instance_norm` + broadcast AdaIN + fp16 inputs: near-surface match,
  no material runtime gain;
- native `instance_norm` + broadcast AdaIN + fp16 inputs + palettization:
  closer surface match, slower and lower quality margin;
- cos-Snake variants: did not create the required M1 placement/runtime win.

Detailed evidence:

- `README/Kokoro-M1-graph-surface-target.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/candidate_frontier_matrix.md`

### More Neural Engine preferred ops is not enough

The exact decoder+vocoder body split gets laishere-like Neural Engine preferred
ops, but warmed runtime is slower. That rejects the simple hypothesis that
partial NE placement alone wins. The missing ingredient is likely boundary,
layout, synchronization, or a narrower runtime contract.

## Research Questions

1. What package-boundary or tensor-layout change can remove `27-29 ms` on
   Irvine M1 `3s/7s` without adding another hot Core ML call?
2. Can the source/body simplification and HAR-post rewrite be fused into one
   runtime-positive package path rather than treated as additive projections?
3. Can we eliminate or bypass the expensive Swift upstream/runtime component on
   `3s/7s` without changing audio quality?
4. What exact MIL pattern causes laishere to beat the first-party path on M1
   short buckets despite first-party wins on M2 Studio?
5. Is there a smaller first-party source representation that preserves
   listening quality but lets Core ML run the vocoder body with fewer syncs?

## Candidate Requirements

A useful proposal must specify:

- exact package boundary and model inputs/outputs;
- tensor ranks and static shapes for `3s` and `7s`;
- whether it preserves strict waveform parity or requires no-ASR listening;
- expected impact on the `27.0 ms` and `29.0 ms` remaining gaps;
- why it avoids the already-measured split-boundary penalty;
- Core ML deployment target, precision, compute units, and expected
  `MLComputePlan` signature;
- how it will be tested locally before Irvine timing.

## Acceptance Gate

Do not promote a candidate unless all applicable gates pass:

- local M2 Studio warm timing is positive or there is a justified M1-specific
  partition hypothesis with compute-plan evidence;
- waveform strict gate passes, or exact generated WAVs receive no-ASR human
  listening acceptance;
- Irvine timing is run only when `check_remote_host_quiet.py` reports quiet;
- `competitive_frontier` is refreshed from warmed full-duration rows;
- result is recorded in the generated frontier artifacts and the rejection or
  promotion is durable.

## Expected Output

The research output should be an implementation plan with one of these shapes:

- a single-package strict `GeneratorFromHar` replacement that closes `3s/7s`;
- a source/body path that preserves quality and closes `3s/7s` after the
  HAR-post rewrite;
- a proof that laishere's `3s/7s` paper rows are stale or non-equivalent, with
  exact warmed rerun evidence;
- a concrete negative result that rules out a new boundary/layout/partition
  hypothesis.

Do not return general Core ML advice. Tie every recommendation to the measured
Irvine `3s/7s` gap and the files listed above.
