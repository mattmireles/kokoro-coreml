# Kokoro Irvine M1 3s/7s Paper Frontier Guide

This guide ingests the restarted external report focused on the Irvine M1
short-bucket paper frontier. Treat the raw report as research input, not
canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-irvine-m1-3s-and-7s-paper-frontier-optimization/2026-06-06T23-29-25-659Z/raw-report.md`

## Executive Summary

The raw report's useful conclusion matches the repo evidence: the paper-facing
target is not "beat MLX" anymore. Corrected warmed Mac rows already remove MLX
as the blocker. The remaining lower-end Mac target is laishere on Irvine M1,
especially `3s` and `7s`.

This guide should be read with the current generated frontier artifacts:

- `outputs/external_bakeoff/competitive_frontier.md`
- `outputs/external_bakeoff/goal_frontier_status.md`
- `outputs/external_bakeoff/irvine_next_targets.md`
- `outputs/external_bakeoff/irvine_3s_placement_target.md`
- `outputs/external_bakeoff/strict_win_budget_after_overlap_rewrite.md`

## Current Working Theory

The short-bucket gap is dominated by fixed runtime cost: prediction boundaries,
host-to-Core ML handoffs, buffer movement, and scheduler decisions inside a
small number of hot model calls. Large graph-surface changes that add more
Core ML predictions have repeatedly lost even when their compute plan looks
closer to laishere.

Use this priority order for the next strict pass:

1. Prove the host is quiet.
2. Run the `outputBackings` harness ablation for `3s` and `7s`.
3. Promote the production upsample ConvTranspose rewrite to Irvine only under
   quiet warmed conditions.
4. Continue single-package body reshaping inside the existing hot boundary.
5. Keep quality-failing F0/source speed branches separate from strict parity.

## Paper-Facing Rules

| Rule | Reason |
| --- | --- |
| Warm each bucket separately. | Bucket shape and Core ML specialization matter. |
| Compare full-duration output only. | Partial-output rows make MLX and Core ML incomparable. |
| Exclude compile/cache time. | The paper hypothesis is inference speed. |
| Keep laishere, MLX, Soniqo, and first-party rows from the same benchmark contract. | Cross-contract deltas are not publishable. |
| Attach quiet-host evidence to Irvine M1 promotion rows. | Short-bucket deltas are too small to trust without host-state proof. |

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Optimize for measured `3s` and `7s` medians first. | Assuming a `30s` win transfers to short buckets. |
| Treat MLComputePlan as placement evidence, not speed evidence. | Promoting a candidate because it has more NE-preferred ops. |
| Keep strict and listening-approved branches labeled separately. | Mixing quality-failing speedups into the strict frontier. |
| Recompute the frontier after each accepted candidate. | Reusing stale gap numbers after new warmed rows land. |

## Related Documentation

- [Apple Silicon warmed-inference benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Kokoro M1 vocoder runtime boundary guide](Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Kokoro M1 vocoder boundary research brief](../../Notes/Kokoro-M1-vocoder-boundary-research-brief.md)
- [Kokoro M1 paper frontier prompt](../../Notes/Kokoro-M1-paper-frontier-3s-7s-deep-research-prompt.md)
