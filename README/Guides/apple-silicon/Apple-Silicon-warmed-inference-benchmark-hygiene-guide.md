# Apple Silicon Warmed-Inference Benchmark Hygiene Guide

This guide ingests the restarted external benchmark-hygiene report for the
Kokoro lower-end Mac bakeoff. Treat the raw report as research input, not
canonical truth.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/apple-silicon-warmed-inference-benchmark-hygiene-for-kokoro-core-ml-and-mlx/2026-06-06T23-29-40-666Z/raw-report.md`

## Executive Summary

Paper-facing Kokoro comparisons must measure warmed inference only. Core ML
compile, first-run graph specialization, cache fills, stale package rows, and
host noise are not implementation speed. This matters most on Irvine M1, where
the remaining publishable fight is short-bucket warmed latency against laishere,
not MLX.

The repo source of truth remains the generated bakeoff artifacts, especially:

- `outputs/external_bakeoff/competitive_frontier.md`
- `outputs/external_bakeoff/frontier_freshness.md`
- `outputs/external_bakeoff/fixed_cost_latency_fit.md`
- `outputs/external_bakeoff/remote_host_quiet_latest.md`
- `outputs/external_bakeoff/candidate_frontier_matrix.md`

## Benchmark Contract

Use the fixed runtime buckets independently:

- `3s`
- `7s`
- `10s`
- `15s`
- `30s`

Each bucket gets its own warmup and timing sequence. Do not warm one shape and
report another shape. Do not mix cold package compilation with warmed
prediction.

## Quiet-Host Gate

Before promoting a lower-end Mac result, regenerate the quiet-host report:

```bash
python3 scripts/external_bakeoff/check_remote_host_quiet.py
```

Only run publishable timings when the target host reports `quiet=yes`.

The current gate is allowed to reject on practical host conditions:

| Check | Why it matters |
| --- | --- |
| High load or low idle CPU | Short-bucket fixed cost is easy to pollute. |
| Known noisy processes such as Spotlight or media analysis | Background CPU and disk work can dominate a `3s` inference delta. |
| Swap activity or memory pressure | Memory movement is part of the Core ML/MLX hypothesis; swap invalidates it. |
| Battery / AC-power mismatch | Power policy changes CPU/GPU/ANE behavior. |
| Thermal warnings where available | Thermal throttling turns benchmark rows into host-state rows. |

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Report medians and raw per-run timings for each bucket. | Reporting only an average or one lucky run. |
| Keep compile/cache time separate from warmed inference. | Treating first prediction as implementation speed. |
| Record host quiet status beside Irvine M1 rows. | Rerunning on a noisy host and averaging the pollution away. |
| Compare Config F, MLX, Soniqo, and laishere on equivalent full-duration audio. | Comparing padded, truncated, stale, or compile-inclusive rows. |
| Preserve the raw JSON/Markdown evidence bundle. | Only copying headline numbers into prose. |

## Claims Left As Heuristics

The raw report contains useful operational advice, but several claims are
outside the current repo evidence and should stay non-canonical until measured
locally:

- exact thresholds for all Apple background services;
- iPhone foreground and Low Power Mode benchmark policy;
- whether Instruments, `powermetrics`, or tethering materially changes each
  measured bucket;
- any recommendation that requires disabling platform security features.

## Related Documentation

- [Restarted Kokoro guide triage](../../Notes/kokoro-restarted-guide-triage-2026-06-06.md)
- [Kokoro M1 vocoder boundary research brief](../../Notes/Kokoro-M1-vocoder-boundary-research-brief.md)
- [Fixed-cost latency fit](../../Notes/fixed-cost-latency-fit.md)
- [Core ML vs MLX vocoder scheduling](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
- [iPhone Core ML device lab runbook](iPhone-CoreML-device-lab-runbook.md)
- [Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)
- [Apple Silicon NVMe and energy measurement](apple-silicon-nvme-energy-measurement-guide.md)
