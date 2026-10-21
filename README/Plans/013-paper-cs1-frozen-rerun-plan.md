# Surgical Inference CS1 Frozen Rerun Plan

**Date:** 2026-10-20
**Status:** Complete for the bucketed matrix; flexible-generator rerun on OS 27 Macs waits on an Apple fix
**Progress model:** Phase task checkboxes only

## Executive Summary

Re-measure every Kokoro (case study 1) number in `paper/main.tex` on the
current shipped pipeline, on the paper's own devices, under one frozen
protocol, then replace the paper's tables with the new results. The paper's
numbers come from April–July builds that predate PRs #6–#12, which changed
nearly every stage.

## Problem Statement

- **Symptom:** The paper's Config F tables (June bakeoff, April ledger, July
  iPhone) describe a pipeline that no longer ships. Newer numbers exist only
  as single-device community measurements or loaded-host upper bounds.
- **Root Cause:** The pipeline changed after the 2026-07-17 paper freeze;
  nobody re-ran the paper's matrix.
- **Impact:** The paper either understates the current system or overstates
  the old one. Neither number is measured under one protocol.

## Goals and Non-Goals

### Goals

- One frozen run per device and arm, raw JSON kept, warm medians and IQR.
- New values for paper Tables 3–9 and 11, the per-stage table, and the
  abstract/conclusion ratios.

### Non-Goals

- No pipeline changes. A run that exposes a bug records it and stops.
- No new competitors, no new devices, no quality study beyond the parity and
  spot-check WAVs the adapters already write.
- LFM2.5 material stays out of the paper (owner decision 2026-10-20).

## Frozen Protocol (Do Not Change After Phase 1 Starts)

| Item | Value |
| --- | --- |
| Source | `main` at the commit recorded in Phase 0 (no local diff in pipeline code) |
| Models | HF `mattmireles/kokoro-coreml` revision `dadebca4`, via `scripts/download_models.py` |
| Config F | `run_config_f_reference.py --compute-units staged --use-padded-duration-models` — the shipped default: padded masked duration, flexible generator when the OS supports it |
| Inputs | The five frozen runtime inputs (`3s`, `7s`, `10s`, `15s`, `30s`), voice `af_heart`, speed 1.0 |
| Timing | 3 discarded preflight calls, 1 cold marker, 10 warm calls per input; report warm median and IQR, cold separately |
| MLX | mlx-audio 0.4.3 at `862dfbe`, `mlx-community/Kokoro-82M-bf16` (same pin as the paper) |
| PyTorch arms | `bakeoff_harness.py run --configs a,d,e --iterations 5 --order-seed 0`, `PYTORCH_ENABLE_MPS_FALLBACK=1` |
| External | laishere `484907d`, soniqo speech-swift `0d09a2e` (same pins as the paper) |
| Quiet gate | 1-minute load ≤ 1.5 before each arm (`check_remote_host_quiet.py` semantics); record `vm.loadavg` and thermal state before and after each arm |
| Botnet | `SupervisorCLI pause` before timed runs; restore each host's prior mode afterwards |
| iPhone | Release builds only; arm order alternated per bucket; phone cooled to nominal between arms |

A host that cannot pass the quiet gate is still run, but its rows are labeled
`upper bound (load X)` and are not used for ratios without that label.

## Hosts

| Paper device | Host | OS now | Prior Botnet mode |
| --- | --- | --- | --- |
| M2 Ultra 64 GB | `m2-studio` (seat) | macOS 26.7.1 (25G241) | `paused` (manual, set 2026-10-20 12:45 PT) |
| M2 Air 24 GB | `m2-air` | macOS 27.0.1 | `adaptive` (manual override) |
| M1 Mini 16 GB | `irvine-m1` | macOS 27.0.1 | `adaptive` (default) |
| A17 Pro | iPhone 15 Pro Max, cabled to `m2-studio` | iOS 27.0.1 | — |
| A14 | iPhone 12 Pro, cabled to `m2-studio` | iOS 27.0.1 | — |

Every OS moved since the paper (26.5 / 15.7.7). The two laptops/minis are on
macOS 27.0.1 (26A434); m2-studio is still on 26.7.1.

**2026-10-20 finding (irvine-m1):** the shipped flexible generator aborts the
process on M1 + macOS 27.0.1 on its first call — MPSGraph `'mps.expand_dims' op
invalid axis: <garbage>` then `failed assertion 'shape for TensorData is not
static'`. laishere's own flexible model hits the same assertion during
conversion, so this looks like an OS regression, not our package. Per the stop
gate, irvine-m1's Config F matrix is blocked pending a decision. The new tables state the new
OS; the paper must not attribute the whole delta to code.

## Implementation Phases

### Phase 0: Freeze and setup

- [ ] Record the source commit, HF revision and host table above in the run log.
- [ ] On each Mac: sync the repo to the frozen commit, install the bakeoff env,
  download models at `dadebca4`, rebuild `kokoro-bench` release, prepare inputs.
- [ ] On each Mac: MLX venv at the pinned commit; laishere and soniqo clones at
  the pinned commits, laishere conversion done.
- [ ] Smoke: one Config F call per input passes the 2% duration check.

### Phase 1: Mac matrix

- [ ] Pause Botnet on the three Macs; record prior modes.
- [ ] Per Mac, in this order, each behind the quiet gate: Config F, MLX,
  laishere, soniqo, then A/D/E.
- [ ] Per-stage timings for Config F at every bucket.

### Phase 2: iPhone matrix

- [ ] Rebuild bench fixtures (`prepare_swift_bench_inputs.py`,
  `ios-bench/prepare_resources.sh`) from the frozen models.
- [ ] A17 Pro and A14: Config F vs kokoro-ios MLX, Release, alternated order,
  cooled between arms.

### Phase 3: Restore and record

- [ ] Restore each host's prior Botnet mode and verify with `SupervisorCLI status`.
- [ ] Write the run log and results into `README/Notes/` (one consolidated note).
- [ ] Replace the paper's CS1 numbers with the new results, each with its OS
  and date; label any upper-bound rows.

## Kill and Stop Gates

- A Config F parity or duration-check failure stops that device's matrix; the
  failure is recorded, not worked around.
- A competitor that fails to build at its pinned commit is reported as
  unavailable; its old row is either removed or kept with its old date, never
  silently mixed with new rows.

## Success Criteria

- Every number in paper CS1 is either new (frozen protocol, raw JSON path
  cited) or explicitly dated as older.
- Botnet is back in each host's prior mode.
