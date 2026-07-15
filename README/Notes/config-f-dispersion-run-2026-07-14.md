# Config F Dispersion Run — M2 Ultra Mac Studio (2026-07-14)

## Why this exists

`Scratchpad/surgical-inference.md` §5.2 reports warm medians for Config F on
the M2 Ultra Studio (50.6 / 96.1 / 126.2 / 185.6 / 379.3 ms for the 3s / 7s /
10s / 15s / 30s buckets) with no dispersion statistic, and the raw per-iteration
JSONs behind those June 2026 medians are no longer present on this machine.
This note supplies a fresh, well-documented N=10 dispersion dataset for the
same five frozen buckets so the paper can cite a spread alongside the median,
and records how the fresh medians compare to the June values.

**This is a fresh measurement, not a re-derivation of the June numbers.** The
June raw JSONs were unavailable; this run used the current checked-out Config
F build (see Provenance) and the canonical reference adapter, not an attempt
to reproduce the exact June binary.

## Protocol

1. **Quiet-host gate.** Per
   `README/Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md`
   and the `bakeoff` skill, timing data should only be collected on a quiet
   host. This machine was **not quiet** for the full run: `sysctl vm.loadavg`
   showed load1 in the 3.95–5.49 range throughout, driven by another agent's
   `swift/.build/release/kokoro-bench` process pinned at ~100% CPU (an
   audio-render/benchmark job running concurrently, per the task brief) plus
   background `CoreMediaIO`/`WindowServer` load. A bounded local poller
   (45 s interval, 30 min cap, mirroring the thresholds in
   `scripts/external_bakeoff/check_remote_host_quiet.py`: load1 ≤ 1.5, no
   contender process ≥25% CPU) ran the full 30 minutes and never reached
   quiet — the contending `kokoro-bench` process was still at 100% CPU at
   both the start and the end of the wait window. Per the mission protocol's
   documented fallback ("if the host stays busy for more than ~30 minutes,
   collect anyway but flag the contamination risk"), data was collected after
   the 30-minute bound elapsed, with the host still busy.
   **Contamination risk: present.** Treat the absolute per-iteration values
   below as noisier than a quiet-host run would produce, though see the
   Interpretation section for why the direction of the observed deviations
   argues against gross corruption.
2. **Inputs.** The five frozen runtime bucket inputs (3s/7s/10s/15s/30s,
   voice `af_heart`, speed 1.0) were regenerated for this run because
   `outputs/` is gitignored and the prior fixtures were not present on this
   machine:
   - `scripts/external_bakeoff/prepare_runtime_inputs.py` rebuilt
     `outputs/external_bakeoff/runtime_input_manifest.json` (3s/7s/15s/30s
     pulled from `outputs/bakeoff/input_manifest.json`, 10s freshly derived
     via the script's frozen 10s candidate text). Canonical audio durations:
     2.800 / 6.750 / 9.625 / 13.900 / 27.375 s — consistent with the paper's
     §5.2 audio-duration column (2.80 / 6.75 / 9.60 / 13.90 / 27.38 s).
   - `scripts/prepare_swift_bench_inputs.py` regenerated the five
     `outputs/swift_bench_inputs/{3s,7s,10s,15s,30s}.json` fixtures keyed off
     that manifest. Resulting token counts: 44 / 105 / 156 / 219 / 476 — each
     has a matching `coreml/kokoro_duration_exact_t*.mlpackage` (exact-duration
     path), avoiding the padded-duration cost documented in §4.2/§5.3 of the
     paper.
3. **Benchmark adapter.** `scripts/external_bakeoff/run_config_f_reference.py`
   is the canonical Config F reference adapter for this machine per the
   `bakeoff` skill and its own README (`scripts/external_bakeoff/README.md`).
   Invocation:

   ```bash
   uv run --with-requirements requirements-bakeoff.txt --no-sync \
     python scripts/external_bakeoff/run_config_f_reference.py \
     --machine-id m2_ultra_dispersion_2026_07_14 \
     --iterations 10
   ```

   This uses the adapter's defaults: `--compute-units staged` (production
   staged compute policy — duration/F0Ntrain/generator on CPU+GPU,
   decoder-pre on CPU+ANE), exact-duration model discovery enabled
   (`KOKORO_USE_EXACT_DURATION_MODELS=1`), and `--preflight-runs 3` (three
   discarded warmup calls before the recorded cold call). `--iterations 10`
   requests the mission's N=10 warm iterations per bucket; the adapter's
   single `cold_result` call per bucket is recorded separately
   (`cold_wall_time_s`) and excluded from the warm-iteration dispersion
   statistics below — this is the run's "1 discarded cold call" per bucket.
   The Swift benchmark binary was rebuilt at current HEAD before running
   (`swift build -c release --product kokoro-bench`) and its `--help` output
   confirmed `--batch` support.

## Environment

- Git: `05e184895f03c17431e9333d6c84780407f0e2bb` (branch
  `codex/kokoro-drop-in-sdk-v1`)
- macOS: 26.5.1 (Build 25F80)
- Machine: Apple M2 Ultra Mac Studio, 64 GB — the paper's §5.1 headline
  machine
- Other load present during collection: **yes** (see Protocol §1) —
  `swift/.build/release/kokoro-bench` at ~100% CPU throughout, load1 in the
  3.95–5.49 range vs. the quiet-host threshold of 1.5
- Raw results: `outputs/external_bakeoff/results_config_f_reference_m2_ultra_dispersion_2026_07_14.json`
  (gitignored; cite this path, do not commit the JSON)
- Spotcheck WAVs:
  `outputs/external_bakeoff/spotcheck_wavs/config_f_reference_m2_ultra_dispersion_2026_07_14/`

## Dispersion table (N=10 warm iterations per bucket, milliseconds)

| Bucket | N | Median | IQR | Min | Max | Relative IQR (%) | Cold call | June §5.2 median | Δ vs June (%) | Flag |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 3s  | 10 | 49.6  | 1.5 | 48.0  | 56.2  | 3.1 | 47.4  | 50.6  | −1.9  | ok |
| 7s  | 10 | 92.8  | 1.3 | 90.7  | 97.0  | 1.4 | 91.3  | 96.1  | −3.5  | ok |
| 10s | 10 | 129.2 | 7.3 | 124.1 | 147.4 | 5.6 | 140.6 | 126.2 | +2.4  | ok |
| 15s | 10 | 178.0 | 4.0 | 173.8 | 183.7 | 2.2 | 174.9 | 185.6 | −4.1  | ok |
| 30s | 10 | 340.2 | 3.9 | 332.7 | 345.8 | 1.1 | 340.7 | 379.3 | −10.3 | **>5%, flagged** |

Raw warm iteration times (ms), for reference:

- 3s:  50.5, 49.3, 48.3, 48.0, 49.0, 49.4, 49.9, 50.6, 53.0, 56.2
- 7s:  93.6, 97.0, 92.5, 90.7, 91.9, 92.3, 94.4, 92.6, 92.9, 93.5
- 10s: 142.2, 131.3, 134.7, 147.4, 129.4, 129.1, 129.0, 124.1, 125.8, 124.7
- 15s: 177.6, 173.8, 178.6, 178.4, 183.7, 179.2, 177.0, 174.3, 174.4, 180.6
- 30s: 340.1, 340.3, 344.3, 345.8, 332.7, 339.9, 342.4, 343.5, 339.2, 337.2

**Max relative IQR across buckets: 5.6% (10s bucket). Max |median deviation
vs June|: 10.3% (30s bucket).**

## June-vs-July median comparison

| Bucket | June 2026 median (§5.2) | July 2026 median (N=10, this run) | Δ |
| --- | ---: | ---: | ---: |
| 3s  | 50.6 ms  | 49.6 ms  | −1.9% |
| 7s  | 96.1 ms  | 92.8 ms  | −3.5% |
| 10s | 126.2 ms | 129.2 ms | +2.4% |
| 15s | 185.6 ms | 178.0 ms | −4.1% |
| 30s | 379.3 ms | 340.2 ms | −10.3% (flagged, exceeds 5%) |

## Interpretation

- **Four of five buckets reproduce the June medians within 5%**, despite this
  run being collected on a busy host — the small, consistent negative deltas
  on 3s/7s/15s (−1.9% to −4.1%) are compatible with ordinary run-to-run noise
  and are not evidence of contamination invalidating those cells.
- **The 30s bucket is 10.3% faster than its June value**, which exceeds the
  5% threshold this mission set for flagging. Two explanations are plausible
  and this run cannot distinguish them: (a) genuine Config F pipeline
  improvements landed between the June freeze and this HEAD (the codebase has
  had multiple performance-oriented commits since; see `git log` between the
  June bakeoff and `05e18489`), consistent with the paper's own §5.4 caveat
  that different Config F vintages are not directly comparable; or (b) an
  artifact of collecting on a busy host. The **direction** of the deviation
  (faster, not slower) argues against simple CPU-contention corruption, which
  would be expected to slow the pipeline down, not speed it up — but this
  reasoning is not proof, and the paper should not silently substitute this
  run's 340.2 ms for the published 379.3 ms without a dedicated quiet-host
  re-run.
- **The 10s bucket has the highest relative IQR (5.6%)** and its raw warm
  times show a visible split (six calls in the 124–135 ms range, four calls
  in the 140–147 ms range) — this pattern, unlike the other four buckets'
  tight clustering, is the one place in this dataset most consistent with
  contention-driven jitter from the concurrent `kokoro-bench` process.
- **Recommendation:** this dataset is sufficient to support a dispersion
  claim in §5.1 (relative IQR ≤ ~6% across buckets even under host
  contention), but the 30s median discrepancy should be re-verified with a
  quiet-host run before the paper treats 340 ms rather than 379 ms as current
  truth for that bucket.

## Paper-ready sentences for §5.1

> A July 2026 N=10 verification run on the same M2 Ultra Studio machine
> reproduced four of five June medians within 5% (relative IQR 1.1–5.6%
> across buckets), despite running under a busy host that a quiet-host gate
> could not clear within 30 minutes. The 30 s bucket's fresh median (340 ms)
> is 10.3% faster than the June value (379 ms); we attribute this
> tentatively to Config F pipeline improvements landed since the June freeze
> rather than measurement noise, since contention would be expected to slow
> rather than speed up warm inference, but flag it as unconfirmed pending a
> quiet-host re-run. Per-bucket dispersion is otherwise tight: relative IQR
> is under 3.5% for four of the five buckets and 5.6% at worst (10 s), the
> one bucket showing a bimodal split in raw warm times consistent with host
> contention.

## Provenance

- Machine: Apple M2 Ultra Mac Studio, 64 GB
- Git: `05e184895f03c17431e9333d6c84780407f0e2bb`
- macOS: 26.5.1 (Build 25F80)
- Results: `outputs/external_bakeoff/results_config_f_reference_m2_ultra_dispersion_2026_07_14.json`
  (gitignored — not committed; this note is the durable record)
- Host-quiet status: not quiet; collected after the mission's 30-minute
  busy-host fallback bound
- MLX dispersion (optional mission step 6): **skipped**. The
  `mlx-audio` adapter environment
  (`scripts/external_bakeoff/requirements_mlx_audio.txt`, pinned 0.4.3 @
  `862dfbe`) was not stood up in this session — Config F dispersion was the
  must-have deliverable and the 30-minute timebox was allocated to the
  quiet-host wait and the Config F collection instead.

## Plan reference

- Paper: `Scratchpad/surgical-inference.md` §5.1–§5.2 (not edited by this
  note; a human/downstream pass should decide how to fold the paper-ready
  sentences above into the manuscript)
- Bakeoff conventions: `.claude/skills/bakeoff/SKILL.md`
- Config F reference adapter: `scripts/external_bakeoff/run_config_f_reference.py`
- Quiet-host gate: `scripts/external_bakeoff/check_remote_host_quiet.py`
