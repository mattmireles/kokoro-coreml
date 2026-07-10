# External Bakeoff Phase 2 Run Log

**Date:** 2026-06-05
**Plan:** `README/Plans/kokoro-external-bakeoff-v1.md`
**Status:** M2 Studio, m2-air, and irvine-m1 now have Config F, MLX, Soniqo,
and laishere JSON plus durable spot-check WAVs for every successful result
cell; Phase 2 collection is complete, local hardware-placement evidence is
captured, and human listening evidence is still pending.

## M2 Studio Precheck

Before local collection, botnet health was green:

```json
{
  "ok": true,
  "queueDepth": 0,
  "claimedFresh": 0,
  "freshWorkerCount": 2,
  "canaryStatus": "passing",
  "canaryWorkerId": "operator-prove-live"
}
```

After the spot-check rerun, botnet health was still green:

```json
{
  "ok": true,
  "queueDepth": 0,
  "claimedFresh": 0,
  "freshWorkerCount": 3,
  "canaryStatus": "passing",
  "canaryWorkerId": "operator-prove-live"
}
```

## M2 Studio Result Files

Generated, uncommitted result files:

- `outputs/external_bakeoff/results_config_f_reference_m2-studio.json`
- `outputs/external_bakeoff/results_mlx_audio_m2-studio.json`
- `outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_m2-studio.json`
- `outputs/external_bakeoff/results_laishere_kokoro_coreml_m2-studio.json`

Each file validates against `scripts/external_bakeoff/schema.py`.

## Config F Same-Window Result

Config F used the main checkout Core ML artifacts at
`/Users/mm/Documents/GitHub/kokoro-coreml/coreml` and the persistent
`kokoro-bench --batch` adapter.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Bucket |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.125504 | 0.131485 | 5 | 2.800 | 3s |
| 7s | ok | 0.309493 | 0.284174 | 5 | 6.750 | 7s |
| 10s | ok | 0.570868 | 0.548194 | 5 | 9.625 | 10s |
| 15s | ok | 0.647346 | 0.632877 | 5 | 13.900 | 15s |
| 30s | ok | 1.389132 | 1.191795 | 5 | 27.400 | 30s |

The 30s first compile spent roughly 20 minutes in Core ML on-device AOT
compilation before timed synthesis. A sampled stack showed Core ML / Espresso
inside program-library preparation and shortest-path segmentation, so the long
silence was compiler work, not a harness deadlock.

## MLX Result

`Blaizzy/mlx-audio` was run from the pinned current clone
`862dfbe5338e91df6f74ac986b4df8bede7961a6` with `mlx-audio 0.4.3` and
`mlx 0.31.2`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | error | - | - | 0 | - | deterministic broadcast-shape failure |
| 7s | ok | 0.195928 | 0.223944 | 5 | 6.750 | - |
| 10s | ok | 4.737087 | 0.288822 | 5 | 9.600 | - |
| 15s | ok | 0.438077 | 0.376303 | 5 | 13.900 | - |
| 30s | ok | 0.930204 | 0.762699 | 5 | 27.375 | - |

The 3s cell failed on the initial run and a one-input retry with:

```text
ValueError: [broadcast_shapes] Shapes (1,67200,1) and (1,67500,9) cannot be broadcast.
```

Because the adapter is using the public current clone and the shared manifest
text, this is recorded as competitor behavior rather than patched locally.

## Soniqo Speech Swift Result

`soniqo/speech-swift` was run through the generated macOS Swift CLI at pinned
SHA `0d09a2ed5464c7c94cf4545be59043c21f8775ea` with
`KokoroTTSModel.fromPretrained(computeUnits: .all)`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.615211 | 0.071711 | 5 | 2.700 | duration mismatch |
| 7s | ok | 0.432999 | 0.069311 | 5 | 5.000 | truncated versus manifest |
| 10s | ok | 0.397993 | 0.071024 | 5 | 5.000 | truncated versus manifest |
| 15s | ok | 0.411708 | 0.068065 | 5 | 5.000 | truncated versus manifest |
| 30s | ok | 0.414261 | 0.069504 | 5 | 5.000 | truncated versus manifest |

These Soniqo cells are not quality-parity evidence yet. The timing is useful
for implementation behavior, but the emitted audio duration must be resolved or
clearly caveated before using the cells in the paper table.

Source and artifact check:

- `KokoroTTSModel.fromPretrained(...)` downloads `kokoro_5s.mlmodelc/**`.
- `KokoroNetwork` would load `kokoro_10s` or `kokoro_15s` if present, but the
  upstream `aufklarer/Kokoro-82M-CoreML` file listing only contains
  `kokoro_5s.mlmodelc`.
- Local caches under both Hugging Face and `qwen3-speech` contain only
  `kokoro_5s.mlmodelc`.

This makes the 5.0s cap public-comparator behavior for the selected Soniqo
model artifact, not an adapter timing-boundary bug.

## Laishere Core ML Backup Result

`laishere/kokoro-coreml` was probed as the long-bucket Core ML backup at pinned
SHA `484907db6a8347a6afb6e7b86850ea2878c6a3fb`. The repo does not ship
prebuilt `.mlpackage` artifacts, so the seven public Core ML packages were
converted under `/tmp/kokoro-external-bakeoff/laishere-kokoro-coreml/output`
with:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python convert.py --max-frames 2000
```

`convert.py` emitted all seven packages, then its final chained validation hit
a Core ML/Espresso dynamic-shape error in the PostAlbert path:

```text
RuntimeError: Unable to compute the prediction using ML Program
Tile: Shape deduction failed as reps[0]=-1317260229 < 0
```

The repo's standalone `benchmark.py --n-runs 1` still ran successfully against
the generated packages. It rendered six built-in passages from 1.50s to 28.12s
audio with chain times from 62.9ms to 606.7ms on this M2 Studio. The normalized
adapter then ran the shared runtime manifest with N=5 warm calls:

| Input | Status | Cold s | Warm median s | Warm N | Observed s | T_a |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3s | ok | 0.236952 | 0.212307 | 5 | 2.775 | 111 |
| 7s | ok | 0.359018 | 0.403259 | 5 | 6.800 | 272 |
| 10s | ok | 0.839707 | 0.626281 | 5 | 9.625 | 385 |
| 15s | ok | 0.676127 | 0.429827 | 5 | 13.975 | 559 |
| 30s | ok | 1.955135 | 0.925116 | 5 | 27.375 | 1095 |

These cells are the current long-bucket Core ML parity backup. The adapter
times the seven-stage Core ML chain only; G2P and feed preparation are outside
the timed calls, matching laishere's public benchmark boundary. That boundary
must be stated if these numbers are used beside Soniqo or MLX in the paper.

## Spot-Check WAV Support

The M2 Studio collection was rerun after adapters were updated to write durable
spot-check WAV files:

- Config F keeps the last warm `kokoro-bench` WAV per input.
- MLX writes the last warm PCM array as mono 16-bit WAV.
- Soniqo writes the last warm Swift `[Float]` audio as mono 16-bit WAV.

One-input smokes verified valid mono 24 kHz WAV files for Config F, MLX, and
Soniqo. The full M2 Studio rerun then produced durable WAVs for every successful
result cell. MLX has no 3s WAV because that cell errors before audio is
materialized.

## M2 Air Partial Collection

`m2-air` was prepared from a disposable checkout at
`/tmp/kokoro-coreml-bakeoff-run`, using that host's existing Core ML artifacts
from `/Users/mattmireles/Documents/GitHub/kokoro-coreml/coreml`. Before the
run, `pmset -g therm` reported no thermal or performance warning. Config F,
MLX, Soniqo, and laishere completed and were copied back to:

- `outputs/external_bakeoff/results_config_f_reference_m2-air.json`
- `outputs/external_bakeoff/results_mlx_audio_m2-air.json`
- `outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_m2-air.json`
- `outputs/external_bakeoff/results_laishere_kokoro_coreml_m2-air.json`

All four files validate against `scripts/external_bakeoff/schema.py`, and every
successful cell has a mono 24 kHz spot-check WAV.

### M2 Air Config F

| Input | Status | Cold s | Warm median s | Warm N | Observed s |
| --- | --- | ---: | ---: | ---: | ---: |
| 3s | ok | 0.313816 | 0.317402 | 5 | 2.800 |
| 7s | ok | 0.683149 | 0.808074 | 5 | 6.750 |
| 10s | ok | 1.054014 | 1.373335 | 5 | 9.625 |
| 15s | ok | 2.134623 | 2.052364 | 5 | 13.900 |
| 30s | ok | 9.447099 | 9.559135 | 5 | 27.400 |

The 30s bucket compiled successfully after the long Core ML AOT window. The
fanless host stayed thermally nominal during the run.

#### M2 Air Config F Warmed-Inference Correction

The original 30s cell is compile-inclusive and not used for paper-facing warmed
inference. It used the padded `t512` duration model and showed Core ML AOT
compile/load work in the recorded window. A corrected run rebuilt the current
Swift benchmark source on the host, set `KOKORO_USE_EXACT_DURATION_MODELS=1`,
discarded three preflight calls, then recorded 20 warm calls:

| Input | Status | Post-preflight cold s | Warm median s | Warm N | Observed s | Duration model |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 30s | ok | 4.094875 | 3.943964 | 20 | 27.400 | exact_t476 |

The recorded warm range was `3.656564s` to `4.035546s`, with median observed
RTF `0.1440715`. The last recorded call spent `0.049253s` in Duration Core ML
and `3.484556s` in generator Core ML, proving the large original 30s cell was
compile/cache contamination rather than warmed Duration inference.

### M2 Air MLX

`Blaizzy/mlx-audio` ran from pinned SHA
`862dfbe5338e91df6f74ac986b4df8bede7961a6` in a disposable Python 3.12 venv
with `mlx-audio 0.4.3`, `mlx 0.31.2`, and MLX default device `gpu`.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | error | - | - | 0 | - | deterministic broadcast-shape failure |
| 7s | ok | 0.670390 | 0.685626 | 5 | 6.750 | - |
| 10s | ok | 20.802833 | 0.835810 | 5 | 9.600 | model/cache cold start |
| 15s | ok | 1.636798 | 1.520953 | 5 | 13.900 | - |
| 30s | ok | 2.851399 | 2.600340 | 5 | 27.375 | - |

The 3s error matched the M2 Studio failure:

```text
ValueError: [broadcast_shapes] Shapes (1,67200,1) and (1,67500,9) cannot be broadcast.
```

### M2 Air Soniqo

`soniqo/speech-swift` ran from pinned SHA
`0d09a2ed5464c7c94cf4545be59043c21f8775ea`. The first retry was required
because SwiftPM attempted to use the remote keychain credential helper while
downloading Soniqo's public `SpeechCore.xcframework` artifact; rerunning with
the Git credential helper disabled forced anonymous access and completed.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 1.189790 | 1.097402 | 5 | 2.700 | - |
| 7s | ok | 1.210679 | 1.135835 | 5 | 5.000 | public 5s artifact |
| 10s | ok | 1.237192 | 1.122956 | 5 | 5.000 | public 5s artifact |
| 15s | ok | 1.273711 | 1.125471 | 5 | 5.000 | public 5s artifact |
| 30s | ok | 1.233093 | 1.123540 | 5 | 5.000 | public 5s artifact |

### M2 Air laishere

`laishere/kokoro-coreml` ran from pinned SHA
`484907db6a8347a6afb6e7b86850ea2878c6a3fb`. M2 Air did not have `uv`, so a
disposable Python 3.12 venv was created under `/tmp/kokoro-external-bakeoff`.
The repo's editable install failed because setuptools rejected the flat
top-level `assets` and `iOSDemo` layout, so the declared dependencies were
installed directly. Conversion used:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /tmp/kokoro-external-bakeoff/laishere-venv/bin/python \
  convert.py --max-frames 2000
```

The conversion produced all seven packages and reported
`mel_corr=0.993667`. Non-fatal caveats: coremltools warned that scikit-learn
`1.9.0` and torch `2.12.0` are newer than its tested range.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | T_a |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3s | ok | 0.289868 | 0.142010 | 5 | 2.775 | 111 |
| 7s | ok | 0.330730 | 0.316934 | 5 | 6.825 | 273 |
| 10s | ok | 0.710552 | 0.450170 | 5 | 9.650 | 386 |
| 15s | ok | 0.746766 | 0.657251 | 5 | 13.925 | 557 |
| 30s | ok | 1.616250 | 1.476442 | 5 | 27.350 | 1094 |

## Irvine M1 Completed Collection

`irvine-m1` was rerun in a low-pressure window with stdout/stderr redirected
from the start. Botnet health stayed clean during the successful Config F, MLX,
Soniqo, and laishere collection: queue depth `0`, fresh claims `0`, and canary
passing across the sampled polls. Result files copied back:

- `outputs/external_bakeoff/results_config_f_reference_irvine-m1.json`
- `outputs/external_bakeoff/results_mlx_audio_irvine-m1.json`
- `outputs/external_bakeoff/results_soniqo_speech_swift_kokoro_irvine-m1.json`
- `outputs/external_bakeoff/results_laishere_kokoro_coreml_irvine-m1.json`

All four files validate against `scripts/external_bakeoff/schema.py`, and every
successful cell has a mono 24 kHz spot-check WAV. No remote bakeoff processes
were left running after collection.

After copying result artifacts back, disposable bakeoff checkouts, external
clones, virtualenvs, Swift build products, and temporary logs were removed from
`irvine-m1`, `m2-air`, and local `/tmp`. The committed evidence lives in this
note and `README/Notes/performance-notes.md`; generated JSON/WAV artifacts
remain ignored under `outputs/external_bakeoff/`.

### Irvine M1 Config F

The redirected Config F run took about 45 minutes wall-clock because Core ML
compiled several buckets on the host. The 30s bucket alone had a large warm
median because one of the warm calls entered another long Core ML compile/load
path; this is preserved as observed host behavior.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Bucket |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 0.285956 | 0.304639 | 5 | 2.800 | 3s |
| 7s | ok | 0.647701 | 0.696148 | 5 | 6.750 | 7s |
| 10s | ok | 1.035011 | 1.348528 | 5 | 9.625 | 10s |
| 15s | ok | 1.372462 | 1.672729 | 5 | 13.900 | 15s |
| 30s | ok | 9.114338 | 16.119647 | 5 | 27.400 | 30s |

#### Irvine M1 Config F Warmed-Inference Correction

The original 30s cell is compile-inclusive and not used for paper-facing warmed
inference. A corrected run set `KOKORO_USE_EXACT_DURATION_MODELS=1`, discarded
three preflight calls, then recorded 20 warm calls:

| Input | Status | Post-preflight cold s | Warm median s | Warm N | Observed s | Duration model |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 30s | ok | 2.085811 | 2.076132 | 20 | 27.400 | exact_t476 |

The recorded warm range was `2.051301s` to `2.104326s`, with median observed
RTF `0.07584`. The last recorded call spent `0.118366s` in Duration Core ML and
`1.72577s` in generator Core ML.

### Irvine M1 MLX

`Blaizzy/mlx-audio` ran from pinned SHA
`862dfbe5338e91df6f74ac986b4df8bede7961a6` in the disposable Python 3.12 venv.
The 3s cell failed with the same broadcast-shape error seen on M2 Studio and
M2 Air.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | error | - | - | 0 | - | deterministic broadcast-shape failure |
| 7s | ok | 0.807166 | 0.823996 | 5 | 6.750 | - |
| 10s | ok | 20.027821 | 1.124308 | 5 | 9.600 | model/cache cold start |
| 15s | ok | 1.662442 | 1.589512 | 5 | 13.900 | - |
| 30s | ok | 3.293409 | 3.077911 | 5 | 27.375 | - |

### Irvine M1 Soniqo

`soniqo/speech-swift` ran from pinned SHA
`0d09a2ed5464c7c94cf4545be59043c21f8775ea`. The first launch used the wrong
adapter flag and exited before running. A second launch hit Xcode's system Git
credential helper while downloading Soniqo's public `SpeechCore.xcframework`.
The successful run used `GIT_CONFIG_NOSYSTEM=1` plus an empty credential helper
to force anonymous artifact download.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | Caveat |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 3s | ok | 1.395284 | 1.330872 | 5 | 2.700 | - |
| 7s | ok | 1.391859 | 1.343588 | 5 | 5.000 | public 5s artifact |
| 10s | ok | 1.413150 | 1.313932 | 5 | 5.000 | public 5s artifact |
| 15s | ok | 1.460957 | 1.343619 | 5 | 5.000 | public 5s artifact |
| 30s | ok | 1.431827 | 1.351175 | 5 | 5.000 | public 5s artifact |

### Irvine M1 laishere

`laishere/kokoro-coreml` ran from pinned SHA
`484907db6a8347a6afb6e7b86850ea2878c6a3fb`. A disposable Python 3.12 venv was
created under `/tmp/kokoro-external-bakeoff`, declared dependencies were
installed directly, and conversion used:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 \
  /tmp/kokoro-external-bakeoff/laishere-venv/bin/python \
  convert.py --max-frames 2000
```

The conversion produced all seven packages and reported
`mel_corr=0.993145`. Non-fatal caveats matched M2 Air: coremltools warned that
scikit-learn `1.9.0` and torch `2.12.0` are newer than its tested range.

| Input | Status | Cold s | Warm median s | Warm N | Observed s | T_a |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3s | ok | 1.102515 | 0.176330 | 5 | 2.775 | 111 |
| 7s | ok | 1.239340 | 0.394566 | 5 | 6.750 | 270 |
| 10s | ok | 1.877005 | 0.593892 | 5 | 9.625 | 385 |
| 15s | ok | 1.659649 | 0.912001 | 5 | 13.950 | 558 |
| 30s | ok | 2.791752 | 2.135142 | 5 | 27.375 | 1095 |

## Irvine M1 Earlier Aborted Collection

`irvine-m1` was prepared from the same disposable checkout and built the current
`kokoro-bench` successfully. The first Config F attempt reached the 30s bucket,
and stack sampling showed Core ML / Espresso / ANE on-device AOT compilation.
Because the original SSH session had been lost, the process later sat idle
after writing `30s.wav` without producing a result JSON; this was treated as a
detached-pipe stall, not a completed benchmark.

A redirected rerun was started with logs at `/tmp/kokoro-configf-irvine.log`.
It progressed through 10s and 15s and re-entered 30s compilation. During that
rerun, botnet health showed production pressure:

```json
{
  "ok": true,
  "queueDepth": 9,
  "claimedFresh": 11,
  "freshWorkerCount": 3,
  "canaryStatus": "passing"
}
```

Per the plan's no-disruption guardrail, the `irvine-m1` benchmark was killed
before a result JSON was produced. After stopping it, health improved to
`queueDepth=1` with canary still passing. The partial `irvine-m1` WAVs under
the remote `/tmp` directory are not publication data because there is no
schema-valid JSON for that host.

## Continuation Gate: Production Pressure

On 2026-06-05, a follow-up attempt checked for a low-traffic window before
resuming `m2-air` Soniqo/laishere or restarting `irvine-m1`. No remote bakeoff
processes were running on `irvine-m1` or `m2-air`, but botnet health showed
active production claims throughout the polling window:

| Time | Queue depth | Claimed fresh | Fresh workers | Canary |
| --- | ---: | ---: | ---: | --- |
| 00:14:29 | 1 | 14 | 3 | passing |
| 00:15:30 | 2 | 13 | 3 | passing |
| 00:16:31 | 2 | 13 | 3 | passing |
| 00:17:33 | 0 | 15 | 3 | passing |
| 00:18:34 | 0 | 15 | 3 | passing |

Because `claimedFresh` stayed high, no additional benchmark runs were started.
Resume Phase 2 only after both queue depth and active fresh claims are low
enough that a benchmark host will not compete with production TTS work.

## Phase 3 Partial Quality Sanity

While remote collection was paused, the existing spot-check WAVs were run
through `scripts/audio_quality_probe.py`. Config F WAVs were used as
machine-local references, and competitor WAVs were classified against derived
thresholds for RMS, activity, zero-crossing rate, speech-band energy, clipping,
sample rate, and channel count.

Generated, uncommitted reports:

- `outputs/external_bakeoff/quality/m2-studio/audio_quality_report.json`
- `outputs/external_bakeoff/quality/m2-studio/audio_quality_summary.md`
- `outputs/external_bakeoff/quality/m2-air/audio_quality_report.json`
- `outputs/external_bakeoff/quality/m2-air/audio_quality_summary.md`
- `outputs/external_bakeoff/quality/irvine-m1/audio_quality_report.json`
- `outputs/external_bakeoff/quality/irvine-m1/audio_quality_summary.md`

No collected competitor WAV was rejected by the waveform sanity gate. Every
candidate requires human listening before quality parity is claimed. Soniqo's
longer-bucket files remain duration caveats: they look healthy by waveform
metrics but are 5.0s clips for 7s, 10s, 15s, and 30s manifest inputs.

## Local Hardware-Placement Capture

After the warmed 30s correction, local M2 Studio `powermetrics` captures were
collected for Config F, MLX, Soniqo, and laishere placement context. The first
attempted Config F 30s trace entered a long padded `t512` Core ML compile/load
window and was stopped before result JSON; it remains compile-window evidence
only and is not an inference-speed cell.

The usable Config F capture ran the debug `kokoro-bench` binary on the 3s input with
five discarded preflight calls and forty recorded warm calls, while
`powermetrics` sampled every 500 ms with `cpu_power,gpu_power,ane_power`.
Artifacts are ignored under `outputs/external_bakeoff/placement/`:

- `results_config_f_reference_m2-studio_3s_warm_placement.json`
- `config_f_m2-studio_3s_warm_powermetrics.txt`

Summary:

| Signal | N | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Warm wall time | 40 | 1.050493s | 1.075376s | 1.887251s |
| CPU Power | 80 | 9987 mW | 17801 mW | 35342 mW |
| GPU Power | 160 | 813 mW | 2384 mW | 21635 mW |
| ANE Power | 80 | 0 mW | 0 mW | 0 mW |
| GPU HW active residency | 80 | 52.56% | 68.835% | 99.17% |

The run used a debug binary and padded duration artifacts, so the warm wall
time is not a replacement for the paper-facing Config F table. The placement
signal is still useful: this local path was CPU/GPU-dominant, not ANE-bound.
The last recorded call spent `0.082838s` in Duration Core ML, `0.004875s` in
F0Ntrain Core ML, `0.004963s` in DecoderPre Core ML, `0.034667s` in generator
Core ML, and `0.915704s` in Swift HnSF. Fleet health remained passing after the
capture (`queueDepth=0`, `freshWorkerCount=3`, canary passing), though
`claimedFresh` rose from 5 to 11 during the window.

The MLX capture recreated the pinned `Blaizzy/mlx-audio` checkout at
`862dfbe5338e91df6f74ac986b4df8bede7961a6` in `/tmp`, installed
`mlx-audio 0.4.3` in an ignored venv, warmed the model with a successful 7s
smoke, then ran the 7s input with thirty recorded warm calls while
`powermetrics` sampled the same power counters. Artifacts are ignored under
`outputs/external_bakeoff/placement/`:

- `results_mlx_audio_m2-studio_7s_warm_placement.json`
- `mlx_m2-studio_7s_warm_powermetrics.txt`

Summary:

| Signal | N | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Warm wall time | 30 | 0.193810s | 0.2207245s | 0.315039s |
| CPU Power | 114 | 9244 mW | 22585.5 mW | 34538 mW |
| GPU Power | 228 | 593 mW | 791.5 mW | 25391 mW |
| ANE Power | 114 | 0 mW | 0 mW | 0 mW |
| GPU HW active residency | 114 | 32.6% | 49.435% | 98.28% |

This is placement evidence for the primary MLX competitor, not a replacement
latency cell. Fleet health remained passing after the MLX capture
(`queueDepth=0`, `claimedFresh=8`, `freshWorkerCount=3`, canary passing).

The Soniqo capture restored the pinned `soniqo/speech-swift` checkout at
`0d09a2ed5464c7c94cf4545be59043c21f8775ea`, resolved SwiftPM dependencies,
rebuilt the generated `SoniqoKokoroBench` CLI in an ignored work directory, and
ran the 3s input with thirty recorded warm calls using `computeUnits: .all`.
Artifacts are ignored under `outputs/external_bakeoff/placement/`:

- `results_soniqo_m2-studio_3s_warm_placement.json`
- `soniqo_m2-studio_3s_warm_powermetrics.txt`

Summary:

| Signal | N | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Warm wall time | 30 | 0.067434s | 0.0690795s | 0.073503s |
| CPU Power | 120 | 8019 mW | 18148.5 mW | 42770 mW |
| GPU Power | 240 | 552 mW | 751 mW | 49274 mW |
| ANE Power | 120 | 0 mW | 0 mW | 0 mW |
| GPU HW active residency | 120 | 31.55% | 50.465% | 98.15% |

This is placement evidence for the primary iOS/Core ML comparator, not a
replacement latency cell. On this M2 Studio run, Soniqo used Core ML with
`computeUnits: .all` but did not show ANE power draw. Fleet health remained
passing after the Soniqo capture (`claimedFresh=0`, `freshWorkerCount=3`,
canary passing), though queue depth had risen to 8.

The laishere capture restored the pinned `laishere/kokoro-coreml` checkout at
`484907db6a8347a6afb6e7b86850ea2878c6a3fb`, installed its Python dependencies
into the ignored venv, and regenerated the seven `.mlpackage` files locally.
Conversion produced all packages but its own E2E validation failed in
`KokoroPostAlbert` with a Core ML dynamic-shape BNNS/Espresso error under the
newer local stack (`coremltools 9.0`, `torch 2.10.0`). The generated packages
still ran through `run_laishere_kokoro_coreml.py` for the 3s input, so a
placement trace was captured with thirty recorded warm calls. Artifacts are
ignored under `outputs/external_bakeoff/placement/`:

- `results_laishere_m2-studio_3s_warm_placement.json`
- `laishere_m2-studio_3s_warm_powermetrics.txt`

Summary:

| Signal | N | Min | Median | Max |
| --- | ---: | ---: | ---: | ---: |
| Warm wall time | 30 | 0.083475s | 0.091534s | 0.105189s |
| CPU Power | 97 | 4758 mW | 12886 mW | 30393 mW |
| GPU Power | 194 | 540 mW | 809 mW | 4836 mW |
| ANE Power | 97 | 0 mW | 0 mW | 0 mW |
| GPU HW active residency | 97 | 38.54% | 54.23% | 94.25% |

This is placement evidence for the laishere backup Core ML chain, not a
replacement latency cell. Fleet health remained passing after the laishere
capture (`queueDepth=0`, `claimedFresh=5`, `freshWorkerCount=3`, canary
passing).

### M2 Studio Quality Probe

| Impl | Input | Decision | Duration s | RMS |
| --- | --- | --- | ---: | ---: |
| Config F reference | 3s | reference_pass | 2.800 | 4433.4 |
| Config F reference | 7s | reference_pass | 6.750 | 4696.9 |
| Config F reference | 10s | reference_pass | 9.625 | 4261.0 |
| Config F reference | 15s | reference_pass | 13.900 | 5226.7 |
| Config F reference | 30s | reference_pass | 27.400 | 4590.8 |
| MLX | 7s | needs_listening | 6.750 | 3471.8 |
| MLX | 10s | needs_listening | 9.600 | 2355.2 |
| MLX | 15s | needs_listening | 13.900 | 4836.9 |
| MLX | 30s | needs_listening | 27.375 | 3267.7 |
| Soniqo | 3s | needs_listening | 2.700 | 4504.1 |
| Soniqo | 7s | needs_listening | 5.000 | 5136.5 |
| Soniqo | 10s | needs_listening | 5.000 | 4235.7 |
| Soniqo | 15s | needs_listening | 5.000 | 4937.7 |
| Soniqo | 30s | needs_listening | 5.000 | 5243.8 |
| laishere | 3s | needs_listening | 2.775 | 4592.0 |
| laishere | 7s | needs_listening | 6.800 | 3864.8 |
| laishere | 10s | needs_listening | 9.625 | 3279.0 |
| laishere | 15s | needs_listening | 13.975 | 4906.4 |
| laishere | 30s | needs_listening | 27.375 | 3602.0 |

### M2 Air Quality Probe

| Impl | Input | Decision | Duration s | RMS |
| --- | --- | --- | ---: | ---: |
| Config F reference | 3s | reference_pass | 2.800 | 920.8 |
| Config F reference | 7s | reference_pass | 6.750 | 4603.5 |
| Config F reference | 10s | reference_pass | 9.625 | 4204.8 |
| Config F reference | 15s | reference_pass | 13.900 | 6087.4 |
| Config F reference | 30s | reference_pass | 27.400 | 4592.0 |
| MLX | 7s | needs_listening | 6.750 | 3424.5 |
| MLX | 10s | needs_listening | 9.600 | 2346.2 |
| MLX | 15s | needs_listening | 13.900 | 4788.8 |
| MLX | 30s | needs_listening | 27.375 | 3114.0 |
| Soniqo | 3s | needs_listening | 2.700 | 4483.9 |
| Soniqo | 7s | needs_listening | 5.000 | 5140.1 |
| Soniqo | 10s | needs_listening | 5.000 | 4244.0 |
| Soniqo | 15s | needs_listening | 5.000 | 4911.8 |
| Soniqo | 30s | needs_listening | 5.000 | 5331.4 |
| laishere | 3s | needs_listening | 2.775 | 4606.1 |
| laishere | 7s | needs_listening | 6.825 | 3372.6 |
| laishere | 10s | needs_listening | 9.650 | 3060.5 |
| laishere | 15s | needs_listening | 13.925 | 4919.3 |
| laishere | 30s | needs_listening | 27.350 | 3570.8 |

### Irvine M1 Quality Probe

| Impl | Input | Decision | Duration s | RMS |
| --- | --- | --- | ---: | ---: |
| Config F reference | 3s | reference_pass | 2.800 | 4456.3 |
| Config F reference | 7s | reference_pass | 6.750 | 4811.3 |
| Config F reference | 10s | reference_pass | 9.625 | 4252.2 |
| Config F reference | 15s | reference_pass | 13.900 | 5229.9 |
| Config F reference | 30s | reference_pass | 27.400 | 4463.0 |
| MLX | 7s | needs_listening | 6.750 | 3485.3 |
| MLX | 10s | needs_listening | 9.600 | 2298.6 |
| MLX | 15s | needs_listening | 13.900 | 4753.2 |
| MLX | 30s | needs_listening | 27.375 | 3030.6 |
| Soniqo | 3s | needs_listening | 2.700 | 4483.9 |
| Soniqo | 7s | needs_listening | 5.000 | 5140.1 |
| Soniqo | 10s | needs_listening | 5.000 | 4244.0 |
| Soniqo | 15s | needs_listening | 5.000 | 4911.8 |
| Soniqo | 30s | needs_listening | 5.000 | 5331.4 |
| laishere | 3s | needs_listening | 2.775 | 4627.4 |
| laishere | 7s | needs_listening | 6.750 | 3615.4 |
| laishere | 10s | needs_listening | 9.625 | 3087.6 |
| laishere | 15s | needs_listening | 13.950 | 4936.5 |
| laishere | 30s | needs_listening | 27.375 | 3651.0 |

## Remaining Work

- Keep every headline comparison on warmed inference only: compile-inclusive
  and cold-start timings are operational evidence, not ranking inputs.
- Decide whether the MLX 3s public-implementation failure is a paper caveat or
  requires an alternate, predeclared 3s input.
- Decide how the paper table presents Soniqo's high-adoption 5s-only result
  beside laishere's lower-adoption long-bucket Core ML backup.
- Listen to all `needs_listening` WAVs before making quality-parity claims.
