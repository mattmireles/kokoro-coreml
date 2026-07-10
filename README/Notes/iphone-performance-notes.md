# iPhone Performance Notes

Physical-iPhone benchmark evidence for the Kokoro Core ML pipeline. Mac
benchmark history lives in [performance-notes.md](performance-notes.md);
iPhone debugging trails (failure modes, not timings) live in
[iphone-debug-notes.md](iphone-debug-notes.md). Device-lab procedure:
[iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md).

Newest entries first. Warmed inference only — first-load compile/cache effects
are excluded from every table, per the runbook.

## iPhone Bench v2: Release-build Config F ladder (supersedes v1 absolute timings)

**Collected:** 2026-06-10
**Status:** Complete for the staged policy on both phones. These are the
numbers to cite for absolute iPhone latency; the v1 tables below were Debug
(`-Onone`) builds and overstate wall time ~1.4–1.7x.

### Why v1 is superseded

The v1 bench app was built with the Xcode `Debug` configuration. Unoptimized
Swift inflated the CPU-side HnSF stage ~37x (the Core ML stage times were
barely affected — the distortion is all in Swift-side compute), which
overstated total wall ~1.7x on the A17 Pro and ~1.4x on the A14. The v1
tables remain valid for *arm-vs-arm comparison* (both arms paid the same
Debug tax) and for the jetsam/failure findings, but not for absolute
latency. All v2 rows are `Release` builds of the same `ios-bench` app, same
frozen inputs, same warmup discipline (2 warmups discarded, 5 warm calls,
median reported).

### Warm medians — Release, staged policy

Staged = decoder-pre on `.cpuAndNeuralEngine`; duration, f0n, generator on
`.cpuAndGPU` — the production Mac Config F policy. `.all` still hard-fails
on both phones with Core ML error -9 at the duration stage
(`last_vended_stage: "duration"`), identical to v1; see
[iphone-debug-notes.md](iphone-debug-notes.md).

| Bucket | 15 Pro Max median | 15 Pro Max RTF | 12 Pro median | 12 Pro RTF |
| --- | ---: | ---: | ---: | ---: |
| 3s  | 0.426 s | 0.152 | 0.864 s | 0.308 |
| 7s  | 0.865 s | 0.128 | 1.625 s | 0.241 |
| 15s | 1.860 s | 0.134 | 3.727 s | 0.268 |
| 30s | 3.742 s | 0.137 | 8.551 s | 0.312 |

The A17 Pro runs the 30s bucket ~7.3x realtime; the A14 ~3.5x realtime. The
padded-duration stage is the largest single line item at 30s: 0.799 s of
3.742 s (21%) on the A17 Pro, 1.558 s of 8.551 s (18%) on the A14 — the
motivation for the exact-duration experiment in
[kokoro-iphone-performance-v1.md](../Plans/kokoro-iphone-performance-v1.md).

Raw artifacts: `outputs/iphone_bench/results_15pm_ladder_release.json` and
`results_12pro_ladder_release.json` (gitignored; hardware `iPhone16,2` /
`iPhone13,3`, both iOS 26.5). A 15 Pro Max single-stage compute-unit matrix
run (`--mode matrix`) and the exact-duration A/B were still in flight when
this entry was written; results land here when pulled.

## iPhone Bench v1: Config F vs MLX Swift (kokoro-ios) on two iPhones

**Collected:** 2026-06-09
**Status:** Superseded for absolute timings by v2 above (these tables are
Debug builds — see "Why v1 is superseded"). Still the reference for the
MLX-arm comparison and the A14 jetsam finding; the external-bakeoff gap
("Config F iPhone timings remain absent") is closed.

### Setup and provenance

- **Bench app:** `ios-bench/` (XcodeGen project `KokoroIPhoneBench`), headless
  SwiftUI app running both arms in-process from bundled resources, results
  flushed to `Documents/results*.json` after every (arm, bucket) pair.
- **Our arm (Config F):** `KokoroPipeline` via `executeKokoroSynthesis`,
  models Xcode-precompiled to `.mlmodelc` from the shipped runtime
  `.mlpackage` set (duration t64/t128/t256/t512, f0ntrain
  t120/t280/t600/t1200, decoder_pre and decoder_har_post 3s/7s/15s/30s).
  Inputs are the frozen bakeoff JSONs from
  `scripts/prepare_swift_bench_inputs.py` (voice `af_heart`, speed 1.0; no
  10s input exists in that manifest, so the 10s bucket is absent on iPhone).
  Timing boundary: token IDs in to PCM out, identical to the Mac bakeoff.
- **MLX Swift arm:** `mlalma/kokoro-ios` tag `1.0.8` (commit `cd7f5f3`),
  vendored under `ios-bench/Vendor/kokoro-ios` with two build patches
  (declare the `MLXFast` product its source imports; drop the deleted test
  target) and one product change (static instead of dynamic library — see
  [iphone-debug-notes.md](iphone-debug-notes.md)). Weights
  `kokoro-v1_0.safetensors` (fp32) and `voices.npz` from the companion
  `mlalma/KokoroTestApp` clone. Its `generateAudio` API takes raw text, so
  MLX timings include the Misaki G2P pass; our arm starts from pre-tokenized
  IDs. The advantage MLX would need to overcome is small but nonzero —
  disclosed wherever published.
- **Devices:** iPhone 12 Pro (`iPhone13,3`, A14, 4 GB) and iPhone 15 Pro Max
  (`iPhone16,2`, A17 Pro, 8 GB), both iOS 26.5, unlocked, plugged in,
  foregrounded per the runbook. 2 warmups discarded, 5 recorded warm calls,
  median reported. Raw JSON artifacts: `outputs/iphone_bench/*.json`
  (gitignored); the full warm arrays are reproduced below so this note is
  self-contained.

### Warm medians

| Bucket | 15 Pro Max Config F | 15 Pro Max MLX | 12 Pro Config F | 12 Pro MLX |
| --- | ---: | ---: | ---: | ---: |
| 3s  | 0.702 s | 0.919 s | 1.383 s | 1.624 s |
| 7s  | 1.492 s | 1.875 s | 2.966 s | 2.405 s |
| 15s | 3.272 s | 3.805 s | 6.250 s | 5.022 s |
| 30s | 6.374 s | 7.792 s | 12.301 s | OOM (signal 9) |

Config F wins every bucket on the A17 Pro (1.16-1.31x, ~4-4.5x realtime). On
the 4 GB A14 the result is split: Config F wins 3s, MLX wins 7s and 15s
(~1.2x), and MLX cannot complete 30s — jetsam kills it after one iteration,
in a fresh process with MLX GPU cache capped at 256 MB, reproduced twice.
Config F runs 30s in 12.3 s on the same phone.

ALL published Config F iPhone rows ran the `staged` compute policy
(decoder-pre on cpuAndNeuralEngine; duration, f0n, and generator on
cpuAndGPU) because both iPhone ANE compilers reject the `.all` plan — see
the ANECCompile issue in
[iphone-debug-notes.md](iphone-debug-notes.md). The published Mac Config F
rows run the same staged per-stage policy
([performance-notes.md](performance-notes.md): "Config F rows use the
production-shaped staged policy"; `.all` is the historical Config F label
from [coreml-compute-unit-ablation.md](coreml-compute-unit-ablation.md)), so
the Mac and iPhone tables are policy-matched. The remaining confounds when
comparing them are the duration packages (exact-duration on the Mac frontier
rows, padded `t{n}` on iPhone) and the compile route (runtime-compiled
`.mlpackage` on Mac, Xcode-precompiled `.mlmodelc` on iPhone).

### Raw warm iterations (seconds)

iPhone 15 Pro Max (`iPhone16,2`, iOS 26.5):

| Arm | Bucket | Warm calls |
| --- | --- | --- |
| coreml (staged) | 3s | 0.702 0.767 0.716 0.675 0.644 |
| coreml (staged) | 7s | 1.492 1.471 1.536 1.536 1.483 |
| coreml (staged) | 15s | 3.269 3.297 3.264 3.284 3.272 |
| coreml (staged) | 30s | 6.374 6.367 6.319 7.752 6.394 |
| mlx | 3s | 0.837 0.857 0.919 1.020 1.159 |
| mlx | 7s | 2.373 1.813 1.748 1.900 1.875 |
| mlx | 15s | 3.805 3.963 3.786 3.835 3.798 |
| mlx | 30s | 7.865 7.762 7.795 7.777 7.792 |

iPhone 12 Pro (`iPhone13,3`, iOS 26.5):

| Arm | Bucket | Warm calls |
| --- | --- | --- |
| coreml (staged) | 3s | 1.612 1.684 1.383 1.355 1.321 |
| coreml (staged) | 7s | 3.232 2.969 2.966 2.905 2.937 |
| coreml (staged) | 15s | 7.550 6.154 6.463 6.064 6.250 |
| coreml (staged) | 30s | 12.152 12.325 12.128 12.301 14.625 |
| mlx | 3s | 1.285 1.431 1.772 1.848 1.624 |
| mlx | 7s | 2.405 2.393 2.409 2.406 2.404 |
| mlx | 15s | 5.038 5.017 5.021 5.022 5.026 |
| mlx | 30s | one compile-inclusive call ~10.5-11.4 s, then jetsam (twice) |

Process-isolation caveat: the iPhone 12 Pro MLX 7s/15s rows and the 15 Pro
Max MLX 7s/15s/30s rows came from MLX-only relaunches (`--arms mlx`) after
the combined run was jetsammed; the Core ML rows and MLX 3s rows came from
the combined run. Each arm's timings are steady-state within their process.

### Follow-ups

- Isolate which model `ANECCompile` rejects (per-stage smoke test), then try
  an iPhone-targeted export of that stage to unlock a true `.all` policy.
  Strategy and tooling for this (on-device compute-plan dump, ANE admission
  limits, re-chunking designs):
  [Kokoro A14 iPhone generator execution guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md).
- Add a 10s input to the bakeoff manifest so iPhone tables cover all five
  shipped buckets.
- The MLX 3s rows on both phones trend upward across iterations (e.g. 0.837
  → 1.159 on the A17 Pro), suggesting thermal or cache-pressure drift; a
  longer-window rerun would firm those medians up.
