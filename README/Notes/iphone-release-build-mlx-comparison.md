# iPhone Release-Build Comparison: Config F vs MLX Swift (kokoro-ios)

**Collected:** 2026-07-14
**Purpose:** The surgical-inference paper abstract claims our Core ML pipeline
beats the MLX Swift Kokoro port "on every bucket" on the iPhone 15 Pro Max.
The underlying comparison ([iphone-performance-notes.md](iphone-performance-notes.md)
v1, 2026-06-09) was collected with Debug builds on both arms; the v2 Release
ladder (2026-06-10) re-ran only our arm. The MLX arm had never been measured
in Release. This note re-establishes the comparison with Release builds on
both arms and bounds the G2P boundary asymmetry numerically.

**Verdict: the claim survives, with a thermal-protocol condition.** In
thermally matched Release runs on the A17 Pro, Config F (staged) beats MLX
Swift on every bucket by 1.25–1.44x. The Debug-vs-Debug margins (1.16–1.31x)
were not an artifact of the Debug tax — but the tax was NOT symmetric
(MLX sped up ~1.4–1.6x in Release, our arm's wall barely moved), so the
margin survived by measurement, not by the v1 note's symmetry assumption.
One caveat gates the wording: without thermal matching, ordering effects can
flip the 15s/30s buckets (observed once, disclosed below). "Every bucket"
holds for warm medians under the runbook's matched protocol.

## Setup and provenance

- **Device:** iPhone 15 Pro Max (`iPhone16,2`, A17 Pro, 8 GB), iOS 26.5,
  unlocked, Developer Mode on, foregrounded, idle timer disabled. The
  secondary target (iPhone 12 Pro, `iPhone13,3`) was NOT re-run — see
  "iPhone 12 Pro status" below.
- **Host:** macOS Darwin 25.5.0, Xcode 26.6 (17F113). The June v1/v2 runs
  used Xcode 26.5; the on-device OS is unchanged (iOS 26.5). The June
  SwiftBuild `clang -v -E` probe stall did not reproduce under 26.6.
- **Bench app:** `ios-bench/` (`KokoroIPhoneBench`), built with the Xcode
  **Release** configuration (`-configuration Release`, verified
  `** BUILD SUCCEEDED **`), one binary running both arms in separate
  processes via `--arms`, per the jetsam protocol in
  [iphone-debug-notes.md](iphone-debug-notes.md).
- **Our arm (Config F):** `KokoroPipeline` via `executeKokoroSynthesis`,
  repo branch `codex/kokoro-drop-in-sdk-v1` @ `05e18489`, models
  Xcode-precompiled to `.mlmodelc` from the shipped runtime `.mlpackage` set,
  padded duration path (default; NOT `--exact-duration`). Timing boundary:
  token IDs in → 24 kHz PCM out.
- **MLX arm:** `mlalma/kokoro-ios` tag `1.0.8` (commit `cd7f5f3`), vendored
  with the June build patches, `mlx-swift` 0.31.4, fp32
  `kokoro-v1_0.safetensors` + `voices.npz`, GPU cache capped at 256 MB.
  Timing boundary: raw text in → PCM out (includes Misaki G2P — bounded
  below).
- **Bench-only source deltas vs `05e18489`** (uncommitted at run time, both
  additive, neither touches the timed code paths of either arm):
  `ios-bench/Sources/BenchApp.swift` gained a `--mode g2p` loop, and the
  vendored `KokoroTTS.swift` gained `phonemizeOnlyForBench` — used only by
  the G2P isolation runs. The run-1 sweeps predate even these (first build).
- **Inputs/protocol:** frozen bakeoff JSONs (voice `af_heart`, speed 1.0,
  44/105/219/476 tokens for 3s/7s/15s/30s), 2 warmups discarded, 5 warm
  calls, median reported — identical to the June protocol.
- **Compute policy:** the ladder reproduced the June behavior exactly on
  every Core ML run: `.all` fails at first predict with
  `ANECCompile() FAILED` (error -9, `last_vended_stage: "duration"`), and
  the ladder settles on **staged** (decoder-pre on `.cpuAndNeuralEngine`;
  duration, f0n, generator on `.cpuAndGPU`). Same policy behind the
  published Mac Config F rows.

## Headline result — matched-thermal Release pair (run 3)

Each arm ran alone in a fresh process after an identical ~14-minute idle
cooldown (Core ML first, then MLX, each from the same cool baseline).

| Bucket | Config F median | Config F RTF | MLX median | MLX RTF | MLX/Config F |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s  | 0.466 s | 0.166 | 0.584 s | 0.208 | **1.25x** |
| 7s  | 0.894 s | 0.132 | 1.290 s | 0.193 | **1.44x** |
| 15s | 1.909 s | 0.137 | 2.517 s | 0.181 | **1.32x** |
| 30s | 4.034 s | 0.147 | 5.278 s | 0.193 | **1.31x** |

Config F wins every bucket. Its run-3 medians are within 4–9% of the June
v2 Release ladder (0.426/0.865/1.860/3.742), cross-validating both sessions.

## Supporting runs and the thermal caveat

Three same-day pairs were collected; only run 3 is thermally matched.

| Bucket | R1 CF | R1 MLX | R1 ratio | R2 CF | R2 MLX | R2 ratio | R3 CF | R3 MLX | R3 ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3s  | 0.444 | 0.797 | 1.79 | 0.477 | 0.589 | 1.23 | 0.466 | 0.584 | 1.25 |
| 7s  | 0.936 | 1.780 | 1.90 | 1.227 | 1.452 | 1.18 | 0.894 | 1.290 | 1.44 |
| 15s | 1.895 | 3.539 | 1.87 | 2.974 | 2.926 | **0.98** | 1.909 | 2.517 | 1.32 |
| 30s | 5.266 | 8.023 | 1.52 | 7.224 | 6.083 | **0.84** | 4.034 | 5.278 | 1.31 |

- **Run 1** (Config F first, MLX immediately after; phone already hot from
  the 667 MB install): Config F wins everything, but the second arm ran
  more heat-soaked, biasing against MLX.
- **Run 2** (order reversed: MLX first, Config F immediately after): the
  bias flips — Config F, now the heat-soaked second arm, loses 15s and 30s.
  Its 30s median degraded 79% vs run 3 (7.224 vs 4.034 s).
- **Run 3** (cooldown before each arm): the matched pair above.

Lesson recorded for future sessions: on this fanless phone, run order is a
thermal treatment. Back-to-back arm sweeps hand the second arm a
double-digit-percent penalty at the 15s/30s buckets — enough to flip signs.
Any published iPhone arm-vs-arm row must come from cooldown-separated,
single-arm processes. `ProcessInfo.thermalState` read `serious` during every
Core ML iteration of all three runs (it is too coarse to certify matching —
the equal cooldown treatment is what matches run 3, corroborated by run 3's
agreement with the June v2 ladder). The MLX arm does not record thermal
state; the bench app should add that before the next session.

## G2P boundary asymmetry: bounded at ~1 ms

The MLX `generateAudio` API takes raw text, so its timings include the
Misaki G2P pass; our arm starts from pre-tokenized IDs. Previously disclosed
as "small but nonzero" — now measured per frozen input with `--mode g2p`
(warm medians over 5 calls after 2 warmups, `phonemizeOnlyForBench`):

| Input | Warm G2P median | Phonemes | Share of MLX run-3 median |
| --- | ---: | ---: | ---: |
| 3s  | 0.17 ms | 42  | 0.03% |
| 7s  | 0.25 ms | 103 | 0.02% |
| 15s | 0.56 ms | 216 | 0.02% |
| 30s | 1.05 ms | 474 | 0.02% |

Phoneme counts track the frozen inputs' token counts (44/105/219/476),
confirming the same effective sequence lengths. The asymmetry is three
orders of magnitude below every win margin; subtracting it changes no ratio
at the reported precision. The paper can state: "the MLX timings include a
G2P pass ours excludes; we measured it at ≤1.1 ms per input (≤0.03% of MLX
wall time)."

## iPhone 12 Pro status

Not re-run. The abstract's "every bucket" claim is specific to the A17 Pro;
the June Debug data already showed the A14 result was split (MLX won 7s/15s
there) and MLX cannot complete 30s on 4 GB (jetsam, documented as OOM), so
no A14 clean-sweep claim exists to defend. If a Release A14 table is wanted
later, the same binary and protocol apply (device `iPhone13,3`, UDID via
`xcrun devicectl list devices`), with the arms-in-separate-processes rule
being mandatory there, and cooldown separation per the lesson above.

## Raw artifacts and reproduction

Raw JSONs (gitignored): `outputs/iphone_bench/results_{coreml,mlx}_release_15pm.json`
(run 1), `..._run2.json`, `..._run3.json` (headline pair),
`results_g2p_release_15pm.json`. Each embeds hardware, iOS version, warmup
discipline, per-iteration warm arrays, per-stage medians, and (Core ML arm)
per-iteration thermal states.

```bash
# Build (Release, generic iOS device) — resources must be staged first
cd ios-bench && ./prepare_resources.sh && xcodegen generate --spec project.yml
xcodebuild -project KokoroIPhoneBench.xcodeproj -scheme KokoroIPhoneBench \
  -configuration Release -destination 'generic/platform=iOS' \
  -derivedDataPath /tmp/kokoro_ios_build -allowProvisioningUpdates build

# Install + run (phone unlocked, plugged in; one arm per process,
# >=14 min idle cooldown before EACH arm)
xcrun devicectl device install app --device <UDID> \
  /tmp/kokoro_ios_build/Build/Products/Release-iphoneos/KokoroIPhoneBench.app
xcrun devicectl device process launch --console --terminate-existing \
  --device <UDID> com.mattmireles.KokoroIPhoneBench \
  --arms coreml --keys 3s,7s,15s,30s --out results_coreml.json
# ... cooldown ...
xcrun devicectl device process launch --console --terminate-existing \
  --device <UDID> com.mattmireles.KokoroIPhoneBench \
  --arms mlx --keys 3s,7s,15s,30s --out results_mlx.json
# G2P isolation
xcrun devicectl device process launch --console --terminate-existing \
  --device <UDID> com.mattmireles.KokoroIPhoneBench \
  --mode g2p --keys 3s,7s,15s,30s --out results_g2p.json
# Pull results
xcrun devicectl device copy from --device <UDID> \
  --domain-type appDataContainer \
  --domain-identifier com.mattmireles.KokoroIPhoneBench \
  --source Documents/<name>.json --destination outputs/iphone_bench/<name>.json
```

## Verdict line

**The "beats the MLX port on every bucket on the iPhone 15 Pro Max" claim
survives in Release builds** — 1.25–1.44x on warm medians under the
thermally matched protocol (and 1.52–1.90x in the both-hot pair). The paper
should (a) cite the run-3 table, (b) replace "small but nonzero" G2P
language with the ≤1.1 ms bound, and (c) state the warm-median,
thermally controlled protocol, since an adversarial reviewer running arms
back-to-back can reproduce a 15s/30s flip (our run 2).
