# Restarted Kokoro Deep Research Guide Triage

June 6, 2026

> **Scope:** Eighteen create-guide raw reports landed across twelve distinct
> topics after the workflow reset. Treat them as draft research input. This
> note records the implementation triage and points to the repo-native guides
> produced by guide-ingest.

## Status

All six restarted runs completed on the second attempt and have been folded
into repo-native guides. Six older same-topic runs were superseded by the
restart outputs and are tracked below for auditability, not imported as
separate canonical guides. The 15-minute heartbeat was deleted after all
reports landed.

## Raw Reports

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-m1-partition-and-boundary-mechanics-for-strict-1d-kokoro-vocoder-bodies/2026-06-06T23-29-25-644Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-irvine-m1-3s-and-7s-paper-frontier-optimization/2026-06-06T23-29-25-659Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-source-body-core-ml-implementation-path-against-laishere/2026-06-06T23-29-25-697Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-vocoder-runtime-boundary-strategies-for-lower-end-apple-silicon/2026-06-06T23-29-40-654Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/apple-silicon-warmed-inference-benchmark-hygiene-for-kokoro-core-ml-and-mlx/2026-06-06T23-29-40-666Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-har-stft-strict-repair-and-lightweight-distillation-for-core-ml-source-bo/2026-06-06T23-29-40-702Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-har-stft-contract-repair-for-strict-core-ml-source-body-vocoder-path/2026-06-06T22-37-46-144Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-compiler-failure-triage-for-large-ml-program-models-retry/2026-06-06T23-25-15-016Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-transformer-layout-and-op-compatibility-retry/2026-06-06T23-25-22-946Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-split-graphs-and-multifunction-packaging-for-streaming-inference-retry/2026-06-06T23-25-30-108Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/iphone-core-ml-device-lab-runbook-for-signing-profiling-and-evidence-capture-ret/2026-06-06T23-25-36-950Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-ane-temporal-escape-hatches-for-stateful-streaming-transformers-on-iphon/2026-06-06T22-48-22-802Z/raw-report.md`

## Superseded Duplicate Raw Reports

Older duplicate runs also landed for several restarted topics. Their
same-topic drafts were superseded by the restarted raw reports above and were
not imported as separate canonical guides:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/core-ml-m1-partition-and-boundary-mechanics-for-strict-1d-kokoro-vocoder-bodies/2026-06-06T23-26-24-084Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-irvine-m1-3s-and-7s-paper-frontier-optimization/2026-06-06T23-26-28-193Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-source-body-core-ml-implementation-path-against-laishere/2026-06-06T23-26-32-266Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/apple-silicon-warmed-inference-benchmark-hygiene-for-kokoro-core-ml-and-mlx/2026-06-06T23-26-36-396Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-har-stft-strict-repair-and-lightweight-distillation-for-core-ml-source-bo/2026-06-06T23-26-40-384Z/raw-report.md`
- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-m1-vocoder-runtime-boundary-strategies-for-lower-end-apple-silicon/2026-06-06T23-26-44-290Z/raw-report.md`

## Immediate Decisions

1. Keep the benchmark target on warmed inference only. Per-bucket warmup must be
   separate for `3s`, `7s`, `10s`, `15s`, and `30s`; do not mix bucket shapes
   in one warmup cycle.
2. Prioritize lower-end Mac wins. Skip iPhone work for this pass unless a Mac
   candidate becomes clearly worth mobile validation.
3. Do not use Whisper ASR as the acceptance gate. Quality-failing speed branches
   need strict objective recovery or explicit no-ASR listening review.
4. Reject another broad multi-package split unless it removes a hot-path Core ML
   call or proves a positive warmed runtime on Irvine M1. Current evidence says
   added prediction boundaries erase the ANE savings on short buckets.
5. Treat `.all` and `.cpuAndNeuralEngine` as hypotheses, not proof. Every
   promoted candidate needs MLComputePlan evidence plus warmed timing.

## Actionable Implementation Queue

| Priority | Track | Why |
| --- | --- | --- |
| 1 | Quiet-host benchmark gate and evidence bundle | Any Irvine M1 number is publishable only if background CPU, thermal, swap, and raw per-run timing are recorded. Current gate already rejects `mediaanalysisd`, `mds_stores`, load, swap, power, and thermal blockers. |
| 2 | Zero-copy/output-backing probe | Xcode SDK headers verify `MLPredictionOptions.outputBackings` and `MLMultiArray(pixelBuffer:shape:)`; the harness-only generator output-backing ablation now exists, but local CPU+GPU short-bucket timing did not clear the promotion gate. Keep it as a measurement tool, not a production optimization. |
| 3 | Production upsample ConvTranspose rewrite on lower-end Macs | It is already a strict local win and combines with other independent savings. Promote only with quiet Irvine evidence. |
| 4 | Single-package body reshaping | The reports agree with repo evidence: change the graph surface inside the current call boundary instead of adding more prediction calls. |
| 5 | HAR/STFT learned repair probe | Direct scalar Nyquist formulas failed. A tiny `Conv1d(kernel_size=1)` adapter is plausible, but it is a training/parity experiment, not an immediate benchmark win. |
| 6 | Source/body compact representation | Useful only if it avoids the padded HAR payload and stays inside one runtime-positive package boundary. |

## Verified API Facts

Verification used the current local toolchain: Xcode `MacOSX26.5.sdk` Core ML
headers and installed `coremltools 9.0`.

- `MLPredictionOptions.outputBackings` exists in the local Xcode SDK and is
  available on macOS 11.0+ / iOS 16.0+.
- The SDK header says the backing object must match the output feature type:
  `CVPixelBuffer` for image outputs or `MLMultiArray` for multi-array outputs.
- `MLMultiArray.init(pixelBuffer:shape:)` exists and the SDK header describes it
  as an IOSurface-backed initializer that can reduce inference latency by
  avoiding buffer copies.
- The same header limits the pixel-buffer backed path mostly to FP16
  multi-arrays; this must be verified against the actual generator input/output
  dtypes before implementation.

## Output-Backing Probe Result

June 6, 2026 local M2 Studio result: built a harness-only `outputBackings`
ablation for generator prediction in `kokoro-bench --generator-input-dump` via
`--generator-output-backing`.

Method:

1. Preallocate the generator `waveform` output `MLMultiArray` from the model
   output constraint.
2. Call `genModel.prediction(from:options:)` with
   `MLPredictionOptions.outputBackings = ["waveform": backing]`.
3. Compare dumped baseline and output-backed waveforms.
4. Benchmark warmed generator-only `3s` and `7s`, CPU+GPU, warmup `5`,
   iterations `15`.

Results:

| Bucket | Baseline median | Output-backed median | Delta | Decision |
| --- | ---: | ---: | ---: | --- |
| `3s` | `30.032 ms` | `30.109 ms` | `-0.077 ms` | Reject as local win. |
| `7s` | `60.870 ms` | `60.456 ms` | `+0.414 ms` | Below promotion gate. |

Parity check: dumped `3s` `waveform_full` and trimmed `waveform` are
bit-identical (`max_abs=0.0`) between baseline and output-backed paths.

Artifacts:

- `outputs/generator_output_backing/3s_baseline_cpugpu.json`
- `outputs/generator_output_backing/3s_output_backing_cpugpu.json`
- `outputs/generator_output_backing/7s_baseline_cpugpu.json`
- `outputs/generator_output_backing/7s_output_backing_cpugpu.json`

Decision: keep the harness flag for future device checks, but do not promote
`outputBackings` into production or Irvine timing unless a lower-end Mac shows a
material result that contradicts this local short-bucket rejection.

Do not use `CVPixelBuffer` input rewrites until output backing proves measurable
on the current model. Input-buffer changes are higher risk because they can
change dtype/layout and invalidate existing strict parity assumptions.

## Claims To Verify Before Canonical Use

- `CVPixelBuffer` / `IOSurface` zero-copy advice for arbitrary tensor features:
  verify against actual Core ML input/output contracts before attempting it.
- "Replace all Linear with Conv1d guarantees ANE placement": too broad. Check
  actual MIL, ranks, dtypes, and MLComputePlan per bucket.
- "Native instance_norm maps flawlessly": too broad. Existing repo probes
  removed manual reduction/tile surface but did not produce a material local
  speed win by itself.
- Palette and LUT claims: prior first-party pal8 visible-surface matching
  regressed local `3s`; do not revisit without a materially different package
  or memory-boundary hypothesis.
- iOS-specific foreground, Low Power Mode, Developer Disk Image, and padding
  claims need current Apple/API verification before a paper-facing guide uses
  them.

## Ingested Guides

The restarted reports are now folded into repo-native guides. These guides
preserve raw report paths, mark speculative claims, and link back to the
current Kokoro evidence instead of importing the raw reports as canonical truth:

- [Apple Silicon warmed-inference benchmark hygiene](../Guides/apple-silicon/Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md)
- [Kokoro Irvine M1 3s/7s paper frontier](../Guides/apple-silicon/Kokoro-Irvine-M1-3s-7s-paper-frontier-guide.md)
- [Kokoro M1 source/body Core ML](../Guides/apple-silicon/Kokoro-M1-source-body-coreml-guide.md)
- [Kokoro M1 vocoder partition and boundary](../Guides/apple-silicon/Kokoro-M1-vocoder-partition-boundary-guide.md)
- [Kokoro M1 vocoder runtime boundary](../Guides/apple-silicon/Kokoro-M1-vocoder-runtime-boundary-guide.md)
- [Kokoro HAR/STFT strict repair and distillation](../Guides/apple-silicon/Kokoro-HAR-STFT-strict-repair-distillation-guide.md)
- [Kokoro M1 HAR/STFT contract repair](../Guides/apple-silicon/Kokoro-M1-HAR-STFT-contract-repair-guide.md)

Additional completed Core ML guide reports from the same batch are also
ingested:

- [Core ML ANE compiler failure triage](../Guides/apple-silicon/CoreML-ANE-compiler-failure-triage-guide.md)
- [Core ML ANE transformer layout and op compatibility](../Guides/apple-silicon/CoreML-ANE-transformer-layout-op-compatibility-guide.md)
- [Core ML split graphs and multifunction packaging](../Guides/apple-silicon/CoreML-split-graphs-multifunction-packaging-guide.md)
- [iPhone Core ML device lab runbook](../Guides/apple-silicon/iPhone-CoreML-device-lab-runbook.md)
- [Core ML ANE temporal escape hatches](../Guides/apple-silicon/CoreML-ANE-temporal-escape-hatches-guide.md)
