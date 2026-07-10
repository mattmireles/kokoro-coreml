# Kokoro A14 iPhone Generator Execution Guide

This guide ingests the external deep-research report on executing the Kokoro
ISTFTNet-style generator on A14-class iPhones (iPhone 12 Pro, 4 GB, iOS 26.x).
Treat the raw report as research input, not canonical truth: it is thinly
cited (two opaque grounding links), and several of its strongest statements
are flagged below as heuristics.

Raw report:

- `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-istftnet-vocoder-core-ml-execution-on-a14-iphones-espresso-9-semantics-an/2026-06-10T05-47-44-034Z/raw-report.md`

Commissioning brief:
[Kokoro A14 iPhone generator create-guide brief](../../Notes/Kokoro-A14-iphone-generator-create-guide-brief.md).
Ingest triage and verification decisions:
[kokoro-a14-iphone-guide-triage-2026-06-10](../../Notes/kokoro-a14-iphone-guide-triage-2026-06-10.md).

## Executive Summary

Compute-unit flags cannot close the iPhone gap. The report's verdict matches
the repo's arithmetic: the realistic A14 GPU ceiling for this workload is RTF
~0.19-0.24, and M1-class RTF (~0.072) on A14 is structurally impossible on
the GPU — it requires ANE admission of the generator, which in turn requires
re-chunking the time axis under the 16,384 per-axis limit.

The report's honest gaps (marked `DATA UNAVAILABLE` in the source): whether
iOS 26 changed compute-unit fallback semantics; first-load ANE compile
time/memory cost; whether laishere's `x_pre`-anchor design loads ANE-resident
on A14-generation silicon; whether decoder-pre's CPU+NE pin is actually
honored on the phone.

## iOS Failure Semantics

On macOS an `ANECCompile()` failure silently reroutes to CPU/GPU; on iOS the
same failure hard-aborts the prediction. The report attributes the iOS
hard-fail to jetsam-adjacent memory strictness rather than a different op
compiler.

Espresso error taxonomy per the report (weakly sourced — see heuristics):

| Code | Report's reading |
| --- | --- |
| `-1` | Dynamic resizing / axis-out-of-bounds failure at runtime. |
| `-5` | Generic graph configuration error (precision or op mismatch). |
| `-9` | "Could not create inference context": initialization hard-fail when a graph violates ANE constraints under memory pressure. |
| `-14` | Compiler timeout or hardware queue rejection. |
| `0x20004` | ANECF compilation failure on a malformed or rejected MIL op. |

Repo evidence is more specific than the report here: on both test iPhones the
`-9` surfaced with the verbatim `MILCompilerForANE ... ANECCompile() FAILED`
string at first predict — the same signature M-series Macs emit (then
silently reroute) — see
[iphone-debug-notes.md](../../Notes/iphone-debug-notes.md).

## ANE Admission and Memory Budgets

- No single tensor dimension may exceed 16,384 elements; A14 ANE inference is
  fp16-only.
- The report asserts the 16,384 limit is enforced per-segment at
  `ANECCompile` time, so an under-limit subgraph could get partial residency
  if its boundary tensors conform — but over-limit boundary tensors (Kokoro's
  `har` input 28,801-288,001 and waveform output 72,000-720,000) drag the
  whole connected subgraph into rejection. Treat the per-segment claim as a
  hypothesis to test, not established fact.
- `ANECompilerService` is an XPC daemon subject to jetsam; the report puts
  its practical budget near ~1 GB `phys_footprint` on 4 GB devices
  (heuristic). The 17k-134k-op unrolled-LSTM duration packages are exactly
  the kind of program that spikes it; prefer the 780-op native-`lstm`
  exact-duration packages on iPhone.
- The "Increased Memory Limit" entitlement raises the app's budget, not the
  compiler daemon's.
- The on-device ANE cache invalidates on reboot and app update; pre-warm on
  first run, charged and idle, and never flip `computeUnits` between loads of
  the same model.

## GPU-Path Forensics: the 6.8x Anomaly

Candidate causes for the A14 generator running ~6.8x slower than M1 when raw
GPU specs predict 2-2.7x, in the report's weighting:

1. Partial CPU fallback + IOSurface round-trips at unsupported-op boundaries
   (manual AdaIN `reduce_mean`/`tile`, Snake) — highest weight.
2. `MLMultiArray` input serialization/copy cost on the CPU before dispatch.
3. iOS MPSGraph runtime specialization choosing safer, slower kernels than
   macOS.
4. Memory bandwidth (~34 vs ~68 GB/s) plus thermal downclocking across warm
   iterations.

Repo note: the same anomaly shows on A17 Pro (3.3x slower than M1 against a
~1.2-1.3x raw gap), so treat it as an iPhone-platform effect, not an A14
defect — see
[iphone-performance-notes.md](../../Notes/iphone-performance-notes.md).

## Re-Chunking Designs for ANE Admission

| Design | Dilated convs | AdaIN statistics | ConvTranspose seams | Verdict |
| --- | --- | --- | --- | --- |
| Windowed overlap-add across predict calls | Receptive field resets; needs overlap padding and edge discard | Local stats drift vs full-utterance stats | Click risk at seams; cross-fade required | Most likely to be admitted; highest parity work |
| In-graph rank-4 folding `(1, C, chunks, <=16384)` | Breaks across the folded boundary without heavy in-graph padding | Needs multi-axis reductions the ANE may reject | Stride misalignment baked into the graph | Least promising per the report |
| Stateful sliding window (`MLState`) | Exact parity via carried state | Rolling stats possible | Exact overlap via state | Cleanest math; hardest conversion; iOS 18+ |

## Per-Stage Policy on iPhone

- Pin compute units explicitly per package; never `.all` on unverified
  graphs (it hard-fails at first predict on both test iPhones).
- Duration: avoid the unrolled-LSTM padded packages on device; use the
  exact-native `lstm` variants pinned to `.cpuOnly` or `.cpuAndGPU`.
- Decoder-pre: keep `.cpuAndNeuralEngine`, but verify actual residency with a
  compute plan before crediting the ANE.
- Background TTS: the report claims iOS 26 throttles background GPU harder
  than prior versions; laishere's notes claim the opposite direction
  (backgrounding suspended GPU work before iOS 26). Unresolved conflict —
  do not design around either claim without an on-device test. Either way,
  ANE residency is the only placement that is safe for background synthesis.

## Verified Diagnostic Tooling

Context7-verified against current coremltools docs (the raw report's API
sketch was directionally right but imprecise):

```python
import coremltools as ct

# Host-side: per-op device usage for a compiled model.
plan = ct.models.compute_plan.MLComputePlan.load_from_path(
    path="kokoro_decoder_har_post_3s.mlmodelc",
    compute_units=ct.ComputeUnits.CPU_AND_NE,
)
main = plan.model_structure.program.functions["main"]
for op in main.block.operations:
    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
    cost = plan.get_estimated_cost_for_mlprogram_operation(op)
```

```python
# On-device (experimental): dump the iPhone's own compute plan from the Mac.
from coremltools.models.ml_program.experimental.remote_device import (
    Device, DeviceType,
)
devices = Device.get_connected_development_devices(device_type=DeviceType.IPHONE)
device = await devices[0].prepare_for_model_debugging()
plan = await ct.models.ml_program.experimental.compute_plan_utils.load_compute_plan_from_path_on_device(
    path=compiled_path, device=device,
)
```

The Swift-side equivalent is `MLComputePlan.load(contentsOf:configuration:)`
plus `deviceUsage(for:)` (iOS 17.4+). Device log capture:
`log stream --device --predicate '(subsystem IN {"com.apple.espresso","com.apple.coreml"})' --info --debug`.

## First Two Experiments

1. **ANE admittance proof.** Export a stripped generator body at a hardcoded
   small time axis (every dim < 16,384, fp16, no fp32 noise/tail branches)
   and load it on the A14 pinned to `.cpuAndNeuralEngine`. GO: it predicts —
   the conv stack is admissible and re-chunking is worth building. STOP: it
   still throws the ANECCompile failure — an operator (e.g. AdaIN lowering)
   is structurally incompatible and must be rewritten first.
2. **Compute-plan GPU forensics.** Dump the per-op device usage of the 3s
   generator under `.cpuAndGPU` (ideally on-device via the experimental
   remote-device API). GO: CPU-mapped ops found — rewrite those subgraphs.
   STOP: 100% GPU with no fallback — the anomaly is bandwidth/specialization
   and ANE re-chunking is the only path to M1-class RTF.

## Promotion Gates

Promote no iPhone claim without: a zero-CPU-fallback compute plan for the
stage in question; an Instruments Core ML trace from the tethered phone
showing ANE-track activity without copy gaps; a clean espresso/coreml
`os_log` capture (no `-9`/`-5`/`-1` during init); and a sustained warmed
benchmark (medians, recorded thermal state) per
[benchmark hygiene](Apple-Silicon-warmed-inference-benchmark-hygiene-guide.md).

## Do / Avoid

| Do | Avoid |
| --- | --- |
| Split the pipeline into per-stage packages with explicit compute units. | Mixing fp32 and fp16 stages inside one ANE-targeted graph. |
| Keep every ANE-stage tensor dimension under 16,384. | Passing full-rate audio axes (72k-720k) through an ANE-targeted boundary. |
| Pre-warm once on a charged, idle device and keep the compiled cache. | `.all` on unverified graphs on iOS. |
| Use exact-native `lstm` duration packages on device. | Shipping 17k-134k-op unrolled-LSTM graphs to a 4 GB phone. |

## Claims Left As Heuristics

The following report claims are plausible but under-sourced; treat each as a
hypothesis until reproduced against the actual Kokoro packages on the phone:

- The Espresso numeric-code taxonomy (notably `-9` = "could not create
  inference context", and `-14` = compiler timeout — the earlier
  [ANE compiler failure triage guide](CoreML-ANE-compiler-failure-triage-guide.md)
  raw report instead maps `-14` to load-time execution-plan build failure).
- Per-segment enforcement of the 16,384 limit at `ANECCompile` time.
- The ~1 GB `ANECompilerService` jetsam budget on 4 GB devices.
- "The A14 ANE notoriously rejects multi-axis reductions."
- The iOS 26 background-GPU throttling claim (conflicts with laishere's
  documented pre-iOS-26 framing).

## Related Documentation

- [Core ML ANE compiler failure triage](CoreML-ANE-compiler-failure-triage-guide.md)
- [Core ML compute unit scheduling](CoreML-Compute-Unit-Scheduling-guide.md)
- [Core ML / MLX scheduling for 1D ConvTranspose ISTFTNet vocoders](Core%20ML-MLX-Scheduling-1D-ConvTranspose-ISTFTNet-vocoders-guide.md)
- [Kokoro M1 vocoder partition and boundary guide](Kokoro-M1-vocoder-partition-boundary-guide.md)
- [iPhone Core ML device lab runbook](iPhone-CoreML-device-lab-runbook.md)
- [iPhone performance notes](../../Notes/iphone-performance-notes.md)
- [iPhone debug notes](../../Notes/iphone-debug-notes.md)
