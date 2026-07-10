# Kokoro A14 iPhone Generator Create-Guide Brief

June 9, 2026

Target repo: `kokoro-coreml`
Target guide path after ingest:
`README/Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md`
Purpose: External Deep Research input for `create_guide_v1`; not a guide.
Status: run `7379b598` completed 2026-06-10; ingested as
[Kokoro-A14-iPhone-generator-execution-guide.md](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md)
(triage:
[kokoro-a14-iphone-guide-triage-2026-06-10.md](kokoro-a14-iphone-guide-triage-2026-06-10.md)).
Note: written before
[iphone-debug-notes.md](iphone-debug-notes.md) showed the A17 Pro fails
`.all` identically — the `.all` rejection is iOS-side, not A14-specific.

Create an advanced developer field guide on executing a Kokoro/ISTFTNet-style
Core ML vocoder on A14-class iPhones (iPhone 12 Pro, 4 GB RAM, iOS 26.x). This
is not a generic Core ML optimization guide, and it is not a Mac guide: the
M1/lower-end-Mac version of this problem is already covered by our existing
corpus. This guide must answer what is different on the phone.

## Context

First-party Swift + Core ML Kokoro TTS pipeline, fixed buckets (`3s`, `7s`,
`15s`, `30s`), warmed inference only. Four Core ML stage families: duration
(unrolled-LSTM mlprogram, 17k-134k ops padded variants; 780-op exact native
`lstm` variants exist), F0Ntrain (563 ops, 1 native `lstm`), decoder-pre
(499 ops, max axis 2400), and the generator `GeneratorFromHar`
(`kokoro_decoder_har_post_*`, 2207 ops, 4 `conv_transpose`, manual AdaIN with
88 `reduce_mean` + 96 `tile`, Snake activations, fp16, macOS13 target).

Production placement (identical on Mac and the phone bench): duration/F0 and
generator on `.cpuAndGPU`, decoder-pre on `.cpuAndNeuralEngine`.

First on-device A14 measurements (iPhone 12 Pro, iPhone13,3, iOS 26.5, warmed
medians, 2 warmups + 5 iterations):

| Bucket | A14 Core ML staged | A14 RTF | A14 MLX arm | Irvine M1 Config F | M1 RTF |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3s | `1383.4 ms` | `0.494` | `1624.1 ms` | `233.6 ms` | `0.083` |
| 7s | `2966.4 ms` | `0.439` | `2405.2 ms` | `492.7 ms` | `0.073` |
| 15s | `6250.1 ms` | `0.450` | `5021.5 ms` | `1014.9 ms` | `0.073` |
| 30s | `12301.4 ms` | `0.449` | (not run) | `1959.4 ms` | `0.072` |

Latency fit: A14 is `~46 ms fixed + 447 ms` per audio-second versus Irvine M1
`22.4 ms fixed + 70.7 ms` per audio-second — a `6.3x` slope ratio. The gap is
compute-slope-dominated, not boundary/fixed-cost-dominated.

Per-stage M1 reference (Config F): generator is `167.1/383.7/820.8/1631.9 ms`
at `3s/7s/15s/30s`, i.e. 72-83% of wall. Cross-platform fit puts the A14
generator at 82-90% of phone wall (`~1134/2603/5569/11072 ms`), a fitted
`6.8x` slowdown versus the M1 generator while the raw A14/M1 GPU gap is only
`2-2.7x` (4 vs 8 GPU cores, ~34 vs ~68 GB/s).

Failure evidence:

- Under `.all` for all stages, the A14 rejects at predict time with
  `Espresso error -9`; a per-stage fallback ladder lands on the staged policy.
  The failing stage is not yet captured in device logs.
- On every tested Mac, forcing the generator to `.cpuAndNeuralEngine` fails
  with `MILCompilerForANE error: failed to compile ANE model using ANEF ...
  ANECCompile() FAILED` and runs `~1517 ms` vs `~28 ms` on CPU+GPU (M2
  Studio); macOS silently reroutes under `.all`, iOS hard-fails.
- The generator violates the community-documented 16384 per-axis ANE limit at
  every bucket: `har` input last axis `28801..288001`, waveform output
  `72000..720000`, and the noise branch materializes `(1,128,har_T)` at full
  HAR length because trimming happens after `noise_res`. Prior in-repo
  evidence: `Tensor width goes beyond limit supported (16390 > 16384)`.
- A14 and M1 share the same 16-core 11-TOPS ANE generation, so M1 ANE
  rejection predicts A14 ANE rejection as-shaped.

Public precedent: `laishere/kokoro-coreml` runs a Kokoro vocoder ANE-resident
on iPhone 16 Pro (A18) at 16.9x real-time using seven packages, a dual-output
`x_pre` anchor, fp32 noise stage, fp32 iSTFT tail, and int8 palettization of
the discarded-audio vocoder. The oldest publicly verified device for that
design is A18; nothing verifies it on the A14/M1 ANE generation.

RTF arithmetic (what "fix it" must mean): with the generator free, the
non-generator floor is RTF `0.045-0.089`. Fixing the GPU `6.8x` anomaly down
to raw hardware scaling (`~2.5x`) yields RTF `~0.19-0.24`. Matching M1-class
RTF `0.072` at 30s leaves a generator budget of `~742 ms` — `2.2x` faster
than M1's own GPU generator on a GPU `2-2.7x` slower, i.e. unreachable on the
A14 GPU. M1-class RTF on A14 requires an ANE-admissible generator at a
re-chunked boundary, or an architecturally cheaper generator.

## Core Observation

Compute-unit flags cannot fix the phone. The generator is structurally
inadmissible to the ANE at every bucket because of per-axis tensor limits, and
the GPU it actually runs on is `2-2.7x` weaker than M1's yet measures `6.8x`
slower — so there are two distinct problems: an unexplained GPU-path scaling
anomaly worth `~2.3x`, and an ANE admission problem worth the remaining
`~2.5-3x`. No existing guide in our corpus contains any A14 or iPhone 12
content; `Espresso error -9` appears in no public error taxonomy we found.

## Primary Research Goal

Determine how to execute an ISTFTNet-style Core ML vocoder on A14-class
iPhones at the best achievable RTF: explain the iOS-specific failure
semantics, establish the A14 ANE admission rules for conv-heavy audio graphs,
and identify the re-chunking/boundary designs and GPU-path fixes with the
best evidence.

## Questions To Answer

1. What does predict-time `Espresso error -9` ("could not create inference
   context" family) mean on iOS 26? Map the Espresso/ANECF error taxonomy
   (-1, -5, -9, -14, 0x20004) to root causes, and explain why iOS hard-fails
   under `.all` where macOS silently reroutes. Did iOS 26 change compute-unit
   fallback semantics?
2. Is the 16384 per-axis ANE limit enforced whole-program or per-segment at
   `ANECCompile` time on A14-generation hardware? Could a 3s generator whose
   post-upsample body axis (14401) fits get partial ANE residency if only the
   `har` input and waveform output were re-chunked?
3. What are the ANECompilerService compile-memory budgets on 4 GB devices,
   their jetsam interaction, the effect of the Increased Memory Limit
   entitlement, and the caching/AOT-specialization semantics of first load
   (cost, persistence across reboots and app updates, invalidation)?
4. What graph surface and package boundary does the A14/M1-generation ANE
   compiler actually accept for ISTFTNet-style vocoders? Specifically, does
   laishere's `x_pre`-anchor + fp32-tail + fp32-noise design load ANE-resident
   on A14/M1-generation ANE, or only on newer compilers/silicon?
5. What time-axis re-chunking designs admit a 24 kHz vocoder to the ANE:
   windowed overlap-add chunking across predict calls, in-graph rank-4
   folding `(1, C, chunks, <=16384)`, or sliding-window state? What are the
   boundary semantics for dilated convs, AdaIN statistics, and ConvTranspose
   upsampling, and the parity risk of each?
6. Why would a Core ML CPU+GPU conv plan scale `~6.8x` worse than M1 when the
   raw GPU gap is `2-2.7x`? Candidate causes to weigh: MPSGraph runtime
   specialization differences on iOS, fp16 bandwidth limits, MLMultiArray
   input copy costs, partial CPU fallback inside the plan, thermal throttling
   in a 5-iteration loop. What is the realistic A14 GPU RTF ceiling for this
   workload?
7. What is the right per-stage policy on A14: is decoder-pre's
   `.cpuAndNeuralEngine` pin real on the phone (and how to verify), and are
   17k-134k-op unrolled-LSTM duration programs an Espresso/ANE program-size
   or first-load hazard versus 780-op native-`lstm` packages?
8. What are the iOS-specific scheduling and product constraints: background
   GPU execution on iOS 26 (and earlier), sustained-inference thermal and
   energy tradeoffs of ANE vs GPU placement for TTS, and implications for
   always-on/background TTS apps?
9. What on-device diagnostic workflow proves placement and captures failures:
   `MLComputePlan` on iOS, `log stream`/os_log predicates for
   `com.apple.espresso`/`com.apple.coreml`, Instruments Core ML template
   against a tethered iPhone, and what evidence should gate any promotion
   claim?
10. What first two implementation experiments should we run on the A14 (with
    stop/go criteria), assuming we can modify exports, package boundaries,
    and the Swift host but not retrain the model first?

## Output Format

Start with an executive summary of whether M1-class RTF on A14 is achievable
and by which path. Then provide:

- the iOS Espresso/ANE error taxonomy and failure-semantics explanation;
- A14 ANE admission rules (limits, enforcement granularity, memory budgets);
- re-chunking/boundary design recommendations with parity-risk notes;
- GPU-path forensics for the `6.8x` anomaly and the realistic GPU ceiling;
- per-stage compute-unit policy recommendations for A14;
- on-device profiling and evidence-capture workflow;
- do-this / avoid-this tables;
- first two experiments with stop/go criteria;
- clearly mark speculative ideas separately from evidence-backed guidance.
