# Kokoro A14 iPhone Guide Triage

June 10, 2026

Ingest triage for
[Kokoro A14 iPhone generator execution guide](../Guides/apple-silicon/Kokoro-A14-iPhone-generator-execution-guide.md).
Brief:
[Kokoro-A14-iphone-generator-create-guide-brief.md](Kokoro-A14-iphone-generator-create-guide-brief.md).

## Provenance

- Raw source: `/Users/mm/Documents/GitHub/llm-workflows/outputs/create-guide/kokoro-istftnet-vocoder-core-ml-execution-on-a14-iphones-espresso-9-semantics-an/2026-06-10T05-47-44-034Z/raw-report.md`
  (`create_guide_v1`, run `7379b598-c546-4380-a146-19e18f7e572c`, agent mode
  `max`, ~15 min wall).
- The report cites only two opaque grounding links (one github, one reddit
  redirect). Much of its quantitative content restates the brief's own
  measured tables; the genuinely external content is the error taxonomy, the
  per-segment enforcement claim, the ANECompilerService budget figure, and
  the chunking parity analysis. All four are quarantined under "Claims Left
  As Heuristics" in the ingested guide.

## Post-Brief Device Evidence (changes the framing)

The brief was written before
[iphone-debug-notes.md](iphone-debug-notes.md) landed. That note shows the
**A17 Pro (iPhone 15 Pro Max) fails `.all` identically to the A14** with
`MILCompilerForANE ... ANECCompile() FAILED` at first predict, surfacing as
`com.apple.CoreML Code=0 ... (error code: -9)`, while every tested M-series
Mac accepts the same packages and silently reroutes. Consequences:

- The `.all` hard-fail is an **iOS ANEF-specialization behavior**, not an
  A14-generation limit. The guide's A14-centric framing is correct about ANE
  *admission* limits (A14 = M1 ANE generation) but the failure semantics
  apply to current iPhones generally.
- The brief's open question "what does Espresso -9 mean" is now largely
  answered by the captured log string: `-9` is the ANECCompile failure
  surfacing at first predict, because iOS runs on-device ANEF specialization
  at first prediction and aborts instead of rerouting.
- Decoder-pre's ANEF compile succeeds on both phones (the staged policy
  works); the rejected stage is among duration, F0Ntrain, generator. The
  duration model predicts first, making the 17k-134k-op unrolled-LSTM padded
  packages the prime suspect — consistent with the guide's duration-model
  hazard warning, and testable via the per-stage smoke test in
  [iphone-performance-notes.md](iphone-performance-notes.md) follow-ups.

## Verification Decisions

- **Context7-corrected (adopted with fixes):** the report's
  `MLComputePlan.load()` / `computePlan.deviceUsage(for:)` sketch. Current
  coremltools API is `ct.models.compute_plan.MLComputePlan.load_from_path`
  plus `get_compute_device_usage_for_mlprogram_operation` /
  `get_estimated_cost_for_mlprogram_operation`; there is also an
  experimental on-device loader
  (`load_compute_plan_from_path_on_device` with
  `Device.get_connected_development_devices(device_type=DeviceType.IPHONE)`),
  which is the most useful tool the ingest surfaced for the per-stage
  iPhone probe. Swift-side names (`MLComputePlan.load(contentsOf:configuration:)`,
  `deviceUsage(for:)`, iOS 17.4+) verified against prior guide corpus.
- **Verified:** `compute_precision` defaults to FLOAT16 for `mlprogram`;
  `ct.transform.FP16ComputePrecision(op_selector=...)` exists for selective
  fp32 preservation (relevant if the fp32 noise/tail isolation is ever moved
  in-graph).
- **Adopted as-is:** the re-chunking parity table (overlap-add vs rank-4
  folding vs stateful sliding window) — engineering reasoning, not citable
  fact, but consistent with the repo's own boundary findings.
- **Rejected/flagged:** the report's claim that iOS 26 throttles background
  GPU *harder* than earlier versions. laishere's notes document the opposite
  direction ("GPU work is suspended when apps background **before** iOS 26").
  Left in the guide as an unresolved conflict; both versions agree ANE is
  the only background-safe placement.
- **Flagged conflict:** the report maps Espresso `-14` to "compiler timeout
  or hardware queue rejection"; the earlier compiler-failure-triage raw
  report maps `-14` to load-time "failed to build the model execution plan".
  Both kept visible in the guide's heuristics list.
- **Echo, not evidence:** the hardware table, RTF arithmetic, 6.8x anomaly
  decomposition, and the 0.19-0.24 GPU ceiling restate the brief's inputs.
  Their authority remains the repo's own measurements
  ([iphone-performance-notes.md](iphone-performance-notes.md),
  [performance-notes.md](performance-notes.md)), not the report.

## What This Changes For The Roadmap

Nothing structural; the report independently lands on the repo's working
theory. The actionable additions from this ingest:

1. The experimental coremltools on-device compute-plan dump is the right
   instrument for the per-stage smoke test (which stage rejects ANEF; is
   decoder-pre's ANE pin real on the phone).
2. The "ANE admittance proof" experiment (stripped, under-16,384, fp16-only
   generator body on `.cpuAndNeuralEngine`) is the cheapest go/no-go gate
   before investing in overlap-add re-chunking.
3. Duration's unrolled-LSTM packages should be replaced with exact-native
   `lstm` packages in the iPhone bundle regardless of other outcomes (jetsam
   and compile-budget hazard, plus the Mac-vs-iPhone duration confound
   already noted in [iphone-performance-notes.md](iphone-performance-notes.md)).
