---
title: Runtime Boundary
last_synced: 2026-09-24
sources:
  - README.md
  - README/SDK.md
  - README/Notes/kokoro-runtime-boundary.md
  - README/Notes/debug-notes.md
  - README/Notes/hf-release-2026-09-24.md
  - swift/Package.swift
---

# Runtime Boundary

## Current Belief

Kokoro CoreML is a Swift-first inference runtime. Python is acceptable for
export, probes, and bakeoff tooling, but not for the production inference path.
Apps integrate through the `KokoroTTS` SDK (`swift-tts`, iOS 18 / macOS 15);
`KokoroPipeline.synthesize(...)` is the low-level boundary underneath it.

The pipeline is four Core ML model families plus native Swift stages: duration,
alignment, F0/noise (f0ntrain), decoder-pre, the hn-NSF harmonic source, the
generator, and trim. Placement follows measured compute plans (2026-09-24):

- Decoder-pre runs bucketed on the Neural Engine at every size (3–30 s),
  250 of 250 ops on the ANE. This is a maintainer decision: one placement for
  every machine, even where an M3 Max GPU would be about 30 ms faster at 30 s.
- Duration, f0ntrain and the generator run on CPU+GPU; their LSTMs sit on the
  CPU.
- The generator runs as one flexible-length (`RangeDim`) GPU program at the
  utterance's real length, rounded up to 0.5 s, when the package is present and
  the OS is macOS 15 / iOS 18. Otherwise per-bucket generator packages serve.
- Every padded package takes a required `mask`, so bucket padding stays out of
  every time-axis statistic.

## Do Not Break

- Keep `KokoroTTS` as the app boundary and `KokoroPipeline.synthesize(...)` as
  the pipeline boundary.
- Keep Neural Engine stages bucketed. Use flexible shapes only for a GPU stage
  whose win is measured, and keep a bucketed fallback.
- Keep decoder-pre on `cpuAndNeuralEngine`; `DecoderPreComputeUnitsTests` pins it.
- Treat benchmark claims as stale unless revalidated on the target machine; a
  successful `predict()` is not ANE proof.

## Executable Memory

Regression test:

```bash
swift test --package-path swift
swift test --package-path swift-tts
```

Then record the exact command and machine in notes.
