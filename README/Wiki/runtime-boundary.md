---
title: Runtime Boundary
last_synced: 2026-06-06
sources:
  - README.md
  - README/Notes/kokoro-runtime-boundary.md
  - swift/Package.swift
---

# Runtime Boundary

## Current Belief

Kokoro CoreML is a Swift-first inference runtime. Python is acceptable for
export, probes, and bakeoff tooling, but not for the production inference path.

The durable architecture is five CoreML model families plus native Swift DSP:
duration, alignment, matrix ops, F0/noise prediction, decoder pre, harmonic
source, generator, and trim. The CPU handles small dynamic orchestration; ANE
gets fixed-shape dense math.

## Do Not Break

- Keep `KokoroPipeline.synthesize(...)` as the runtime boundary.
- Keep model bucket geometry explicit; dynamic-shape optimism is not a plan.
- Treat benchmark claims as stale unless revalidated on the target machine.

## Executable Memory

Regression test: run the relevant Swift package test or benchmark command from
`swift/Package.swift`, then record the exact command and machine in notes.
