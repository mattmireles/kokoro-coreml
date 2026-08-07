---
name: coreml-profile
description: >-
  Profile Kokoro and related Core ML packages to establish actual CPU, GPU, and
  Neural Engine placement, latency, and silent fallback behavior. Use for
  device compute plans, scheduling claims, package-shape changes, warmed timing,
  or hardware comparisons.
---

# Profile Kokoro Core ML

Bind every measurement to the package hash, model/configuration, hardware, OS,
Core ML compute-unit request, input shape/duration, warmup policy, and timing
boundary. Inspect a device compute plan or physical trace when available; never
infer Neural Engine placement from a requested compute unit or speed alone.

Start with `scripts/dump_device_compute_plan.py`,
`scripts/inspect_coreml_compute_plan.m`, and
`README/Guides/apple-silicon/CoreML-Compute-Unit-Scheduling-guide.md`.
Keep synthesis, phonemization, I/O, and individual model-stage timing separate.
