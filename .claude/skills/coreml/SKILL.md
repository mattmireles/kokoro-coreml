---
name: coreml
description: Route Core ML work inside kokoro-coreml without depending on a personal ~/.cursor skill. Use for conversion, numerical parity, compute-plan profiling, ANE/GPU/CPU placement, Swift SDK behavior, TTS quality, or bakeoffs. Delegate to the repo-local child skills and checked-in guides.
---

# Core ML in Kokoro

Read `CLAUDE.md`, `.claude/skills/references/repo-profiles.md`, and the smallest
relevant guide or plan.

Route by intent:

- numerical parity or export drift: `coreml-validate`
- compute-unit placement or fallback: `coreml-profile`
- controlled performance comparisons: `bakeoff`
- perceptual TTS evidence: `audio-judge`
- sibling TTS service integration: `botnet`
- unknown failures: `debug`

Require package hashes and physical-device evidence for hardware claims. Keep
repo analysis in `README/Notes/`; ingest external research into guides only
through the documented guide workflow.
