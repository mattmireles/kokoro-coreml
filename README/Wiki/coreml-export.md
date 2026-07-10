---
title: CoreML Export
last_synced: 2026-06-06
sources:
  - README/coreml-conversion-guide.md
  - README/COREML_EXPORT_SUMMARY.md
  - README/Kokoro-to-CoreML-conversion.md
  - README/Guides/apple-silicon/CoreML-LSTM-export-guide.md
---

# CoreML Export

## Current Belief

The export system should redesign the pipeline around CoreML constraints instead
of forcing the original PyTorch graph into one giant dynamic model.

Fixed buckets beat dynamic-shape failure modes for this workload. Keep the
messy, data-dependent work in Swift or Python tooling and give CoreML static
tensors that can actually schedule on Apple Silicon.

## Do Not Break

- Do not reintroduce ONNX as the default conversion path.
- Do not hide unsupported op rewrites in undocumented probes.
- Update the guide or notes when export geometry, buckets, or precision changes.

## Executable Memory

Regression test: rerun the smallest export/probe command that exercises the
changed model family and record exact model package names in notes.
