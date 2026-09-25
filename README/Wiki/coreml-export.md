---
title: CoreML Export
last_synced: 2026-09-24
sources:
  - README/coreml-conversion-guide.md
  - README/COREML_EXPORT_SUMMARY.md
  - README/Notes/debug-notes.md
  - README/Notes/hf-release-2026-09-24.md
  - README/Guides/apple-silicon/CoreML-LSTM-export-guide.md
---

# CoreML Export

## Current Belief

The export system redesigns the pipeline around Core ML constraints instead of
forcing the original PyTorch graph into one giant dynamic model. Direct
`coremltools.convert()` on traced PyTorch; no ONNX.

Fixed buckets are the default because the Neural Engine needs static shapes.
Every bucketed package takes a required `mask` input, and the export wrappers
fail loudly without it. The generator is the one exception: an optional
`RangeDim` program (`--time-axis range`, one bucket only, GPU, macOS 15) that
takes the mask at each internal resolution. The generator's `har` input is at
native length: 60 × x_pre frames + 1.

The 2026-09-24 re-export (HF packages `dadebca4`, metadata `5bb7d75d`) passed:
- parity against masked PyTorch fp32: fixed generators 46 dB, flexible 44 dB,
  decoder-pre 55–59 dB;
- pytest, Swift and SDK tests with zero skips.

## Do Not Break

- Do not reintroduce ONNX as the default conversion path.
- Do not hide unsupported op rewrites in undocumented probes.
- `export_synth/main.py` and `export_decoder_pre.py` exit 0 on conversion errors.
  Check each log for the `Saved` line.
- Update the guide or notes when export geometry, buckets, masks, or precision
  change.

## Executable Memory

Regression test:

```bash
uv run --no-project --python .venv/bin/python --with pytest python -m pytest -q
```

Then rerun the smallest export command for the changed model family and record
the exact package names in notes.
