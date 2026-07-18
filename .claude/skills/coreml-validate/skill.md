---
name: coreml-validate
description: Validate CoreML model numerical correctness against PyTorch reference. Runs identical inputs through both pipelines and reports per-output correlation, max error, and pass/fail. Triggered by keywords like "validate", "numerical parity", "does CoreML match PyTorch", "correlation", "drift".
---

# CoreML Validate

## Parent skill

Entry point for all Core ML work: [coreml](../coreml/SKILL.md). Use this skill
when the routed intent is **numerical parity**, not compute-unit profiling.

## Repo gate

Read [references/repo-profiles.md](../references/repo-profiles.md).

**Stop** unless both exist:

1. A Core ML `.mlpackage` for the model under test (any path in-repo).
2. A reference pipeline (PyTorch, MLX, or C++ per profile).

**crossfade:** No `kokoro` package required. Before any `.mlpackage` exists, use
MLX/C++ parity per `CLAUDE.md` Part 3 — do not run this skill.

**kokoro-coreml:** Typical layout is `coreml/` + `kokoro` package; see kokoro
export scripts in [coreml/reference.md](../coreml/reference.md).

## Purpose

Answer the question: **"Does the CoreML model produce the same output as
PyTorch?"** Run identical inputs through both pipelines and report per-output
tensor correlation, max absolute error, and pass/fail against defined thresholds.

## Use When

- After exporting a new CoreML model and wanting to verify numerical parity
- After changing export settings (compute_precision, minimum_deployment_target)
- Debugging audio quality issues that might be conversion artifacts
- The user says "validate", "does it match", "numerical parity", "correlation"
- Comparing FP16 vs FP32 CoreML output against PyTorch FP32

## Do Not Use When

- The user wants to know which compute units are being used (use `coreml-profile`)
- The user wants to run the full bakeoff benchmark (use `bakeoff`)
- The model hasn't been exported yet (export first, then validate)

## Prerequisites

- Python environment with coremltools, torch, numpy: `uv run python3`
- Both PyTorch model weights and CoreML `.mlpackage` files available
- For full pipeline validation: `kokoro` package importable

## Reference Material

Before validating, consult:

- `CLAUDE.md` Part 4 — Core ML edge cases (FP16 drift, layout, stateful caches)
- `CLAUDE.md` Part 5 — Validate/Profile/Iterate checklist
- kokoro sibling: `../kokoro-coreml/README/Notes/debug-notes.md` — bisection
  methodology (decoder ghost audio, stage isolation)

## Key Concepts

### Correlation vs Max Error
- **Pearson correlation** (r): measures shape similarity. r > 0.99 = good,
  r > 0.999 = excellent. Insensitive to scale offset.
- **Max absolute error**: catches catastrophic outliers. Should be < 0.01 for
  FP16, < 1e-5 for FP32.
- **Use both**: high correlation + low max error = faithful conversion.
  High correlation + high max error = systematic scale shift.
  Low correlation = fundamentally broken conversion.

### FP16 Drift
CoreML `FLOAT16` compute precision introduces quantization noise. Most models
tolerate this. Judge by **task metrics** (PESQ for audio, WER for ASR), not
element-wise equality. If drift is unacceptable, use `op_selector` for
mixed-precision on sensitive layers.

### Stage Bisection
When overall correlation is low, bisect the pipeline to isolate which stage
diverges. The methodology from debug-notes.md:
1. Wrap individual stages as separate models
2. Feed identical inputs to both PyTorch and CoreML for each stage
3. Find the first stage where correlation drops below threshold
4. Focus debugging effort there

## Procedure

### 1. Identify what to validate

Ask the user (or infer):
- Which model? (e.g., `kokoro_decoder_har_post_10s.mlpackage`)
- Against what reference? (PyTorch eager, PyTorch traced, or another CoreML)
- What precision? (FP16 default, FP32 for debugging)
- What input? (real data preferred over random — catches preprocessing issues)

### 2. Single-model validation (most common)

For a single CoreML model against its PyTorch source:

```python
uv run python3 -c "
import coremltools as ct
import numpy as np
import torch

# --- Config ---
COREML_PATH = 'coreml/MODEL_NAME.mlpackage'
# Set compute_units to CPU_ONLY for FP32 baseline comparison
COMPUTE_UNITS = ct.ComputeUnit.CPU_ONLY

# --- Load CoreML model and inspect inputs ---
ml_model = ct.models.MLModel(COREML_PATH, compute_units=COMPUTE_UNITS)
spec = ml_model.get_spec()

print('Model inputs:')
for inp in spec.description.input:
    shape = list(inp.type.multiArrayType.shape)
    print(f'  {inp.name}: {shape}')
print('Model outputs:')
for out in spec.description.output:
    shape = list(out.type.multiArrayType.shape)
    print(f'  {out.name}: {shape}')

# --- Build matching inputs ---
np.random.seed(42)
inputs_np = {}
inputs_pt = {}
for inp in spec.description.input:
    shape = list(inp.type.multiArrayType.shape)
    arr = np.random.randn(*shape).astype(np.float32)
    inputs_np[inp.name] = arr
    inputs_pt[inp.name] = torch.from_numpy(arr)

# --- Run CoreML ---
coreml_out = ml_model.predict(inputs_np)

# --- Run PyTorch ---
# (User must fill in the PyTorch model loading and forward pass here)
# pytorch_out = ...

# --- Compare ---
for name in coreml_out:
    cml = np.array(coreml_out[name]).flatten()
    # pt = pytorch_out[name].detach().numpy().flatten()
    # corr = np.corrcoef(cml, pt)[0, 1]
    # max_err = np.max(np.abs(cml - pt))
    # print(f'{name}: correlation={corr:.6f}  max_error={max_err:.6f}')
    print(f'{name}: shape={cml.shape}  range=[{cml.min():.4f}, {cml.max():.4f}]')
"
```

### 3. Full pipeline validation (Kokoro-specific)

Compare the full Kokoro TTS pipeline output between Python and CoreML paths:

```python
uv run python3 -c "
import numpy as np
import torch

torch.manual_seed(0)

# --- Python HAR-post path (Config A reference) ---
from kokoro.pipeline import HybridTTSPipeline
pipe = HybridTTSPipeline()
text = 'Hello world, this is a test of the speech synthesis system.'
voice = 'af_heart'

# Run full pipeline
audio_a, sr = pipe(text, voice=voice, speed=1.0)
audio_a = np.array(audio_a)

# --- Compare with Config F (Swift) output ---
# Run the Swift binary and load its output WAV, or compare intermediate tensors

print(f'Config A output: {audio_a.shape}, range=[{audio_a.min():.4f}, {audio_a.max():.4f}]')
print(f'Sample rate: {sr}')
"
```

### 4. Per-stage bisection (when overall correlation is low)

If the full pipeline shows poor correlation, isolate which stage diverges.
The pattern from debug-notes.md:

```python
# For each stage (duration, f0ntrain, decoder_pre, generator):
# 1. Extract intermediate tensors from PyTorch path
# 2. Feed same tensors to CoreML model
# 3. Compare outputs
# 4. Find first divergence point

# Stage names for Kokoro pipeline:
STAGES = [
    ('duration',     'kokoro_duration_t128.mlpackage'),
    ('f0ntrain',     'kokoro_f0ntrain_t400.mlpackage'),
    ('decoder_pre',  'kokoro_decoder_pre_10s.mlpackage'),
    ('generator',    'kokoro_decoder_har_post_10s.mlpackage'),
]
```

### 5. FP16 vs FP32 comparison

When you suspect FP16 quantization is the issue:

```python
uv run python3 -c "
import coremltools as ct
import numpy as np

MODEL = 'coreml/MODEL_NAME.mlpackage'

# Same input for both
np.random.seed(42)
spec = ct.models.MLModel(MODEL).get_spec()
inputs = {}
for inp in spec.description.input:
    shape = list(inp.type.multiArrayType.shape)
    inputs[inp.name] = np.random.randn(*shape).astype(np.float32)

# FP16 (default — runs on ALL compute units including ANE)
model_fp16 = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.ALL)
out_fp16 = model_fp16.predict(inputs)

# FP32 (CPU_ONLY forces FP32 execution)
model_fp32 = ct.models.MLModel(MODEL, compute_units=ct.ComputeUnit.CPU_ONLY)
out_fp32 = model_fp32.predict(inputs)

for name in out_fp16:
    a = np.array(out_fp16[name]).flatten()
    b = np.array(out_fp32[name]).flatten()
    corr = np.corrcoef(a, b)[0, 1]
    max_err = np.max(np.abs(a - b))
    print(f'{name}: FP16 vs FP32 correlation={corr:.6f}  max_error={max_err:.6f}')
"
```

## Pass/Fail Thresholds

| Metric | Excellent | Acceptable | Investigate | Broken |
| --- | --- | --- | --- | --- |
| Pearson correlation | > 0.999 | > 0.99 | > 0.90 | < 0.90 |
| Max absolute error (FP16) | < 0.001 | < 0.01 | < 0.1 | > 0.1 |
| Max absolute error (FP32) | < 1e-5 | < 1e-4 | < 1e-3 | > 1e-3 |

For **audio/TTS models**, perceptual metrics matter more than element-wise:
- PESQ > 4.0 = transparent
- PESQ 3.5-4.0 = acceptable
- MCD < 5 dB = good alignment
- A/B listening test: ultimate ground truth

## Output Template

```text
## CoreML Validation: [model name]

### Config
- CoreML: [path] ([FP16/FP32], [compute_units])
- Reference: [PyTorch eager / traced / other CoreML]
- Input: [real data / random seed 42]

### Per-Output Results
| Output | Correlation | Max Error | Verdict |
| ------ | ----------- | --------- | ------- |
| ...    | ...         | ...       | ...     |

### Overall Verdict: [PASS / INVESTIGATE / FAIL]

### Recommendation
- [Model is numerically faithful / FP16 drift on layer X — use mixed precision / etc.]
```

## Anti-patterns

- **Comparing with random inputs only** — real data catches preprocessing
  mismatches that random noise doesn't. Always try both.
- **Judging TTS by correlation alone** — a phase-shifted waveform has low
  correlation but sounds identical. Use perceptual metrics for audio.
- **Comparing FP16 CoreML vs FP32 PyTorch and blaming "CoreML"** — first
  compare FP16 vs FP32 CoreML to isolate quantization from conversion issues.
- **Ignoring output scale** — if CoreML outputs are 1000x the PyTorch values,
  correlation will be perfect but the model is wrong. Check ranges.
- **Testing one input only** — edge cases (very short, very long, silence,
  extreme pitch) often expose issues that typical inputs don't.

## Canonical References

- CLAUDE.md Part 4: Core ML / ANE edge cases
- CLAUDE.md Part 5: Validate/Profile/Iterate
- Kokoro debug notes (bisection): `../kokoro-coreml/README/Notes/debug-notes.md`
- Test files: `test_coreml_export_verify.py`, `test_coreml_numeric_validate.py`
- Export scripts: `export_f0ntrain.py`, `export_duration.py`, `examples/export_coreml.py`
